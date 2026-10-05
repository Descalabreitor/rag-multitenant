"""Soft-deleted documents and the source_hash key, checked at the SQL level (ADR 0008).

Setting documents.deleted_at must hide the document and its chunks from app_rw
without a join in the chunks policy: the ACL rows go, the chunk trigger yields
an empty acl_principals, and the documents policy filters deleted_at.
"""

from uuid import UUID

import asyncpg
import pytest

from tests.db.pg import dsn, fetchrow, session
from tests.leaks.conftest import (
    INGEST,
    READER,
    World,
    canaries,
    canary,
    insert_chunk,
    source_hash,
    visible_chunks,
)

pytestmark = [pytest.mark.db, pytest.mark.leaks]


async def soft_delete(tenant: UUID, document: UUID) -> None:
    async with session(INGEST, tenant) as conn:
        await conn.execute("UPDATE documents SET deleted_at = now() WHERE id = $1", document)


async def visible_titles(tenant: UUID, user: str) -> set[str]:
    async with session(READER, tenant, user) as conn:
        rows = await conn.fetch("SELECT title FROM documents")
    return {row["title"] for row in rows}


async def writer_chunk_acls(tenant: UUID, document: UUID) -> set[tuple[str, ...]]:
    async with session(INGEST, tenant) as conn:
        rows = await conn.fetch(
            "SELECT acl_principals FROM chunks WHERE document_id = $1", document
        )
    return {tuple(r["acl_principals"]) for r in rows}


# --- visibility ---------------------------------------------------------------


async def test_deleted_document_and_its_chunks_disappear_for_readers(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["public_a"]
    await soft_delete(tenant, doc)

    # public_a was tenant-wide: nobody may see it any more, not even through the ACL.
    assert await visible_chunks(tenant, "alice") == canaries("hr_a")
    assert await visible_chunks(tenant, "carol") == set()
    assert await visible_titles(tenant, "alice") == {"hr_a"}
    async with session(READER, tenant, "alice") as conn:
        acl = await conn.fetchval("SELECT count(*) FROM document_acl WHERE document_id = $1", doc)
        by_id = await conn.fetchval("SELECT count(*) FROM documents WHERE id = $1", doc)
    assert (acl, by_id) == (0, 0)


async def test_the_writer_keeps_the_document_with_empty_chunk_acls(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["hr_a"]
    await soft_delete(tenant, doc)
    async with session(INGEST, tenant) as conn:
        deleted_at = await conn.fetchval("SELECT deleted_at FROM documents WHERE id = $1", doc)
        acl = await conn.fetchval("SELECT count(*) FROM document_acl WHERE document_id = $1", doc)
    assert deleted_at is not None
    assert acl == 0
    assert await writer_chunk_acls(tenant, doc) == {()}


async def test_acl_added_to_a_deleted_document_does_not_reach_its_chunks(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["hr_a"]
    await soft_delete(tenant, doc)
    async with session(INGEST, tenant) as conn:
        await conn.execute(
            "INSERT INTO document_acl (tenant_id, document_id, principal)"
            " VALUES ($1, $2, 'tenant:*')",
            tenant,
            doc,
        )
    assert await writer_chunk_acls(tenant, doc) == {()}
    assert canary("hr_a") not in await visible_chunks(tenant, "carol")
    assert "hr_a" not in await visible_titles(tenant, "carol")


async def test_chunk_written_to_a_deleted_document_gets_an_empty_acl(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["public_a"]
    await soft_delete(tenant, doc)
    async with session(INGEST, tenant) as conn:
        await insert_chunk(conn, tenant, doc, "CANARY-late-chunk", 7)
        await conn.execute("UPDATE chunks SET content = content WHERE document_id = $1", doc)
    assert await writer_chunk_acls(tenant, doc) == {()}
    assert "CANARY-late-chunk" not in await visible_chunks(tenant, "carol")


async def test_restoring_a_document_brings_no_access_back(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["public_a"]
    await soft_delete(tenant, doc)
    async with session(INGEST, tenant) as conn:
        await conn.execute("UPDATE documents SET deleted_at = NULL WHERE id = $1", doc)
    assert await visible_chunks(tenant, "carol") == set()

    async with session(INGEST, tenant) as conn:
        await conn.execute(
            "INSERT INTO document_acl (tenant_id, document_id, principal)"
            " VALUES ($1, $2, 'tenant:*')",
            tenant,
            doc,
        )
    assert await visible_chunks(tenant, "carol") == canaries("public_a")


async def test_deleting_in_one_tenant_leaves_the_other_alone(world: World) -> None:
    await soft_delete(world.tenants["a"], world.documents["hr_a"])
    assert await visible_chunks(world.tenants["b"], "alice") == canaries("hr_b", "public_b")


async def test_reader_cannot_delete_or_restore(world: World) -> None:
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with session(READER, world.tenants["a"], "alice") as conn:
            await conn.execute("UPDATE documents SET deleted_at = now()")


async def test_chunks_policy_has_no_join() -> None:
    """ADR 0003: the chunks policy stays a per-row check, deleted documents or not."""
    row = await fetchrow(
        dsn(READER),
        "SELECT qual FROM pg_policies WHERE tablename = 'chunks' AND policyname = $1",
        "chunks_app_rw_select",
    )
    assert "documents" not in row["qual"]
    assert "document_acl" not in row["qual"]


# --- source_hash --------------------------------------------------------------


async def _insert_document(conn: asyncpg.Connection, tenant: UUID, hash_: str) -> UUID:
    doc: UUID = await conn.fetchval(
        "INSERT INTO documents (tenant_id, title, source_hash) VALUES ($1, 'dup', $2) RETURNING id",
        tenant,
        hash_,
    )
    return doc


async def test_source_hash_is_unique_among_live_documents(world: World) -> None:
    tenant = world.tenants["a"]
    with pytest.raises(asyncpg.UniqueViolationError):
        async with session(INGEST, tenant) as conn:
            await _insert_document(conn, tenant, source_hash("hr_a"))

    # The same file may come back once the first copy is deleted.
    await soft_delete(tenant, world.documents["hr_a"])
    async with session(INGEST, tenant) as conn:
        await _insert_document(conn, tenant, source_hash("hr_a"))


async def test_source_hash_is_unique_per_tenant_only(world: World) -> None:
    # hr_b's hash already exists in tenant b; tenant a may hold the same file.
    tenant = world.tenants["a"]
    async with session(INGEST, tenant) as conn:
        await _insert_document(conn, tenant, source_hash("hr_b"))


@pytest.mark.parametrize("value", ["", "abc", "A" * 64, "legacy"])
async def test_source_hash_must_be_sha256_hex(world: World, value: str) -> None:
    tenant = world.tenants["a"]
    with pytest.raises(asyncpg.CheckViolationError):
        async with session(INGEST, tenant) as conn:
            await _insert_document(conn, tenant, value)


async def test_source_hash_is_required(world: World) -> None:
    tenant = world.tenants["a"]
    with pytest.raises(asyncpg.NotNullViolationError):
        async with session(INGEST, tenant) as conn:
            await conn.execute("INSERT INTO documents (tenant_id, title) VALUES ($1, 'x')", tenant)
