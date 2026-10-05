"""IngestService against PostgreSQL: writes as app_ingest, reads back as app_rw.

The service runs on the real writer engine with fake ports (tests/ingest_helpers.py).
What a reader sees is checked the way a user request would see it: through
app_rw with the user's context (`visible_chunks`), so the triggers and policies
are part of every assertion.

audit_events can't be read by any runtime role (insert-only), so the audit tests
capture the parameters of the service's own INSERTs on its engine. A call that
returns has committed them.
"""

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ragmt.ingest.service import (
    DocumentNotFoundError,
    IngestService,
    InvalidAclError,
    source_hash,
)
from ragmt.settings import Settings
from tests.db.pg import session
from tests.ingest_helpers import (
    CountingEmbeddings,
    FakeConverter,
    chunk_paragraphs,
    markdown_document,
)
from tests.leaks.conftest import INGEST, READER, World, canaries, visible_chunks

pytestmark = [pytest.mark.db, pytest.mark.leaks]

ADMIN = "admin-sub"


@dataclass
class Audit:
    tenant_id: UUID
    actor_sub: str
    action: str
    details: dict[str, Any]


@dataclass
class Harness:
    service: IngestService
    embedder: CountingEmbeddings
    audits: list[Audit] = field(default_factory=list)


def _capture_audits(engine: AsyncEngine, audits: list[Audit]) -> None:
    def before_cursor_execute(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        if statement.startswith("INSERT INTO audit_events"):
            tenant, actor, action, details = parameters
            audits.append(Audit(tenant, actor, action, json.loads(details)))

    event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    url = os.environ.get(INGEST)
    if not url:
        pytest.skip(f"{INGEST} is not set")
    settings = Settings()
    engine = create_async_engine(url, pool_size=1, max_overflow=0)
    embedder = CountingEmbeddings(settings.embedding_dim)
    h = Harness(
        IngestService(engine, FakeConverter(), chunk_paragraphs, embedder, settings), embedder
    )
    _capture_audits(engine, h.audits)
    try:
        yield h
    finally:
        await engine.dispose()


def new_document(*paragraphs: str) -> tuple[bytes, set[str]]:
    """Bytes of a Markdown document whose chunks are unique canaries."""
    tag = uuid4().hex[:8]
    contents = [f"CANARY-{tag}-{p}" for p in paragraphs]
    return markdown_document(f"Doc {tag}", *contents), set(contents)


async def writer_fetch(tenant: UUID, query: str, *args: object) -> list[Any]:
    async with session(INGEST, tenant) as conn:
        return list(await conn.fetch(query, *args))


async def acl_of(tenant: UUID, document: UUID) -> set[str]:
    rows = await writer_fetch(
        tenant, "SELECT principal FROM document_acl WHERE document_id = $1", document
    )
    return {r["principal"] for r in rows}


async def chunk_acls(tenant: UUID, document: UUID) -> set[tuple[str, ...]]:
    rows = await writer_fetch(
        tenant, "SELECT acl_principals FROM chunks WHERE document_id = $1", document
    )
    return {tuple(r["acl_principals"]) for r in rows}


async def reader_sees_document(tenant: UUID, user: str, document: UUID) -> bool:
    async with session(READER, tenant, user) as conn:
        return bool(await conn.fetchval("SELECT count(*) FROM documents WHERE id = $1", document))


# --- ACLs on ingest -------------------------------------------------------------------


async def test_default_acl_lets_only_the_uploader_read(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, contents = new_document("one", "two")

    result = await harness.service.ingest(tenant, "dave", data, "notes.md")

    assert (result.chunks, result.unchanged) == (2, False)
    assert result.title.startswith("Doc ")
    assert await visible_chunks(tenant, "dave") == contents | canaries("public_a")
    assert await visible_chunks(tenant, "alice") == canaries("hr_a", "public_a")
    assert await acl_of(tenant, result.document_id) == {"user:dave"}
    row = (
        await writer_fetch(
            tenant,
            "SELECT created_by, source_hash FROM documents WHERE id = $1",
            result.document_id,
        )
    )[0]
    assert (row["created_by"], row["source_hash"]) == ("dave", source_hash(data))


async def test_explicit_acl_replaces_the_default(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, contents = new_document("one")

    result = await harness.service.ingest(
        tenant, ADMIN, data, "hr.md", acl=["group:hr", "user:erin", "group:hr"]
    )

    assert await acl_of(tenant, result.document_id) == {"group:hr", "user:erin"}
    assert await chunk_acls(tenant, result.document_id) == {("group:hr", "user:erin")}
    assert contents <= await visible_chunks(tenant, "alice")  # through group hr
    assert contents <= await visible_chunks(tenant, "erin")
    assert not contents & await visible_chunks(tenant, ADMIN)  # the uploader is not added


@pytest.mark.parametrize(
    "acl", [[], ["admins"], ["user:"], ["group: "], ["tenant:x"], ["user:a\nb"]]
)
async def test_invalid_acl_writes_nothing(world: World, harness: Harness, acl: list[str]) -> None:
    tenant = world.tenants["a"]
    data, _ = new_document("one")
    with pytest.raises(InvalidAclError):
        await harness.service.ingest(tenant, ADMIN, data, "x.md", acl=acl)
    rows = await writer_fetch(
        tenant, "SELECT id FROM documents WHERE source_hash = $1", source_hash(data)
    )
    assert rows == []
    assert harness.embedder.document_calls == 0


# --- duplicates and replace ------------------------------------------------------------


async def test_same_bytes_twice_is_one_document(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, _ = new_document("one", "two")

    first = await harness.service.ingest(tenant, ADMIN, data, "a.md", acl=["tenant:*"])
    second = await harness.service.ingest(tenant, "other-admin", data, "b.md")

    assert second.unchanged
    assert (second.document_id, second.title, second.chunks) == (
        first.document_id,
        first.title,
        2,
    )
    assert harness.embedder.document_calls == 1  # the duplicate was never embedded
    rows = await writer_fetch(
        tenant, "SELECT id FROM documents WHERE source_hash = $1", source_hash(data)
    )
    assert [r["id"] for r in rows] == [first.document_id]
    chunk_count = await writer_fetch(
        tenant, "SELECT count(*) AS n FROM chunks WHERE document_id = $1", first.document_id
    )
    assert chunk_count[0]["n"] == 2
    # The second call's ACL default did not touch the first document's ACL.
    assert await acl_of(tenant, first.document_id) == {"tenant:*"}
    assert [a.action for a in harness.audits] == ["ingest"]


async def test_replace_swaps_chunks_and_keeps_the_acl(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    old_data, old = new_document("one", "two")
    new_data, new = new_document("three")
    doc = (
        await harness.service.ingest(tenant, ADMIN, old_data, "a.md", acl=["group:hr"])
    ).document_id

    result = await harness.service.replace(tenant, ADMIN, doc, new_data, "a.md")

    assert (result.document_id, result.chunks, result.unchanged) == (doc, 1, False)
    seen = await visible_chunks(tenant, "alice")
    assert new <= seen
    assert not old & seen
    assert await acl_of(tenant, doc) == {"group:hr"}
    assert await chunk_acls(tenant, doc) == {("group:hr",)}
    hashes = await writer_fetch(tenant, "SELECT source_hash FROM documents WHERE id = $1", doc)
    assert hashes[0]["source_hash"] == source_hash(new_data)

    again = await harness.service.replace(tenant, ADMIN, doc, new_data, "a.md")
    assert again.unchanged
    assert harness.embedder.document_calls == 2


# --- ACL changes -------------------------------------------------------------------


async def test_set_acl_moves_visibility_immediately(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, contents = new_document("one", "two")
    doc = (await harness.service.ingest(tenant, ADMIN, data, "a.md", acl=["group:hr"])).document_id
    assert contents <= await visible_chunks(tenant, "alice")

    stored = await harness.service.set_acl(tenant, ADMIN, doc, ["user:dave", "user:dave"])

    assert stored == ("user:dave",)
    assert not contents & await visible_chunks(tenant, "alice")
    assert not await reader_sees_document(tenant, "alice", doc)
    assert contents <= await visible_chunks(tenant, "dave")
    assert await chunk_acls(tenant, doc) == {("user:dave",)}
    with pytest.raises(InvalidAclError):
        await harness.service.set_acl(tenant, ADMIN, doc, [])
    assert await acl_of(tenant, doc) == {"user:dave"}


# --- deletes -------------------------------------------------------------------------


async def test_soft_delete_hides_document_and_chunks(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, contents = new_document("one")
    doc = (await harness.service.ingest(tenant, ADMIN, data, "a.md", acl=["tenant:*"])).document_id

    await harness.service.soft_delete(tenant, ADMIN, doc)

    assert await visible_chunks(tenant, "carol") == canaries("public_a")
    assert not await reader_sees_document(tenant, "carol", doc)
    assert await chunk_acls(tenant, doc) == {()}
    for call in (
        harness.service.soft_delete(tenant, ADMIN, doc),
        harness.service.set_acl(tenant, ADMIN, doc, ["tenant:*"]),
        harness.service.replace(tenant, ADMIN, doc, b"# new\n\nother", "a.md"),
    ):
        with pytest.raises(DocumentNotFoundError):
            await call
    assert await chunk_acls(tenant, doc) == {()}

    # The same file can be uploaded again as a new document.
    again = await harness.service.ingest(tenant, ADMIN, data, "a.md", acl=["tenant:*"])
    assert not again.unchanged
    assert again.document_id != doc
    assert contents <= await visible_chunks(tenant, "carol")


async def test_hard_delete_removes_the_chunks(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, _ = new_document("one", "two")
    doc = (await harness.service.ingest(tenant, ADMIN, data, "a.md")).document_id
    await harness.service.soft_delete(tenant, ADMIN, doc)  # hard delete also purges these

    await harness.service.hard_delete(tenant, ADMIN, doc)

    assert await writer_fetch(tenant, "SELECT id FROM chunks WHERE document_id = $1", doc) == []
    assert await writer_fetch(tenant, "SELECT id FROM documents WHERE id = $1", doc) == []
    with pytest.raises(DocumentNotFoundError):
        await harness.service.hard_delete(tenant, ADMIN, doc)


# --- tenant isolation ------------------------------------------------------------------


async def test_tenant_a_cannot_touch_tenant_b(world: World, harness: Harness) -> None:
    a, b = world.tenants["a"], world.tenants["b"]
    b_doc = world.documents["hr_b"]
    data, _ = new_document("one")
    calls = [
        harness.service.replace(a, ADMIN, b_doc, data, "x.md"),
        harness.service.set_acl(a, ADMIN, b_doc, ["tenant:*"]),
        harness.service.soft_delete(a, ADMIN, b_doc),
        harness.service.hard_delete(a, ADMIN, b_doc),
    ]
    for call in calls:
        with pytest.raises(DocumentNotFoundError) as caught:
            await call
        # The same error as for an id that never existed.
        assert str(caught.value) == str(DocumentNotFoundError(b_doc))

    assert await visible_chunks(b, "alice") == canaries("hr_b", "public_b")
    assert await acl_of(b, b_doc) == {"group:hr"}
    rows = await writer_fetch(b, "SELECT deleted_at FROM documents WHERE id = $1", b_doc)
    assert rows[0]["deleted_at"] is None
    assert harness.embedder.document_calls == 0  # refused before any embedding
    assert harness.audits == []


async def test_unknown_document_id_is_not_found(world: World, harness: Harness) -> None:
    with pytest.raises(DocumentNotFoundError):
        await harness.service.set_acl(world.tenants["a"], ADMIN, uuid4(), ["tenant:*"])


async def test_same_bytes_in_two_tenants_are_two_documents(world: World, harness: Harness) -> None:
    data, contents = new_document("one")
    in_a = await harness.service.ingest(world.tenants["a"], ADMIN, data, "a.md", acl=["tenant:*"])
    in_b = await harness.service.ingest(world.tenants["b"], ADMIN, data, "a.md", acl=["tenant:*"])
    assert not in_b.unchanged
    assert in_a.document_id != in_b.document_id
    assert contents <= await visible_chunks(world.tenants["b"], "carol")


# --- audit ----------------------------------------------------------------------------


async def test_every_write_is_audited_without_content(world: World, harness: Harness) -> None:
    tenant = world.tenants["a"]
    data, contents = new_document("one", "two")
    new_data, new_contents = new_document("three")

    result = await harness.service.ingest(tenant, ADMIN, data, "secret-name.md")
    doc = result.document_id
    await harness.service.replace(tenant, ADMIN, doc, new_data, "secret-name.md")
    await harness.service.set_acl(tenant, ADMIN, doc, ["group:hr", "tenant:*"])
    await harness.service.soft_delete(tenant, ADMIN, doc)
    await harness.service.hard_delete(tenant, ADMIN, doc)

    assert [a.action for a in harness.audits] == [
        "ingest",
        "replace",
        "acl_change",
        "soft_delete",
        "hard_delete",
    ]
    assert all(a.tenant_id == tenant and a.actor_sub == ADMIN for a in harness.audits)
    assert all(a.details["document_id"] == str(doc) for a in harness.audits)

    ingest, replace, acl_change, soft, hard = (a.details for a in harness.audits)
    assert (ingest["chunks"], ingest["source_hash"]) == (2, source_hash(data))
    assert ingest["principals"] == [f"user:{ADMIN}"]
    assert (replace["chunks"], replace["previous_chunks"]) == (1, 2)
    assert replace["source_hash"] == source_hash(new_data)
    assert acl_change["old_principals"] == [f"user:{ADMIN}"]
    assert acl_change["new_principals"] == ["group:hr", "tenant:*"]
    assert soft["old_principals"] == ["group:hr", "tenant:*"]
    assert hard["old_principals"] == []  # removed by the soft delete

    dumped = json.dumps([a.details for a in harness.audits])
    for secret in (*contents, *new_contents, result.title, "secret-name"):
        assert secret not in dumped
