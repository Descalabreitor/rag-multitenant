"""RLS policies, grants and the ACL trigger, checked at the SQL level.

app_rw is the role user requests use; app_ingest is the writer (ADR 0004).
"""

from uuid import UUID, uuid4

import asyncpg
import pytest

from tests.db.pg import dsn, session
from tests.leaks.conftest import (
    EMBEDDING,
    INGEST,
    READER,
    World,
    canaries,
    canary,
    insert_chunk,
    visible_chunks,
)

pytestmark = [pytest.mark.db, pytest.mark.leaks]

MIGRATOR = "MIGRATOR_DATABASE_URL"
TABLES = ("tenants", "memberships", "documents", "document_acl", "chunks")


# --- context ------------------------------------------------------------------


@pytest.mark.parametrize("var", [READER, INGEST])
@pytest.mark.usefixtures("world")
async def test_no_context_returns_zero_rows(var: str) -> None:
    async with session(var) as conn:
        for table in TABLES:
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0, table  # noqa: S608 -- table names are constants above


async def test_context_does_not_survive_the_transaction(world: World) -> None:
    """A pooled connection reused without context sees nothing, and doesn't error.

    After SET LOCAL the setting reads as '' (not NULL) for the rest of the
    session, which the policies must treat as "no context".
    """
    conn = await asyncpg.connect(dsn(READER))
    try:
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.tenant_id', $1, true)", str(world.tenants["a"])
            )
            await conn.execute("SELECT set_config('app.user_sub', 'alice', true)")
            assert await conn.fetchval("SELECT count(*) FROM chunks") > 0
        async with conn.transaction():
            assert await conn.fetchval("SELECT count(*) FROM chunks") == 0
            assert await conn.fetchval("SELECT count(*) FROM documents") == 0
    finally:
        await conn.close()


async def test_tenant_without_user_returns_zero_rows(world: World) -> None:
    async with session(READER, world.tenants["a"]) as conn:
        for table in ("documents", "document_acl", "chunks", "memberships"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0, table  # noqa: S608 -- constant table names


# --- reads through app_rw -----------------------------------------------------


async def test_group_member_sees_group_and_tenant_wide_chunks(world: World) -> None:
    assert await visible_chunks(world.tenants["a"], "alice") == canaries("hr_a", "public_a")


async def test_user_without_groups_sees_only_tenant_wide_chunks(world: World) -> None:
    assert await visible_chunks(world.tenants["a"], "carol") == canaries("public_a")


async def test_user_principal_grants_access(world: World) -> None:
    assert await visible_chunks(world.tenants["a"], "bob") == canaries("bob_a", "public_a")


async def test_same_sub_and_group_in_other_tenant_stay_separate(world: World) -> None:
    assert await visible_chunks(world.tenants["b"], "alice") == canaries("hr_b", "public_b")


async def test_documents_and_acl_are_filtered_like_chunks(world: World) -> None:
    async with session(READER, world.tenants["a"], "carol") as conn:
        titles = {r["title"] for r in await conn.fetch("SELECT title FROM documents")}
        principals = {
            r["principal"] for r in await conn.fetch("SELECT principal FROM document_acl")
        }
        memberships = await conn.fetchval("SELECT count(*) FROM memberships")
    assert titles == {"public_a"}
    assert principals == {"tenant:*"}
    assert memberships == 0


async def test_user_sees_only_own_memberships(world: World) -> None:
    tenant = world.tenants["a"]
    async with session(INGEST, tenant) as conn:
        await conn.execute(
            "INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES ($1, 'bob', 'eng')",
            tenant,
        )
    async with session(READER, tenant, "alice") as conn:
        rows = await conn.fetch("SELECT user_sub, group_name FROM memberships")
    assert [tuple(r) for r in rows] == [("alice", "hr")]


async def test_revoked_membership_takes_effect_on_next_query(world: World) -> None:
    tenant = world.tenants["a"]
    async with session(INGEST, tenant) as conn:
        await conn.execute("DELETE FROM memberships WHERE user_sub = 'alice' AND group_name = 'hr'")
    assert await visible_chunks(tenant, "alice") == canaries("public_a")


# --- ACL trigger (ADR 0003) ---------------------------------------------------


async def chunk_acls(tenant: UUID, document: UUID) -> set[tuple[str, ...]]:
    async with session(INGEST, tenant) as conn:
        rows = await conn.fetch(
            "SELECT acl_principals FROM chunks WHERE document_id = $1", document
        )
    return {tuple(r["acl_principals"]) for r in rows}


async def test_acl_changes_propagate_to_chunks(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["hr_a"]

    async with session(INGEST, tenant) as conn:
        await conn.execute(
            "INSERT INTO document_acl (tenant_id, document_id, principal)"
            " VALUES ($1, $2, 'user:carol')",
            tenant,
            doc,
        )
    assert await chunk_acls(tenant, doc) == {("group:hr", "user:carol")}
    assert canary("hr_a") in await visible_chunks(tenant, "carol")

    async with session(INGEST, tenant) as conn:
        await conn.execute(
            "UPDATE document_acl SET principal = 'group:eng'"
            " WHERE document_id = $1 AND principal = 'group:hr'",
            doc,
        )
    assert await chunk_acls(tenant, doc) == {("group:eng", "user:carol")}
    assert canary("hr_a") not in await visible_chunks(tenant, "alice")

    async with session(INGEST, tenant) as conn:
        await conn.execute("DELETE FROM document_acl WHERE document_id = $1", doc)
    assert await chunk_acls(tenant, doc) == {()}
    assert canary("hr_a") not in await visible_chunks(tenant, "carol")


async def test_chunk_acl_cannot_be_written_directly(world: World) -> None:
    tenant, doc = world.tenants["a"], world.documents["bob_a"]
    async with session(INGEST, tenant) as conn:
        await conn.execute(
            """
            INSERT INTO chunks (tenant_id, document_id, ordinal, content, embedding,
                                acl_principals)
            VALUES ($1, $2, 99, 'CANARY-forged', $3::text::vector, '{tenant:*}')
            """,
            tenant,
            doc,
            EMBEDDING,
        )
        await conn.execute(
            "UPDATE chunks SET acl_principals = '{tenant:*}' WHERE document_id = $1", doc
        )
    assert await chunk_acls(tenant, doc) == {("user:bob",)}
    assert "CANARY-forged" not in await visible_chunks(tenant, "carol")


async def test_every_chunk_matches_its_document_acl(world: World) -> None:
    """The drift check from ADR 0003, over the whole tenant."""
    tenant = world.tenants["a"]
    async with session(INGEST, tenant) as conn:
        drifted = await conn.fetchval(
            """
            SELECT count(*) FROM chunks c
            WHERE c.acl_principals IS DISTINCT FROM ARRAY(
                SELECT a.principal FROM document_acl a
                WHERE a.document_id = c.document_id ORDER BY a.principal)
            """
        )
        total = await conn.fetchval("SELECT count(*) FROM chunks")
    assert total > 0
    assert drifted == 0


# --- writes -------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO tenants (id, name) VALUES (gen_random_uuid(), 'x')",
        "INSERT INTO memberships (tenant_id, user_sub, group_name)"
        " VALUES (current_setting('app.tenant_id')::uuid, 'alice', 'admins')",
        "INSERT INTO documents (tenant_id, title, source_hash)"
        " VALUES (current_setting('app.tenant_id')::uuid, 'x', repeat('0', 64))",
        "UPDATE chunks SET content = 'x'",
        "DELETE FROM chunks",
        "DELETE FROM document_acl",
    ],
)
async def test_reader_cannot_write_tenant_data(world: World, statement: str) -> None:
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with session(READER, world.tenants["a"], "alice") as conn:
            await conn.execute(statement)


async def test_writer_cannot_reach_other_tenant(world: World) -> None:
    a, b = world.tenants["a"], world.tenants["b"]
    async with session(INGEST, a) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks WHERE tenant_id = $1", b) == 0
        assert await conn.execute("DELETE FROM documents WHERE tenant_id = $1", b) == "DELETE 0"

    with pytest.raises(asyncpg.InsufficientPrivilegeError, match="row-level security"):
        async with session(INGEST, a) as conn:
            await conn.execute(
                "INSERT INTO documents (tenant_id, title, source_hash)"
                " VALUES ($1, 'x', repeat('0', 64))",
                b,
            )

    with pytest.raises(asyncpg.InsufficientPrivilegeError, match="row-level security"):
        async with session(INGEST, a) as conn:
            await conn.execute("UPDATE documents SET tenant_id = $1", b)


async def test_chunk_cannot_point_at_other_tenants_document(world: World) -> None:
    # An unused ordinal, so the unique key (checked before the foreign key) passes.
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        async with session(INGEST, world.tenants["a"]) as conn:
            await insert_chunk(conn, world.tenants["a"], world.documents["public_b"], "x", 99)


async def test_deleting_a_document_removes_its_chunks(world: World) -> None:
    tenant = world.tenants["a"]
    async with session(INGEST, tenant) as conn:
        await conn.execute("DELETE FROM documents WHERE id = $1", world.documents["public_a"])
    assert await visible_chunks(tenant, "carol") == set()


async def test_migrator_sees_no_tenant_rows(world: World) -> None:
    """FORCE RLS with no migrator policies: the owner is locked out of the data too."""
    async with session(MIGRATOR, world.tenants["a"], "alice") as conn:
        for table in TABLES:
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0, table  # noqa: S608 -- constant table names


# --- audit_events -------------------------------------------------------------

INSERT_AUDIT = (
    "INSERT INTO audit_events (tenant_id, actor_sub, action) VALUES ($1, $2, 'retrieval')"
)


async def test_reader_can_insert_own_audit_event(world: World) -> None:
    async with session(READER, world.tenants["a"], "alice") as conn:
        await conn.execute(INSERT_AUDIT, world.tenants["a"], "alice")


@pytest.mark.parametrize("target", ["other_actor", "other_tenant"])
async def test_reader_cannot_forge_audit_event(world: World, target: str) -> None:
    tenant, actor = world.tenants["a"], "alice"
    if target == "other_actor":
        actor = "bob"
    else:
        tenant = world.tenants["b"]
    with pytest.raises(asyncpg.InsufficientPrivilegeError, match="row-level security"):
        async with session(READER, world.tenants["a"], "alice") as conn:
            await conn.execute(INSERT_AUDIT, tenant, actor)


@pytest.mark.parametrize("var", [READER, INGEST])
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE audit_events SET action = 'x'",
        "DELETE FROM audit_events",
        "TRUNCATE audit_events",
    ],
)
async def test_audit_events_are_insert_only(world: World, var: str, statement: str) -> None:
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with session(var, world.tenants["a"], "alice") as conn:
            await conn.execute(statement)


async def test_writer_cannot_read_audit_events(world: World) -> None:
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with session(INGEST, world.tenants["a"]) as conn:
            await conn.execute("SELECT * FROM audit_events")


# Only tenant admins read audit_events through app_rw (migration b7d41c2e9f05).


async def _make_admin(tenant: UUID, sub: str) -> None:
    async with session(INGEST, tenant) as conn:
        await conn.execute(
            "INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES ($1, $2, 'admins')",
            tenant,
            sub,
        )


async def _audit_actors(tenant: UUID | None, user: str | None) -> list[str]:
    async with session(READER, tenant, user) as conn:
        rows = await conn.fetch("SELECT actor_sub FROM audit_events")
    return sorted(row["actor_sub"] for row in rows)


async def test_tenant_admin_reads_only_their_tenants_audit_events(world: World) -> None:
    a, b = world.tenants["a"], world.tenants["b"]
    for tenant in (a, b):
        for actor in ("alice", "bob"):
            async with session(READER, tenant, actor) as conn:
                await conn.execute(INSERT_AUDIT, tenant, actor)
    await _make_admin(a, "bob")

    assert await _audit_actors(a, "bob") == ["alice", "bob"]
    # Same sub, other tenant: bob isn't an admin in B, so he reads nothing there.
    assert await _audit_actors(b, "bob") == []


@pytest.mark.parametrize(
    ("tenant_key", "user"),
    [
        ("a", "alice"),  # a member of hr, not of admins
        ("a", "nobody"),  # no memberships at all
        ("a", None),  # no user set
        (None, None),  # no context at all
    ],
)
async def test_non_admins_read_no_audit_events(
    world: World, tenant_key: str | None, user: str | None
) -> None:
    a = world.tenants["a"]
    async with session(READER, a, "alice") as conn:
        await conn.execute(INSERT_AUDIT, a, "alice")
    await _make_admin(world.tenants["b"], "alice")  # an admin elsewhere changes nothing here
    tenant = None if tenant_key is None else world.tenants[tenant_key]
    assert await _audit_actors(tenant, user) == []


async def test_unknown_tenant_sees_nothing() -> None:
    assert await visible_chunks(uuid4(), "alice") == set()
