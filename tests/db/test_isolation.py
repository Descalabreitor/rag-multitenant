"""Isolation at the SQL level, through `tenant_session`: two tenants, three groups.

Every read here is a bare `SELECT * FROM <table>` with no WHERE clause: the
database alone decides what comes back (the phase 1 exit criterion).
"""

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.tenancy import tenant_session

pytestmark = [pytest.mark.db, pytest.mark.leaks]

TENANT_TABLES = ("tenants", "memberships", "documents", "document_acl", "chunks", "audit_events")
READABLE_TABLES = TENANT_TABLES[:-1]  # audit_events has no SELECT grant at all

# (tenant, user) -> groups. alice is in different groups in each tenant, and the
# group names are the same in both: neither may carry across.
MEMBERSHIPS: dict[tuple[str, str], set[str]] = {
    ("a", "alice"): {"hr"},
    ("a", "bob"): {"eng", "finance"},
    ("b", "alice"): {"eng"},
    ("b", "dave"): {"hr"},
}

# document name (also its chunks' content and its title) -> (tenant, ACL)
DOCUMENTS: dict[str, tuple[str, list[str]]] = {
    "a-hr": ("a", ["group:hr"]),
    "a-eng": ("a", ["group:eng"]),
    "a-finance": ("a", ["group:finance"]),
    "a-hr+eng": ("a", ["group:hr", "group:eng"]),
    "a-carol": ("a", ["user:carol"]),
    "a-all": ("a", ["tenant:*"]),
    "b-hr": ("b", ["group:hr"]),
    "b-eng": ("b", ["group:eng"]),
    "b-finance": ("b", ["group:finance"]),
    "b-all": ("b", ["tenant:*"]),
}

# What each user sees in each tenant, written out by hand rather than derived.
EXPECTED: dict[tuple[str, str], set[str]] = {
    ("a", "alice"): {"a-hr", "a-hr+eng", "a-all"},
    ("a", "bob"): {"a-eng", "a-finance", "a-hr+eng", "a-all"},
    ("a", "carol"): {"a-carol", "a-all"},
    ("a", "dave"): {"a-all"},
    ("b", "alice"): {"b-eng", "b-all"},
    ("b", "bob"): {"b-all"},
    ("b", "carol"): {"b-all"},
    ("b", "dave"): {"b-hr", "b-all"},
}

EMBEDDING = "[" + ",".join(["0.1"] * 768) + "]"


@pytest.fixture
async def tenants(writer: AsyncEngine) -> dict[str, UUID]:
    """Seed both tenants through app_ingest, as ingest and permsync will."""
    ids = {"a": uuid4(), "b": uuid4()}
    for key, tenant in ids.items():
        async with tenant_session(writer, tenant) as conn:
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant, "name": f"Tenant {key}"},
            )
            for (t, user), groups in MEMBERSHIPS.items():
                for group in groups if t == key else ():
                    await conn.execute(
                        text(
                            "INSERT INTO memberships (tenant_id, user_sub, group_name)"
                            " VALUES (:t, :u, :g)"
                        ),
                        {"t": tenant, "u": user, "g": group},
                    )
            for name, (t, acl) in DOCUMENTS.items():
                if t == key:
                    await _add_document(conn, tenant, name, acl)
    return ids


async def _add_document(conn: AsyncConnection, tenant: UUID, name: str, acl: list[str]) -> None:
    doc = (
        await conn.execute(
            text(
                "INSERT INTO documents (tenant_id, title, source_hash)"
                " VALUES (:t, :n, encode(sha256(convert_to(:n, 'UTF8')), 'hex')) RETURNING id"
            ),
            {"t": tenant, "n": name},
        )
    ).scalar_one()
    for principal in acl:
        await conn.execute(
            text(
                "INSERT INTO document_acl (tenant_id, document_id, principal) VALUES (:t, :d, :p)"
            ),
            {"t": tenant, "d": doc, "p": principal},
        )
    for ordinal in range(2):
        await conn.execute(
            text(
                "INSERT INTO chunks (tenant_id, document_id, ordinal, content, embedding)"
                " VALUES (:t, :d, :o, :c, CAST(CAST(:e AS text) AS vector))"
            ),
            {"t": tenant, "d": doc, "o": ordinal, "c": name, "e": EMBEDDING},
        )


async def _chunks(conn: AsyncConnection) -> set[str]:
    rows = (await conn.execute(text("SELECT * FROM chunks"))).mappings()
    return {row["content"] for row in rows}


async def _documents(conn: AsyncConnection) -> set[str]:
    rows = (await conn.execute(text("SELECT * FROM documents"))).mappings()
    return {row["title"] for row in rows}


async def _count(conn: AsyncConnection, table: str) -> int:
    rows = (await conn.execute(text(f"SELECT * FROM {table}"))).all()  # noqa: S608 -- constant table names
    return len(rows)


# --- who sees what ------------------------------------------------------------


@pytest.mark.parametrize(("tenant", "user"), sorted(EXPECTED))
async def test_each_user_sees_exactly_their_chunks(
    reader: AsyncEngine, tenants: dict[str, UUID], tenant: str, user: str
) -> None:
    async with tenant_session(reader, tenants[tenant], user) as conn:
        assert await _chunks(conn) == EXPECTED[tenant, user]
        assert await _documents(conn) == EXPECTED[tenant, user]


@pytest.mark.parametrize("tenant", ["a", "b"])
async def test_writer_sees_its_whole_tenant_and_nothing_else(
    writer: AsyncEngine, tenants: dict[str, UUID], tenant: str
) -> None:
    own = {name for name, (t, _) in DOCUMENTS.items() if t == tenant}
    async with tenant_session(writer, tenants[tenant]) as conn:
        assert await _chunks(conn) == own
        assert await _documents(conn) == own


async def test_reader_without_user_sees_nothing(
    reader: AsyncEngine, tenants: dict[str, UUID]
) -> None:
    async with tenant_session(reader, tenants["a"]) as conn:
        for table in ("memberships", "documents", "document_acl", "chunks"):
            assert await _count(conn, table) == 0, table


async def test_unknown_tenant_sees_nothing(reader: AsyncEngine) -> None:
    async with tenant_session(reader, uuid4(), "alice") as conn:
        for table in READABLE_TABLES:
            assert await _count(conn, table) == 0, table


# --- no context ---------------------------------------------------------------


@pytest.mark.usefixtures("tenants")
async def test_no_context_returns_zero_rows(reader: AsyncEngine, writer: AsyncEngine) -> None:
    for engine in (reader, writer):
        async with engine.connect() as conn, conn.begin():
            for table in READABLE_TABLES:
                assert await _count(conn, table) == 0, (engine.url.username, table)


async def test_context_ends_with_the_session(reader: AsyncEngine, tenants: dict[str, UUID]) -> None:
    """The next transaction on the same pooled connection starts with no context."""
    async with tenant_session(reader, tenants["a"], "alice") as conn:
        assert await _chunks(conn)
        pid = (await conn.execute(text("SELECT pg_backend_pid()"))).scalar_one()

    async with reader.connect() as conn, conn.begin():
        assert (await conn.execute(text("SELECT pg_backend_pid()"))).scalar_one() == pid
        settings = (
            await conn.execute(
                text(
                    "SELECT current_setting('app.tenant_id', true),"
                    " current_setting('app.user_sub', true)"
                )
            )
        ).one()
        assert tuple(settings) == ("", "")
        assert await _chunks(conn) == set()


async def test_session_overrides_settings_left_on_the_connection(
    reader: AsyncEngine, tenants: dict[str, UUID]
) -> None:
    """Even after a plain SET left another tenant and user behind (the bug the
    helper exists to prevent), a session sees only its own context, including
    a session with no user."""
    async with reader.connect() as conn, conn.begin():
        await conn.execute(text(f"SET app.tenant_id = '{tenants['b']}'"))
        await conn.execute(text("SET app.user_sub = 'dave'"))

    async with tenant_session(reader, tenants["a"], "carol") as conn:
        assert await _chunks(conn) == EXPECTED["a", "carol"]
    async with tenant_session(reader, tenants["a"]) as conn:
        assert await _chunks(conn) == set()


# --- app_rw cannot widen its own view -----------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "SET ROLE migrator",
        "SET ROLE app_ingest",
        "SET ROLE postgres",
        "SET SESSION AUTHORIZATION migrator",
        "SELECT set_config('role', 'app_ingest', false)",
    ],
)
async def test_reader_cannot_switch_role(
    reader: AsyncEngine, tenants: dict[str, UUID], statement: str
) -> None:
    with pytest.raises(DBAPIError, match="permission denied"):
        async with tenant_session(reader, tenants["a"], "carol") as conn:
            await conn.execute(text(statement))

    async with tenant_session(reader, tenants["a"], "carol") as conn:
        assert (await conn.execute(text("SELECT current_user"))).scalar_one() == "app_rw"
        assert await _chunks(conn) == EXPECTED["a", "carol"]


@pytest.mark.parametrize(
    "statement",
    [
        "RESET ROLE",
        "RESET SESSION AUTHORIZATION",
        "RESET ALL",
        "RESET app.tenant_id",
        "RESET app.user_sub",
        # Only app.tenant_id and app.user_sub mean anything to the policies;
        # groups and principals can't be injected through made-up settings.
        "SET app.principals = 'group:hr,group:eng,group:finance,tenant:*'",
        "SET app.groups = 'hr,eng,finance'",
        "SET app.user_sub = '*'",
        "SET app.user_sub = 'carol'' OR true --'",
        "SET app.tenant_id = '*'",
        "SET app.tenant_id = ''",
        "SET row_security = off",
    ],
)
async def test_reader_cannot_widen_with_set_or_reset(
    reader: AsyncEngine, tenants: dict[str, UUID], statement: str
) -> None:
    """After the statement, carol sees a subset of what she saw before, or the
    query fails. Either way, nothing new comes back."""
    try:
        async with tenant_session(reader, tenants["a"], "carol") as conn:
            await conn.execute(text(statement))
            assert (await conn.execute(text("SELECT current_user"))).scalar_one() == "app_rw"
            seen = await _chunks(conn) | await _documents(conn)
    except DBAPIError:
        return  # failing is closed too
    assert seen <= EXPECTED["a", "carol"]
