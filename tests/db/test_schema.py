"""The migrated schema matches the security invariants (catalog checks, no data).

Needs `alembic upgrade head` to have run against the database.
"""

import os

import pytest

from tests.db.pg import dsn, fetch, fetchrow

pytestmark = pytest.mark.db

TENANT_TABLES = ("tenants", "memberships", "documents", "document_acl", "chunks", "audit_events")


async def _tables() -> dict[str, dict[str, object]]:
    rows = await fetch(
        dsn("DATABASE_URL"),
        """
        SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
               pg_get_userbyid(c.relowner) AS owner
        FROM pg_class c
        WHERE c.relnamespace = 'public'::regnamespace
          AND c.relkind = 'r'
          AND c.relname = ANY($1::text[])
        """,
        list(TENANT_TABLES),
    )
    tables = {row["relname"]: dict(row) for row in rows}
    missing = set(TENANT_TABLES) - tables.keys()
    assert not missing, f"tables missing (did `alembic upgrade head` run?): {sorted(missing)}"
    return tables


@pytest.mark.parametrize("table", TENANT_TABLES)
async def test_tenant_table_has_forced_rls(table: str) -> None:
    info = (await _tables())[table]
    assert info["relrowsecurity"] is True
    # Without FORCE, the table owner would skip the policies.
    assert info["relforcerowsecurity"] is True


@pytest.mark.parametrize("table", TENANT_TABLES)
async def test_tenant_table_is_owned_by_migrator(table: str) -> None:
    assert (await _tables())[table]["owner"] == "migrator"


async def test_embedding_dimension_matches_settings() -> None:
    expected = os.environ.get("EMBEDDING_DIM")
    if not expected:
        pytest.skip("EMBEDDING_DIM is not set")
    row = await fetchrow(
        dsn("DATABASE_URL"),
        """
        SELECT format_type(atttypid, atttypmod) AS type
        FROM pg_attribute
        WHERE attrelid = 'public.chunks'::regclass AND attname = 'embedding'
        """,
    )
    assert row["type"] == f"vector({expected})"
