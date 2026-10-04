"""The database roles and extensions match what the security invariants rely on."""

import pytest

from tests.db.pg import dsn, fetchrow

pytestmark = pytest.mark.db


@pytest.mark.parametrize("var", ["DATABASE_URL", "INGEST_DATABASE_URL"])
async def test_runtime_roles_cannot_bypass_rls(var: str) -> None:
    row = await fetchrow(
        dsn(var),
        "SELECT rolsuper, rolbypassrls, rolinherit FROM pg_roles WHERE rolname = current_user",
    )
    assert row["rolsuper"] is False
    assert row["rolbypassrls"] is False
    assert row["rolinherit"] is False


async def test_migrator_is_not_superuser() -> None:
    row = await fetchrow(
        dsn("MIGRATOR_DATABASE_URL"),
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user",
    )
    assert row["rolsuper"] is False
    assert row["rolbypassrls"] is False


async def test_pgvector_supports_iterative_index_scans() -> None:
    row = await fetchrow(
        dsn("DATABASE_URL"),
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    )
    version = tuple(int(part) for part in row["extversion"].split("."))
    assert version >= (0, 8, 0)
