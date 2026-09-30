"""The database roles and extensions match what the security invariants rely on."""

import os

import asyncpg
import pytest

pytestmark = pytest.mark.db


def _dsn(var: str) -> str:
    url = os.environ.get(var)
    if not url:
        pytest.skip(f"{var} is not set")
    # asyncpg takes a plain libpq URL, not the SQLAlchemy dialect form.
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _fetchrow(dsn: str, query: str) -> asyncpg.Record:
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(query)
    finally:
        await conn.close()
    assert row is not None
    return row


async def test_app_role_cannot_bypass_rls() -> None:
    row = await _fetchrow(
        _dsn("DATABASE_URL"),
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user",
    )
    assert row["rolsuper"] is False
    assert row["rolbypassrls"] is False


async def test_migrator_is_not_superuser() -> None:
    row = await _fetchrow(
        _dsn("MIGRATOR_DATABASE_URL"),
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user",
    )
    assert row["rolsuper"] is False
    assert row["rolbypassrls"] is False


async def test_pgvector_supports_iterative_index_scans() -> None:
    row = await _fetchrow(
        _dsn("DATABASE_URL"),
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    )
    version = tuple(int(part) for part in row["extversion"].split("."))
    assert version >= (0, 8, 0)
