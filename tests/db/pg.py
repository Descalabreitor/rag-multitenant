"""Helpers for SQL-level tests that talk to PostgreSQL directly."""

import os

import asyncpg
import pytest


def dsn(var: str) -> str:
    """Return the URL in `var` as a libpq DSN, or skip the test if it is not set."""
    url = os.environ.get(var)
    if not url:
        pytest.skip(f"{var} is not set")
    # asyncpg takes a plain libpq URL, not the SQLAlchemy dialect form.
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def fetch(dsn: str, query: str, *args: object) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetch(query, *args)
    finally:
        await conn.close()


async def fetchrow(dsn: str, query: str, *args: object) -> asyncpg.Record:
    rows = await fetch(dsn, query, *args)
    assert len(rows) == 1
    return rows[0]
