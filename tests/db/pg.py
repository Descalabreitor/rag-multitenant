"""Helpers for SQL-level tests that talk to PostgreSQL directly."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

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


@asynccontextmanager
async def session(
    var: str, tenant: UUID | None = None, user: str | None = None
) -> AsyncIterator[asyncpg.Connection]:
    """Open a connection as the role in `var` and run one transaction on it.

    The context is set the way the tenancy helper will set it: transaction-local
    (`set_config(..., true)`), never a plain SET.
    """
    conn = await asyncpg.connect(dsn(var))
    try:
        async with conn.transaction():
            if tenant is not None:
                await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant))
            if user is not None:
                await conn.execute("SELECT set_config('app.user_sub', $1, true)", user)
            yield conn
    finally:
        await conn.close()
