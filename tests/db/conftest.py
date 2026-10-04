"""SQLAlchemy engines for the runtime roles, as the application will create them."""

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


async def _engine(var: str) -> AsyncIterator[AsyncEngine]:
    url = os.environ.get(var)
    if not url:
        pytest.skip(f"{var} is not set")
    # One pooled connection, so consecutive sessions are guaranteed to reuse it:
    # that is where a leftover setting would leak.
    engine = create_async_engine(url, pool_size=1, max_overflow=0)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def reader() -> AsyncIterator[AsyncEngine]:
    """app_rw (DATABASE_URL): user requests."""
    async for engine in _engine("DATABASE_URL"):
        yield engine


@pytest.fixture
async def writer() -> AsyncIterator[AsyncEngine]:
    """app_ingest (INGEST_DATABASE_URL): ingest and permsync."""
    async for engine in _engine("INGEST_DATABASE_URL"):
        yield engine
