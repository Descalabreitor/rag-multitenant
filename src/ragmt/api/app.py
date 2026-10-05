"""FastAPI application factory.

The API creates two database engines in its lifespan (ADR 0004, 0008):

- app_rw, on DATABASE_URL, for every request. Reads see only what the user's
  principals allow, and the admin check of the write routes runs here too.
- app_ingest, on INGEST_DATABASE_URL, owned by `ragmt.tenancy.writer` and
  reachable only by the write routes after their admin check. No GET route
  depends on it (`tests/unit/test_writer_boundary.py`).

MIGRATOR_DATABASE_URL is not even a setting. The embedding provider is built
once here and checked against EMBEDDING_DIM before the app accepts requests:
startup fails if it is unreachable or answers with vectors of another size.

Run it with `uvicorn --factory ragmt.api.app:create_app`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.adapters.llm import build_embedding_provider
from ragmt.api import documents, health, writes
from ragmt.settings import Settings, get_settings
from ragmt.tenancy.writer import open_writer


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Settings default to the environment (`get_settings()`)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_async_engine(
            settings.database_url.get_secret_value(),
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
        )
        # Read by ragmt.tenancy.get_engine; routes never touch it directly.
        app.state.engine = engine
        embedder = build_embedding_provider(settings)
        try:
            await embedder.check()
            async with open_writer(app, settings, embedder):
                yield
        finally:
            del app.state.engine
            await embedder.aclose()
            await engine.dispose()

    app = FastAPI(title="ragmt", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(writes.router)
    return app
