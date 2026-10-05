"""FastAPI application factory.

The API creates exactly one database engine, on DATABASE_URL (app_rw). It never
builds an engine from INGEST_DATABASE_URL, and MIGRATOR_DATABASE_URL is not
even a setting, so no request can run with a role that sees more than the user
may read (ADR 0004).

Run it with `uvicorn --factory ragmt.api.app:create_app`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.api import documents, health
from ragmt.settings import Settings, get_settings


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
        try:
            yield
        finally:
            del app.state.engine
            await engine.dispose()

    app = FastAPI(title="ragmt", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    return app
