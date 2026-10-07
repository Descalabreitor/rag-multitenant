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
The chat provider is built once too, but not contacted: a chat model that is
down only fails `POST /ask` (503), not startup. Both go into the `AskService`
on the app_rw engine (ADR 0009).

Run it with `uvicorn --factory ragmt.api.app:create_app`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.adapters.llm import build_chat_provider, build_embedding_provider
from ragmt.api import ask, audit, documents, health, writes
from ragmt.ask import AskService
from ragmt.retrieval import PgVectorRetriever
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
        chat = build_chat_provider(settings)
        try:
            await embedder.check()
            # Read by ragmt.api.ask.get_ask_service. Searches on app_rw only.
            app.state.ask_service = AskService(
                engine, embedder, PgVectorRetriever.from_settings(settings), chat, settings
            )
            async with open_writer(app, settings, embedder):
                yield
        finally:
            app.state.ask_service = None
            del app.state.engine
            await chat.aclose()
            await embedder.aclose()
            await engine.dispose()

    app = FastAPI(title="ragmt", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(writes.router)
    app.include_router(ask.router)
    app.include_router(audit.router)
    return app
