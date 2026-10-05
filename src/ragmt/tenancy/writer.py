"""The API's only app_ingest engine, and the write service built on it (ADR 0004, 0008).

app_ingest sees its whole tenant, so this engine must never serve a read. It is
created in the app's lifespan (`open_writer`) and reachable only through
`get_ingest_service`, which only the write routes depend on, and only after
their admin check (`ragmt.api.writes`). Two tests hold that line:
`tests/unit/test_writer_boundary.py` checks that no other module under `src/`
reads INGEST_DATABASE_URL, and that no GET route depends on `get_ingest_service`.

The service is stored on the app state, not the engine, so code that reaches
for the state gets document writes that always take a tenant and an actor,
never a connection.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Request
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.domain import EmbeddingProvider, Principal
from ragmt.ingest.chunking import chunk
from ragmt.ingest.convert import UploadConverter
from ragmt.ingest.service import IngestResult, IngestService
from ragmt.settings import Settings

_STATE_KEY = "ingest_service"


@asynccontextmanager
async def open_writer(
    app: FastAPI, settings: Settings, embedder: EmbeddingProvider
) -> AsyncIterator[None]:
    """Create the app_ingest engine and the IngestService for the app's lifetime.

    Creating the engine doesn't connect. The pool reuses the app_rw limits: a
    write holds a writer connection only for its short transactions, not while
    it converts or embeds.
    """
    engine = create_async_engine(
        settings.ingest_database_url.get_secret_value(),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
    )
    try:
        service = IngestService(
            engine, UploadConverter(settings.ingest_max_bytes), chunk, embedder, settings
        )
        setattr(app.state, _STATE_KEY, service)
        try:
            yield
        finally:
            delattr(app.state, _STATE_KEY)
    finally:
        await engine.dispose()


def get_ingest_service(request: Request) -> IngestService:
    """The writer-engine dependency. Only write routes may depend on it, after the admin check."""
    service = getattr(request.app.state, _STATE_KEY, None)
    if not isinstance(service, IngestService):
        raise RuntimeError("no ingest service: was the app started through its lifespan?")
    return service


IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]


@dataclass(frozen=True, slots=True)
class TenantWriter:
    """Document writes bound to one verified admin: the tenant and the actor come
    from `admin`, so a route can't pass any other."""

    service: IngestService
    admin: Principal

    @property
    def max_bytes(self) -> int:
        return self.service.max_bytes

    async def ingest(self, data: bytes, filename: str, acl: list[str] | None) -> IngestResult:
        return await self.service.ingest(self.admin.tenant_id, self.admin.sub, data, filename, acl)

    async def set_acl(self, document_id: UUID, principals: list[str]) -> tuple[str, ...]:
        return await self.service.set_acl(
            self.admin.tenant_id, self.admin.sub, document_id, principals
        )

    async def soft_delete(self, document_id: UUID) -> None:
        await self.service.soft_delete(self.admin.tenant_id, self.admin.sub, document_id)

    async def hard_delete(self, document_id: UUID) -> None:
        await self.service.hard_delete(self.admin.tenant_id, self.admin.sub, document_id)
