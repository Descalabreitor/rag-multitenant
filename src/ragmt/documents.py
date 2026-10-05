"""Document queries for user requests, run on a tenant_session connection.

There is no tenant or ACL filter in any of them, on purpose: the connection is
app_rw with the caller's context set, so RLS returns only the documents the user
may read (ADR 0001). The table definition leaves out `tenant_id`, so these
queries can neither filter on it nor return it.

Results are plain frozen dataclasses, not SQLAlchemy rows, so callers don't
depend on SQLAlchemy's Row typing (which changed between 2.0 and 2.1).
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Text, Uuid, column, select, table
from sqlalchemy.ext.asyncio import AsyncConnection

documents = table(
    "documents",
    column("id", Uuid()),
    column("title", Text()),
    column("source_uri", Text()),
    column("created_at", DateTime(timezone=True)),
)


@dataclass(frozen=True)
class DocumentSummary:
    id: UUID
    title: str


@dataclass(frozen=True)
class DocumentRecord:
    id: UUID
    title: str
    source_uri: str | None
    created_at: datetime


async def list_documents(conn: AsyncConnection) -> list[DocumentSummary]:
    """Id and title of every document the user can see, by title."""
    result = await conn.execute(
        select(documents.c.id, documents.c.title).order_by(documents.c.title, documents.c.id)
    )
    return [DocumentSummary(id=row.id, title=row.title) for row in result]


async def get_document(conn: AsyncConnection, document_id: UUID) -> DocumentRecord | None:
    """The document, or None if it doesn't exist or the user can't see it.

    The two cases are indistinguishable here, and must stay that way.
    """
    result = await conn.execute(
        select(
            documents.c.id, documents.c.title, documents.c.source_uri, documents.c.created_at
        ).where(documents.c.id == document_id)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return DocumentRecord(
        id=row.id, title=row.title, source_uri=row.source_uri, created_at=row.created_at
    )
