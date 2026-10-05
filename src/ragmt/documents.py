"""Document queries for user requests, run on a tenant_session connection.

There is no tenant or ACL filter in any of them, on purpose: the connection is
app_rw with the caller's context set, so RLS returns only the documents the user
may read (ADR 0001). The table definition leaves out `tenant_id`, so these
queries can neither filter on it nor return it.
"""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Row, Text, Uuid, column, select, table
from sqlalchemy.ext.asyncio import AsyncConnection

documents = table(
    "documents",
    column("id", Uuid()),
    column("title", Text()),
    column("source_uri", Text()),
    column("created_at", DateTime(timezone=True)),
)


async def list_documents(conn: AsyncConnection) -> Sequence[Row[tuple[UUID, str]]]:
    """Id and title of every document the user can see, by title."""
    result = await conn.execute(
        select(documents.c.id, documents.c.title).order_by(documents.c.title, documents.c.id)
    )
    return result.all()


async def get_document(
    conn: AsyncConnection, document_id: UUID
) -> Row[tuple[UUID, str, str | None, datetime]] | None:
    """The document, or None if it doesn't exist or the user can't see it.

    The two cases are indistinguishable here, and must stay that way.
    """
    result = await conn.execute(
        select(
            documents.c.id, documents.c.title, documents.c.source_uri, documents.c.created_at
        ).where(documents.c.id == document_id)
    )
    return result.one_or_none()
