"""Documents the caller can read. Visibility is decided by RLS, not here.

No route takes a tenant: it comes from the Principal through `TenantConn`. A
`tenant_id` in the query string or body is not declared, so FastAPI ignores it.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ragmt import documents
from ragmt.tenancy import TenantConn

router = APIRouter(prefix="/documents", tags=["documents"])

# One response for "doesn't exist" and "exists but not for you", so a caller
# can't probe for other tenants' or other users' documents (404, never 403).
NOT_FOUND = "Document not found"


class DocumentSummary(BaseModel):
    id: UUID
    title: str


class Document(BaseModel):
    id: UUID
    title: str
    source_uri: str | None
    created_at: datetime


@router.get("")
async def list_documents(conn: TenantConn) -> list[DocumentSummary]:
    found = await documents.list_documents(conn)
    return [DocumentSummary(id=doc.id, title=doc.title) for doc in found]


@router.get(
    "/{document_id}",
    responses={status.HTTP_404_NOT_FOUND: {"description": NOT_FOUND}},
)
async def get_document(document_id: UUID, conn: TenantConn) -> Document:
    doc = await documents.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)
    return Document(
        id=doc.id, title=doc.title, source_uri=doc.source_uri, created_at=doc.created_at
    )
