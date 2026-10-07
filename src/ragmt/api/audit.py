"""`GET /audit`: the tenant's audit trail, for tenant admins only (ADR 0009).

The admin check reads the caller's own memberships on the request's app_rw
connection, like the write routes (ADR 0008), and a non-admin gets the same 404
as a path that doesn't exist: the route doesn't reveal that it is there. The
database enforces the same rule on its own: app_rw's SELECT policy on
audit_events shows rows only to the tenant's admins, and only that tenant's.

Pages are newest first, keyset-paginated by id: pass `next_before` from one
page as `before` to get the next.
"""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from ragmt import audit
from ragmt.auth.dependencies import get_principal
from ragmt.domain import Principal
from ragmt.tenancy import TenantConn
from ragmt.tenancy.admin import is_tenant_admin

router = APIRouter(prefix="/audit", tags=["audit"])

# FastAPI's own body for an unknown path, so the route looks absent to non-admins.
NOT_FOUND = "Not Found"
MAX_PAGE_SIZE = 200


async def require_audit_reader(
    principal: Annotated[Principal, Depends(get_principal)], conn: TenantConn
) -> Principal:
    """The caller, if they are a tenant admin; 404 otherwise."""
    if not await is_tenant_admin(conn, principal.sub):
        raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
    return principal


class AuditEventOut(BaseModel):
    id: int
    occurred_at: datetime
    actor_sub: str
    action: str
    chunk_ids: list[UUID]
    details: dict[str, Any]


class AuditPageOut(BaseModel):
    events: list[AuditEventOut]
    next_before: int | None


@router.get("", responses={status.HTTP_404_NOT_FOUND: {"description": "Not a tenant admin"}})
async def list_audit_events(
    _admin: Annotated[Principal, Depends(require_audit_reader)],
    conn: TenantConn,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    before: Annotated[
        int | None, Query(ge=1, description="`next_before` of the previous page")
    ] = None,
) -> AuditPageOut:
    page = await audit.list_events(conn, limit=limit, before=before)
    return AuditPageOut(
        events=[
            AuditEventOut(
                id=e.id,
                occurred_at=e.occurred_at,
                actor_sub=e.actor_sub,
                action=e.action,
                chunk_ids=list(e.chunk_ids),
                details=e.details,
            )
            for e in page.events
        ],
        next_before=page.next_before,
    )
