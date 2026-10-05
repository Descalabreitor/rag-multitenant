"""FastAPI dependencies that give a route its tenant-scoped connection.

Routes take `TenantConn` and nothing else: the engine is the app_rw one from
the app's lifespan, and the tenant and user come only from the Principal, never
from the request's path, query string or body.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.auth.dependencies import get_principal
from ragmt.domain import Principal
from ragmt.tenancy.session import tenant_session


def get_engine(request: Request) -> AsyncEngine:
    """The app_rw engine created in the app's lifespan (`ragmt.api.app`)."""
    engine = getattr(request.app.state, "engine", None)
    if not isinstance(engine, AsyncEngine):
        raise RuntimeError("no database engine: was the app started through its lifespan?")
    return engine


async def get_tenant_conn(
    principal: Annotated[Principal, Depends(get_principal)],
    engine: Annotated[AsyncEngine, Depends(get_engine)],
) -> AsyncIterator[AsyncConnection]:
    """Yield a connection in a transaction scoped to the caller's tenant and user.

    tenant_session commits when the route returns and rolls back when it raises
    (an HTTPException such as a 404 included).
    """
    async with tenant_session(engine, principal.tenant_id, principal.sub) as conn:
        yield conn


# scope="function" ends the transaction when the route function returns, before
# the response is sent: a failed commit becomes a 500 instead of a 200 for work
# that was rolled back. The default ("request") would commit after sending.
TenantConn = Annotated[AsyncConnection, Depends(get_tenant_conn, scope="function")]
