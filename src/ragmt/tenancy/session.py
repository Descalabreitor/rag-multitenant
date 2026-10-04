"""Database sessions scoped to one tenant, the only way to touch tenant data.

The context lives in transaction-local settings (`set_config(..., true)`, the same
as SET LOCAL), never a plain SET, which would stay on the pooled connection
after the request. The RLS policies read these settings; the user's group
principals are resolved from `memberships` inside the database (ADR 0002), so
callers pass the user's `sub`, never a list of principals.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


@asynccontextmanager
async def tenant_session(
    engine: AsyncEngine, tenant_id: UUID, user_sub: str | None = None
) -> AsyncIterator[AsyncConnection]:
    """Yield a connection inside a transaction with the tenant context set.

    `tenant_id` must come from the verified JWT. Pass `user_sub` for user
    requests (engine on DATABASE_URL); leave it out for writer jobs (engine on
    INGEST_DATABASE_URL). The transaction commits on success and rolls back if
    the block raises.

    Both settings are set on every call, with '' for a missing user, so nothing
    an earlier user of the pooled connection set can carry over.
    """
    if not isinstance(tenant_id, UUID):
        raise TypeError("tenant_id must be a UUID")
    if user_sub is not None and not user_sub:
        raise ValueError("user_sub must not be empty")

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            select(
                func.set_config("app.tenant_id", str(tenant_id), True),
                func.set_config("app.user_sub", user_sub or "", True),
            )
        )
        yield conn
