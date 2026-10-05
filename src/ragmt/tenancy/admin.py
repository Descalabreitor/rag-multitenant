"""Whether the caller is a tenant admin, read through app_rw (ADR 0008).

The answer comes from `memberships` (synced from Keycloak by permsync), never
from the token, and on the request's own app_rw connection, whose policy shows a
user only their own memberships. It needs no writer engine, so a caller that
fails it never reaches one.
"""

from sqlalchemy import Text, column, exists, select, table
from sqlalchemy.ext.asyncio import AsyncConnection

from ragmt.domain import TENANT_ADMIN_GROUP

memberships = table("memberships", column("user_sub", Text()), column("group_name", Text()))


async def is_tenant_admin(conn: AsyncConnection, user_sub: str) -> bool:
    """True if `user_sub` is in the tenant's admins group.

    `conn` is a `tenant_session` connection on app_rw for that same user. The
    `user_sub` condition repeats what RLS already enforces, so the query reads
    correctly on its own.
    """
    query = select(
        exists().where(
            memberships.c.user_sub == user_sub,
            memberships.c.group_name == TENANT_ADMIN_GROUP,
        )
    )
    return bool((await conn.execute(query)).scalar_one())
