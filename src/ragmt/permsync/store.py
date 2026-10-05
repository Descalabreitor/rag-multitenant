"""Memberships in PostgreSQL, written as app_ingest (INGEST_DATABASE_URL, ADR 0004).

Every call is one transaction through `tenant_session(engine, tenant_id)`, so
RLS confines it to that tenant: even a bug here cannot reach another tenant's
rows. The WHERE clauses repeat the tenant id anyway, so the intent is visible.
"""

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.permsync.directory import Membership, TenantSnapshot
from ragmt.permsync.sync import Changes, diff
from ragmt.tenancy import tenant_session

AUDIT_ACTION = "permsync"


def audit_details(changes: Changes, total: int, **extra: object) -> dict[str, Any]:
    """The audit row's `details`: counts and the changed (sub, group) pairs.

    Subs are the only personal data; usernames and emails never reach the database.
    """
    return {
        "added": len(changes.added),
        "removed": len(changes.removed),
        "memberships": total,
        "added_memberships": sorted([sub, group] for sub, group in changes.added),
        "removed_memberships": sorted([sub, group] for sub, group in changes.removed),
        **extra,
    }


class SqlMembershipStore:
    """`MembershipStore` on an engine connected as app_ingest."""

    def __init__(self, engine: AsyncEngine, actor_sub: str) -> None:
        self._engine = engine
        self._actor_sub = actor_sub

    async def apply(self, snapshot: TenantSnapshot) -> Changes:
        tenant_id = snapshot.tenant_id
        async with tenant_session(self._engine, tenant_id) as conn:
            # Upserting first also locks the tenant row until commit, so two syncs
            # of the same tenant run one after the other and can't both insert.
            # Without a name from Keycloak, an existing name is kept.
            await conn.execute(
                text(
                    "INSERT INTO tenants (id, name)"
                    " VALUES (:id, COALESCE(CAST(:name AS text), :alias))"
                    " ON CONFLICT (id) DO UPDATE"
                    " SET name = COALESCE(CAST(:name AS text), tenants.name)"
                ),
                {"id": tenant_id, "name": snapshot.name, "alias": snapshot.alias},
            )
            changes = diff(await _current(conn, tenant_id), snapshot.memberships)
            await _write(conn, tenant_id, changes)
            # synced_at says when Keycloak last confirmed each row (ADR 0002:
            # the time since the last successful sync must be monitored).
            await conn.execute(
                text("UPDATE memberships SET synced_at = now() WHERE tenant_id = :t"),
                {"t": tenant_id},
            )
            await self._audit(conn, tenant_id, audit_details(changes, len(snapshot.memberships)))
        return changes

    async def purge(self, tenant_id: UUID) -> Changes:
        async with tenant_session(self._engine, tenant_id) as conn:
            changes = diff(await _current(conn, tenant_id), frozenset())
            await _write(conn, tenant_id, changes)
            await self._audit(conn, tenant_id, audit_details(changes, 0, organization_missing=True))
        return changes

    async def _audit(self, conn: AsyncConnection, tenant_id: UUID, details: dict[str, Any]) -> None:
        await conn.execute(
            text(
                "INSERT INTO audit_events (tenant_id, actor_sub, action, details)"
                " VALUES (:t, :actor, :action, CAST(:details AS jsonb))"
            ),
            {
                "t": tenant_id,
                "actor": self._actor_sub,
                "action": AUDIT_ACTION,
                "details": json.dumps(details),
            },
        )


async def _current(conn: AsyncConnection, tenant_id: UUID) -> frozenset[Membership]:
    result = await conn.execute(
        text("SELECT user_sub, group_name FROM memberships WHERE tenant_id = :t"),
        {"t": tenant_id},
    )
    return frozenset((row.user_sub, row.group_name) for row in result)


async def _write(conn: AsyncConnection, tenant_id: UUID, changes: Changes) -> None:
    if changes.removed:
        await conn.execute(
            text(
                "DELETE FROM memberships WHERE tenant_id = :t AND user_sub = :u AND group_name = :g"
            ),
            [{"t": tenant_id, "u": sub, "g": group} for sub, group in changes.removed],
        )
    if changes.added:
        await conn.execute(
            text("INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES (:t, :u, :g)"),
            [{"t": tenant_id, "u": sub, "g": group} for sub, group in changes.added],
        )
