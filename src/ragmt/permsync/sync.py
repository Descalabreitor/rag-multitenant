"""One permsync cycle: read Keycloak, then apply each tenant on its own.

The rules that keep a sync from revoking access by accident:

- A tenant is applied only from a snapshot read without any error. If anything
  fails while reading it, that tenant is left exactly as it is until the next cycle.
- A tenant is purged (all its memberships deleted) only when a complete listing
  of the realm's groups no longer contains it. If that listing fails, nothing is
  purged and nothing is applied.
- Tenants are independent: an error in one doesn't stop the others.

Each change is logged after its transaction commits, with a UTC timestamp from
the log formatter, so the revocation window can be measured from the logs.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from ragmt.permsync.directory import (
    Membership,
    TenantRef,
    TenantSnapshot,
    list_tenants,
    read_tenant,
)
from ragmt.permsync.keycloak import KeycloakAdmin, KeycloakError, Organization

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Changes:
    added: frozenset[Membership] = frozenset()
    removed: frozenset[Membership] = frozenset()


def diff(current: frozenset[Membership], desired: frozenset[Membership]) -> Changes:
    return Changes(added=desired - current, removed=current - desired)


class MembershipStore(Protocol):
    """Where memberships live. Each call is one transaction scoped to one tenant."""

    async def apply(self, snapshot: TenantSnapshot) -> Changes:
        """Make the tenant's memberships equal the snapshot's and return what changed."""
        ...

    async def purge(self, tenant_id: UUID) -> Changes:
        """Delete every membership of a tenant that no longer exists in Keycloak."""
        ...


@dataclass
class CycleReport:
    synced: dict[UUID, Changes] = field(default_factory=dict)
    purged: dict[UUID, Changes] = field(default_factory=dict)
    failed: set[UUID] = field(default_factory=set)
    # True when the realm's groups could not be listed, so nothing was done.
    directory_failed: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.directory_failed


class PermissionSync:
    """Runs cycles and remembers which tenants it has seen, to notice deletions.

    The database can't list tenants for app_ingest (its policies need a tenant
    id first), so a tenant deleted from Keycloak is purged only if this process
    saw it in an earlier cycle. ADR 0007 covers why that gap is harmless.
    """

    def __init__(self, keycloak: KeycloakAdmin, store: MembershipStore) -> None:
        self._keycloak = keycloak
        self._store = store
        self._known: set[UUID] = set()

    async def run_cycle(self) -> CycleReport:
        report = CycleReport()
        started = time.monotonic()
        log.info("cycle started")
        try:
            refs = await list_tenants(self._keycloak)
        except KeycloakError as exc:
            report.directory_failed = True
            log.error("cycle aborted: cannot list Organization groups (%s); nothing changed", exc)
        else:
            await self._sync_all(refs, report)
        added = sum(len(c.added) for c in report.synced.values())
        removed = sum(len(c.removed) for c in (*report.synced.values(), *report.purged.values()))
        log.info(
            "cycle finished ok=%s tenants=%d failed=%d purged=%d added=%d removed=%d"
            " duration_ms=%d",
            report.ok,
            len(report.synced),
            len(report.failed),
            len(report.purged),
            added,
            removed,
            (time.monotonic() - started) * 1000,
        )
        return report

    async def _sync_all(self, refs: dict[UUID, TenantRef], report: CycleReport) -> None:
        organizations: dict[str, list[Organization]] = {}
        for ref in refs.values():
            try:
                snapshot = await read_tenant(self._keycloak, ref, organizations)
            except KeycloakError as exc:
                report.failed.add(ref.tenant_id)
                log.error("tenant=%s not synced, Keycloak read failed: %s", ref.tenant_id, exc)
                continue
            try:
                changes = await self._store.apply(snapshot)
            except (SQLAlchemyError, OSError) as exc:
                report.failed.add(ref.tenant_id)
                log.error("tenant=%s not synced, database error: %s", ref.tenant_id, exc)
                continue
            report.synced[ref.tenant_id] = changes
            _log_changes(ref.tenant_id, changes)

        unpurged: set[UUID] = set()
        for tenant_id in sorted(self._known - refs.keys()):
            log.warning("tenant=%s no longer in Keycloak; deleting its memberships", tenant_id)
            try:
                changes = await self._store.purge(tenant_id)
            except (SQLAlchemyError, OSError) as exc:
                report.failed.add(tenant_id)
                unpurged.add(tenant_id)
                log.error("tenant=%s not purged, database error: %s", tenant_id, exc)
                continue
            report.purged[tenant_id] = changes
            _log_changes(tenant_id, changes)
        # A failed purge stays known, so the next cycle tries again.
        self._known = set(refs) | unpurged


def _log_changes(tenant_id: UUID, changes: Changes) -> None:
    for sub, group in sorted(changes.removed):
        log.info("revoked tenant=%s sub=%s group=%s", tenant_id, sub, group)
    for sub, group in sorted(changes.added):
        log.info("granted tenant=%s sub=%s group=%s", tenant_id, sub, group)
    log.info(
        "tenant=%s synced added=%d removed=%d",
        tenant_id,
        len(changes.added),
        len(changes.removed),
    )
