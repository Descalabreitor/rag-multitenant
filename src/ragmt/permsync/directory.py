"""Read Keycloak's view of each tenant: who is in which group (ADR 0005, ADR 0007).

The realm models an Organization as a top-level group whose `organization_id`
attribute holds the Organization id (= tenant id), with the tenant's groups as
its direct children: `/acme/finance`. A member of `/acme/finance` counts only if
they are an enabled user and a member of the enabled Acme Organization. Anyone
else is skipped and logged, never given a membership.

Logs carry user subs only, never usernames or emails.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from ragmt.permsync.keycloak import KeycloakAdmin, KeycloakError, Organization

log = logging.getLogger(__name__)

ORGANIZATION_ID_ATTRIBUTE = "organization_id"

# (user_sub, group_name), as stored in `memberships`.
type Membership = tuple[str, str]


@dataclass(frozen=True)
class TenantRef:
    """A top-level group that stands for an Organization."""

    tenant_id: UUID
    alias: str
    group_id: str
    sub_group_count: int


@dataclass(frozen=True)
class TenantSnapshot:
    """Every membership Keycloak has for one tenant, read without any error."""

    tenant_id: UUID
    # The Organization's name, as seen through one of its members. None when no
    # member could tell (no members in any group).
    name: str | None
    alias: str
    memberships: frozenset[Membership]


async def list_tenants(keycloak: KeycloakAdmin) -> dict[UUID, TenantRef]:
    """Every Organization group in the realm, by tenant id. Raises KeycloakError.

    Top-level groups without the attribute are not tenants and are ignored. A
    group with a malformed attribute, or two groups claiming the same tenant, is
    a realm misconfiguration: that tenant is left out, so its memberships are
    revoked rather than guessed.
    """
    refs: dict[UUID, TenantRef] = {}
    duplicated: set[UUID] = set()
    for group in await keycloak.top_level_groups():
        values = group.attributes.get(ORGANIZATION_ID_ATTRIBUTE)
        if not values:
            continue
        try:
            (value,) = values
            tenant_id = UUID(value)
        except ValueError:
            log.error("group %s has an invalid %s; ignored", group.id, ORGANIZATION_ID_ATTRIBUTE)
            continue
        if tenant_id in refs:
            duplicated.add(tenant_id)
        refs[tenant_id] = TenantRef(tenant_id, group.name, group.id, group.sub_group_count)
    for tenant_id in duplicated:
        log.error("several groups claim tenant=%s; it is not synced", tenant_id)
        del refs[tenant_id]
    return refs


async def read_tenant(
    keycloak: KeycloakAdmin,
    ref: TenantRef,
    organizations: dict[str, list[Organization]] | None = None,
) -> TenantSnapshot:
    """Read one tenant's memberships in full, or raise KeycloakError.

    `organizations` caches each user's Organizations for the rest of a cycle, so
    a user in several groups is looked up once.
    """
    cache = {} if organizations is None else organizations
    children = await keycloak.child_groups(ref.group_id)
    if len(children) != ref.sub_group_count:
        # The groups changed between the two listings, or a page went missing.
        # Either way this is not a complete picture, so it is not applied.
        raise KeycloakError(
            f"tenant={ref.tenant_id}: listed {len(children)} groups, expected {ref.sub_group_count}"
        )

    name: str | None = None
    memberships: set[Membership] = set()
    for child in children:
        if child.sub_group_count:
            log.warning(
                "tenant=%s group %s has subgroups; only direct members count",
                ref.tenant_id,
                child.name,
            )
        for member in await keycloak.group_members(child.id):
            if not member.enabled:
                log.info("tenant=%s sub=%s is disabled; skipped", ref.tenant_id, member.id)
                continue
            if member.id not in cache:
                cache[member.id] = await keycloak.user_organizations(member.id)
            organization = next((o for o in cache[member.id] if o.id == ref.tenant_id), None)
            if organization is None:
                log.warning(
                    "tenant=%s sub=%s is in group %s but not in the Organization; skipped",
                    ref.tenant_id,
                    member.id,
                    child.name,
                )
                continue
            name = organization.name
            if not organization.enabled:
                log.warning(
                    "tenant=%s Organization is disabled; sub=%s skipped", ref.tenant_id, member.id
                )
                continue
            memberships.add((member.id, child.name))
    return TenantSnapshot(ref.tenant_id, name, ref.alias, frozenset(memberships))
