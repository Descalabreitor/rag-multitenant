"""permsync against PostgreSQL: the real store as app_ingest, a fake Keycloak.

Every sync goes through `SqlMembershipStore` on INGEST_DATABASE_URL, and
access is checked the way a user request reads: app_rw, `visible_chunks`. The
world (tests/leaks/conftest.py) has alice in group hr in both tenants, so a
sync that touched the wrong tenant would show up as a change in the other one.
"""

import os
from collections.abc import AsyncIterator
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ragmt.permsync.directory import Membership, TenantSnapshot
from ragmt.permsync.store import SqlMembershipStore
from ragmt.permsync.sync import Changes, PermissionSync
from tests.db.pg import session
from tests.leaks.conftest import INGEST, World, canaries, visible_chunks
from tests.permsync_helpers import FakeKeycloak, Org, group_id

pytestmark = [pytest.mark.db, pytest.mark.leaks]


@pytest.fixture
async def store() -> AsyncIterator[SqlMembershipStore]:
    url = os.environ.get(INGEST)
    if not url:
        pytest.skip(f"{INGEST} is not set")
    engine: AsyncEngine = create_async_engine(url, pool_size=1, max_overflow=0)
    try:
        yield SqlMembershipStore(engine, actor_sub="service:ragmt-permsync")
    finally:
        await engine.dispose()


def keycloak_for(world: World, *keys: str) -> tuple[FakeKeycloak, dict[str, Org]]:
    """A realm with the world's tenants `keys`, each with alice in hr, as in the database."""
    kc = FakeKeycloak()
    orgs = {}
    for key in keys:
        org = Org(world.tenants[key], f"tenant-{key}", f"Tenant {key}")
        kc.organizations.append(org)
        kc.member("alice", org, "hr")
        orgs[key] = org
    return kc, orgs


async def memberships(tenant: UUID) -> set[Membership]:
    async with session(INGEST, tenant) as conn:
        rows = await conn.fetch("SELECT user_sub, group_name FROM memberships")
    return {(row["user_sub"], row["group_name"]) for row in rows}


async def last_synced(tenant: UUID) -> datetime:
    async with session(INGEST, tenant) as conn:
        value: datetime = await conn.fetchval("SELECT max(synced_at) FROM memberships")
    return value


async def test_sync_writes_keycloak_and_a_second_run_changes_nothing(
    world: World, store: SqlMembershipStore
) -> None:
    kc, orgs = keycloak_for(world, "a")
    kc.member("bob", orgs["a"], "hr", "engineering")
    sync = PermissionSync(kc.client(), store)
    tenant = world.tenants["a"]
    expected = {("alice", "hr"), ("bob", "hr"), ("bob", "engineering")}

    first = await sync.run_cycle()
    assert first.ok
    assert first.synced[tenant] == Changes(added=frozenset(expected - {("alice", "hr")}))
    assert await memberships(tenant) == expected
    synced_at = await last_synced(tenant)

    second = await sync.run_cycle()
    assert second.synced[tenant] == Changes()
    assert await memberships(tenant) == expected
    # Same rows, but Keycloak confirmed them again.
    assert await last_synced(tenant) > synced_at


async def test_leaving_a_group_revokes_access_on_the_next_read(
    world: World, store: SqlMembershipStore
) -> None:
    kc, orgs = keycloak_for(world, "a")
    sync = PermissionSync(kc.client(), store)
    tenant = world.tenants["a"]
    await sync.run_cycle()
    assert await visible_chunks(tenant, "alice") == canaries("hr_a", "public_a")

    orgs["a"].groups["hr"].discard("alice")
    report = await sync.run_cycle()

    assert report.synced[tenant] == Changes(removed=frozenset({("alice", "hr")}))
    assert await visible_chunks(tenant, "alice") == canaries("public_a")


async def test_a_sync_for_one_tenant_cannot_touch_another(
    world: World, store: SqlMembershipStore
) -> None:
    a, b = world.tenants["a"], world.tenants["b"]
    kc, orgs = keycloak_for(world, "a")  # tenant b is not in this realm at all
    orgs["a"].groups["hr"].clear()
    sync = PermissionSync(kc.client(), store)

    await sync.run_cycle()
    # Same user and group name in b: if the delete weren't confined to a, it would go too.
    assert await memberships(a) == set()
    assert await memberships(b) == {("alice", "hr")}
    assert await visible_chunks(b, "alice") == canaries("hr_b", "public_b")

    kc.organizations.clear()
    report = await sync.run_cycle()
    assert set(report.purged) == {a}
    assert await memberships(b) == {("alice", "hr")}


async def test_a_snapshot_is_written_only_to_its_own_tenant(
    world: World, store: SqlMembershipStore
) -> None:
    a, b = world.tenants["a"], world.tenants["b"]
    snapshot = TenantSnapshot(a, "Tenant a", "a", frozenset({("mallory", "hr")}))

    await store.apply(snapshot)

    assert await memberships(a) == {("mallory", "hr")}
    assert await memberships(b) == {("alice", "hr")}
    assert canaries("hr_b") <= await visible_chunks(b, "alice")
    assert canaries("hr_b").isdisjoint(await visible_chunks(b, "mallory"))


async def test_a_keycloak_error_mid_listing_leaves_the_tenant_as_it_was(
    world: World, store: SqlMembershipStore
) -> None:
    kc, orgs = keycloak_for(world, "a", "b")
    for i in range(3):
        kc.member(f"user{i}", orgs["a"], "hr")
    kc.fail[rf"/groups/{group_id('tenant-a', 'hr')}/members\?.*first=2"] = 503
    # In the database but no longer in b's hr group: b's sync must remove it.
    orgs["b"].groups["hr"].clear()

    report = await PermissionSync(kc.client(page_size=2), store).run_cycle()

    assert report.failed == {world.tenants["a"]}
    assert await memberships(world.tenants["a"]) == {("alice", "hr")}
    assert await memberships(world.tenants["b"]) == set()


async def test_a_new_organization_gets_its_tenant_row(store: SqlMembershipStore) -> None:
    tenant = uuid4()
    kc = FakeKeycloak()
    org = Org(tenant, "fresh", "Fresh Org")
    kc.organizations.append(org)
    kc.member("zoe", org, "ops")

    await PermissionSync(kc.client(), store).run_cycle()

    async with session(INGEST, tenant) as conn:
        assert await conn.fetchval("SELECT name FROM tenants") == "Fresh Org"
    assert await memberships(tenant) == {("zoe", "ops")}


async def test_a_deleted_organization_loses_its_memberships(
    world: World, store: SqlMembershipStore
) -> None:
    kc, _ = keycloak_for(world, "a")
    sync = PermissionSync(kc.client(), store)
    tenant = world.tenants["a"]
    await sync.run_cycle()

    kc.organizations.clear()
    report = await sync.run_cycle()

    assert report.purged == {tenant: Changes(removed=frozenset({("alice", "hr")}))}
    assert await memberships(tenant) == set()
    assert await visible_chunks(tenant, "alice") == canaries("public_a")
    # app_ingest cannot delete tenants; the row stays.
    async with session(INGEST, tenant) as conn:
        assert await conn.fetchval("SELECT count(*) FROM tenants") == 1
