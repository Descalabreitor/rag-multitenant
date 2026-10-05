"""permsync without a database: the Keycloak client, the snapshot and the cycle rules.

Keycloak is `FakeKeycloak` behind httpx.MockTransport; the database is an
in-memory `FakeStore`, so these tests check what permsync decides to write.
What the SQL does with it is in tests/leaks/test_permsync.py.
"""

import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

from ragmt.permsync.directory import Membership, TenantRef, TenantSnapshot, read_tenant
from ragmt.permsync.keycloak import KeycloakAdmin, KeycloakError, keycloak_location
from ragmt.permsync.store import audit_details
from ragmt.permsync.sync import Changes, PermissionSync, diff
from tests.permsync_helpers import (
    BASE_URL,
    CLIENT_SECRET,
    REALM,
    TOKEN_LIFETIME,
    FakeKeycloak,
    Org,
    User,
    group_id,
)

A = UUID("7f66d4f3-54bf-47da-9604-0551ebdb756b")
B = UUID("17033638-c3cc-4d1e-b1fb-539941854a6e")


@dataclass
class FakeStore:
    rows: dict[UUID, set[Membership]] = field(default_factory=dict)
    names: dict[UUID, str | None] = field(default_factory=dict)
    applied: list[UUID] = field(default_factory=list)
    purged: list[UUID] = field(default_factory=list)
    broken: set[UUID] = field(default_factory=set)

    async def apply(self, snapshot: TenantSnapshot) -> Changes:
        self._check(snapshot.tenant_id)
        changes = diff(frozenset(self.rows.get(snapshot.tenant_id, set())), snapshot.memberships)
        self.rows[snapshot.tenant_id] = set(snapshot.memberships)
        self.names[snapshot.tenant_id] = snapshot.name
        self.applied.append(snapshot.tenant_id)
        return changes

    async def purge(self, tenant_id: UUID) -> Changes:
        self._check(tenant_id)
        changes = diff(frozenset(self.rows.get(tenant_id, set())), frozenset())
        self.rows[tenant_id] = set()
        self.purged.append(tenant_id)
        return changes

    def _check(self, tenant_id: UUID) -> None:
        if tenant_id in self.broken:
            raise OperationalError("INSERT ...", {}, ConnectionResetError())


def two_tenants() -> tuple[FakeKeycloak, Org, Org]:
    kc = FakeKeycloak()
    acme = Org(A, "acme", "Acme Logistics")
    umbra = Org(B, "umbra", "Umbra Biotech")
    kc.organizations = [acme, umbra]
    kc.member("alice", acme, "finance")
    kc.member("bob", acme, "engineering", "finance")
    kc.member("erin", acme)  # in the Organization, in no group
    kc.member("carol", umbra, "research")
    kc.member("dave", umbra, "finance")
    return kc, acme, umbra


ACME_ROWS = {("alice", "finance"), ("bob", "engineering"), ("bob", "finance")}
UMBRA_ROWS = {("carol", "research"), ("dave", "finance")}


async def cycle(kc: FakeKeycloak, store: FakeStore, page_size: int = 100) -> PermissionSync:
    sync = PermissionSync(kc.client(page_size=page_size), store)
    await sync.run_cycle()
    return sync


# --- the Keycloak client -------------------------------------------------------


async def test_token_is_reused_until_shortly_before_it_expires() -> None:
    kc, _, _ = two_tenants()
    now = [1000.0]
    client = kc.client(clock=lambda: now[0])

    await client.top_level_groups()
    now[0] += TOKEN_LIFETIME - 31
    await client.top_level_groups()
    assert kc.tokens_issued == 1

    now[0] += 2  # within 30 s of expiry
    await client.top_level_groups()
    assert kc.tokens_issued == 2


async def test_a_rejected_token_is_replaced_once() -> None:
    kc, _, _ = two_tenants()
    client = kc.client()
    await client.top_level_groups()
    kc.valid_tokens.clear()  # e.g. Keycloak restarted

    assert len(await client.top_level_groups()) == 2
    assert kc.tokens_issued == 2


async def test_a_persistent_401_is_an_error() -> None:
    kc, _, _ = two_tenants()
    kc.fail[r"/groups\?"] = 401
    with pytest.raises(KeycloakError, match="401"):
        await kc.client().top_level_groups()
    assert kc.tokens_issued == 2


async def test_bad_client_credentials_are_an_error_without_the_secret() -> None:
    kc, _, _ = two_tenants()
    http = httpx.AsyncClient(transport=httpx.MockTransport(kc.handle), base_url=BASE_URL)
    wrong = SecretStr("wrong-" + CLIENT_SECRET)
    client = KeycloakAdmin(http, realm=REALM, client_id="ragmt-permsync", client_secret=wrong)
    with pytest.raises(KeycloakError, match="token request: HTTP 401") as excinfo:
        await client.top_level_groups()
    assert CLIENT_SECRET not in str(excinfo.value)


async def test_every_page_is_read() -> None:
    kc, acme, _ = two_tenants()
    for i in range(5):
        kc.member(f"user{i}", acme, "finance")
    client = kc.client(page_size=2)

    members = await client.group_members(group_id("acme", "finance"))

    assert {m.id for m in members} == {"alice", "bob"} | {f"user{i}" for i in range(5)}
    pages = [r for r in kc.requests if "/members?" in r]
    assert [p.split("first=")[1].split("&")[0] for p in pages] == ["0", "2", "4", "6"]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, {"error": "boom"}),
        (403, {"error": "HTTP 403 Forbidden"}),
        (200, {"not": "a list"}),
        (200, [{"id": "x", "name": "g"}]),  # no subGroupCount: can't check completeness
        (200, "not json"),
    ],
)
async def test_unusable_responses_are_errors(status: int, body: object) -> None:
    kc, _, _ = two_tenants()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/groups"):
            if body == "not json":
                return httpx.Response(status, content=b"<html>")
            return httpx.Response(status, json=body)
        return kc.handle(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    client = KeycloakAdmin(
        http, realm=REALM, client_id="ragmt-permsync", client_secret=SecretStr(CLIENT_SECRET)
    )
    with pytest.raises(KeycloakError):
        await client.top_level_groups()


async def test_an_unreachable_keycloak_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    client = KeycloakAdmin(
        http, realm=REALM, client_id="ragmt-permsync", client_secret=SecretStr(CLIENT_SECRET)
    )
    with pytest.raises(KeycloakError, match="ConnectError"):
        await client.top_level_groups()


@pytest.mark.parametrize(
    ("issuer", "override", "expected"),
    [
        ("http://localhost:8080/realms/ragmt", None, ("http://localhost:8080", "ragmt")),
        ("https://sso.example/auth/realms/r1", None, ("https://sso.example/auth", "r1")),
        (
            "http://localhost:8080/realms/ragmt",
            "http://keycloak:8080/",
            ("http://keycloak:8080", "ragmt"),
        ),
    ],
)
def test_keycloak_location(issuer: str, override: str | None, expected: tuple[str, str]) -> None:
    assert keycloak_location(issuer, override) == expected


@pytest.mark.parametrize("issuer", ["http://localhost:8080", "http://kc/realms/", "http://kc/x/y"])
def test_keycloak_location_needs_a_realm_issuer(issuer: str) -> None:
    with pytest.raises(ValueError, match="realms"):
        keycloak_location(issuer)


# --- what a snapshot contains ----------------------------------------------------


async def test_snapshot_has_every_group_member_and_the_organization_name() -> None:
    kc, _, _ = two_tenants()
    store = FakeStore()
    await cycle(kc, store)
    assert store.rows == {A: ACME_ROWS, B: UMBRA_ROWS}
    assert store.names == {A: "Acme Logistics", B: "Umbra Biotech"}


async def test_disabled_users_and_non_members_get_no_membership() -> None:
    kc, acme, _ = two_tenants()
    kc.users["bob"].enabled = False
    # In /acme/finance, but a member of Umbra only (ADR 0005: such users are skipped).
    kc.users["mallory"] = User(organizations={B})
    acme.groups["finance"].add("mallory")
    store = FakeStore()

    await cycle(kc, store)

    assert store.rows[A] == {("alice", "finance")}


async def test_a_disabled_organization_grants_nothing() -> None:
    kc, acme, _ = two_tenants()
    acme.enabled = False
    store = FakeStore(rows={A: set(ACME_ROWS)})

    await cycle(kc, store)

    assert store.rows[A] == set()
    assert store.rows[B] == UMBRA_ROWS


async def test_a_children_listing_that_does_not_match_the_count_is_an_error() -> None:
    kc, _, _ = two_tenants()
    ref = TenantRef(A, "acme", group_id("acme"), sub_group_count=3)  # Keycloak lists 2
    with pytest.raises(KeycloakError, match="expected 3"):
        await read_tenant(kc.client(), ref)


async def test_groups_that_are_not_valid_organizations_are_ignored() -> None:
    kc, _, _ = two_tenants()
    kc.extra_groups = [
        {"id": "g1", "name": "staff", "subGroupCount": 0},  # no organization_id
        {"id": "g2", "name": "bad", "subGroupCount": 0, "attributes": {"organization_id": ["x"]}},
        {
            "id": "g3",
            "name": "two",
            "subGroupCount": 0,
            "attributes": {"organization_id": [str(uuid4()), str(uuid4())]},
        },
    ]
    store = FakeStore()
    await cycle(kc, store)
    assert set(store.applied) == {A, B}


async def test_two_groups_claiming_one_tenant_revoke_instead_of_guessing() -> None:
    kc, _, _ = two_tenants()
    store = FakeStore()
    sync = await cycle(kc, store)
    kc.extra_groups = [
        {
            "id": "g9",
            "name": "acme2",
            "subGroupCount": 0,
            "attributes": {"organization_id": [str(A)]},
        }
    ]

    await sync.run_cycle()

    assert store.rows[A] == set()
    assert store.purged == [A]


# --- what a cycle changes ----------------------------------------------------------


async def test_cycle_applies_exactly_the_difference() -> None:
    kc, _, _ = two_tenants()
    store = FakeStore(rows={A: {("alice", "finance"), ("erin", "finance")}})
    report = await PermissionSync(kc.client(), store).run_cycle()

    assert report.ok
    assert report.synced[A] == Changes(
        added=frozenset({("bob", "engineering"), ("bob", "finance")}),
        removed=frozenset({("erin", "finance")}),
    )
    assert report.synced[B] == Changes(added=frozenset(UMBRA_ROWS))


async def test_a_second_cycle_changes_nothing() -> None:
    kc, _, _ = two_tenants()
    store = FakeStore()
    sync = await cycle(kc, store)

    report = await sync.run_cycle()

    assert report.synced == {A: Changes(), B: Changes()}
    assert store.rows == {A: ACME_ROWS, B: UMBRA_ROWS}


async def test_error_mid_listing_leaves_that_tenant_untouched() -> None:
    kc, acme, _ = two_tenants()
    for i in range(3):
        kc.member(f"user{i}", acme, "finance")
    before = {("alice", "finance"), ("erin", "engineering")}
    store = FakeStore(rows={A: set(before)})
    finance = group_id("acme", "finance")
    # The first page of /acme/finance loads, the second one fails.
    kc.fail[rf"/groups/{finance}/members\?.*first=2"] = 503

    report = await PermissionSync(kc.client(page_size=2), store).run_cycle()

    assert report.failed == {A}
    assert not report.ok
    assert store.applied == [B]
    assert store.rows[A] == before


async def test_a_failed_organization_lookup_leaves_that_tenant_untouched() -> None:
    kc, _, _ = two_tenants()
    kc.fail[r"/organizations/members/dave/"] = 500
    store = FakeStore(rows={B: {("carol", "research")}})

    report = await PermissionSync(kc.client(), store).run_cycle()

    assert report.failed == {B}
    assert store.applied == [A]
    assert store.rows[B] == {("carol", "research")}


async def test_unlisted_groups_change_nothing_and_purge_nothing() -> None:
    kc, _, _ = two_tenants()
    store = FakeStore()
    sync = await cycle(kc, store)
    kc.fail[r"/groups\?"] = 500

    report = await sync.run_cycle()

    assert report.directory_failed
    assert store.applied == [A, B]  # only the first cycle's
    assert store.purged == []
    assert store.rows == {A: ACME_ROWS, B: UMBRA_ROWS}


async def test_a_deleted_organization_loses_its_memberships_once() -> None:
    kc, acme, _ = two_tenants()
    store = FakeStore()
    sync = await cycle(kc, store)
    kc.organizations.remove(acme)

    report = await sync.run_cycle()
    assert report.purged == {A: Changes(removed=frozenset(ACME_ROWS))}
    assert store.rows[A] == set()

    await sync.run_cycle()
    assert store.purged == [A]


async def test_a_database_error_in_one_tenant_does_not_stop_the_others() -> None:
    kc, _, _ = two_tenants()
    store = FakeStore(broken={A})

    report = await PermissionSync(kc.client(), store).run_cycle()

    assert report.failed == {A}
    assert report.synced == {B: Changes(added=frozenset(UMBRA_ROWS))}


async def test_a_failed_purge_is_retried() -> None:
    kc, acme, _ = two_tenants()
    store = FakeStore()
    sync = await cycle(kc, store)
    kc.organizations.remove(acme)
    store.broken.add(A)

    assert (await sync.run_cycle()).failed == {A}
    store.broken.clear()
    assert A in (await sync.run_cycle()).purged


# --- audit -----------------------------------------------------------------------


def test_audit_details_hold_counts_and_subs_only() -> None:
    changes = Changes(
        added=frozenset({("s2", "hr"), ("s1", "hr")}), removed=frozenset({("s3", "x")})
    )
    details = audit_details(changes, total=4)
    assert details == {
        "added": 2,
        "removed": 1,
        "memberships": 4,
        "added_memberships": [["s1", "hr"], ["s2", "hr"]],
        "removed_memberships": [["s3", "x"]],
    }
    json.dumps(details)
