"""Phase 2 end to end: real Keycloak tokens, the real API, RLS in PostgreSQL.

Who may read what is written out by hand from seed/corpus.py, as in
tests/leaks/test_seed.py. Both tenants have a `finance` group: alice's is
Acme's and dave's is Umbra's, so the same group name must not cross tenants.
"""

import base64
import json
import os
import time
from typing import Any

import httpx
import pytest

from ragmt.auth.tokens import LEEWAY_SECONDS
from seed.corpus import ALICE, TENANTS, UMBRA
from tests.e2e.conftest import Keycloak, RealmAdmin, sync_permissions

pytestmark = [pytest.mark.e2e, pytest.mark.db]

VISIBLE = {
    "alice": {"Employee handbook", "Q3 budget", "Performance review: Alice"},
    "bob": {"Employee handbook", "Routing service runbook"},
    "erin": {"Employee handbook"},
    "carol": {"Lab safety policy", "Compound UB-7 trial results"},
    "dave": {"Lab safety policy", "Annual budget"},
}
DOCUMENT_IDS = {doc.title: doc.id for tenant in TENANTS for doc in tenant.documents}


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def titles(api: httpx.Client, token: str, **params: str) -> set[str]:
    response = api.get("/documents", headers=bearer(token), params=params)
    assert response.status_code == httpx.codes.OK, response.text
    return {item["title"] for item in response.json()}


def _b64decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def claims(token: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(_b64decode(token.split(".")[1]))
    return payload


# --- who sees what ------------------------------------------------------------------


@pytest.mark.parametrize("username", list(VISIBLE))
def test_each_user_sees_only_their_permitted_documents(
    api: httpx.Client, keycloak: Keycloak, username: str
) -> None:
    assert titles(api, keycloak.user_token(username)) == VISIBLE[username]


def test_another_tenants_document_is_not_found(api: httpx.Client, keycloak: Keycloak) -> None:
    dave = keycloak.user_token("dave")
    for title in VISIBLE["alice"]:
        response = api.get(f"/documents/{DOCUMENT_IDS[title]}", headers=bearer(dave))
        assert response.status_code == httpx.codes.NOT_FOUND, title


def test_a_tenant_id_in_the_request_is_ignored(api: httpx.Client, keycloak: Keycloak) -> None:
    alice = keycloak.user_token("alice")
    assert titles(api, alice, tenant_id=str(UMBRA)) == VISIBLE["alice"]


# --- tokens that must not get in ------------------------------------------------------


def test_a_token_with_one_signature_byte_changed_is_rejected(
    api: httpx.Client, keycloak: Keycloak
) -> None:
    token = keycloak.user_token("alice")
    assert api.get("/documents", headers=bearer(token)).status_code == httpx.codes.OK
    header, payload, signature = token.split(".")
    raw = bytearray(_b64decode(signature))
    raw[len(raw) // 2] ^= 0x01
    tampered = f"{header}.{payload}.{_b64encode(bytes(raw))}"
    assert api.get("/documents", headers=bearer(tampered)).status_code == 401


def test_a_payload_moved_to_another_organization_is_rejected(
    api: httpx.Client, keycloak: Keycloak
) -> None:
    """Alice's token re-encoded to claim Umbra, with the original signature."""
    token = keycloak.user_token("alice")
    assert api.get("/documents", headers=bearer(token)).status_code == httpx.codes.OK
    header, _, signature = token.split(".")
    forged = claims(token)
    forged["organization"] = {"umbra": {"id": str(UMBRA)}}
    payload = _b64encode(json.dumps(forged, separators=(",", ":")).encode())

    response = api.get("/documents", headers=bearer(f"{header}.{payload}.{signature}"))

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_an_expired_token_is_rejected(
    api: httpx.Client, keycloak: Keycloak, admin: RealmAdmin
) -> None:
    short_lived = {
        "clientId": "ragmt-e2e-short-lived",
        "name": "DEV ONLY - e2e test, deleted afterwards",
        "publicClient": True,
        "standardFlowEnabled": False,
        "directAccessGrantsEnabled": True,
        "fullScopeAllowed": False,
        "attributes": {"access.token.lifespan": "1"},
        "defaultClientScopes": ["basic", "profile", "organization"],
        "protocolMappers": [_audience_mapper("ragmt-api")],
    }
    with admin.temporary_client(short_lived) as client_id:
        token = keycloak.user_token("alice", client_id)
    assert claims(token)["aud"] == "ragmt-api"
    # Accepted while inside the leeway, so the only thing that changes below is the time.
    assert api.get("/documents", headers=bearer(token)).status_code == httpx.codes.OK
    # Past exp plus the validator's clock-skew allowance.
    time.sleep(max(0.0, claims(token)["exp"] + LEEWAY_SECONDS + 1 - time.time()))

    assert api.get("/documents", headers=bearer(token)).status_code == 401


def test_a_token_for_another_audience_is_rejected(
    api: httpx.Client, keycloak: Keycloak, admin: RealmAdmin
) -> None:
    other = {
        "clientId": "ragmt-e2e-other-audience",
        "name": "DEV ONLY - e2e test, deleted afterwards",
        "publicClient": True,
        "standardFlowEnabled": False,
        "directAccessGrantsEnabled": True,
        "fullScopeAllowed": False,
        "defaultClientScopes": ["basic", "profile", "organization"],
        "protocolMappers": [_audience_mapper("some-other-api")],
    }
    with admin.temporary_client(other) as client_id:
        token = keycloak.user_token("alice", client_id)
    # A genuine, unexpired token for the right user and organization; only `aud` differs.
    assert claims(token)["aud"] == "some-other-api"
    assert claims(token)["organization"]

    assert api.get("/documents", headers=bearer(token)).status_code == 401


def test_permsyncs_own_token_is_rejected(api: httpx.Client, keycloak: Keycloak) -> None:
    assert api.get("/documents", headers=bearer(keycloak.service_token())).status_code == 401


def _audience_mapper(audience: str) -> dict[str, Any]:
    return {
        "name": f"{audience} audience",
        "protocol": "openid-connect",
        "protocolMapper": "oidc-audience-mapper",
        "config": {"included.custom.audience": audience, "access.token.claim": "true"},
    }


# --- revocation ------------------------------------------------------------------------


def test_leaving_a_group_in_keycloak_revokes_access_after_one_sync(
    api: httpx.Client, keycloak: Keycloak, admin: RealmAdmin
) -> None:
    """The same token, issued before the change, loses the group's documents (ADR 0002)."""
    token = keycloak.user_token("alice")
    budget = f"/documents/{DOCUMENT_IDS['Q3 budget']}"
    assert "Q3 budget" in titles(api, token)
    assert api.get(budget, headers=bearer(token)).status_code == httpx.codes.OK

    admin.leave_group(ALICE, "/acme/finance")
    try:
        sync_permissions()

        assert titles(api, token) == VISIBLE["alice"] - {"Q3 budget"}
        assert api.get(budget, headers=bearer(token)).status_code == httpx.codes.NOT_FOUND
    finally:
        admin.join_group(ALICE, "/acme/finance")
        sync_permissions()
    assert titles(api, token) == VISIBLE["alice"]


def test_the_real_cli_client_refuses_the_password_grant(keycloak: Keycloak) -> None:
    """Correct credentials, but only the dev-only client may use them this way."""
    response = httpx.post(
        keycloak.token_url,
        data={
            "grant_type": "password",
            "client_id": "ragmt-cli",
            "username": "alice",
            "password": os.environ["KC_DEMO_USER_PASSWORD"],
        },
    )
    assert response.status_code in (400, 401)
    assert response.json()["error"] == "unauthorized_client"
