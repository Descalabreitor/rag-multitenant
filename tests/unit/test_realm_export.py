"""keycloak/realm-export.json: no secrets in git, safe clients, and agreement with the seed.

The file is imported with `--import-realm`, which resolves `${VAR}` and
`${VAR:default}` placeholders from the Keycloak container's environment
(compose.yaml passes them from .env). That is how secrets stay out of the file.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from seed.corpus import TENANTS

REALM_FILE = Path(__file__).resolve().parents[2] / "keycloak" / "realm-export.json"
REALM: dict[str, Any] = json.loads(REALM_FILE.read_text(encoding="utf-8"))

PLACEHOLDER = re.compile(r"^\$\{[A-Z][A-Z0-9_]*(:[^}]*)?\}$")
SECRET_KEYS = {"secret", "password", "value", "privateKey", "certificate", "clientSecret"}

CLIENTS = {c["clientId"]: c for c in REALM["clients"]}
ORGS = {o["alias"]: o for o in REALM["organizations"]}
PEOPLE = [u for u in REALM["users"] if "serviceAccountClientId" not in u]


def _secret_values(node: object, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SECRET_KEYS and isinstance(value, str):
                found.append((f"{path}.{key}", value))
            found += _secret_values(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found += _secret_values(item, f"{path}[{i}]")
    return found


# --- No secrets in git --------------------------------------------------------


def test_every_secret_is_an_environment_placeholder() -> None:
    values = _secret_values(REALM)
    assert values, "expected the client secret and user passwords as placeholders"
    for path, value in values:
        assert PLACEHOLDER.match(value), f"{path} holds a literal value"


def test_no_signing_keys_are_exported() -> None:
    # Keycloak generates the realm keys on import. A full export puts the private
    # keys under `components`; this file must never contain them.
    assert "components" not in REALM


# --- Clients --------------------------------------------------------------------


def test_cli_is_public_with_pkce_and_no_password_grant() -> None:
    cli = CLIENTS["ragmt-cli"]
    assert cli["publicClient"] is True
    assert "secret" not in cli
    assert cli["standardFlowEnabled"] is True
    assert cli["attributes"]["pkce.code.challenge.method"] == "S256"
    assert cli["implicitFlowEnabled"] is False
    assert cli["serviceAccountsEnabled"] is False
    assert cli["directAccessGrantsEnabled"] is False
    assert cli["fullScopeAllowed"] is False
    assert all(uri.startswith("http://127.0.0.1/") for uri in cli["redirectUris"])
    assert cli["webOrigins"] == []


def test_password_grant_is_a_separate_dev_only_client_off_by_default() -> None:
    dev = CLIENTS["ragmt-dev-password"]
    # Off unless the environment turns it on at import (local development only).
    assert dev["enabled"] == "${KC_DEV_PASSWORD_CLIENT:false}"
    assert dev["name"].startswith("DEV ONLY")
    assert dev["description"].startswith("DEVELOPMENT ONLY")
    assert dev["publicClient"] is True
    assert "secret" not in dev
    assert dev["directAccessGrantsEnabled"] is True
    assert dev["standardFlowEnabled"] is False
    assert dev["implicitFlowEnabled"] is False
    assert dev["serviceAccountsEnabled"] is False
    assert dev["fullScopeAllowed"] is False
    assert dev["redirectUris"] == []
    # No other client may use the password grant.
    others = [c["clientId"] for c in REALM["clients"] if c["clientId"] != "ragmt-dev-password"]
    assert all(CLIENTS[c].get("directAccessGrantsEnabled") is False for c in others)


@pytest.mark.parametrize("client_id", ["ragmt-cli", "ragmt-dev-password"])
def test_user_tokens_carry_the_api_audience_and_organization_ids(client_id: str) -> None:
    client = CLIENTS[client_id]
    mappers = client["protocolMappers"]
    audience = [m for m in mappers if m["protocolMapper"] == "oidc-audience-mapper"]
    assert [m["config"]["included.custom.audience"] for m in audience] == ["ragmt-api"]
    assert "organization" in client["defaultClientScopes"]

    scopes = {s["name"]: s for s in REALM["clientScopes"]}
    (mapper,) = scopes["organization"]["protocolMappers"]
    assert mapper["protocolMapper"] == "oidc-organization-membership-mapper"
    assert mapper["config"]["addOrganizationId"] == "true"


def test_permsync_is_a_read_only_service_account() -> None:
    permsync = CLIENTS["ragmt-permsync"]
    assert permsync["publicClient"] is False
    assert permsync["serviceAccountsEnabled"] is True
    assert permsync["standardFlowEnabled"] is False
    assert permsync["implicitFlowEnabled"] is False
    assert permsync["directAccessGrantsEnabled"] is False
    (account,) = [u for u in REALM["users"] if u.get("serviceAccountClientId") == "ragmt-permsync"]
    assert account["clientRoles"] == {"realm-management": ["view-users", "query-groups"]}
    assert "realmRoles" not in account


# --- Organizations, users and groups --------------------------------------------


def test_organization_ids_are_the_seed_tenant_ids() -> None:
    assert {(o["id"], o["name"]) for o in ORGS.values()} == {(str(t.id), t.name) for t in TENANTS}


def test_each_user_belongs_to_exactly_one_organization() -> None:
    members = [m["username"] for o in ORGS.values() for m in o["members"]]
    assert sorted(members) == sorted(u["username"] for u in PEOPLE)


def test_top_level_groups_are_organizations() -> None:
    assert {g["name"]: g["attributes"]["organization_id"] for g in REALM["groups"]} == {
        alias: [org["id"]] for alias, org in ORGS.items()
    }


def test_users_are_only_in_groups_of_their_own_organization() -> None:
    alias_of = {m["username"]: alias for alias, o in ORGS.items() for m in o["members"]}
    for user in PEOPLE:
        for path in user["groups"]:
            assert path.startswith(f"/{alias_of[user['username']]}/"), (user["username"], path)


def test_group_memberships_match_the_seed() -> None:
    alias_of = {m["username"]: alias for alias, o in ORGS.items() for m in o["members"]}
    realm = {
        (ORGS[alias_of[u["username"]]]["id"], u["id"], path.rsplit("/", 1)[1])
        for u in PEOPLE
        for path in u["groups"]
    }
    seed = {(str(t.id), sub, group) for t in TENANTS for sub, group in t.memberships}
    assert realm == seed


def test_each_organization_has_admins_who_are_not_in_finance() -> None:
    """/<alias>/admins marks tenant admins (ADR 0008). Each tenant has one, and
    none is in finance, so tests can tell "may write" from "may read"."""
    for alias in ORGS:
        (top,) = [g for g in REALM["groups"] if g["name"] == alias]
        assert "admins" in {g["name"] for g in top["subGroups"]}
        admins = [u for u in PEOPLE if f"/{alias}/admins" in u["groups"]]
        assert len(admins) == 1, alias
        assert f"/{alias}/finance" not in admins[0]["groups"]
