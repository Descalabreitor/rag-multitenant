"""A fake Keycloak Admin API for permsync tests, served through httpx.MockTransport.

It answers the endpoints `ragmt.permsync.keycloak` calls, with the response
shapes Keycloak 26.0 returns, `first`/`max` pagination included. Tests edit
`organizations`, `users` and `fail` between cycles to change what it returns.
"""

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid5

import httpx
from pydantic import SecretStr

from ragmt.permsync.keycloak import KeycloakAdmin

BASE_URL = "http://keycloak.test"
REALM = "ragmt"
CLIENT_ID = "ragmt-permsync"
CLIENT_SECRET = "test-secret"  # noqa: S105 -- fake credential for the fake server
TOKEN_LIFETIME = 300

_ADMIN = f"/admin/realms/{REALM}/"
_TOKEN_PATH = f"/realms/{REALM}/protocol/openid-connect/token"
_GROUP_IDS = uuid5(UUID(int=0), "groups")


@dataclass
class Org:
    """An Organization, its top-level group and the members of each child group."""

    tenant_id: UUID
    alias: str
    name: str
    groups: dict[str, set[str]] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class User:
    enabled: bool = True
    # Tenant ids of the Organizations the user is a member of.
    organizations: set[UUID] = field(default_factory=set)


def group_id(*path: str) -> str:
    return str(uuid5(_GROUP_IDS, "/".join(path)))


@dataclass
class FakeKeycloak:
    organizations: list[Org] = field(default_factory=list)
    users: dict[str, User] = field(default_factory=dict)
    # Path regexes that answer with the given status instead of data. A key
    # with `first=N` in it only matches that page.
    fail: dict[str, int] = field(default_factory=dict)
    # Extra top-level groups, for shapes the Org model can't express.
    extra_groups: list[dict[str, Any]] = field(default_factory=list)
    tokens_issued: int = 0
    requests: list[str] = field(default_factory=list)
    # Tokens the server accepts. Clearing it makes every current token stale.
    valid_tokens: set[str] = field(default_factory=set)

    def member(self, sub: str, org: Org, *groups: str) -> None:
        """Make `sub` a member of `org` and of each of its `groups`."""
        self.users.setdefault(sub, User()).organizations.add(org.tenant_id)
        for group in groups:
            org.groups.setdefault(group, set()).add(sub)

    def client(self, page_size: int = 100, clock: Any = None) -> KeycloakAdmin:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self.handle), base_url=BASE_URL)
        kwargs: dict[str, Any] = {} if clock is None else {"clock": clock}
        return KeycloakAdmin(
            http,
            realm=REALM,
            client_id=CLIENT_ID,
            client_secret=SecretStr(CLIENT_SECRET),
            page_size=page_size,
            **kwargs,
        )

    # --- the fake server ---------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        query = request.url.query.decode()
        self.requests.append(f"{request.method} {path}?{query}")
        for pattern, status in self.fail.items():
            if re.search(pattern, f"{path}?{query}"):
                return httpx.Response(status, json={"error": "injected"})
        if path == _TOKEN_PATH:
            return self._token(request)
        if request.headers.get("Authorization", "").removeprefix("Bearer ") not in (
            self.valid_tokens
        ):
            return httpx.Response(401, json={"error": "HTTP 401 Unauthorized"})
        if not path.startswith(_ADMIN):
            return httpx.Response(404)
        return self._admin(path.removeprefix(_ADMIN), request.url.params)

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = dict(httpx.QueryParams(request.content.decode()))
        if form != {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }:
            return httpx.Response(401, json={"error": "unauthorized_client"})
        self.tokens_issued += 1
        token = f"token-{self.tokens_issued}"
        self.valid_tokens.add(token)
        return httpx.Response(
            200,
            json={"access_token": token, "expires_in": TOKEN_LIFETIME, "token_type": "Bearer"},
        )

    def _admin(self, path: str, params: httpx.QueryParams) -> httpx.Response:
        parts = path.split("/")
        match parts:
            case ["groups"]:
                return self._page(self._top_level_groups(), params)
            case ["groups", gid, "children"]:
                org = self._org_by_group(gid)
                if org is None:
                    return httpx.Response(404)
                children = [
                    _group(group_id(org.alias, name), name, f"/{org.alias}/{name}")
                    for name in sorted(org.groups)
                ]
                return self._page(children, params)
            case ["groups", gid, "members"]:
                for org in self.organizations:
                    for name, subs in org.groups.items():
                        if group_id(org.alias, name) == gid:
                            members = [
                                {
                                    "id": sub,
                                    "username": f"user-{sub}",
                                    "enabled": self._enabled(sub),
                                }
                                for sub in sorted(subs)
                            ]
                            return self._page(members, params)
                return httpx.Response(404)
            case ["organizations", "members", sub, "organizations"]:
                user = self.users.get(sub)
                if user is None:
                    return httpx.Response(404)
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": str(o.tenant_id),
                            "name": o.name,
                            "alias": o.alias,
                            "enabled": o.enabled,
                        }
                        for o in self.organizations
                        if o.tenant_id in user.organizations
                    ],
                )
        return httpx.Response(404)

    def _top_level_groups(self) -> list[dict[str, Any]]:
        groups = [
            {
                **_group(group_id(org.alias), org.alias, f"/{org.alias}", len(org.groups)),
                "attributes": {"organization_id": [str(org.tenant_id)]},
            }
            for org in self.organizations
        ]
        return groups + self.extra_groups

    def _org_by_group(self, gid: str) -> Org | None:
        return next((o for o in self.organizations if group_id(o.alias) == gid), None)

    def _enabled(self, sub: str) -> bool:
        user = self.users.get(sub)
        return user.enabled if user else True

    @staticmethod
    def _page(items: list[dict[str, Any]], params: httpx.QueryParams) -> httpx.Response:
        first = int(params.get("first", "0"))
        size = int(params.get("max", "100"))
        return httpx.Response(200, json=items[first : first + size])


def _group(gid: str, name: str, path: str, sub_group_count: int = 0) -> dict[str, Any]:
    return {
        "id": gid,
        "name": name,
        "path": path,
        "subGroupCount": sub_group_count,
        "subGroups": [],
        "attributes": {},
    }
