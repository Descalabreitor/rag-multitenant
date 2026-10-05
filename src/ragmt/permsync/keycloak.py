"""The parts of the Keycloak Admin REST API that permsync reads.

Checked against Keycloak 26.0 (the version in compose.yaml), with only the
service account roles in keycloak/realm-export.json: `view-users` and
`query-groups`. With those, `/groups`, `/groups/{id}/children`,
`/groups/{id}/members` and `/organizations/members/{user}/organizations` answer,
while `/organizations` and `/organizations/{id}/members` return 403 (ADR 0005,
ADR 0007). Nothing here writes.

Every failure, whether a transport error, a status other than 200 or a body
that doesn't parse, raises `KeycloakError`. An error never turns into an
empty list, because an empty list would revoke every membership.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

DEFAULT_PAGE_SIZE = 100
# Refresh the access token this long before it expires (or at half its lifetime,
# if that is shorter), so a token never expires halfway through a cycle.
_REFRESH_MARGIN_SECONDS = 30.0


class KeycloakError(Exception):
    """Keycloak could not be reached, refused the request or answered something unusable."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class Group(_Model):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    # Required: it is how a caller can tell that a children listing is complete.
    sub_group_count: int = Field(alias="subGroupCount", ge=0)


class Member(_Model):
    # The Keycloak user id, which is also the `sub` of the user's tokens.
    id: str = Field(min_length=1)
    enabled: bool


class Organization(_Model):
    id: UUID
    name: str = Field(min_length=1)
    enabled: bool


class _Token(_Model):
    access_token: str = Field(min_length=1)
    expires_in: float = Field(gt=0)


_GROUPS = TypeAdapter(list[Group])
_MEMBERS = TypeAdapter(list[Member])
_ORGANIZATIONS = TypeAdapter(list[Organization])


def keycloak_location(issuer: str, base_url: str | None = None) -> tuple[str, str]:
    """Return (base URL, realm) for an issuer like `http://host:8080/realms/<realm>`.

    `base_url` overrides the issuer's origin, for when permsync reaches Keycloak
    under another name than the one in the tokens (inside compose).
    """
    head, sep, realm = issuer.rstrip("/").rpartition("/realms/")
    if not sep or not realm or "/" in realm:
        raise ValueError("OIDC_ISSUER must end with /realms/<realm>")
    if base_url is None:
        parts = urlsplit(head)
        base_url = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return base_url.rstrip("/"), realm


def _segment(value: str) -> str:
    """Quote an id taken from a response before putting it in a path."""
    return quote(value, safe="")


class KeycloakAdmin:
    """Read-only Admin API client authenticated as the permsync service account.

    `http` must have Keycloak's base URL as its `base_url`. The caller owns it
    and closes it. The access token is fetched with the client-credentials grant,
    reused until shortly before it expires, and fetched again once if a request
    gets a 401.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        realm: str,
        client_id: str,
        client_secret: SecretStr,
        page_size: int = DEFAULT_PAGE_SIZE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self._http = http
        self._realm = _segment(realm)
        self._client_id = client_id
        self._client_secret = client_secret
        self._page_size = page_size
        self._clock = clock
        self._token: str | None = None
        self._refresh_at = 0.0
        self._lock = asyncio.Lock()

    # --- reads ------------------------------------------------------------------

    async def top_level_groups(self) -> list[Group]:
        return await self._paged("groups", {"briefRepresentation": "false"}, _GROUPS)

    async def child_groups(self, group_id: str) -> list[Group]:
        return await self._paged(
            f"groups/{_segment(group_id)}/children", {"briefRepresentation": "false"}, _GROUPS
        )

    async def group_members(self, group_id: str) -> list[Member]:
        return await self._paged(
            f"groups/{_segment(group_id)}/members", {"briefRepresentation": "true"}, _MEMBERS
        )

    async def user_organizations(self, user_id: str) -> list[Organization]:
        """The Organizations the user is a member of (not paginated in Keycloak 26.0)."""
        path = f"organizations/members/{_segment(user_id)}/organizations"
        return _parse(_ORGANIZATIONS, await self._get(path, {}), path)

    # --- plumbing ---------------------------------------------------------------

    async def _paged[T](
        self, path: str, params: dict[str, str], adapter: TypeAdapter[list[T]]
    ) -> list[T]:
        """Fetch every page. A failed page fails the whole listing."""
        items: list[T] = []
        first = 0
        while True:
            page_params = {**params, "first": str(first), "max": str(self._page_size)}
            page = _parse(adapter, await self._get(path, page_params), path)
            items.extend(page)
            if len(page) < self._page_size:
                return items
            first += len(page)

    async def _get(self, path: str, params: dict[str, str]) -> Any:
        url = f"admin/realms/{self._realm}/{path}"
        for attempt in range(2):
            token = await self._access_token()
            try:
                response = await self._http.get(
                    url, params=params, headers={"Authorization": f"Bearer {token}"}
                )
            except httpx.HTTPError as exc:
                raise KeycloakError(f"GET {path}: {type(exc).__name__}") from exc
            if response.status_code == httpx.codes.UNAUTHORIZED and attempt == 0:
                # Revoked or expired early (e.g. Keycloak restarted): get a new token once.
                self._token = None
                continue
            if response.status_code != httpx.codes.OK:
                raise KeycloakError(f"GET {path}: HTTP {response.status_code}")
            try:
                return response.json()
            except ValueError as exc:
                raise KeycloakError(f"GET {path}: body is not JSON") from exc
        raise KeycloakError(f"GET {path}: HTTP 401 with a fresh token")

    async def _access_token(self) -> str:
        async with self._lock:
            token = self._token
            if token is None or self._clock() >= self._refresh_at:
                token, self._refresh_at = await self._fetch_token()
                self._token = token
            return token

    async def _fetch_token(self) -> tuple[str, float]:
        """Return a new access token and the clock time at which to refresh it."""
        path = f"realms/{self._realm}/protocol/openid-connect/token"
        try:
            response = await self._http.post(
                path,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret.get_secret_value(),
                },
            )
        except httpx.HTTPError as exc:
            raise KeycloakError(f"token request: {type(exc).__name__}") from exc
        if response.status_code != httpx.codes.OK:
            # The body may echo request details; only the status is reported.
            raise KeycloakError(f"token request: HTTP {response.status_code}")
        try:
            token = _Token.model_validate_json(response.content)
        except ValidationError:
            # Not chained: the validation error would quote the response, token included.
            raise KeycloakError("token request: unexpected response") from None
        lifetime = token.expires_in
        refresh_at = self._clock() + lifetime - min(_REFRESH_MARGIN_SECONDS, lifetime / 2)
        return token.access_token, refresh_at


def _parse[T](adapter: TypeAdapter[T], data: Any, path: str) -> T:
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise KeycloakError(f"GET {path}: unexpected response shape") from exc
