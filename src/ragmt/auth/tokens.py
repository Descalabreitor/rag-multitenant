"""Bearer token → verified Principal(sub, tenant_id).

Two failure classes, mapped to HTTP by `ragmt.auth.dependencies` (ADR 0006):

- InvalidTokenError (401): the token proves nothing. It is missing, malformed,
  unsigned, signed with the wrong key or algorithm, expired, or meant for
  another issuer or audience.
- NoTenantError (403): the token is genuine but does not name exactly one
  organization, so there is no tenant to scope the request to.

The `reason` on both is for server logs only. It never contains token contents.
"""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import jwt

from ragmt.auth.jwks import JwksCache, UnknownKeyError
from ragmt.domain import Principal
from ragmt.settings import SAFE_ALGORITHMS

# Clock skew tolerated on exp/nbf/iat between Keycloak and this service.
LEEWAY_SECONDS = 10
_REQUIRED_CLAIMS = ["exp", "iss", "aud", "sub"]
ORGANIZATION_CLAIM = "organization"


class InvalidTokenError(Exception):
    """The token cannot be trusted. Maps to 401."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class NoTenantError(Exception):
    """A genuine token without exactly one well-formed organization. Maps to 403."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TokenValidator:
    """Verifies Keycloak access tokens against the cached JWKS."""

    def __init__(
        self,
        jwks: JwksCache,
        *,
        issuer: str,
        audience: str,
        algorithms: Sequence[str],
    ) -> None:
        # Settings already reject these; checked again so no other caller can
        # build a validator that accepts "none" or HS* (RS256 -> HS256 confusion).
        unsafe = sorted(set(algorithms) - SAFE_ALGORITHMS)
        if not algorithms or unsafe:
            raise ValueError(f"unsafe or empty algorithm list: {unsafe}")
        self._jwks = jwks
        self._issuer = issuer
        self._audience = audience
        self._algorithms = tuple(algorithms)

    async def validate(self, token: str) -> Principal:
        claims = await self._verified_claims(token)
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub.strip():
            raise InvalidTokenError("empty sub")
        # The `groups` claim is ignored on purpose: memberships come from the database (ADR 0002).
        return Principal(sub=sub, tenant_id=tenant_from_claims(claims))

    async def _verified_claims(self, token: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise InvalidTokenError("malformed token") from None
        # Checked before the key lookup, so a token that could never verify
        # cannot trigger a JWKS refetch.
        if header.get("alg") not in self._algorithms:
            raise InvalidTokenError("algorithm not allowed")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise InvalidTokenError("missing kid")
        try:
            key = await self._jwks.get_key(kid)
        except UnknownKeyError:
            raise InvalidTokenError("unknown kid") from None
        try:
            return jwt.decode(
                token,
                key,
                algorithms=list(self._algorithms),
                issuer=self._issuer,
                audience=self._audience,
                leeway=LEEWAY_SECONDS,
                options={"require": _REQUIRED_CLAIMS},
            )
        except jwt.PyJWTError as exc:
            # The class name (ExpiredSignatureError, InvalidAudienceError, ...) is
            # enough to debug, and never carries token data.
            raise InvalidTokenError(type(exc).__name__) from None


def tenant_from_claims(claims: Mapping[str, Any]) -> UUID:
    """The tenant is the id of the single organization in the `organization` claim.

    Keycloak's Organization Membership mapper, with "Add organization id" on,
    emits `{"<alias>": {"id": "<uuid>", ...}}`. Anything else is refused:
    no claim (the `organization` scope was not granted), a list of aliases (the
    mapper lacks the id), zero or several organizations, or an id that is not a
    canonical UUID.
    """
    organizations = claims.get(ORGANIZATION_CLAIM)
    if organizations is None:
        raise NoTenantError("no organization claim")
    if not isinstance(organizations, dict):
        raise NoTenantError("organization claim is not an object keyed by alias")
    if len(organizations) != 1:
        raise NoTenantError(f"{len(organizations)} organizations, expected exactly 1")
    (organization,) = organizations.values()
    org_id = organization.get("id") if isinstance(organization, dict) else None
    if not isinstance(org_id, str):
        raise NoTenantError("organization has no id")
    try:
        tenant_id = UUID(org_id)
    except ValueError:
        raise NoTenantError("organization id is not a UUID") from None
    # UUID() also accepts braces, "urn:uuid:" and bare hex; Keycloak never sends those.
    if str(tenant_id) != org_id:
        raise NoTenantError("organization id is not a canonical UUID")
    return tenant_id
