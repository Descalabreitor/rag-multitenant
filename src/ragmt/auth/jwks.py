"""The issuer's signing keys (JWKS), fetched with httpx and cached in memory.

PyJWKClient is not used: it fetches with urllib, which would block the event loop.

Fetches are rate-limited. Outside the first one, at most one HTTP request goes to
the issuer per `min_refetch_interval_seconds`, whatever the reason (TTL expiry,
unknown `kid`, an earlier failure). A flood of tokens with forged `kid`s therefore
costs Keycloak one request every 30 seconds, not one per token.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)

# Only asymmetric keys can verify a token here. An "oct" (symmetric) entry in the
# JWKS is skipped even if the issuer publishes one.
_ASYMMETRIC_KEY_TYPES = frozenset({"RSA", "EC", "OKP"})
_FETCH_TIMEOUT_SECONDS = 5.0


class JwksUnavailableError(Exception):
    """The JWKS could not be fetched or held no usable key. Not the token's fault."""


class UnknownKeyError(Exception):
    """No signing key has the token's `kid`, and a refetch was not allowed or didn't help."""


def keycloak_jwks_url(issuer: str) -> str:
    """Keycloak publishes a realm's keys at a fixed path under the issuer URL."""
    return f"{issuer}/protocol/openid-connect/certs"


class JwksCache:
    """Signing keys by `kid`, refreshed every `ttl_seconds`."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        ttl_seconds: float,
        min_refetch_interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._client = client
        self._url = url
        self._ttl = ttl_seconds
        # A TTL shorter than the interval would leave the cache expired with no way to refresh.
        self._min_refetch_interval = min(min_refetch_interval_seconds, ttl_seconds)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at: float | None = None  # last successful fetch
        self._attempted_at: float | None = None  # last fetch, successful or not

    async def get_key(self, kid: str) -> jwt.PyJWK:
        """Return the key for `kid`. Raises UnknownKeyError or JwksUnavailableError."""
        # The lock makes concurrent requests share one fetch instead of each starting
        # their own. Outside a fetch it only guards a dict lookup.
        async with self._lock:
            if self._expired():
                if not self._may_fetch():
                    raise JwksUnavailableError("JWKS expired and the last refresh failed recently")
                await self._fetch()
            key = self._keys.get(kid)
            if key is None and self._may_fetch():
                # Possibly a key rotation in Keycloak; possibly a forged kid.
                logger.info("unknown kid, refetching JWKS")
                await self._fetch()
                key = self._keys.get(kid)
        if key is None:
            raise UnknownKeyError
        return key

    def _expired(self) -> bool:
        return self._fetched_at is None or self._clock() - self._fetched_at >= self._ttl

    def _may_fetch(self) -> bool:
        return (
            self._attempted_at is None
            or self._clock() - self._attempted_at >= self._min_refetch_interval
        )

    async def _fetch(self) -> None:
        self._attempted_at = self._clock()
        try:
            response = await self._client.get(self._url, timeout=_FETCH_TIMEOUT_SECONDS)
            response.raise_for_status()
            keys = _parse_jwks(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("JWKS fetch failed: %s", type(exc).__name__)
            raise JwksUnavailableError from exc
        self._keys = keys
        self._fetched_at = self._attempted_at


def _parse_jwks(document: Any) -> dict[str, jwt.PyJWK]:
    """Keep the asymmetric signing keys that have a `kid` and that PyJWT can load."""
    if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
        raise ValueError("not a JWK set")
    keys: dict[str, jwt.PyJWK] = {}
    for entry in document["keys"]:
        if not isinstance(entry, dict):
            continue
        kid = entry.get("kid")
        if (
            not isinstance(kid, str)
            or not kid
            or entry.get("kty") not in _ASYMMETRIC_KEY_TYPES
            or entry.get("use", "sig") != "sig"
        ):
            continue
        try:
            keys[kid] = jwt.PyJWK(entry)
        except (jwt.PyJWTError, ValueError):
            # e.g. Keycloak's RSA-OAEP encryption key, or an algorithm PyJWT lacks.
            continue
    if not keys:
        raise ValueError("no usable signing keys")
    return keys
