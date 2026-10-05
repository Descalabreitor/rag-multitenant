"""Test doubles for auth: RSA keys made in-process and a fake Keycloak JWKS endpoint.

No Keycloak is needed. `FakeJwks` serves the keys through httpx.MockTransport
and counts requests, so tests can check caching and rate limiting.
"""

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from ragmt.auth.jwks import JwksCache, keycloak_jwks_url
from ragmt.auth.tokens import TokenValidator

ISSUER = "http://keycloak.test/realms/ragmt"
AUDIENCE = "ragmt-api"
JWKS_URL = keycloak_jwks_url(ISSUER)
TTL_SECONDS = 300
TENANT_ID = UUID("7f66d4f3-54bf-47da-9604-0551ebdb756b")
SUB = "6c1f0a52-3c4e-4d1b-9a43-0f3a8e2d7b11"


@dataclass
class SigningKey:
    kid: str
    private_key: rsa.RSAPrivateKey = field(
        default_factory=lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048)
    )

    def jwk(self) -> dict[str, Any]:
        public = json.loads(RSAAlgorithm.to_jwk(self.private_key.public_key()))
        return {**public, "kid": self.kid, "use": "sig", "alg": "RS256"}

    def sign(self, claims: dict[str, Any], **header: Any) -> str:
        return jwt.encode(
            claims, self.private_key, algorithm="RS256", headers={"kid": self.kid, **header}
        )


def claims(drop: tuple[str, ...] = (), **overrides: Any) -> dict[str, Any]:
    """Claims shaped like a Keycloak 26 access token with the organization mapper."""
    now = int(time.time())
    base: dict[str, Any] = {
        "iss": ISSUER,
        "aud": [AUDIENCE, "account"],
        "sub": SUB,
        "iat": now,
        "exp": now + 300,
        "typ": "Bearer",
        "azp": "ragmt-cli",
        "organization": {"acme": {"id": str(TENANT_ID)}},
        # Present in real tokens and ignored by the validator (ADR 0002).
        "groups": ["/hr", "/admins"],
    }
    base.update(overrides)
    for name in drop:
        del base[name]
    return base


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forge(header: dict[str, Any], payload: dict[str, Any], signature: bytes = b"") -> str:
    """Build a token by hand, for shapes PyJWT refuses to produce."""
    head = b64url(json.dumps(header).encode())
    body = b64url(json.dumps(payload).encode())
    return f"{head}.{body}.{b64url(signature)}"


def forge_hs256(payload: dict[str, Any], secret: bytes, kid: str) -> str:
    """HMAC-signed token, e.g. with the RSA public key as the secret (alg confusion)."""
    head = b64url(json.dumps({"alg": "HS256", "typ": "JWT", "kid": kid}).encode())
    body = b64url(json.dumps(payload).encode())
    mac = hmac.new(secret, f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{b64url(mac)}"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeJwks:
    """A JWKS endpoint that counts requests and can be switched off."""

    def __init__(self, *keys: SigningKey) -> None:
        self.keys = list(keys)
        self.extra_entries: list[dict[str, Any]] = []
        self.calls = 0
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == JWKS_URL
        self.calls += 1
        if self.down:
            return httpx.Response(503)
        entries = [key.jwk() for key in self.keys] + self.extra_entries
        return httpx.Response(200, json={"keys": entries})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def make_validator(
    fake: FakeJwks, clock: FakeClock | None = None, algorithms: tuple[str, ...] = ("RS256",)
) -> TokenValidator:
    cache = JwksCache(fake.client(), JWKS_URL, ttl_seconds=TTL_SECONDS, clock=clock or FakeClock())
    return TokenValidator(cache, issuer=ISSUER, audience=AUDIENCE, algorithms=algorithms)
