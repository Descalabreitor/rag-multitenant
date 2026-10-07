"""PKCE (RFC 7636, S256 only) and the OAuth `state` value."""

import base64
import hashlib
import secrets


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_verifier() -> str:
    """A fresh code verifier: 86 characters from the unreserved set (RFC 7636 allows 43-128)."""
    return _b64url(secrets.token_bytes(64))


def challenge_s256(verifier: str) -> str:
    """BASE64URL(SHA256(ASCII(verifier))), without padding."""
    if not 43 <= len(verifier) <= 128:
        raise ValueError("a PKCE verifier has 43 to 128 characters")
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def make_state() -> str:
    """An unguessable `state`, checked on the callback against login CSRF."""
    return _b64url(secrets.token_bytes(32))
