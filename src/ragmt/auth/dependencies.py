"""FastAPI dependency: `Authorization: Bearer <token>` → verified Principal.

Status codes (ADR 0006):
- 401 + `WWW-Authenticate: Bearer`: no token, or one that fails verification.
- 403: a genuine token that does not name exactly one organization.
- 503: the issuer's JWKS cannot be fetched. Nothing is accepted meanwhile.

Response bodies are generic on purpose. The reason is logged at INFO, never the token.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ragmt.auth.jwks import JwksCache, JwksUnavailableError, keycloak_jwks_url
from ragmt.auth.tokens import InvalidTokenError, NoTenantError, TokenValidator
from ragmt.domain import Principal
from ragmt.settings import Settings

logger = logging.getLogger(__name__)

# auto_error=False: a missing or non-Bearer header gets the same 401 as a bad token.
_bearer = HTTPBearer(auto_error=False, description="Keycloak access token")

_VALIDATOR_KEY = "token_validator"
_HTTP_KEY = "jwks_http"


def build_token_validator(settings: Settings, client: httpx.AsyncClient) -> TokenValidator:
    jwks = JwksCache(
        client,
        keycloak_jwks_url(settings.oidc_issuer),
        ttl_seconds=settings.jwks_cache_ttl_seconds,
    )
    return TokenValidator(
        jwks,
        issuer=settings.oidc_issuer,
        audience=settings.oidc_audience,
        algorithms=settings.oidc_algorithms,
    )


@asynccontextmanager
async def open_token_validator(app: FastAPI, settings: Settings) -> AsyncIterator[None]:
    """Create the JWKS HTTP client and the validator for the app's lifetime.

    One validator per app, so the JWKS cache (and its refetch rate limit) is
    shared by all requests. Creating the client doesn't connect; the client is
    closed when the lifespan ends.
    """
    client = httpx.AsyncClient()
    try:
        setattr(app.state, _HTTP_KEY, client)
        setattr(app.state, _VALIDATOR_KEY, build_token_validator(settings, client))
        try:
            yield
        finally:
            delattr(app.state, _VALIDATOR_KEY)
            delattr(app.state, _HTTP_KEY)
    finally:
        await client.aclose()


async def get_token_validator(request: Request) -> TokenValidator:
    """The validator built in the app's lifespan (`open_token_validator`).

    Tests replace it with `app.dependency_overrides[get_token_validator]`.
    """
    validator = getattr(request.app.state, _VALIDATOR_KEY, None)
    if not isinstance(validator, TokenValidator):
        raise RuntimeError("no token validator: was the app started through its lifespan?")
    return validator


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    validator: Annotated[TokenValidator, Depends(get_token_validator)],
) -> Principal:
    """The verified caller. The tenant comes from the token, never from the request."""
    if credentials is None:
        logger.info("token rejected: missing bearer token")
        raise _unauthorized()
    try:
        return await validator.validate(credentials.credentials)
    except InvalidTokenError as exc:
        logger.info("token rejected: %s", exc.reason)
        raise _unauthorized() from None
    except NoTenantError as exc:
        logger.info("no tenant in token: %s", exc.reason)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden") from None
    except JwksUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable",
        ) from None
