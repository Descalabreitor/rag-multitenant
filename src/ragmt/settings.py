"""Application settings, read from the environment and then `.env` (pydantic-settings).

Only what code under `src/` needs lives here. MIGRATOR_DATABASE_URL is
deliberately absent: `migrations/env.py` reads it on its own, so the application
has no way to connect as the schema owner.

Database URLs and secrets are `SecretStr`, so they don't show up in reprs, logs
or tracebacks. Call `.get_secret_value()` only where the value is used.
"""

from functools import lru_cache
from typing import Annotated, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import make_url

# Asymmetric algorithms only. "none" accepts unsigned tokens, and with HS* a token
# signed with the public key as the HMAC secret would pass (RS256 -> HS256 confusion).
SAFE_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"}
)

_ASYNC_DRIVER = "postgresql+asyncpg"


class Settings(BaseSettings):
    """Everything the application reads from its environment. Values are immutable."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # --- PostgreSQL (ADR 0004) -------------------------------------------------
    # app_rw: user requests. Reads only what the user's principals allow.
    database_url: SecretStr
    # app_ingest: ingest, ACL changes and permsync. Sees the whole tenant, so it
    # must never serve a user's query.
    ingest_database_url: SecretStr
    # Must match the vector(n) column in the migrations.
    embedding_dim: int = Field(default=768, gt=0)

    # --- OIDC ------------------------------------------------------------------
    # Compared verbatim with the token's `iss` claim.
    oidc_issuer: str
    oidc_audience: str = Field(min_length=1)
    # Comma-separated in the environment (OIDC_ALGORITHMS=RS256,ES256).
    oidc_algorithms: Annotated[tuple[str, ...], NoDecode] = ("RS256",)
    jwks_cache_ttl_seconds: int = Field(default=300, gt=0)

    # --- permsync (Keycloak service account) -----------------------------------
    permsync_client_id: str = Field(min_length=1)
    permsync_client_secret: SecretStr

    @field_validator("database_url", "ingest_database_url")
    @classmethod
    def _async_postgres(cls, value: SecretStr) -> SecretStr:
        if make_url(value.get_secret_value()).drivername != _ASYNC_DRIVER:
            # The URL itself is not echoed: it contains the password.
            raise ValueError(f"must be a {_ASYNC_DRIVER}:// URL")
        return value

    @field_validator("oidc_issuer")
    @classmethod
    def _issuer_is_http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("must be an http(s) URL")
        if value.endswith("/"):
            raise ValueError("must not end with '/': it is compared verbatim with `iss`")
        return value

    @field_validator("oidc_algorithms", mode="before")
    @classmethod
    def _split_algorithms(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(alg.strip() for alg in value.split(",") if alg.strip())
        return value

    @field_validator("oidc_algorithms")
    @classmethod
    def _only_safe_algorithms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one algorithm is required")
        unsafe = sorted(set(value) - SAFE_ALGORITHMS)
        if unsafe:
            raise ValueError(f"not allowed: {', '.join(unsafe)} (asymmetric algorithms only)")
        return value

    @field_validator("permsync_client_secret")
    @classmethod
    def _secret_not_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def _separate_roles(self) -> Self:
        # The same role in both URLs would serve user queries with the writer's
        # whole-tenant view, which defeats the point of ADR 0004.
        reader = make_url(self.database_url.get_secret_value()).username
        writer = make_url(self.ingest_database_url.get_secret_value()).username
        if reader == writer:
            raise ValueError("DATABASE_URL and INGEST_DATABASE_URL must use different roles")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load the settings once per process. Raises if any value is missing or invalid."""
    return Settings()
