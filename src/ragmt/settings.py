"""Application settings, read from the environment and then `.env` (pydantic-settings).

Only what code under `src/` needs lives here. MIGRATOR_DATABASE_URL is
deliberately absent: `migrations/env.py` reads it on its own, so the application
has no way to connect as the schema owner.

Database URLs and secrets are `SecretStr`, so they don't show up in reprs, logs
or tracebacks. Call `.get_secret_value()` only where the value is used.
"""

from functools import lru_cache
from typing import Annotated, Literal, Self
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

MIB = 1024 * 1024


def _require_http_url(value: str) -> None:
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("must be an http(s) URL")


class Settings(BaseSettings):
    """Everything the application reads from its environment. Values are immutable."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # --- PostgreSQL (ADR 0004) -------------------------------------------------
    # app_rw: user requests. Reads only what the user's principals allow.
    database_url: SecretStr
    # app_ingest: ingest, ACL changes and permsync. Sees the whole tenant, so it
    # must never serve a user's query.
    ingest_database_url: SecretStr
    # Pool of each of the API's engines: app_rw, and app_ingest for the write routes.
    database_pool_size: int = Field(default=5, gt=0)
    database_max_overflow: int = Field(default=5, ge=0)
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
    # Keycloak's base URL as permsync reaches it (http://keycloak:8080 inside
    # compose). Unset means the origin of OIDC_ISSUER; the realm always comes
    # from OIDC_ISSUER.
    permsync_keycloak_url: str | None = None
    # Time from the start of one cycle to the start of the next. This bounds the
    # revocation window (ADR 0002).
    permsync_interval_seconds: int = Field(default=60, gt=0)

    # --- Ingestion (ADR 0008) --------------------------------------------------
    # Largest upload accepted, in bytes, checked before conversion.
    ingest_max_bytes: int = Field(default=10 * MIB, gt=0)
    # Chunk size in characters, and how many characters consecutive chunks of
    # one section share. The overlap must be smaller than the chunk.
    chunk_max_chars: int = Field(default=2000, gt=0)
    chunk_overlap_chars: int = Field(default=200, ge=0)
    # Texts per embedding request.
    embed_batch_size: int = Field(default=32, gt=0)

    # --- LLM providers ----------------------------------------------------------
    # "fake": hash-derived embeddings for CI and tests, meaningless for ranking.
    llm_provider: Literal["ollama", "openai_compat", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = Field(default="nomic-embed-text", min_length=1)
    ollama_chat_model: str = Field(default="llama3.1:8b", min_length=1)
    # Only read when LLM_PROVIDER=openai_compat. Empty values in .env mean unset.
    openai_compat_base_url: str | None = None
    openai_compat_api_key: SecretStr | None = None
    openai_compat_embed_model: str | None = None
    openai_compat_chat_model: str | None = None

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
        _require_http_url(value)
        if value.endswith("/"):
            raise ValueError("must not end with '/': it is compared verbatim with `iss`")
        return value

    @field_validator("permsync_keycloak_url")
    @classmethod
    def _keycloak_url_is_http_url(cls, value: str | None) -> str | None:
        if value is not None:
            _require_http_url(value)
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _ollama_url_is_http_url(cls, value: str) -> str:
        _require_http_url(value)
        return value

    @field_validator(
        "openai_compat_base_url",
        "openai_compat_api_key",
        "openai_compat_embed_model",
        "openai_compat_chat_model",
        mode="before",
    )
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        # .env.example lists these as `OPENAI_COMPAT_BASE_URL=` (empty).
        return None if value == "" else value

    @field_validator("openai_compat_base_url")
    @classmethod
    def _openai_compat_url_is_http_url(cls, value: str | None) -> str | None:
        if value is not None:
            _require_http_url(value)
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

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> Self:
        # With overlap >= size, a chunker that steps by (size - overlap) never advances.
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_MAX_CHARS")
        return self

    @model_validator(mode="after")
    def _openai_compat_is_complete(self) -> Self:
        # The API key stays optional: local OpenAI-compatible servers often need none.
        if self.llm_provider == "openai_compat":
            missing = [
                name.upper()
                for name in (
                    "openai_compat_base_url",
                    "openai_compat_embed_model",
                    "openai_compat_chat_model",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"LLM_PROVIDER=openai_compat needs {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load the settings once per process. Raises if any value is missing or invalid."""
    return Settings()
