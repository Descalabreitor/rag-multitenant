"""Settings load from the environment and reject unsafe or inconsistent values."""

import pytest
from pydantic import ValidationError

from ragmt.settings import Settings

VALID = {
    "DATABASE_URL": "postgresql+asyncpg://app_rw:secret-rw@127.0.0.1:5432/ragmt",
    "INGEST_DATABASE_URL": "postgresql+asyncpg://app_ingest:secret-ingest@127.0.0.1:5432/ragmt",
    "EMBEDDING_DIM": "768",
    "OIDC_ISSUER": "http://localhost:8080/realms/ragmt",
    "OIDC_AUDIENCE": "ragmt-api",
    "OIDC_ALGORITHMS": "RS256",
    "JWKS_CACHE_TTL_SECONDS": "300",
    "PERMSYNC_CLIENT_ID": "ragmt-permsync",
    "PERMSYNC_CLIENT_SECRET": "secret-permsync",
}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete, valid environment; tests override or remove single variables.

    tests/conftest.py may have loaded .env into os.environ, so every variable is
    set here explicitly, and `.env` itself is never read (`_env_file=None`).
    """
    monkeypatch.delenv("MIGRATOR_DATABASE_URL", raising=False)
    for key, value in VALID.items():
        monkeypatch.setenv(key, value)


def load() -> Settings:
    return Settings(_env_file=None)


def test_loads_a_valid_environment() -> None:
    settings = load()
    assert settings.database_url.get_secret_value() == VALID["DATABASE_URL"]
    assert settings.ingest_database_url.get_secret_value() == VALID["INGEST_DATABASE_URL"]
    assert settings.oidc_issuer == VALID["OIDC_ISSUER"]
    assert settings.oidc_algorithms == ("RS256",)
    assert settings.jwks_cache_ttl_seconds == 300
    assert settings.permsync_client_secret.get_secret_value() == "secret-permsync"


def test_secrets_stay_out_of_repr() -> None:
    shown = repr(load())
    for secret in ("secret-rw", "secret-ingest", "secret-permsync"):
        assert secret not in shown


def test_the_migrator_url_is_not_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIGRATOR_DATABASE_URL", "postgresql+asyncpg://migrator:x@h/ragmt")
    assert "migrator_database_url" not in Settings.model_fields
    assert "migrator" not in repr(load().model_dump())


@pytest.mark.parametrize(
    "variable",
    ["DATABASE_URL", "INGEST_DATABASE_URL", "OIDC_ISSUER", "OIDC_AUDIENCE", "PERMSYNC_CLIENT_ID"],
)
def test_required_variables(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.delenv(variable)
    with pytest.raises(ValidationError, match=variable.lower()):
        load()


def test_algorithms_are_a_comma_separated_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OIDC_ALGORITHMS", " RS256, ES256 ,")
    assert load().oidc_algorithms == ("RS256", "ES256")


@pytest.mark.parametrize("algorithms", ["none", "HS256", "RS256,HS512", "rs256", "", " , "])
def test_unsafe_or_empty_algorithms_are_rejected(
    monkeypatch: pytest.MonkeyPatch, algorithms: str
) -> None:
    monkeypatch.setenv("OIDC_ALGORITHMS", algorithms)
    with pytest.raises(ValidationError, match="oidc_algorithms"):
        load()


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://app_rw:secret-rw@127.0.0.1:5432/ragmt",  # sync driver
        "postgresql+psycopg://app_rw:secret-rw@127.0.0.1:5432/ragmt",
        "sqlite+aiosqlite:///ragmt.db",
    ],
)
def test_database_urls_must_use_asyncpg(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("DATABASE_URL", url)
    with pytest.raises(ValidationError, match="database_url") as excinfo:
        load()
    assert "secret-rw" not in str(excinfo.value)


def test_reader_and_writer_must_be_different_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID["INGEST_DATABASE_URL"])
    with pytest.raises(ValidationError, match="different roles"):
        load()


@pytest.mark.parametrize(
    "issuer",
    ["http://localhost:8080/realms/ragmt/", "localhost:8080/realms/ragmt", "ftp://kc/realms/r"],
)
def test_issuer_must_be_an_exact_http_url(monkeypatch: pytest.MonkeyPatch, issuer: str) -> None:
    monkeypatch.setenv("OIDC_ISSUER", issuer)
    with pytest.raises(ValidationError, match="oidc_issuer"):
        load()


@pytest.mark.parametrize(
    ("variable", "value"),
    [("PERMSYNC_CLIENT_SECRET", ""), ("JWKS_CACHE_TTL_SECONDS", "0"), ("EMBEDDING_DIM", "-1")],
)
def test_empty_or_non_positive_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError, match=variable.lower()):
        load()


def test_permsync_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERMSYNC_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("PERMSYNC_KEYCLOAK_URL", raising=False)
    settings = load()
    assert settings.permsync_interval_seconds == 60
    assert settings.permsync_keycloak_url is None


@pytest.mark.parametrize("value", ["0", "-5"])
def test_permsync_interval_must_be_positive(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PERMSYNC_INTERVAL_SECONDS", value)
    with pytest.raises(ValidationError, match="permsync_interval_seconds"):
        load()


@pytest.mark.parametrize("url", ["keycloak:8080", "ftp://keycloak:8080"])
def test_permsync_keycloak_url_must_be_http(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("PERMSYNC_KEYCLOAK_URL", url)
    with pytest.raises(ValidationError, match="permsync_keycloak_url"):
        load()


# --- Ingestion ------------------------------------------------------------------

INGEST_VARIABLES = (
    "INGEST_MAX_BYTES",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "EMBED_BATCH_SIZE",
)


def test_ingest_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in INGEST_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    settings = load()
    assert settings.ingest_max_bytes == 10 * 1024 * 1024
    assert settings.chunk_overlap_chars < settings.chunk_max_chars
    assert settings.embed_batch_size > 0


@pytest.mark.parametrize("variable", ["INGEST_MAX_BYTES", "CHUNK_MAX_CHARS", "EMBED_BATCH_SIZE"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_ingest_sizes_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError, match=variable.lower()):
        load()


def test_overlap_may_be_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "0")
    assert load().chunk_overlap_chars == 0


@pytest.mark.parametrize(("size", "overlap"), [("500", "500"), ("500", "501"), ("500", "-1")])
def test_overlap_must_be_smaller_than_the_chunk(
    monkeypatch: pytest.MonkeyPatch, size: str, overlap: str
) -> None:
    monkeypatch.setenv("CHUNK_MAX_CHARS", size)
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", overlap)
    with pytest.raises(ValidationError, match=r"(?i)chunk_overlap_chars"):
        load()


# --- LLM providers --------------------------------------------------------------

OPENAI_COMPAT = {
    "OPENAI_COMPAT_BASE_URL": "https://llm.example/v1",
    "OPENAI_COMPAT_API_KEY": "secret-llm",
    "OPENAI_COMPAT_EMBED_MODEL": "embed-small",
    "OPENAI_COMPAT_CHAT_MODEL": "chat-large",
}
LLM_VARIABLES = (
    "LLM_PROVIDER",
    "CHAT_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_CHAT_MODEL",
)


def test_llm_defaults_to_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (*LLM_VARIABLES, *OPENAI_COMPAT):
        monkeypatch.delenv(variable, raising=False)
    settings = load()
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_embed_model == "nomic-embed-text"
    assert settings.openai_compat_api_key is None
    assert settings.chat_provider is None
    assert settings.chat_provider_name == "ollama"


def test_empty_chat_provider_follows_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # As in .env.example: `CHAT_PROVIDER=`.
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("CHAT_PROVIDER", "")
    assert load().chat_provider_name == "fake"


def test_chat_provider_overrides_llm_provider_for_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CHAT_PROVIDER", "fake")
    settings = load()
    assert settings.llm_provider == "ollama"
    assert settings.chat_provider_name == "fake"


def test_openai_compat_chat_needs_only_the_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CHAT_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", OPENAI_COMPAT["OPENAI_COMPAT_BASE_URL"])
    monkeypatch.setenv("OPENAI_COMPAT_EMBED_MODEL", "")
    monkeypatch.setenv("OPENAI_COMPAT_CHAT_MODEL", "")
    with pytest.raises(ValidationError, match="OPENAI_COMPAT_CHAT_MODEL"):
        load()
    monkeypatch.setenv("OPENAI_COMPAT_CHAT_MODEL", "chat-large")
    assert load().chat_provider_name == "openai_compat"


def test_empty_openai_compat_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # As in .env.example: `OPENAI_COMPAT_BASE_URL=` and friends.
    for variable in OPENAI_COMPAT:
        monkeypatch.setenv(variable, "")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    settings = load()
    assert settings.openai_compat_base_url is None
    assert settings.openai_compat_api_key is None


def test_openai_compat_loads_with_a_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compat")
    for variable, value in OPENAI_COMPAT.items():
        monkeypatch.setenv(variable, value)
    settings = load()
    assert settings.openai_compat_api_key is not None
    assert settings.openai_compat_api_key.get_secret_value() == "secret-llm"
    assert "secret-llm" not in repr(settings)


def test_openai_compat_key_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compat")
    for variable, value in OPENAI_COMPAT.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "")
    assert load().openai_compat_api_key is None


@pytest.mark.parametrize(
    "variable", ["OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_EMBED_MODEL", "OPENAI_COMPAT_CHAT_MODEL"]
)
def test_openai_compat_needs_url_and_models(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compat")
    for name, value in OPENAI_COMPAT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(variable, "")
    with pytest.raises(ValidationError, match=variable):
        load()


def test_unknown_llm_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValidationError, match="llm_provider"):
        load()


@pytest.mark.parametrize(
    ("variable", "url"),
    [("OLLAMA_BASE_URL", "localhost:11434"), ("OPENAI_COMPAT_BASE_URL", "ftp://llm.example")],
)
def test_llm_urls_must_be_http(monkeypatch: pytest.MonkeyPatch, variable: str, url: str) -> None:
    monkeypatch.setenv(variable, url)
    with pytest.raises(ValidationError, match=variable.lower()):
        load()


# --- Retrieval and generation (ADR 0009) ----------------------------------------

RETRIEVAL_VARIABLES = (
    "RETRIEVAL_K",
    "HNSW_EF_SEARCH",
    "HNSW_ITERATIVE_SCAN",
    "HNSW_MAX_SCAN_TUPLES",
    "ASK_MAX_QUESTION_CHARS",
    "ASK_MAX_CONTEXT_CHARS",
    "CHAT_TIMEOUT_SECONDS",
    "AUDIT_STORE_QUERY_TEXT",
    "CHUNK_MAX_CHARS",
)


def test_retrieval_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in RETRIEVAL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    settings = load()
    assert settings.retrieval_k == 5
    assert settings.hnsw_ef_search >= settings.retrieval_k
    assert settings.hnsw_iterative_scan == "relaxed_order"
    assert settings.hnsw_max_scan_tuples > 0
    assert settings.ask_max_context_chars >= settings.chunk_max_chars
    assert settings.chat_timeout_seconds > 0
    assert settings.audit_store_query_text is False


def test_retrieval_values_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_K", "8")
    monkeypatch.setenv("HNSW_EF_SEARCH", "100")
    monkeypatch.setenv("HNSW_ITERATIVE_SCAN", "strict_order")
    monkeypatch.setenv("CHAT_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("AUDIT_STORE_QUERY_TEXT", "true")
    settings = load()
    assert (settings.retrieval_k, settings.hnsw_ef_search) == (8, 100)
    assert settings.hnsw_iterative_scan == "strict_order"
    assert settings.chat_timeout_seconds == 2.5
    assert settings.audit_store_query_text is True


@pytest.mark.parametrize("mode", ["off", "relaxed_order", "strict_order"])
def test_iterative_scan_modes(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("HNSW_ITERATIVE_SCAN", mode)
    assert load().hnsw_iterative_scan == mode


@pytest.mark.parametrize("mode", ["", "relaxed", "RELAXED_ORDER", "on"])
def test_unknown_iterative_scan_mode_is_rejected(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("HNSW_ITERATIVE_SCAN", mode)
    with pytest.raises(ValidationError, match="hnsw_iterative_scan"):
        load()


@pytest.mark.parametrize(
    "variable",
    [
        "RETRIEVAL_K",
        "HNSW_EF_SEARCH",
        "HNSW_MAX_SCAN_TUPLES",
        "ASK_MAX_QUESTION_CHARS",
        "ASK_MAX_CONTEXT_CHARS",
        "CHAT_TIMEOUT_SECONDS",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_retrieval_values_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError, match=variable.lower()):
        load()


@pytest.mark.parametrize(
    ("variable", "value"), [("RETRIEVAL_K", "101"), ("HNSW_EF_SEARCH", "1001")]
)
def test_retrieval_upper_bounds(monkeypatch: pytest.MonkeyPatch, variable: str, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_K", "5")
    monkeypatch.setenv("HNSW_EF_SEARCH", "40")
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError, match=variable.lower()):
        load()


def test_ef_search_must_cover_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_K", "20")
    monkeypatch.setenv("HNSW_EF_SEARCH", "19")
    with pytest.raises(ValidationError, match="HNSW_EF_SEARCH"):
        load()
    monkeypatch.setenv("HNSW_EF_SEARCH", "20")
    assert load().hnsw_ef_search == 20


def test_context_must_fit_one_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_MAX_CHARS", "2000")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "200")
    monkeypatch.setenv("ASK_MAX_CONTEXT_CHARS", "1999")
    with pytest.raises(ValidationError, match="ASK_MAX_CONTEXT_CHARS"):
        load()
    monkeypatch.setenv("ASK_MAX_CONTEXT_CHARS", "2000")
    assert load().ask_max_context_chars == 2000


def test_chat_models_are_reused_for_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    assert load().ollama_chat_model == "qwen2.5:7b"
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "")
    with pytest.raises(ValidationError, match="ollama_chat_model"):
        load()
