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
