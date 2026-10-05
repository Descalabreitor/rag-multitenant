"""Routes that must answer without a database: health, and 401 without credentials.

Both engines are created in the lifespan but never connect here: /healthz
doesn't use them, and the auth stub rejects the request before a connection is
opened. Embeddings are the fake provider, which needs no network.
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from ragmt.adapters.llm import EmbeddingError
from ragmt.api.app import create_app
from ragmt.settings import Settings

# Nothing listens on port 1: a request that reached the database would fail loudly.
SETTINGS = Settings(
    _env_file=None,
    database_url=SecretStr("postgresql+asyncpg://app_rw:x@127.0.0.1:1/none"),
    ingest_database_url=SecretStr("postgresql+asyncpg://app_ingest:x@127.0.0.1:1/none"),
    oidc_issuer="http://localhost:8080/realms/ragmt",
    oidc_audience="ragmt-api",
    permsync_client_id="ragmt-permsync",
    permsync_client_secret=SecretStr("x"),
    llm_provider="fake",
)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(SETTINGS)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


async def test_healthz_needs_no_auth(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/documents"),
        ("GET", f"/documents/{uuid4()}"),
        ("POST", "/documents"),
        ("PUT", f"/documents/{uuid4()}/acl"),
        ("DELETE", f"/documents/{uuid4()}"),
    ],
)
async def test_documents_without_credentials_is_401(
    client: httpx.AsyncClient, method: str, path: str
) -> None:
    response = await client.request(method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_startup_fails_when_the_embedding_provider_is_unreachable() -> None:
    settings = SETTINGS.model_copy(
        update={"llm_provider": "ollama", "ollama_base_url": "http://127.0.0.1:1"}
    )
    app = create_app(settings)
    with pytest.raises(EmbeddingError):
        async with app.router.lifespan_context(app):
            pass
    assert not hasattr(app.state, "engine")
