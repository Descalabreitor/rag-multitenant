"""get_principal over HTTP: status codes, WWW-Authenticate, and nothing echoed back."""

import logging
from typing import Annotated

import httpx
import pytest
from fastapi import Depends, FastAPI

from ragmt.auth.dependencies import get_principal, get_token_validator
from ragmt.domain import Principal
from tests.auth_helpers import SUB, TENANT_ID, FakeJwks, SigningKey, claims, make_validator


@pytest.fixture(scope="module")
def key() -> SigningKey:
    return SigningKey(kid="kc-1")


@pytest.fixture
def fake(key: SigningKey) -> FakeJwks:
    return FakeJwks(key)


@pytest.fixture
def client(fake: FakeJwks) -> httpx.AsyncClient:
    app = FastAPI()

    @app.get("/me")
    async def me(principal: Annotated[Principal, Depends(get_principal)]) -> dict[str, str]:
        return {"sub": principal.sub, "tenant_id": str(principal.tenant_id)}

    validator = make_validator(fake)
    app.dependency_overrides[get_token_validator] = lambda: validator
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_valid_token(client: httpx.AsyncClient, key: SigningKey) -> None:
    response = await client.get("/me", headers=bearer(key.sign(claims())))
    assert response.status_code == 200
    assert response.json() == {"sub": SUB, "tenant_id": str(TENANT_ID)}


async def test_tenant_in_the_request_is_ignored(client: httpx.AsyncClient, key: SigningKey) -> None:
    other = "17033638-c3cc-4d1e-b1fb-539941854a6e"
    response = await client.get(
        "/me",
        params={"tenant_id": other},
        headers={**bearer(key.sign(claims())), "X-Tenant-Id": other},
    )
    assert response.json()["tenant_id"] == str(TENANT_ID)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer not-a-jwt"},
    ],
)
async def test_missing_or_malformed_header_is_401(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    response = await client.get("/me", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json() == {"detail": "Not authenticated"}


@pytest.mark.leaks
async def test_rejected_token_is_not_echoed_or_logged(
    client: httpx.AsyncClient, key: SigningKey, caplog: pytest.LogCaptureFixture
) -> None:
    token = key.sign(claims(exp=claims()["iat"] - 3600))
    with caplog.at_level(logging.INFO, logger="ragmt.auth"):
        response = await client.get("/me", headers=bearer(token))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    # The client learns nothing about why; the log says why without the token.
    assert response.json() == {"detail": "Not authenticated"}
    assert "token rejected: ExpiredSignatureError" in caplog.text
    for part in token.split("."):
        assert part not in caplog.text
        assert part not in response.text


@pytest.mark.leaks
async def test_two_organizations_is_403(
    client: httpx.AsyncClient, key: SigningKey, caplog: pytest.LogCaptureFixture
) -> None:
    organizations = {
        "acme": {"id": str(TENANT_ID)},
        "umbra": {"id": "17033638-c3cc-4d1e-b1fb-539941854a6e"},
    }
    with caplog.at_level(logging.INFO, logger="ragmt.auth"):
        response = await client.get(
            "/me", headers=bearer(key.sign(claims(organization=organizations)))
        )
    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
    assert "17033638" not in response.text
    assert "2 organizations" in caplog.text


async def test_missing_organization_is_403(client: httpx.AsyncClient, key: SigningKey) -> None:
    response = await client.get("/me", headers=bearer(key.sign(claims(drop=("organization",)))))
    assert response.status_code == 403


async def test_jwks_outage_is_503(
    client: httpx.AsyncClient, fake: FakeJwks, key: SigningKey
) -> None:
    fake.down = True
    response = await client.get("/me", headers=bearer(key.sign(claims())))
    assert response.status_code == 503
