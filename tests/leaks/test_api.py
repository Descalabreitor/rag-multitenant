"""The documents API, end to end against PostgreSQL, under fixed Principals.

JWT validation is replaced through `dependency_overrides`: the test header
`X-Test-Principal` picks the tenant and user of a fixed Principal. Everything
after that (the app_rw engine, tenant_session, RLS) is the real request path.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

import httpx
import pytest
from fastapi import Header

from ragmt.api.app import create_app
from ragmt.auth.dependencies import get_principal
from ragmt.domain import Principal
from ragmt.settings import Settings
from tests.leaks.conftest import World

pytestmark = [pytest.mark.db, pytest.mark.leaks]

# (tenant key, user sub) -> titles that user may list. Alice is in group hr in
# both tenants (tests/leaks/conftest.py); bob has no groups.
VISIBLE = {
    ("a", "alice"): {"hr_a", "public_a"},
    ("a", "bob"): {"bob_a", "public_a"},
    ("b", "alice"): {"hr_b", "public_b"},
}


def as_user(tenant: str, sub: str) -> dict[str, str]:
    return {"X-Test-Principal": f"{tenant}:{sub}"}


def principal_override(world: World) -> Callable[[str], Principal]:
    def principal(x_test_principal: Annotated[str, Header()]) -> Principal:
        tenant, sub = x_test_principal.split(":", 1)
        return Principal(sub=sub, tenant_id=world.tenants[tenant])

    return principal


@asynccontextmanager
async def api(world: World, pool_size: int = 2) -> AsyncIterator[httpx.AsyncClient]:
    """The real app (lifespan included) on a small app_rw pool."""
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL is not set")
    app = create_app(Settings(database_pool_size=pool_size, database_max_overflow=0))
    app.dependency_overrides[get_principal] = principal_override(world)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


async def titles(client: httpx.AsyncClient, tenant: str, sub: str, **request: Any) -> set[str]:
    response = await client.request("GET", "/documents", headers=as_user(tenant, sub), **request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert all(set(item) == {"id", "title"} for item in body)
    return {item["title"] for item in body}


# --- what each user sees ----------------------------------------------------------


@pytest.mark.parametrize(("tenant", "sub"), list(VISIBLE))
async def test_list_shows_only_permitted_documents(world: World, tenant: str, sub: str) -> None:
    async with api(world) as client:
        assert await titles(client, tenant, sub) == VISIBLE[(tenant, sub)]


async def test_get_returns_a_visible_document_without_embeddings(world: World) -> None:
    doc = world.documents["hr_a"]
    async with api(world) as client:
        response = await client.get(f"/documents/{doc}", headers=as_user("a", "alice"))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "title", "source_uri", "created_at"}
    assert (UUID(body["id"]), body["title"]) == (doc, "hr_a")


# --- 404, never 403 -----------------------------------------------------------------


async def test_invisible_documents_are_404_with_identical_bodies(world: World) -> None:
    """Another tenant's document and a same-tenant one without access look alike."""
    async with api(world) as client:
        other_tenant = await client.get(
            f"/documents/{world.documents['hr_b']}", headers=as_user("a", "alice")
        )
        no_access = await client.get(
            f"/documents/{world.documents['bob_a']}", headers=as_user("a", "alice")
        )
    assert other_tenant.status_code == no_access.status_code == 404
    assert other_tenant.content == no_access.content
    assert other_tenant.headers["content-type"] == no_access.headers["content-type"]


# --- leak case 4: a tenant_id in the request is ignored ------------------------------


async def test_tenant_id_in_query_or_body_is_ignored(world: World) -> None:
    other = str(world.tenants["b"])
    expected = VISIBLE[("a", "alice")]
    async with api(world) as client:
        assert await titles(client, "a", "alice", params={"tenant_id": other}) == expected
        assert await titles(client, "a", "alice", json={"tenant_id": other}) == expected
        response = await client.request(
            "GET",
            f"/documents/{world.documents['hr_b']}",
            params={"tenant_id": other},
            json={"tenant_id": other},
            headers=as_user("a", "alice"),
        )
    assert response.status_code == 404


# --- leak case 6: pooled connections under concurrency -------------------------------


async def test_concurrent_requests_on_a_small_pool_never_mix_tenants(world: World) -> None:
    """300 interleaved requests from two tenants share two pooled connections.

    If a tenant context survived on a connection, or were set on the wrong one,
    some response would list (or find) the other tenant's documents.
    """
    users = [("a", "alice"), ("b", "alice")]
    foreign = {"a": world.documents["hr_b"], "b": world.documents["hr_a"]}

    async with api(world, pool_size=2) as client:

        async def one(i: int) -> None:
            tenant, sub = users[i % 2]
            if i % 3 == 0:
                response = await client.get(
                    f"/documents/{foreign[tenant]}", headers=as_user(tenant, sub)
                )
                assert response.status_code == 404, (i, response.text)
            else:
                assert await titles(client, tenant, sub) == VISIBLE[(tenant, sub)], i

        await asyncio.gather(*(one(i) for i in range(300)))
