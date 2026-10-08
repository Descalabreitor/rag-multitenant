"""The write routes end to end against PostgreSQL, under fixed Principals.

The app is the real one (lifespan, both engines, RLS, triggers), with fake
embeddings and JWT validation replaced as in test_api.py. The `admins` fixture
(conftest.py) adds ada in tenant A and bea in tenant B to the leak world, both in
group `admins` and nothing else, so they read only what an ACL gives them.

What readers see is checked twice: over HTTP (`/documents`) and as chunk
contents through app_rw (`visible_chunks`). What was written is checked as
app_ingest, which sees the whole tenant.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from ragmt.api.app import create_app
from ragmt.auth.dependencies import get_principal
from ragmt.settings import Settings
from tests.db.pg import session
from tests.leaks.conftest import INGEST, World, visible_chunks
from tests.leaks.test_api import as_user, principal_override, titles
from tests.upload_helpers import STREAM_CONTENT_TYPE, CountingStream

pytestmark = [pytest.mark.db, pytest.mark.leaks]

NOT_FOUND = {"detail": "Document not found"}


@asynccontextmanager
async def api(world: World, **settings: Any) -> AsyncIterator[httpx.AsyncClient]:
    if not os.environ.get("DATABASE_URL") or not os.environ.get(INGEST):
        pytest.skip("DATABASE_URL and INGEST_DATABASE_URL must be set")
    app = create_app(Settings(llm_provider="fake", **settings))
    app.dependency_overrides[get_principal] = principal_override(world)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


def document(title: str) -> bytes:
    """A Markdown file whose only paragraph is a unique canary."""
    return f"# {title}\n\nCANARY-{title}\n".encode()


async def upload(
    client: httpx.AsyncClient,
    tenant: str,
    sub: str,
    title: str,
    acl: str | None = None,
    filename: str = "doc.md",
    content: bytes | None = None,
    **extra: str,
) -> httpx.Response:
    data = dict(extra)
    if acl is not None:
        data["acl"] = acl
    return await client.post(
        "/documents",
        files={"file": (filename, content if content is not None else document(title))},
        data=data,
        headers=as_user(tenant, sub),
    )


async def replace(
    client: httpx.AsyncClient,
    tenant: str,
    sub: str,
    document_id: UUID | str,
    content: bytes,
    **extra: str,
) -> httpx.Response:
    """PUT /documents/{id}/content as a given user."""
    return await client.put(
        f"/documents/{document_id}/content",
        files={"file": ("doc.md", content)},
        data=extra,
        headers=as_user(tenant, sub),
    )


async def ask_text(client: httpx.AsyncClient, tenant: str, sub: str) -> str:
    """The whole /ask response. FakeChat echoes its prompt, so this holds every
    chunk that reached the model, and the citations."""
    response = await client.post(
        "/ask", json={"question": "What does it say?"}, headers=as_user(tenant, sub)
    )
    assert response.status_code == 200, response.text
    return response.text


# Room for every chunk a leak-world user can read, so /ask retrieves all of them.
ASK_ALL = {"retrieval_k": 20}


async def stored(tenant: UUID) -> dict[UUID, tuple[str, bool, int]]:
    """Every document of `tenant` as app_ingest: id -> (title, deleted, chunk count)."""
    async with session(INGEST, tenant) as conn:
        rows = await conn.fetch(
            "SELECT d.id, d.title, d.deleted_at IS NOT NULL AS deleted,"
            " (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunks"
            " FROM documents d"
        )
    return {r["id"]: (r["title"], r["deleted"], r["chunks"]) for r in rows}


async def reads_canary(tenant: UUID, sub: str, title: str) -> bool:
    """Whether `sub` can read a chunk holding the canary of `title`, through app_rw."""
    return any(f"CANARY-{title}" in chunk for chunk in await visible_chunks(tenant, sub))


async def acl_of(tenant: UUID, document_id: UUID) -> set[str]:
    async with session(INGEST, tenant) as conn:
        rows = await conn.fetch(
            "SELECT principal FROM document_acl WHERE document_id = $1", document_id
        )
    return {r["principal"] for r in rows}


# --- uploads ------------------------------------------------------------------------


async def test_an_admin_upload_is_readable_only_by_the_admin(admins: World) -> None:
    tenant = admins.tenants["a"]
    async with api(admins) as client:
        response = await upload(client, "a", "ada", "ada_notes")
        assert response.status_code == 201, response.text
        body = response.json()
        assert set(body) == {"id", "title", "chunks", "unchanged"}
        assert (body["title"], body["chunks"], body["unchanged"]) == ("ada_notes", 1, False)

        assert "ada_notes" in await titles(client, "a", "ada")
        for sub in ("alice", "bob"):
            assert "ada_notes" not in await titles(client, "a", sub)
        assert "ada_notes" not in await titles(client, "b", "alice")

    assert await acl_of(tenant, UUID(body["id"])) == {"user:ada"}
    assert await reads_canary(tenant, "ada", "ada_notes")
    assert not await reads_canary(tenant, "alice", "ada_notes")
    assert not await reads_canary(admins.tenants["b"], "bea", "ada_notes")


async def test_an_explicit_acl_replaces_the_default(admins: World) -> None:
    tenant = admins.tenants["a"]
    async with api(admins) as client:
        response = await upload(client, "a", "ada", "for_alice", acl='["user:alice"]')
        assert response.status_code == 201, response.text
        assert "for_alice" in await titles(client, "a", "alice")
        assert "for_alice" not in await titles(client, "a", "ada")
        assert "for_alice" not in await titles(client, "a", "bob")
    assert await acl_of(tenant, UUID(response.json()["id"])) == {"user:alice"}
    assert await reads_canary(tenant, "alice", "for_alice")
    assert not await reads_canary(tenant, "ada", "for_alice")


async def test_a_tenant_id_in_the_form_or_query_is_ignored(admins: World) -> None:
    other = str(admins.tenants["b"])
    async with api(admins) as client:
        response = await client.post(
            "/documents",
            params={"tenant_id": other},
            files={"file": ("doc.md", document("landed"))},
            data={"tenant_id": other, "acl": '["tenant:*"]'},
            headers=as_user("a", "ada"),
        )
        assert response.status_code == 201, response.text
        assert "landed" in await titles(client, "a", "bob")
        assert "landed" not in await titles(client, "b", "alice")
    document_id = UUID(response.json()["id"])
    assert document_id in await stored(admins.tenants["a"])
    assert document_id not in await stored(admins.tenants["b"])


async def test_a_non_admin_upload_is_403_and_writes_nothing(admins: World) -> None:
    before = await stored(admins.tenants["a"])
    async with api(admins) as client:
        response = await upload(client, "a", "alice", "sneaky", acl='["tenant:*"]')
        # bea is an admin, but of tenant B: in tenant A she is nobody.
        cross = await upload(client, "a", "bea", "sneaky2", acl='["tenant:*"]')
    assert response.status_code == cross.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
    assert await stored(admins.tenants["a"]) == before


async def test_a_duplicate_upload_is_unchanged(admins: World) -> None:
    async with api(admins) as client:
        first = await upload(client, "a", "ada", "twice")
        second = await upload(client, "a", "ada", "twice", acl='["tenant:*"]')
    assert first.status_code == second.status_code == 201
    assert second.json() == {**first.json(), "unchanged": True}
    # The existing document keeps its ACL.
    assert await acl_of(admins.tenants["a"], UUID(first.json()["id"])) == {"user:ada"}


@pytest.mark.parametrize("acl", ["not json", "[]", '"user:alice"', "[1]", '["admin"]', '["user:"]'])
async def test_an_invalid_acl_is_422_and_writes_nothing(admins: World, acl: str) -> None:
    before = await stored(admins.tenants["a"])
    async with api(admins) as client:
        response = await upload(client, "a", "ada", "bad_acl", acl=acl)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid ACL"}
    assert await stored(admins.tenants["a"]) == before


async def test_a_pdf_is_415_without_echoing_the_file(admins: World) -> None:
    before = await stored(admins.tenants["a"])
    pdf = b"%PDF-1.7\n1 0 obj << /Type /Catalog >> endobj\nCANARY-pdf\n%%EOF\n"
    async with api(admins) as client:
        as_pdf = await upload(client, "a", "ada", "pdf", filename="CANARY-name.pdf", content=pdf)
        renamed = await upload(client, "a", "ada", "pdf", filename="CANARY-name.md", content=pdf)
    for response in (as_pdf, renamed):
        assert response.status_code == 415
        assert response.json() == {"detail": "Unsupported document type"}
        assert b"CANARY" not in response.content
    assert await stored(admins.tenants["a"]) == before


async def test_an_oversized_upload_is_413_without_reading_the_whole_body(admins: World) -> None:
    limit = 4096
    before = await stored(admins.tenants["a"])
    stream = CountingStream(total=1024 * limit, piece=1024)
    async with api(admins, ingest_max_bytes=limit) as client:
        response = await client.post(
            "/documents",
            content=stream,
            headers={"Content-Type": STREAM_CONTENT_TYPE, **as_user("a", "ada")},
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "Document too large"}
    assert stream.sent <= limit + stream.piece < stream.total
    assert await stored(admins.tenants["a"]) == before


async def test_a_non_admin_replace_never_reads_the_body(admins: World) -> None:
    stream = CountingStream(total=1024 * 1024)
    async with api(admins) as client:
        response = await client.put(
            f"/documents/{admins.documents['public_a']}/content",
            content=stream,
            headers={"Content-Type": STREAM_CONTENT_TYPE, **as_user("a", "alice")},
        )
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert stream.sent == 0


async def test_a_non_admin_upload_never_reads_the_body(admins: World) -> None:
    stream = CountingStream(total=1024 * 1024)
    async with api(admins) as client:
        response = await client.post(
            "/documents",
            content=stream,
            headers={"Content-Type": STREAM_CONTENT_TYPE, **as_user("a", "alice")},
        )
    assert response.status_code == 403
    assert stream.sent == 0


async def test_responses_never_carry_embeddings(admins: World) -> None:
    async with api(admins) as client:
        created = await upload(client, "a", "ada", "no_vectors")
        document_id = created.json()["id"]
        acl = await client.put(
            f"/documents/{document_id}/acl",
            json={"principals": ["user:ada"]},
            headers=as_user("a", "ada"),
        )
        read = await client.get(f"/documents/{document_id}", headers=as_user("a", "ada"))
    for response in (created, acl, read):
        assert "embedding" not in response.text


# --- routes with an id: one 404 for everything a caller may not touch ----------------


async def _not_found_body(client: httpx.AsyncClient) -> bytes:
    response = await client.get(f"/documents/{uuid4()}", headers=as_user("a", "ada"))
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    return response.content


async def _id_routes(
    client: httpx.AsyncClient, tenant: str, sub: str, document_id: UUID | str
) -> list[httpx.Response]:
    headers = as_user(tenant, sub)
    return [
        await client.put(
            f"/documents/{document_id}/acl", json={"principals": ["tenant:*"]}, headers=headers
        ),
        await client.delete(f"/documents/{document_id}", headers=headers),
        await client.delete(f"/documents/{document_id}", params={"purge": "true"}, headers=headers),
        await replace(client, tenant, sub, document_id, document("intruder")),
    ]


async def test_a_non_admin_gets_404_on_acl_change_and_delete(admins: World) -> None:
    tenant = admins.tenants["a"]
    target = admins.documents["public_a"]  # alice can read it, but not change it
    before = (await stored(tenant), await acl_of(tenant, target))
    async with api(admins) as client:
        expected = await _not_found_body(client)
        for response in await _id_routes(client, "a", "alice", target):
            assert response.status_code == 404
            assert response.content == expected
        assert "public_a" in await titles(client, "a", "bob")
    assert (await stored(tenant), await acl_of(tenant, target)) == before


async def test_an_admin_of_a_gets_404_on_documents_of_b(admins: World) -> None:
    tenant_b = admins.tenants["b"]
    target = admins.documents["hr_b"]
    before = (await stored(tenant_b), await acl_of(tenant_b, target))
    async with api(admins) as client:
        expected = await _not_found_body(client)
        for response in await _id_routes(client, "a", "ada", target):
            assert response.status_code == 404
            assert response.content == expected
        assert "hr_b" in await titles(client, "b", "alice")
    assert (await stored(tenant_b), await acl_of(tenant_b, target)) == before
    assert await reads_canary(tenant_b, "alice", "hr_b")


async def test_unknown_and_deleted_ids_are_the_same_404(admins: World) -> None:
    async with api(admins) as client:
        expected = await _not_found_body(client)
        for response in await _id_routes(client, "a", "ada", uuid4()):
            assert response.status_code == 404
            assert response.content == expected

        deleted = admins.documents["bob_a"]
        response = await client.delete(f"/documents/{deleted}", headers=as_user("a", "ada"))
        assert response.status_code == 204
        headers = as_user("a", "ada")
        for response in (
            await client.put(
                f"/documents/{deleted}/acl", json={"principals": ["tenant:*"]}, headers=headers
            ),
            await client.delete(f"/documents/{deleted}", headers=headers),
            await replace(client, "a", "ada", deleted, document("revived")),
        ):
            assert response.status_code == 404
            assert response.content == expected
    assert await acl_of(admins.tenants["a"], deleted) == set()


# --- ACL changes and deletes -----------------------------------------------------------


async def test_an_acl_change_moves_access_from_the_old_reader_to_the_new(admins: World) -> None:
    tenant = admins.tenants["a"]
    async with api(admins) as client:
        created = await upload(client, "a", "ada", "moving", acl='["user:alice"]')
        document_id = created.json()["id"]
        assert "moving" in await titles(client, "a", "alice")

        response = await client.put(
            f"/documents/{document_id}/acl",
            json={"principals": ["user:bob", "user:bob"], "tenant_id": str(admins.tenants["b"])},
            headers=as_user("a", "ada"),
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"id": document_id, "principals": ["user:bob"]}

        assert "moving" not in await titles(client, "a", "alice")
        assert "moving" in await titles(client, "a", "bob")
        gone = await client.get(f"/documents/{document_id}", headers=as_user("a", "alice"))
        assert gone.status_code == 404
    assert not await reads_canary(tenant, "alice", "moving")
    assert await reads_canary(tenant, "bob", "moving")


async def test_an_invalid_acl_change_is_422_and_changes_nothing(admins: World) -> None:
    tenant = admins.tenants["a"]
    target = admins.documents["hr_a"]
    async with api(admins) as client:
        for principals in ([], ["everyone"]):
            response = await client.put(
                f"/documents/{target}/acl",
                json={"principals": principals},
                headers=as_user("a", "ada"),
            )
            assert response.status_code == 422
            assert response.json() == {"detail": "Invalid ACL"}
    assert await acl_of(tenant, target) == {"group:hr"}


async def test_a_soft_delete_hides_the_document_and_keeps_the_rows(admins: World) -> None:
    tenant = admins.tenants["a"]
    target = admins.documents["public_a"]
    async with api(admins) as client:
        response = await client.delete(f"/documents/{target}", headers=as_user("a", "ada"))
        assert response.status_code == 204
        assert response.content == b""
        for sub in ("alice", "bob"):
            assert "public_a" not in await titles(client, "a", sub)
            gone = await client.get(f"/documents/{target}", headers=as_user("a", sub))
            assert gone.status_code == 404
    assert not await reads_canary(tenant, "bob", "public_a")
    assert (await stored(tenant))[target] == ("public_a", True, 2)


async def test_a_purge_removes_the_document_and_its_chunks(admins: World) -> None:
    tenant = admins.tenants["a"]
    target = admins.documents["hr_a"]
    async with api(admins) as client:
        response = await client.delete(
            f"/documents/{target}", params={"purge": "true"}, headers=as_user("a", "ada")
        )
        assert response.status_code == 204
        assert "hr_a" not in await titles(client, "a", "alice")
    assert target not in await stored(tenant)
    assert not await reads_canary(tenant, "alice", "hr_a")


async def test_a_soft_deleted_document_can_still_be_purged(admins: World) -> None:
    tenant = admins.tenants["a"]
    target = admins.documents["bob_a"]
    async with api(admins) as client:
        headers = as_user("a", "ada")
        assert (await client.delete(f"/documents/{target}", headers=headers)).status_code == 204
        response = await client.delete(
            f"/documents/{target}", params={"purge": "true"}, headers=headers
        )
    assert response.status_code == 204
    assert target not in await stored(tenant)


# --- new content -------------------------------------------------------------------------


async def test_new_content_replaces_the_chunks_and_keeps_the_acl(admins: World) -> None:
    tenant = admins.tenants["a"]
    async with api(admins, **ASK_ALL) as client:
        created = await upload(client, "a", "ada", "old_version", acl='["user:alice"]')
        document_id = created.json()["id"]
        assert "CANARY-old_version" in await ask_text(client, "a", "alice")

        response = await replace(client, "a", "ada", document_id, document("new_version"))
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": document_id,
            "title": "new_version",
            "chunks": 1,
            "unchanged": False,
        }

        answer = await ask_text(client, "a", "alice")
        assert "CANARY-old_version" not in answer
        assert "CANARY-new_version" in answer
        assert "new_version" in await titles(client, "a", "alice")
        # Same ACL as before: the admin who replaced it still can't read it.
        for sub in ("ada", "bob"):
            assert "new_version" not in await titles(client, "a", sub)
            assert "_version" not in await ask_text(client, "a", sub)

    assert await acl_of(tenant, UUID(document_id)) == {"user:alice"}
    assert (await stored(tenant))[UUID(document_id)] == ("new_version", False, 1)
    assert not await reads_canary(tenant, "alice", "old_version")
    assert await reads_canary(tenant, "alice", "new_version")


async def test_the_same_content_is_unchanged_and_writes_nothing(admins: World) -> None:
    tenant = admins.tenants["a"]
    async with api(admins) as client:
        created = await upload(client, "a", "ada", "same_bytes")
        before = await stored(tenant)
        response = await replace(client, "a", "ada", created.json()["id"], document("same_bytes"))
        audit = await client.get("/audit", headers=as_user("a", "ada"))
    assert response.status_code == 200, response.text
    assert response.json() == {**created.json(), "unchanged": True}
    assert await stored(tenant) == before
    assert [e["action"] for e in audit.json()["events"]][:1] == ["ingest"]


async def test_content_of_another_live_document_is_409(admins: World) -> None:
    tenant = admins.tenants["a"]
    async with api(admins) as client:
        await upload(client, "a", "ada", "first")
        second = await upload(client, "a", "ada", "second")
        before = await stored(tenant)
        response = await replace(client, "a", "ada", second.json()["id"], document("first"))
    assert response.status_code == 409
    assert response.json() == {"detail": "Another document has the same content"}
    assert await stored(tenant) == before


async def test_an_acl_with_new_content_is_422_and_changes_nothing(admins: World) -> None:
    tenant = admins.tenants["a"]
    target = admins.documents["hr_a"]
    before = await stored(tenant)
    async with api(admins) as client:
        response = await replace(
            client, "a", "ada", target, document("with_acl"), acl='["tenant:*"]'
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid upload"}
    assert await stored(tenant) == before
    assert await acl_of(tenant, target) == {"group:hr"}
