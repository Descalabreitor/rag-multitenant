"""Phase 3 end to end: uploads, ACL changes and deletes through the real API.

Real Keycloak tokens, the API embedding with Ollama, RLS in PostgreSQL. Bob is
Acme's admin and carol Umbra's (seed/corpus.py); neither is in finance, so an
admin reads only what an ACL gives them, uploads included (ADR 0008).

Each test uploads its own .docx, built by seed/docx.py with a fresh nonce, so
its bytes are new on every run and no earlier upload can deduplicate it. The
`uploads` fixture purges whatever a test uploaded, and `make seed` (the
`seeded` fixture) hard-deletes any leftover from an interrupted run.
"""

import json
from collections.abc import Callable, Iterator
from uuid import UUID, uuid4

import httpx
import pytest

from seed.corpus import ERIN
from seed.docx import build_docx
from tests.e2e.conftest import Keycloak
from tests.e2e.test_stack import VISIBLE, bearer, titles

pytestmark = [pytest.mark.e2e, pytest.mark.db]

USERS = list(VISIBLE)
NOT_FOUND = {"detail": "Document not found"}
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# Generous: the first upload may wait for Ollama to load the model.
UPLOAD_TIMEOUT = 120.0

Upload = Callable[..., httpx.Response]


def sample_docx(title: str) -> bytes:
    nonce = uuid4().hex
    return build_docx(
        f"# {title}\n\n"
        "## Route changes\n\n"
        "The north depot moves its night shift to the new sorting line in November.\n\n"
        f"Reference: CANARY-E2E-{nonce}\n"
    )


def who_sees(api: httpx.Client, tokens: dict[str, str], title: str) -> set[str]:
    """The seed users whose GET /documents lists `title`."""
    return {user for user, token in tokens.items() if title in titles(api, token)}


@pytest.fixture
def tokens(keycloak: Keycloak) -> dict[str, str]:
    """One token per seed user, issued before anything in the test changes."""
    return {user: keycloak.user_token(user) for user in USERS}


@pytest.fixture
def uploads(api: httpx.Client, tokens: dict[str, str]) -> Iterator[Upload]:
    """POST /documents as a given user; every document created is purged afterwards."""
    created: set[UUID] = set()

    def upload(
        user: str, data: bytes, filename: str = "notes.docx", acl: list[str] | None = None
    ) -> httpx.Response:
        form = {"acl": json.dumps(acl)} if acl is not None else None
        response = api.post(
            "/documents",
            headers=bearer(tokens[user]),
            files={"file": (filename, data, DOCX)},
            data=form,
            timeout=UPLOAD_TIMEOUT,
        )
        if response.status_code == httpx.codes.CREATED:
            created.add(UUID(response.json()["id"]))
        return response

    yield upload
    for document_id in created:
        response = api.delete(
            f"/documents/{document_id}", headers=bearer(tokens["bob"]), params={"purge": "true"}
        )
        assert response.status_code in (204, 404), response.text


# --- upload ------------------------------------------------------------------------------


def test_an_uploaded_docx_is_visible_to_exactly_its_acl(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    title = "Depot shift plan"
    response = uploads("bob", sample_docx(title), "depot-shift-plan.docx", ["group:finance"])

    assert response.status_code == httpx.codes.CREATED, response.text
    body = response.json()
    assert body["title"] == title
    assert body["chunks"] >= 1
    assert body["unchanged"] is False
    assert set(body) == {"id", "title", "chunks", "unchanged"}  # never a vector

    # Acme's finance only: not bob, who uploaded it, and not dave, in Umbra's finance.
    assert who_sees(api, tokens, title) == {"alice"}
    path = f"/documents/{body['id']}"
    assert api.get(path, headers=bearer(tokens["alice"])).status_code == httpx.codes.OK
    for user in ("bob", "erin", "carol", "dave"):
        response = api.get(path, headers=bearer(tokens[user]))
        assert response.status_code == httpx.codes.NOT_FOUND, user
        assert response.json() == NOT_FOUND


def test_an_upload_without_an_acl_is_readable_only_by_the_uploader(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    title = "Admin scratch notes"
    response = uploads("bob", sample_docx(title))

    assert response.status_code == httpx.codes.CREATED, response.text
    assert who_sees(api, tokens, title) == {"bob"}


def test_uploading_the_same_file_twice_stores_it_once(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    title = "Fleet maintenance calendar"
    data = sample_docx(title)
    first = uploads("bob", data, "fleet.docx", ["group:finance"])
    assert first.status_code == httpx.codes.CREATED, first.text

    # Same bytes under another name and another ACL: nothing is written.
    second = uploads("bob", data, "fleet-copy.docx", [f"user:{ERIN}"])

    assert second.status_code == httpx.codes.CREATED, second.text
    assert second.json() == {**first.json(), "unchanged": True}
    listed = [d["title"] for d in api.get("/documents", headers=bearer(tokens["alice"])).json()]
    assert listed.count(title) == 1
    assert who_sees(api, tokens, title) == {"alice"}  # the ACL is left as it was


# --- ACL changes -------------------------------------------------------------------------


def test_changing_the_acl_moves_access_for_the_same_tokens(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    title = "Carrier contract renewal"
    response = uploads("bob", sample_docx(title), acl=["group:finance"])
    assert response.status_code == httpx.codes.CREATED, response.text
    document_id = response.json()["id"]
    path = f"/documents/{document_id}"
    assert who_sees(api, tokens, title) == {"alice"}

    changed = api.put(
        f"{path}/acl", headers=bearer(tokens["bob"]), json={"principals": [f"user:{ERIN}"]}
    )

    assert changed.status_code == httpx.codes.OK, changed.text
    assert changed.json() == {"id": document_id, "principals": [f"user:{ERIN}"]}
    assert who_sees(api, tokens, title) == {"erin"}
    assert api.get(path, headers=bearer(tokens["alice"])).status_code == httpx.codes.NOT_FOUND
    assert api.get(path, headers=bearer(tokens["erin"])).status_code == httpx.codes.OK


# --- non-admins --------------------------------------------------------------------------


def test_a_non_admins_upload_is_forbidden(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    title = "Not an admin's to upload"
    for user in ("alice", "erin", "dave"):
        response = uploads(user, sample_docx(title), acl=["tenant:*"])
        assert response.status_code == httpx.codes.FORBIDDEN, user
        assert response.json() == {"detail": "Forbidden"}
    assert who_sees(api, tokens, title) == set()


def test_a_non_admins_acl_change_or_delete_is_not_found(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    """Alice can read the document, and still can't learn more about it than a stranger."""
    title = "Fuel surcharge table"
    response = uploads("bob", sample_docx(title), acl=["group:finance"])
    assert response.status_code == httpx.codes.CREATED, response.text
    path = f"/documents/{response.json()['id']}"

    # Alice and erin are Acme non-admins; carol is Umbra's admin.
    for user in ("alice", "erin", "carol"):
        widened = api.put(
            f"{path}/acl", headers=bearer(tokens[user]), json={"principals": ["tenant:*"]}
        )
        assert widened.status_code == httpx.codes.NOT_FOUND, user
        assert widened.json() == NOT_FOUND
        deleted = api.delete(path, headers=bearer(tokens[user]), params={"purge": "true"})
        assert deleted.status_code == httpx.codes.NOT_FOUND, user

    assert who_sees(api, tokens, title) == {"alice"}


# --- deletes -----------------------------------------------------------------------------


def test_soft_delete_hides_the_document_and_purge_removes_it(
    api: httpx.Client, tokens: dict[str, str], uploads: Upload
) -> None:
    title = "Warehouse lease terms"
    data = sample_docx(title)
    response = uploads("bob", data, acl=["tenant:*"])
    assert response.status_code == httpx.codes.CREATED, response.text
    path = f"/documents/{response.json()['id']}"
    assert who_sees(api, tokens, title) == {"alice", "bob", "erin"}

    soft = api.delete(path, headers=bearer(tokens["bob"]))

    assert soft.status_code == httpx.codes.NO_CONTENT
    assert who_sees(api, tokens, title) == set()
    for user in ("alice", "bob", "erin"):
        assert api.get(path, headers=bearer(tokens[user])).status_code == httpx.codes.NOT_FOUND
    # Hidden from the API, but still there to purge.
    assert api.delete(path, headers=bearer(tokens["bob"])).status_code == httpx.codes.NOT_FOUND

    purged = api.delete(path, headers=bearer(tokens["bob"]), params={"purge": "true"})

    assert purged.status_code == httpx.codes.NO_CONTENT
    # Gone: a second purge finds nothing, soft-deleted or not.
    again = api.delete(path, headers=bearer(tokens["bob"]), params={"purge": "true"})
    assert again.status_code == httpx.codes.NOT_FOUND
    # And the same bytes are a new document, not a duplicate of the purged one.
    reupload = uploads("bob", data, acl=["tenant:*"])
    assert reupload.status_code == httpx.codes.CREATED, reupload.text
    assert reupload.json()["unchanged"] is False
    assert f"/documents/{reupload.json()['id']}" != path
