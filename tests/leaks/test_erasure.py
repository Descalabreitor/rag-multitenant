"""Erasure: after `DELETE /documents/{id}?purge=true` a document's content is nowhere.

This is the GDPR path. A soft delete hides a document but keeps its rows; a
purge must leave nothing of its content behind for any reader: not in
`/documents`, not in `/ask` (answers or citations), not in the tables. The
audit trail keeps what it always kept, the chunk ids of past retrievals and the
ACL the purge removed, and never any content.

The world and the app are those of test_api_writes.py: the real app with fake
embeddings, and a FakeChat that echoes its prompt, so a canary that reached
the model shows up in the answer.
"""

from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from tests.db.pg import session
from tests.leaks.conftest import INGEST, World
from tests.leaks.test_api_writes import ASK_ALL, NOT_FOUND, api, as_user, ask_text, upload

pytestmark = [pytest.mark.db, pytest.mark.leaks]

# Every user of tenant A who could read the tenant-wide document: ada (admin),
# alice (hr) and bob (his own document).
READERS = ("ada", "alice", "bob")


async def rows_of(tenant: UUID, document_id: UUID, content: str) -> dict[str, int]:
    """What app_ingest, which sees the whole tenant (deleted rows too), still holds."""
    async with session(INGEST, tenant) as conn:
        return {
            "documents": await conn.fetchval(
                "SELECT count(*) FROM documents WHERE id = $1", document_id
            ),
            "document_acl": await conn.fetchval(
                "SELECT count(*) FROM document_acl WHERE document_id = $1", document_id
            ),
            "chunks": await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE document_id = $1", document_id
            ),
            "chunks_with_content": await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE strpos(content, $1) > 0", content
            ),
        }


async def chunk_ids_of(tenant: UUID, document_id: UUID) -> set[str]:
    async with session(INGEST, tenant) as conn:
        rows = await conn.fetch("SELECT id FROM chunks WHERE document_id = $1", document_id)
    return {str(r["id"]) for r in rows}


async def audit_events(client: httpx.AsyncClient) -> tuple[list[dict[str, Any]], str]:
    response = await client.get("/audit", params={"limit": 200}, headers=as_user("a", "ada"))
    assert response.status_code == 200, response.text
    return response.json()["events"], response.text


@pytest.mark.parametrize("soft_deleted_first", [False, True])
async def test_a_purged_document_leaves_no_content_behind(
    admins: World, soft_deleted_first: bool
) -> None:
    tenant = admins.tenants["a"]
    # The title is part of the canary (`CANARY-<title>`), so checking for the
    # title catches both.
    title = f"erase_{uuid4().hex}"
    async with api(admins, **ASK_ALL) as client:
        created = await upload(client, "a", "ada", title, acl='["tenant:*"]')
        assert created.status_code == 201, created.text
        document_id = UUID(created.json()["id"])
        path = f"/documents/{document_id}"
        chunk_ids = await chunk_ids_of(tenant, document_id)
        assert chunk_ids

        # Before: every reader gets it, so the checks below aren't vacuous.
        for sub in READERS:
            answer = await ask_text(client, "a", sub)
            assert f"CANARY-{title}" in answer
            assert str(document_id) in answer  # cited

        headers = as_user("a", "ada")
        if soft_deleted_first:
            assert (await client.delete(path, headers=headers)).status_code == 204
        purged = await client.delete(path, params={"purge": "true"}, headers=headers)
        assert purged.status_code == 204

        # After: nothing, for anyone.
        for sub in READERS:
            listing = await client.get("/documents", headers=as_user("a", sub))
            assert listing.status_code == 200
            single = await client.get(path, headers=as_user("a", sub))
            assert single.status_code == 404
            assert single.json() == NOT_FOUND
            answer = await ask_text(client, "a", sub)
            for text in (listing.text, single.text, answer):
                assert title not in text, sub
                assert str(document_id) not in text, sub
            assert not any(chunk_id in answer for chunk_id in chunk_ids)

        events, raw = await audit_events(client)

    assert await rows_of(tenant, document_id, f"CANARY-{title}") == {
        "documents": 0,
        "document_acl": 0,
        "chunks": 0,
        "chunks_with_content": 0,
    }

    # The audit trail: ids, never content or the title.
    assert title not in raw
    asks = [e for e in events if e["action"] == "ask"]  # newest first
    after, before = asks[: len(READERS)], asks[len(READERS) : 2 * len(READERS)]
    for event in before:
        # The retrievals before the purge still name the purged chunks.
        assert chunk_ids <= set(event["chunk_ids"])
    for event in after:
        assert not chunk_ids & set(event["chunk_ids"])

    deletes = {
        e["action"]: e["details"]
        for e in events
        if e["action"] in {"soft_delete", "hard_delete"}
        and e["details"]["document_id"] == str(document_id)
    }
    # The ACL the delete removed is in its row (ADR 0008). After a soft delete the
    # ACL is already gone, so the purge removes none and the soft delete holds it.
    removed_by = "soft_delete" if soft_deleted_first else "hard_delete"
    assert set(deletes) == {removed_by, "hard_delete"}
    assert deletes[removed_by]["old_principals"] == ["tenant:*"]
    if soft_deleted_first:
        assert deletes["hard_delete"]["old_principals"] == []
