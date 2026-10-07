"""Phase 4 end to end: POST /ask and GET /audit with real tokens, the API and Ollama.

The question is embedded by OLLAMA_EMBED_MODEL and answered by
OLLAMA_CHAT_MODEL (or FakeChat with `make e2e FAKE_CHAT=1`). Nothing here checks
the model's wording: the assertions are about canaries (seed/corpus.py), the
citations the API builds from the retrieved set, and the audit rows.

The seed is embedded with FakeEmbeddings (conftest.py), so ranking is
meaningless. It doesn't need to mean anything: the API runs with RETRIEVAL_K =
E2E_RETRIEVAL_K (10), more than any seed user can read (alice: 8 chunks), so
every search returns everything the user may read. That makes what reaches the
model, and what the audit row records, exact.

Each question carries a nonce, so its audit rows (found by the question's
SHA-256) are this run's and no other's.
"""

import hashlib
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from ragmt.cli.main import EXIT_OK
from ragmt.cli.main import main as ragctl
from ragmt.domain import NO_CONTEXT_ANSWER
from seed.corpus import ALICE, DAVE, ERIN, TENANTS
from tests.e2e.conftest import E2E_RETRIEVAL_K, Keycloak, RealmAdmin, sync_permissions
from tests.e2e.test_stack import VISIBLE, bearer

pytestmark = [pytest.mark.e2e, pytest.mark.db]

# A completion on a CPU can take a while; CHAT_TIMEOUT_SECONDS bounds it server side.
ASK_TIMEOUT = httpx.Timeout(10.0, read=300.0)

CANARIES = {doc.title: doc.canary for tenant in TENANTS for doc in tenant.documents}
SUBS = {"alice": ALICE, "dave": DAVE, "erin": ERIN}
# One user per answer shape: Acme finance, Umbra finance, Acme with no groups.
ASKERS = ("alice", "dave", "erin")


def sha256(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


def nonce() -> str:
    return uuid4().hex[:8]


def ask(api: httpx.Client, token: str, question: str) -> dict[str, Any]:
    response = api.post(
        "/ask", json={"question": question}, headers=bearer(token), timeout=ASK_TIMEOUT
    )
    assert response.status_code == httpx.codes.OK, response.text
    body: dict[str, Any] = response.json()
    assert set(body) == {"answer", "citations", "model"}
    return body


def canaries_in(body: dict[str, Any]) -> set[str]:
    """Every seed canary anywhere in the response: answer, titles, headings."""
    text = " ".join(
        [body["answer"]] + [f"{c['title']} {c['heading'] or ''}" for c in body["citations"]]
    )
    return {canary for canary in CANARIES.values() if canary in text}


def cited_titles(body: dict[str, Any]) -> set[str]:
    return {c["title"] for c in body["citations"]}


def assert_only_allowed(body: dict[str, Any], username: str, seeded: dict[str, UUID]) -> None:
    """Citations and canaries name only documents `username` may read."""
    allowed = VISIBLE[username]
    assert cited_titles(body) <= allowed, body["citations"]
    assert {UUID(c["document_id"]) for c in body["citations"]} <= {seeded[t] for t in allowed}
    assert canaries_in(body) <= {CANARIES[t] for t in allowed}, body["answer"]


def audit_rows(api: httpx.Client, admin_token: str, question: str) -> list[dict[str, Any]]:
    """The admin's view of the ask rows for `question`, newest first."""
    response = api.get("/audit", headers=bearer(admin_token), params={"limit": 200})
    assert response.status_code == httpx.codes.OK, response.text
    return [
        event
        for event in response.json()["events"]
        if event["action"] == "ask" and event["details"]["question_sha256"] == sha256(question)
    ]


# --- the same question, three users ---------------------------------------------------


@pytest.fixture(scope="module")
def asked(api: httpx.Client, keycloak: Keycloak) -> tuple[str, dict[str, dict[str, Any]]]:
    """One question, asked once by each of ASKERS: (question, username -> response)."""
    question = f"What does each document say, and what is its reference code? (run {nonce()})"
    return question, {user: ask(api, keycloak.user_token(user), question) for user in ASKERS}


def test_the_same_question_gets_each_user_only_their_own_sources(
    asked: tuple[str, dict[str, dict[str, Any]]], seeded: dict[str, UUID], chat_model: str
) -> None:
    _, answers = asked
    for user, body in answers.items():
        assert_only_allowed(body, user, seeded)
        assert body["citations"], f"{user} got no citations: {body['answer']!r}"
        assert body["model"] == chat_model

    citation_sets = {frozenset(cited_titles(body)) for body in answers.values()}
    assert len(citation_sets) == len(ASKERS), citation_sets
    assert len({body["answer"] for body in answers.values()}) == len(ASKERS)
    # Erin is in no group: the tenant-wide handbook is all she can be shown.
    assert cited_titles(answers["erin"]) == {"Employee handbook"}


def test_admins_see_the_questions_in_the_audit_trail(
    api: httpx.Client,
    keycloak: Keycloak,
    asked: tuple[str, dict[str, dict[str, Any]]],
    chat_model: str,
) -> None:
    question, _ = asked
    acme = audit_rows(api, keycloak.user_token("bob"), question)
    umbra = audit_rows(api, keycloak.user_token("carol"), question)

    # Each admin sees their own tenant's askers, never the other tenant's.
    assert sorted(row["actor_sub"] for row in acme) == sorted([ALICE, ERIN])
    assert [row["actor_sub"] for row in umbra] == [DAVE]

    by_user = {row["actor_sub"]: row for row in acme + umbra}
    for user in ASKERS:
        row = by_user[SUBS[user]]
        assert 0 < len(row["chunk_ids"]) < E2E_RETRIEVAL_K
        assert len(row["details"]["scores"]) == len(row["chunk_ids"])
        assert row["details"]["model"] == chat_model
        assert "question" not in row["details"]  # AUDIT_STORE_QUERY_TEXT defaults to false
    # Everything erin can read, alice can read too (alice has more besides).
    assert set(by_user[ERIN]["chunk_ids"]) < set(by_user[ALICE]["chunk_ids"])
    assert not set(by_user[DAVE]["chunk_ids"]) & set(by_user[ALICE]["chunk_ids"])


@pytest.mark.parametrize("username", ["alice", "dave", "erin"])
def test_the_audit_trail_is_not_found_for_non_admins(
    api: httpx.Client, keycloak: Keycloak, username: str
) -> None:
    response = api.get("/audit", headers=bearer(keycloak.user_token(username)))
    assert response.status_code == httpx.codes.NOT_FOUND
    assert response.json() == {"detail": "Not Found"}


# --- revocation -----------------------------------------------------------------------


def test_leaving_a_group_in_keycloak_removes_its_citations_after_one_sync(
    api: httpx.Client, keycloak: Keycloak, admin: RealmAdmin, seeded: dict[str, UUID]
) -> None:
    """The same token, issued before the change, can no longer get the budget cited."""
    token = keycloak.user_token("alice")
    bob = keycloak.user_token("bob")
    before_q = f"What is the Q3 fleet budget, and what is its reference code? (run {nonce()})"
    before = ask(api, token, before_q)
    assert "Q3 budget" in cited_titles(before), before
    assert_only_allowed(before, "alice", seeded)

    admin.leave_group(ALICE, "/acme/finance")
    try:
        sync_permissions()
        after_q = f"What is the Q3 fleet budget, and what is its reference code? (run {nonce()})"
        after = ask(api, token, after_q)

        assert "Q3 budget" not in cited_titles(after)
        assert CANARIES["Q3 budget"] not in canaries_in(after)
        assert_only_allowed(after, "alice", seeded)
        # What was retrieved shrank: the budget's chunks are gone, the rest stayed.
        [before_row] = audit_rows(api, bob, before_q)
        [after_row] = audit_rows(api, bob, after_q)
        assert set(after_row["chunk_ids"]) < set(before_row["chunk_ids"])
    finally:
        admin.join_group(ALICE, "/acme/finance")
        sync_permissions()


# --- nothing readable -----------------------------------------------------------------


@pytest.fixture
def handbook_hidden(
    api: httpx.Client, keycloak: Keycloak, seeded: dict[str, UUID]
) -> Iterator[None]:
    """For the test's duration, the handbook is engineering-only: erin can read nothing."""
    bob = keycloak.user_token("bob")
    acl = f"/documents/{seeded['Employee handbook']}/acl"

    def set_acl(principals: list[str]) -> None:
        response = api.put(acl, json={"principals": principals}, headers=bearer(bob))
        assert response.status_code == httpx.codes.OK, response.text

    set_acl(["group:engineering"])
    try:
        yield
    finally:
        set_acl(["tenant:*"])


@pytest.mark.usefixtures("handbook_hidden")
def test_a_user_with_nothing_readable_gets_the_fixed_answer_without_a_model(
    api: httpx.Client, keycloak: Keycloak
) -> None:
    question = f"What are the core working hours? (run {nonce()})"
    body = ask(api, keycloak.user_token("erin"), question)

    assert body == {"answer": NO_CONTEXT_ANSWER, "citations": [], "model": None}
    # The retrieval is still audited: nothing retrieved, no model called.
    [row] = audit_rows(api, keycloak.user_token("bob"), question)
    assert row["actor_sub"] == ERIN
    assert row["chunk_ids"] == []
    assert row["details"]["model"] is None


# --- the CLI --------------------------------------------------------------------------


def test_ragctl_asks_the_api_as_a_dev_user(
    api: httpx.Client, keycloak: Keycloak, tmp_path: Path
) -> None:
    """`ragctl --dev-user erin ask ...` against the running API prints her sources."""
    out, err = io.StringIO(), io.StringIO()
    code = ragctl(
        [
            "--api-url",
            str(api.base_url).rstrip("/"),
            "--issuer",
            keycloak.issuer,
            "--dev-user",
            "erin",
            "ask",
            f"What are the core working hours? (run {nonce()})",
        ],
        environ={
            "KC_DEMO_USER_PASSWORD": os.environ["KC_DEMO_USER_PASSWORD"],
            "RAGCTL_CONFIG_DIR": str(tmp_path),
        },
        stdout=out,
        stderr=err,
    )

    assert code == EXIT_OK, err.getvalue()
    printed = out.getvalue()
    assert "Sources:" in printed
    assert "Employee handbook" in printed
    hidden = [title for title in CANARIES if title not in VISIBLE["erin"]]
    assert not [title for title in hidden if title in printed or CANARIES[title] in printed]
