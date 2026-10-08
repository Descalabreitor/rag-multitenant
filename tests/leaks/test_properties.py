"""Leak case 13: random worlds and random operations, checked against an oracle.

A Hypothesis state machine builds 2 or 3 fresh tenants (random ids, so examples
never see each other's rows) that share user subs and group names on purpose.
It then runs random operations through the real write paths, as app_ingest:

- uploads, with or without an ACL, and uploads of a document's bytes again
  (`IngestService` with FakeEmbeddings and the test chunker, one chunk per paragraph);
- content replacements, ACL changes, soft deletes and purges;
- membership changes (`SqlMembershipStore`, the store permsync writes with);
- writes that name another tenant than the document's.

Each operation is replayed on the oracle (`tests/leaks/oracle.py`), which works
out in plain Python who may read what. After setup and after every step, every
user of every tenant is checked against it through the read paths, as app_rw:

- `GET /documents` lists exactly the oracle's readable documents, and
  `GET /documents/{id}` is 404 for a document the user may not read;
- `SELECT … FROM chunks` returns exactly the oracle's readable chunks, and the
  retriever returns a subset of them;
- `POST /ask` (FakeChat, which echoes its whole prompt) cites only readable
  chunks and holds no canary of anything else.

The searches are aimed at what the user must not see: the query is the exact
text of a chunk they may not read, and FakeEmbeddings maps equal texts to equal
vectors, so a leaked chunk would come first. Every title and paragraph is a
unique canary, `CANARY-<nonce>-t` or `CANARY-<nonce>-p<i>`.

The app runs once per test (lifespan, app_rw engine, RLS, the pgvector
retriever, AskService), with `get_principal` replaced as in test_ask.py:
`X-Test-Principal: <tenant id>:<sub>`. Each check is counted in
`tests.leaks.conftest.TALLY`, which `make leaks` writes to docs/results/leaks.md.
Rows are left behind like the rest of the leak suite does (see conftest.py).
"""

import asyncio
import os
import re
from collections.abc import Awaitable, Coroutine, Iterator, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Header
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    multiple,
    rule,
    run_state_machine_as_test,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import ragmt.api.app
from ragmt.adapters.llm import FakeChat
from ragmt.adapters.llm.fake import FakeEmbeddings
from ragmt.auth.dependencies import get_principal
from ragmt.domain import NO_CONTEXT_ANSWER, ChatMessage, Principal
from ragmt.ingest.service import DocumentNotFoundError, IngestService
from ragmt.permsync.directory import TenantSnapshot
from ragmt.permsync.store import SqlMembershipStore
from ragmt.retrieval import PgVectorRetriever
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session
from tests.ingest_helpers import FakeConverter, chunk_paragraphs, markdown_document
from tests.leaks.conftest import INGEST, READER, TALLY
from tests.leaks.oracle import Oracle

pytestmark = [pytest.mark.db, pytest.mark.leaks]

# The users checked in every tenant. The same subs and group names in every
# tenant, so a leak through "same name, other tenant" shows up.
USERS = ("alice", "bob", "carol")
# `admins` is the tenant admin group: it must grant no read access beyond its ACLs.
GROUPS = ("hr", "finance", "admins")
# Principals an ACL may hold; `user:mallory` and `group:nobody` match no checked user.
PRINCIPALS = (
    *(f"user:{u}" for u in USERS),
    "user:mallory",
    *(f"group:{g}" for g in GROUPS),
    "group:nobody",
    "tenant:*",
)

users = st.sampled_from(USERS)
acls = st.lists(st.sampled_from(PRINCIPALS), min_size=1, max_size=3)
memberships = st.frozensets(st.tuples(users, st.sampled_from(GROUPS)), max_size=5)
paragraph_counts = st.integers(min_value=1, max_value=3)
# A tenant, as an index into the example's tenants (taken modulo their number).
tenant_picks = st.integers(min_value=0, max_value=2)

ACTOR = "property-admin"
PERMSYNC_ACTOR = "property-permsync"
# Above any tenant's readable chunk count here, so a search could return them all.
SEARCH_K = 40

CANARY = re.compile(r"CANARY-[0-9a-f]{12}-(?:t|p\d+)")
MARKER = re.compile(r"\[doc:([0-9a-f-]{36})#(\d+)\]")


class EchoChat(FakeChat):
    """FakeChat without its call log, which would grow with every /ask of the run."""

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        reply = await super().complete(system, messages)
        self.calls.clear()
        return reply


@dataclass
class Harness:
    """What every example shares: the running app and the writers, on one event loop."""

    runner: asyncio.Runner
    client: httpx.AsyncClient
    reader: AsyncEngine
    retriever: PgVectorRetriever
    service: IngestService
    store: SqlMembershipStore
    embedder: FakeEmbeddings

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        return self.runner.run(coro)


async def _open(runner: asyncio.Runner, stack: AsyncExitStack, settings: Settings) -> Harness:
    app = ragmt.api.app.create_app(settings)

    def principal(x_test_principal: Annotated[str, Header()]) -> Principal:
        tenant, sub = x_test_principal.split(":", 1)
        return Principal(sub=sub, tenant_id=UUID(tenant))

    app.dependency_overrides[get_principal] = principal
    await stack.enter_async_context(app.router.lifespan_context(app))
    client = await stack.enter_async_context(
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    )
    writer = create_async_engine(
        settings.ingest_database_url.get_secret_value(), pool_size=2, max_overflow=0
    )
    stack.push_async_callback(writer.dispose)
    embedder = FakeEmbeddings(settings.embedding_dim)
    return Harness(
        runner=runner,
        client=client,
        reader=app.state.engine,
        retriever=PgVectorRetriever.from_settings(settings),
        service=IngestService(writer, FakeConverter(), chunk_paragraphs, embedder, settings),
        store=SqlMembershipStore(writer, PERMSYNC_ACTOR),
        embedder=embedder,
    )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    for var in (READER, INGEST):
        if not os.environ.get(var):
            pytest.skip(f"{var} is not set")
    monkeypatch.setattr(ragmt.api.app, "build_chat_provider", lambda _settings: EchoChat())
    # Nine users are checked at once, each holding at most one app_rw connection.
    settings = Settings(
        database_pool_size=len(USERS) * 3 + 1,
        database_max_overflow=0,
        llm_provider="fake",
        hnsw_ef_search=SEARCH_K,
    )
    with asyncio.Runner() as runner:
        stack = AsyncExitStack()
        h = runner.run(_open(runner, stack, settings))
        try:
            yield h
        finally:
            runner.run(stack.aclose())


@dataclass(frozen=True)
class DocRef:
    tenant: str
    id: UUID

    def __repr__(self) -> str:  # short, for Hypothesis' printout of a failing run
        return f"DocRef({self.tenant!r}, {str(self.id)[:8]})"


async def _found(write: Awaitable[object]) -> bool:
    """Whether a write found its document (it raises DocumentNotFoundError if not)."""
    try:
        await write
    except DocumentNotFoundError:
        return False
    return True


class LeakMachine(RuleBasedStateMachine):
    documents = Bundle("documents")

    def __init__(self, harness: Harness) -> None:
        super().__init__()
        self.h = harness
        self.oracle = Oracle()
        self.tenant_ids: dict[str, UUID] = {}
        self.checks_run = 0
        # Every paragraph written in this example, replaced ones included: the probes.
        self.written: list[str] = []
        TALLY.examples += 1

    # --- setup and writes ---------------------------------------------------------

    # The setup writes a populated world in one step: tenants, their memberships
    # and a few documents. Built up rule by rule instead, most short examples
    # would only shuffle memberships of tenants with nothing in them.
    @initialize(
        target=documents,
        count=st.integers(min_value=2, max_value=3),
        members=st.lists(memberships, min_size=3, max_size=3),
        uploads=st.lists(
            st.tuples(tenant_picks, users, st.none() | acls, paragraph_counts),
            min_size=1,
            max_size=6,
        ),
    )
    def setup(
        self,
        count: int,
        members: list[frozenset[tuple[str, str]]],
        uploads: list[tuple[int, str, list[str] | None, int]],
    ) -> object:
        for i in range(count):
            key = f"t{i}"
            self.tenant_ids[key] = uuid4()
            self.oracle.add_tenant(key)
            self._sync(key, members[i])
        return multiple(
            *(self._upload(self._tenant(t), uploader, acl, n) for t, uploader, acl, n in uploads)
        )

    @rule(
        target=documents,
        pick=tenant_picks,
        uploader=users,
        acl=st.none() | acls,
        n=paragraph_counts,
    )
    def upload(self, pick: int, uploader: str, acl: list[str] | None, n: int) -> DocRef:
        return self._upload(self._tenant(pick), uploader, acl, n)

    @rule(target=documents, doc=documents, uploader=users, acl=st.none() | acls)
    def upload_again(self, doc: DocRef, uploader: str, acl: list[str] | None) -> object:
        """Upload a document's current bytes again, in its own tenant."""
        old = self.oracle.documents[doc.id]
        duplicate = self.oracle.live_duplicate(doc.tenant, old.data)
        result = self.h.run(
            self.h.service.ingest(self.tenant_ids[doc.tenant], uploader, old.data, "doc.md", acl)
        )
        if duplicate is not None:
            # The live copy comes back as it is: the new ACL is not applied.
            assert result.unchanged
            assert result.document_id == duplicate.id
            return multiple()
        assert not result.unchanged
        self.oracle.upload(
            result.document_id, doc.tenant, uploader, old.title, old.data, old.paragraphs, acl
        )
        return DocRef(doc.tenant, result.document_id)

    @rule(doc=documents, n=paragraph_counts)
    def replace_content(self, doc: DocRef, n: int) -> None:
        self._replace(doc.tenant, doc, n)

    @rule(doc=documents, acl=acls)
    def change_acl(self, doc: DocRef, acl: list[str]) -> None:
        self._set_acl(doc.tenant, doc, acl)

    @rule(doc=documents)
    def soft_delete(self, doc: DocRef) -> None:
        self._soft_delete(doc.tenant, doc)

    @rule(doc=documents)
    def purge(self, doc: DocRef) -> None:
        self._purge(doc.tenant, doc)

    @rule(
        doc=documents,
        pick=st.integers(min_value=0, max_value=1),
        op=st.sampled_from(["replace", "acl", "soft_delete", "purge"]),
        acl=acls,
    )
    def write_from_other_tenant(self, doc: DocRef, pick: int, op: str, acl: list[str]) -> None:
        """A write that names another tenant than the document's must find nothing."""
        others = [t for t in self.tenant_ids if t != doc.tenant]
        tenant = others[pick % len(others)]
        match op:
            case "replace":
                found = self._replace(tenant, doc, 1)
            case "acl":
                found = self._set_acl(tenant, doc, acl)
            case "soft_delete":
                found = self._soft_delete(tenant, doc)
            case _:
                found = self._purge(tenant, doc)
        assert not found
        TALLY.checks["foreign_write"] += 1

    @rule(pick=tenant_picks, members=memberships)
    def sync_memberships(self, pick: int, members: frozenset[tuple[str, str]]) -> None:
        self._sync(self._tenant(pick), members)

    @rule(pick=tenant_picks, user=users, group=st.sampled_from(GROUPS))
    def toggle_membership(self, pick: int, user: str, group: str) -> None:
        tenant = self._tenant(pick)
        self._sync(tenant, self.oracle.memberships[tenant] ^ {(user, group)})

    def _tenant(self, pick: int) -> str:
        keys = list(self.tenant_ids)
        return keys[pick % len(keys)]

    def _upload(self, tenant: str, uploader: str, acl: list[str] | None, n: int) -> DocRef:
        title, paragraphs, data = self._new_content(n)
        result = self.h.run(
            self.h.service.ingest(self.tenant_ids[tenant], uploader, data, "doc.md", acl)
        )
        assert not result.unchanged
        self.oracle.upload(result.document_id, tenant, uploader, title, data, paragraphs, acl)
        return DocRef(tenant, result.document_id)

    # Each helper below runs the write, replays it on the oracle and checks that
    # both agree on whether the document was found. Returns that answer.

    def _replace(self, tenant: str, doc: DocRef, n: int) -> bool:
        title, paragraphs, data = self._new_content(n)
        found = self.h.run(
            _found(self.h.service.replace(self.tenant_ids[tenant], ACTOR, doc.id, data, "doc.md"))
        )
        assert found == self.oracle.replace(tenant, doc.id, title, data, paragraphs), doc
        return found

    def _set_acl(self, tenant: str, doc: DocRef, acl: list[str]) -> bool:
        found = self.h.run(
            _found(self.h.service.set_acl(self.tenant_ids[tenant], ACTOR, doc.id, acl))
        )
        assert found == self.oracle.set_acl(tenant, doc.id, acl), doc
        return found

    def _soft_delete(self, tenant: str, doc: DocRef) -> bool:
        found = self.h.run(
            _found(self.h.service.soft_delete(self.tenant_ids[tenant], ACTOR, doc.id))
        )
        assert found == self.oracle.soft_delete(tenant, doc.id), doc
        return found

    def _purge(self, tenant: str, doc: DocRef) -> bool:
        found = self.h.run(
            _found(self.h.service.hard_delete(self.tenant_ids[tenant], ACTOR, doc.id))
        )
        assert found == self.oracle.purge(tenant, doc.id), doc
        return found

    def _sync(self, tenant: str, members: frozenset[tuple[str, str]]) -> None:
        snapshot = TenantSnapshot(self.tenant_ids[tenant], None, f"property-{tenant}", members)
        self.h.run(self.h.store.apply(snapshot))
        self.oracle.set_memberships(tenant, members)

    # --- the checks -----------------------------------------------------------------

    @invariant()
    def every_user_sees_exactly_what_the_oracle_allows(self) -> None:
        if not self.tenant_ids:  # before the tenants exist
            return
        TALLY.steps += 1
        self.checks_run += 1

        async def check_everyone() -> None:
            await asyncio.gather(
                *(self._check(tenant, user) for tenant in self.tenant_ids for user in USERS)
            )

        self.h.run(check_everyone())

    async def _check(self, tenant: str, user: str) -> None:
        who = f"{user} in {tenant}"
        headers = {"X-Test-Principal": f"{self.tenant_ids[tenant]}:{user}"}
        readable = self.oracle.readable(tenant, user)
        readable_ids = {d.id for d in readable}
        readable_chunks = self.oracle.readable_chunks(tenant, user)
        readable_texts = self.oracle.readable_texts(tenant, user)
        client = self.h.client

        # GET /documents: exactly the readable documents, with their current titles.
        response = await client.get("/documents", headers=headers)
        assert response.status_code == 200, response.text
        listed = {(UUID(d["id"]), d["title"]) for d in response.json()}
        assert listed == {(d.id, d.title) for d in readable}, who
        TALLY.checks["documents"] += 1

        # GET /documents/{id}: 404 for a document they may not read (any tenant,
        # deleted and purged ones included), a different one at each step.
        hidden = [d for d in self.oracle.documents.values() if d.id not in readable_ids]
        if hidden:
            doc = hidden[self.checks_run % len(hidden)]
            response = await client.get(f"/documents/{doc.id}", headers=headers)
            assert response.status_code == 404, (who, doc)
            TALLY.checks["document_by_id"] += 1

        # The search query: the text of a chunk they may not read, if there is one.
        probe = self._probe(readable_chunks)
        vector = await self.h.embedder.embed_query(probe)
        async with tenant_session(self.h.reader, self.tenant_ids[tenant], user) as conn:
            rows = await conn.execute(text("SELECT document_id, ordinal, content FROM chunks"))
            in_table = {(row.document_id, row.ordinal, row.content) for row in rows}
            found = await self.h.retriever.search(conn, vector, SEARCH_K)
        assert in_table == readable_chunks, who
        TALLY.checks["chunks"] += 1
        retrieved = {(c.document_id, c.ordinal, c.content) for c in found}
        assert retrieved <= readable_chunks, (who, retrieved - readable_chunks)
        TALLY.checks["retrieval"] += 1

        # POST /ask with the same text as the question.
        response = await client.post("/ask", json={"question": probe}, headers=headers)
        assert response.status_code == 200, response.text
        body = response.json()
        cited = {UUID(c["document_id"]) for c in body["citations"]}
        assert cited <= readable_ids, (who, cited - readable_ids)
        # FakeChat echoes the prompt, whose source blocks carry each chunk's marker.
        markers = {(UUID(d), int(o)) for d, o in MARKER.findall(response.text)}
        allowed = {(d, o) for d, o, _ in readable_chunks}
        assert markers <= allowed, (who, markers - allowed)
        TALLY.checks["ask_citations"] += 1
        # The question is echoed too; its own canary is the caller's input, not a leak.
        canaries = set(CANARY.findall(response.text)) - set(CANARY.findall(probe))
        assert canaries <= readable_texts, (who, canaries - readable_texts)
        if not readable_chunks:
            assert body == {"answer": NO_CONTEXT_ANSWER, "citations": [], "model": None}, who
        TALLY.checks["ask_canaries"] += 1

    def _probe(self, readable: frozenset[tuple[UUID, int, str]]) -> str:
        """A paragraph of this example the user may not read, else any, else filler."""
        allowed = {content for _, _, content in readable}
        candidates = [p for p in self.written if p not in allowed] or self.written
        if not candidates:
            return "What do the documents say?"
        return candidates[self.checks_run % len(candidates)]

    def _new_content(self, paragraphs: int) -> tuple[str, tuple[str, ...], bytes]:
        """A title and paragraphs that are fresh canaries, so the bytes are unique too."""
        nonce = uuid4().hex[:12]
        title = f"CANARY-{nonce}-t"
        texts = tuple(f"CANARY-{nonce}-p{i}" for i in range(paragraphs))
        self.written.extend(texts)
        return title, texts, markdown_document(title, *texts)


def test_no_user_ever_reads_what_the_oracle_forbids(harness: Harness) -> None:
    run_state_machine_as_test(lambda: LeakMachine(harness))  # type: ignore[no-untyped-call]
