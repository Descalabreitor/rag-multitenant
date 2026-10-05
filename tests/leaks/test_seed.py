"""The seed corpus loads through the ingestion pipeline, repeatably, and each canary
reaches exactly its readers.

Later phases (retrieval, eval, the API leak tests) rely on this corpus, so who
may read what is written out by hand here rather than derived from corpus.py.
The seed runs with FakeEmbeddings, whatever LLM_PROVIDER says: these tests check
who sees what, not ranking.
"""

import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ragmt.adapters.llm import FakeEmbeddings
from ragmt.ingest.service import source_hash
from ragmt.settings import Settings
from seed.corpus import ACME, ALICE, BOB, CAROL, DAVE, ERIN, TENANTS, UMBRA
from seed.load import Loaded, document_bytes, load
from tests.db.pg import session

pytestmark = [pytest.mark.db, pytest.mark.leaks]

READER = "DATABASE_URL"
WRITER = "INGEST_DATABASE_URL"

EXPECTED: dict[tuple[UUID, str], set[str]] = {
    (ACME, ALICE): {"CANARY-ACME-HANDBOOK", "CANARY-ACME-BUDGET", "CANARY-ACME-ALICE-REVIEW"},
    (ACME, BOB): {"CANARY-ACME-HANDBOOK", "CANARY-ACME-RUNBOOK"},
    (ACME, ERIN): {"CANARY-ACME-HANDBOOK"},
    (UMBRA, CAROL): {"CANARY-UMBRA-SAFETY", "CANARY-UMBRA-TRIAL"},
    # Both tenants have a finance group; dave's is Umbra's, alice's is Acme's.
    (UMBRA, DAVE): {"CANARY-UMBRA-SAFETY", "CANARY-UMBRA-BUDGET"},
}

# title -> ACL, per tenant, as the documents must be stored.
ACLS: dict[UUID, dict[str, set[str]]] = {
    ACME: {
        "Employee handbook": {"tenant:*"},
        "Q3 budget": {"group:finance"},
        "Routing service runbook": {"group:engineering"},
        "Performance review: Alice": {f"user:{ALICE}"},
    },
    UMBRA: {
        "Lab safety policy": {"tenant:*"},
        "Compound UB-7 trial results": {"group:research"},
        "Annual budget": {"group:finance"},
    },
}

# Tenant admins (ADR 0008), who also upload the seed. Being an admin adds no read
# access: bob and carol read exactly what their other groups allow, as EXPECTED shows.
ADMINS = {ACME: {BOB}, UMBRA: {CAROL}}

ALL_CANARIES = {doc.canary for tenant in TENANTS for doc in tenant.documents}
CANARY_OF = {doc.title: doc.canary for tenant in TENANTS for doc in tenant.documents}


@pytest.fixture
async def writer_engine() -> AsyncIterator[AsyncEngine]:
    url = os.environ.get(WRITER)
    if not url:
        pytest.skip(f"{WRITER} is not set")
    engine = create_async_engine(url)
    try:
        yield engine
    finally:
        await engine.dispose()


async def _load(engine: AsyncEngine) -> list[Loaded]:
    settings = Settings()
    return await load(engine, FakeEmbeddings(settings.embedding_dim), settings)


@pytest.fixture
async def seeded(writer_engine: AsyncEngine) -> list[Loaded]:
    """The corpus, loaded as `make seed` would (with fake embeddings)."""
    return await _load(writer_engine)


async def _writer_fetch(tenant: UUID, query: str, *args: object) -> list[Any]:
    async with session(WRITER, tenant) as conn:
        return list(await conn.fetch(query, *args))


async def _snapshot(tenant: UUID) -> dict[str, list[tuple[Any, ...]]]:
    """Everything the seed owns in `tenant`, as app_ingest sees it."""
    queries = {
        "tenant": "SELECT id, name FROM tenants",
        "memberships": "SELECT user_sub, group_name FROM memberships ORDER BY 1, 2",
        "documents": "SELECT id, title, source_hash, created_by, deleted_at, created_at"
        " FROM documents ORDER BY id",
        "acl": "SELECT document_id, principal FROM document_acl ORDER BY 1, 2",
        "chunks": "SELECT id, document_id, ordinal, heading, content, acl_principals,"
        " md5(embedding::text) FROM chunks ORDER BY id",
    }
    return {
        name: [tuple(row) for row in await _writer_fetch(tenant, query)]
        for name, query in queries.items()
    }


async def _visible_canaries(tenant: UUID, user: str) -> set[str]:
    async with session(READER, tenant, user) as conn:
        rows = await conn.fetch("SELECT content FROM chunks")
    return {c for c in ALL_CANARIES if any(c in row["content"] for row in rows)}


# --- the corpus ------------------------------------------------------------------------


def test_every_canary_is_unique() -> None:
    canaries = [doc.canary for tenant in TENANTS for doc in tenant.documents]
    assert len(canaries) == len(set(canaries))
    assert set().union(*EXPECTED.values()) == ALL_CANARIES


@pytest.mark.parametrize("tenant", TENANTS, ids=lambda t: t.name)
def test_the_corpus_acls_are_the_ones_written_here(tenant: Any) -> None:
    assert {doc.title: set(doc.acl) for doc in tenant.documents} == ACLS[tenant.id]


# --- what a load leaves --------------------------------------------------------------


@pytest.mark.parametrize(("tenant", "user"), list(EXPECTED))
async def test_canaries_reach_exactly_their_readers(
    seeded: list[Loaded], tenant: UUID, user: str
) -> None:
    assert await _visible_canaries(tenant, user) == EXPECTED[(tenant, user)]


@pytest.mark.parametrize("tenant", list(ACLS))
async def test_documents_and_acls_are_the_corpus(seeded: list[Loaded], tenant: UUID) -> None:
    rows = await _writer_fetch(
        tenant,
        "SELECT d.title, array_agg(a.principal) AS acl FROM documents d"
        " JOIN document_acl a ON a.document_id = d.id GROUP BY d.title",
    )
    assert {row["title"]: set(row["acl"]) for row in rows} == ACLS[tenant]


@pytest.mark.parametrize("tenant", TENANTS, ids=lambda t: t.name)
async def test_documents_are_uploads_by_the_admin(seeded: list[Loaded], tenant: Any) -> None:
    rows = await _writer_fetch(
        tenant.id, "SELECT title, source_hash, created_by, deleted_at FROM documents"
    )
    hashes = {doc.title: source_hash(document_bytes(doc)) for doc in tenant.documents}
    assert {row["title"]: row["source_hash"] for row in rows} == hashes
    assert {row["created_by"] for row in rows} == ADMINS[tenant.id]
    assert all(row["deleted_at"] is None for row in rows)


@pytest.mark.parametrize("tenant", TENANTS, ids=lambda t: t.name)
async def test_every_document_has_chunks_with_its_canary(seeded: list[Loaded], tenant: Any) -> None:
    rows = await _writer_fetch(
        tenant.id,
        "SELECT d.title, c.content FROM chunks c JOIN documents d ON d.id = c.document_id",
    )
    chunks: dict[str, list[str]] = {}
    for row in rows:
        chunks.setdefault(row["title"], []).append(row["content"])
    assert set(chunks) == set(ACLS[tenant.id])
    for title, contents in chunks.items():
        assert any(CANARY_OF[title] in content for content in contents), title
        others = ALL_CANARIES - {CANARY_OF[title]}
        assert not any(c in content for c in others for content in contents), title


async def test_the_result_matches_the_database(seeded: list[Loaded]) -> None:
    for tenant, loaded in zip(TENANTS, seeded, strict=True):
        documents = await _writer_fetch(tenant.id, "SELECT id, title FROM documents")
        chunks = await _writer_fetch(tenant.id, "SELECT count(*) AS n FROM chunks")
        assert loaded.documents == {row["title"]: row["id"] for row in documents}
        assert loaded.chunks == chunks[0]["n"] >= len(tenant.documents)


@pytest.mark.parametrize("tenant", list(ADMINS))
async def test_each_tenant_has_its_admins(seeded: list[Loaded], tenant: UUID) -> None:
    rows = await _writer_fetch(
        tenant, "SELECT user_sub FROM memberships WHERE group_name = 'admins'"
    )
    assert {row["user_sub"] for row in rows} == ADMINS[tenant]


# --- loading again ---------------------------------------------------------------------


async def test_loading_twice_changes_nothing(
    seeded: list[Loaded], writer_engine: AsyncEngine
) -> None:
    before = {tenant.id: await _snapshot(tenant.id) for tenant in TENANTS}
    again = await _load(writer_engine)
    after = {tenant.id: await _snapshot(tenant.id) for tenant in TENANTS}

    assert after == before
    assert [loaded.changes for loaded in again] == [0] * len(TENANTS)
    assert [(x.documents, x.chunks) for x in again] == [(x.documents, x.chunks) for x in seeded]


async def test_loading_resets_what_changed_since(
    seeded: list[Loaded], writer_engine: AsyncEngine
) -> None:
    """An upload, a soft delete, an ACL change and membership edits since the last
    load are all undone; untouched documents keep their ids."""
    acme = seeded[0].documents
    async with session(WRITER, ACME) as conn:
        stray: UUID = await conn.fetchval(
            "INSERT INTO documents (tenant_id, title, source_hash) VALUES ($1, 'Stray', $2)"
            " RETURNING id",
            ACME,
            "f" * 64,
        )
        await conn.execute(
            "INSERT INTO document_acl (tenant_id, document_id, principal) VALUES ($1, $2, $3)",
            ACME,
            stray,
            "tenant:*",
        )
        await conn.execute(
            "UPDATE documents SET deleted_at = now() WHERE id = $1", acme["Employee handbook"]
        )
        await conn.execute(
            "INSERT INTO document_acl (tenant_id, document_id, principal) VALUES ($1, $2, $3)",
            ACME,
            acme["Q3 budget"],
            "tenant:*",
        )
        await conn.execute(
            "DELETE FROM memberships WHERE user_sub = $1 AND group_name = 'finance'", ALICE
        )
        await conn.execute(
            "INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES ($1, $2, 'finance')",
            ACME,
            ERIN,
        )

    reloaded = await _load(writer_engine)

    # Stray and the soft-deleted handbook removed, handbook re-ingested, budget ACL
    # reset, alice back in finance, erin out of it.
    assert reloaded[0].changes == 6
    assert reloaded[1].changes == 0
    ids = reloaded[0].documents
    assert stray not in ids.values()
    assert ids["Employee handbook"] != acme["Employee handbook"]
    assert {t: i for t, i in ids.items() if t != "Employee handbook"} == {
        t: i for t, i in acme.items() if t != "Employee handbook"
    }
    rows = await _writer_fetch(ACME, "SELECT count(*) AS n FROM documents")
    assert rows[0]["n"] == len(ACLS[ACME])
    for (tenant, user), canaries in EXPECTED.items():
        assert await _visible_canaries(tenant, user) == canaries, user
