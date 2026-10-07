"""PgVectorRetriever against PostgreSQL, as app_rw under RLS (ADR 0001, 0009).

Documents are written through IngestService (app_ingest) with an embedder whose
vectors the test controls: each test draws a fresh random query direction and
places its chunks at a chosen cosine similarity to it. Leftover chunks from
other tests and earlier runs point in unrelated directions (similarity near 0),
so the nearest chunks in the shared index are always this test's own.

The chunks a user must not read are put closer to the query than the ones they
may read, so a missing filter would show up at the top of the results.
"""

import math
import os
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ragmt.adapters.llm.fake import FakeEmbeddings
from ragmt.domain import RetrievedChunk
from ragmt.ingest.service import IngestService
from ragmt.retrieval import PgVectorRetriever
from ragmt.retrieval.pgvector import IterativeScan
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session
from tests.ingest_helpers import FakeConverter, chunk_paragraphs, markdown_document
from tests.leaks.conftest import INGEST, READER, World, canaries

pytestmark = [pytest.mark.db, pytest.mark.leaks]

ADMIN = "admin-sub"
_FAKE = FakeEmbeddings(int(os.environ.get("EMBEDDING_DIM", "768")))


class ControlledEmbeddings(FakeEmbeddings):
    """FakeEmbeddings, except for the texts given a vector with `place`."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.vectors: dict[str, list[float]] = {}

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        fallback = await super().embed_documents(texts)
        return [self.vectors.get(t, v) for t, v in zip(texts, fallback, strict=True)]


async def random_direction() -> list[float]:
    return await _FAKE.embed_query(uuid4().hex)


async def at_similarity(direction: list[float], similarity: float) -> list[float]:
    """A unit vector whose cosine similarity to `direction` is exactly `similarity`."""
    noise = await random_direction()
    along = sum(n * d for n, d in zip(noise, direction, strict=True))
    orthogonal = [n - along * d for n, d in zip(noise, direction, strict=True)]
    norm = math.sqrt(math.fsum(o * o for o in orthogonal))
    side = math.sqrt(1.0 - similarity * similarity)
    return [similarity * d + side * o / norm for d, o in zip(direction, orthogonal, strict=True)]


@dataclass
class Harness:
    service: IngestService
    embedder: ControlledEmbeddings
    reader: AsyncEngine
    settings: Settings
    query: list[float]

    async def add(
        self,
        tenant: UUID,
        acl: list[str],
        similarities: Iterable[float],
    ) -> tuple[UUID, set[str]]:
        """Ingest one document with a chunk per similarity; return its id and chunk texts."""
        tag = uuid4().hex[:8]
        contents: list[str] = []
        for i, similarity in enumerate(similarities):
            content = f"CANARY-{tag}-{i}-{similarity:.2f}"
            self.embedder.vectors[content] = await at_similarity(self.query, similarity)
            contents.append(content)
        result = await self.service.ingest(
            tenant, ADMIN, markdown_document(f"Doc {tag}", *contents), f"{tag}.md", acl
        )
        assert result.chunks == len(contents)
        return result.document_id, set(contents)

    async def search(
        self,
        tenant: UUID,
        user: str | None,
        k: int,
        *,
        iterative_scan: IterativeScan | None = None,
        force_index: bool = False,
    ) -> list[RetrievedChunk]:
        retriever = PgVectorRetriever(
            dim=self.settings.embedding_dim,
            ef_search=self.settings.hnsw_ef_search,
            iterative_scan=iterative_scan or self.settings.hnsw_iterative_scan,
            max_scan_tuples=self.settings.hnsw_max_scan_tuples,
        )
        async with tenant_session(self.reader, tenant, user) as conn:
            if force_index:
                # With a few thousand rows in its tenant, the planner reads them
                # through chunks_tenant_id_idx and sorts them all: exact, so it can't
                # overfilter. Penalising sorts leaves the HNSW scan, the plan a larger
                # tenant gets on its own (see the EXPLAIN in ADR 0009).
                await conn.execute(text("SELECT set_config('enable_sort', 'off', true)"))
            return await retriever.search(conn, self.query, k)


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    ingest_url, reader_url = os.environ.get(INGEST), os.environ.get(READER)
    if not ingest_url or not reader_url:
        pytest.skip(f"{INGEST} and {READER} must be set")
    settings = Settings()
    writer = create_async_engine(ingest_url, pool_size=1, max_overflow=0)
    # One pooled connection: every search reuses it, as requests share a pool.
    reader = create_async_engine(reader_url, pool_size=1, max_overflow=0)
    embedder = ControlledEmbeddings(settings.embedding_dim)
    service = IngestService(writer, FakeConverter(), chunk_paragraphs, embedder, settings)
    try:
        yield Harness(service, embedder, reader, settings, await random_direction())
    finally:
        await writer.dispose()
        await reader.dispose()


def contents(chunks: Iterable[RetrievedChunk]) -> set[str]:
    return {c.content for c in chunks}


# --- who sees what -------------------------------------------------------------------


async def test_only_chunks_the_user_may_read_come_back(world: World, harness: Harness) -> None:
    """Alice is in group hr in both tenants (the world fixture). Everything she may
    not read is closer to the query than what she may."""
    a, b = world.tenants["a"], world.tenants["b"]
    hr, hr_text = await harness.add(a, ["group:hr"], [0.80, 0.79])
    public, public_text = await harness.add(a, ["tenant:*"], [0.70, 0.69])
    await harness.add(a, ["group:finance"], [0.99, 0.98])
    await harness.add(a, ["user:bob"], [0.99, 0.98])
    deleted, _ = await harness.add(a, ["tenant:*"], [0.99, 0.98])
    await harness.service.soft_delete(a, ADMIN, deleted)
    await harness.add(b, ["group:hr"], [0.99, 0.98])
    await harness.add(b, ["tenant:*"], [0.99, 0.98])

    found = await harness.search(a, "alice", 4)

    assert contents(found) == hr_text | public_text
    assert {c.document_id for c in found} == {hr, public}
    # The title is the chunk's own document's: "Doc <tag>", the tag being in the content.
    assert all(c.title == f"Doc {c.content.split('-')[1]}" for c in found)


async def test_each_user_gets_their_own_slice(world: World, harness: Harness) -> None:
    a, b = world.tenants["a"], world.tenants["b"]
    _, bob_text = await harness.add(a, ["user:bob"], [0.95, 0.94])
    _, public_text = await harness.add(a, ["tenant:*"], [0.70, 0.69])
    _, hr_text = await harness.add(a, ["group:hr"], [0.99, 0.98])
    _, hr_b_text = await harness.add(b, ["group:hr"], [0.60, 0.59])

    assert contents(await harness.search(a, "bob", 4)) == bob_text | public_text
    assert contents(await harness.search(a, "carol", 2)) == public_text
    # Same sub and group name in tenant B: only B's rows.
    assert contents(await harness.search(b, "alice", 2)) == hr_b_text
    assert hr_text.isdisjoint(contents(await harness.search(b, "alice", 40)))


async def test_no_user_in_the_session_finds_nothing(world: World, harness: Harness) -> None:
    a = world.tenants["a"]
    await harness.add(a, ["tenant:*"], [0.99, 0.98])
    assert await harness.search(a, None, 5) == []


async def test_a_deleted_document_is_never_retrieved(world: World, harness: Harness) -> None:
    a = world.tenants["a"]
    doc, doc_text = await harness.add(a, ["tenant:*"], [0.99, 0.98])
    assert contents(await harness.search(a, "carol", 2)) == doc_text

    await harness.service.soft_delete(a, ADMIN, doc)

    assert doc_text.isdisjoint(contents(await harness.search(a, "carol", 40)))


# --- ranking ---------------------------------------------------------------------------


async def test_results_are_closest_first_with_cosine_scores(world: World, harness: Harness) -> None:
    a = world.tenants["a"]
    similarities = [0.55, 0.95, 0.75, 0.85, 0.65, 0.90, 0.60, 0.80, 0.70]
    doc, _ = await harness.add(a, ["tenant:*"], similarities)

    found = await harness.search(a, "carol", 5)

    expected = sorted(similarities, reverse=True)[:5]
    assert [round(c.score, 3) for c in found] == expected
    assert all(c.document_id == doc for c in found)
    # Ordinals follow paragraph order, so each chunk is the one placed at its score.
    assert [round(similarities[c.ordinal], 3) for c in found] == expected


@pytest.mark.parametrize("k", [1, 3, 7])
async def test_k_is_respected(world: World, harness: Harness, k: int) -> None:
    a = world.tenants["a"]
    await harness.add(a, ["tenant:*"], [0.9 - i / 100 for i in range(10)])
    found = await harness.search(a, "carol", k)
    assert len(found) == k
    assert [c.score for c in found] == sorted((c.score for c in found), reverse=True)


# --- overfiltering (ADR 0001) ----------------------------------------------------------

# About 1% of the tenant's chunks near the query is readable, spread over the
# same range as the rest but never in its top ef_search (40), so one HNSW scan
# returns only unreadable candidates.
#
# Synthetic vectors need care to keep the HNSW graph navigable, as real
# embeddings are: both sets fill one continuous cone around the query, and the
# documents are ingested interleaved, so every readable chunk gets edges from
# chunks inserted after it. Readable chunks inserted as one far-off batch can end
# up with no incoming edges, and no scan setting reaches them.
_ROUNDS = 10
_HIDDEN = [0.95 - 0.35 * i / 999 for i in range(1000)]  # 0.95 .. 0.60
_READABLE = [0.80 - 0.20 * i / 9 for i in range(_ROUNDS)]  # 0.80 .. 0.60


async def test_iterative_scan_finds_k_chunks_when_few_are_readable(
    world: World, harness: Harness
) -> None:
    a = world.tenants["a"]
    k = harness.settings.retrieval_k
    ef_search = harness.settings.hnsw_ef_search
    assert sum(s > _READABLE[0] for s in _HIDDEN) > ef_search
    readable: set[str] = set()
    for i in range(_ROUNDS):
        _, texts = await harness.add(a, ["tenant:*"], [_READABLE[i]])
        readable |= texts
        await harness.add(a, ["group:finance"], _HIDDEN[i::_ROUNDS])

    without = await harness.search(a, "carol", k, iterative_scan="off", force_index=True)
    relaxed = await harness.search(a, "carol", k, iterative_scan="relaxed_order", force_index=True)
    strict = await harness.search(a, "carol", k, iterative_scan="strict_order", force_index=True)

    # An exact plan always finds k readable chunks; fewer means the HNSW scan ran.
    assert len(without) < k
    assert contents(without) <= readable
    # relaxed_order keeps scanning until k rows pass RLS, and the query re-sorts them.
    assert len(relaxed) == k
    assert contents(relaxed) <= readable
    assert [c.score for c in relaxed] == sorted((c.score for c in relaxed), reverse=True)
    # strict_order drops tuples that leave the graph out of order, so in this setup
    # it sometimes returns fewer than k, or farther chunks (the world's public_a,
    # which carol may read too). Measured in ADR 0009; it's why relaxed_order is
    # the default. Here it only has to leak nothing.
    assert contents(strict) <= readable | canaries("public_a")


# --- HNSW settings stay in their transaction -------------------------------------------

_HNSW = ("hnsw.ef_search", "hnsw.iterative_scan", "hnsw.max_scan_tuples")
_SHOW = text(
    "SELECT pg_backend_pid() AS pid,"
    " current_setting('hnsw.ef_search') AS ef_search,"
    " current_setting('hnsw.iterative_scan') AS iterative_scan,"
    " current_setting('hnsw.max_scan_tuples') AS max_scan_tuples"
)
_RESET_VALUES = text(
    "SELECT name, reset_val FROM pg_settings WHERE name = ANY(:names) ORDER BY name"
)


async def test_hnsw_settings_do_not_survive_the_transaction(world: World, harness: Harness) -> None:
    a = world.tenants["a"]
    await harness.add(a, ["tenant:*"], [0.9])
    retriever = PgVectorRetriever(
        dim=harness.settings.embedding_dim,
        ef_search=77,
        iterative_scan="strict_order",
        max_scan_tuples=4321,
    )

    async with tenant_session(harness.reader, a, "carol") as conn:
        await retriever.search(conn, harness.query, 1)
        during = (await conn.execute(_SHOW)).one()
    assert (during.ef_search, during.iterative_scan, during.max_scan_tuples) == (
        "77",
        "strict_order",
        "4321",
    )

    # The same pooled connection, next request: back to the session defaults.
    async with tenant_session(harness.reader, a, "carol") as conn:
        after = (await conn.execute(_SHOW)).one()
        defaults = {
            row.name: row.reset_val
            for row in await conn.execute(_RESET_VALUES, {"names": list(_HNSW)})
        }
    assert after.pid == during.pid
    assert (after.ef_search, after.iterative_scan, after.max_scan_tuples) == (
        defaults["hnsw.ef_search"],
        defaults["hnsw.iterative_scan"],
        defaults["hnsw.max_scan_tuples"],
    )
    assert after.ef_search != "77"
    assert after.iterative_scan != "strict_order"
    assert after.max_scan_tuples != "4321"
