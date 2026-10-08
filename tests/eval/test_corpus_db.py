"""Loading the synthetic corpus as app_ingest, and reading it back as app_rw.

Each module run uses its own namespace, so its tenants are fresh and a benchmark
corpus already in the database (the default namespace) is left alone. The
tenants are emptied at the end; the tenant rows and audit rows stay, as with
the leak suite's fixtures.
"""

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import asyncpg
import pytest
from pgvector.asyncpg import register_vector
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from eval.corpus import Config, Corpus, generate, load, reset, tenant_ids
from tests.db.pg import session

pytestmark = pytest.mark.db

READER = "DATABASE_URL"
WRITER = "INGEST_DATABASE_URL"


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(
        total=2_000, seed=11, topics=6, chunks_per_document=5, queries=8, k=5, namespace=uuid4()
    )


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


@pytest.fixture
async def loaded(writer_engine: AsyncEngine, config: Config) -> AsyncIterator[Corpus]:
    corpus = generate(config)
    await load(writer_engine, corpus)
    try:
        yield corpus
    finally:
        await reset(writer_engine, config.namespace)


async def _writer_rows(tenant: UUID, query: str) -> list[asyncpg.Record]:
    async with session(WRITER, tenant) as conn:
        return list(await conn.fetch(query, tenant))


async def test_each_tenant_holds_exactly_its_rows(loaded: Corpus) -> None:
    for tenant in loaded.tenants:
        async with session(WRITER, tenant.id) as conn:
            counts = await conn.fetchrow(
                "SELECT (SELECT count(*) FROM documents WHERE tenant_id = $1) AS documents,"
                " (SELECT count(*) FROM document_acl WHERE tenant_id = $1) AS acls,"
                " (SELECT count(*) FROM chunks WHERE tenant_id = $1) AS chunks,"
                " (SELECT count(*) FROM memberships WHERE tenant_id = $1) AS memberships,"
                " (SELECT count(*) FROM chunks) AS visible",
                tenant.id,
            )
        assert counts is not None
        assert counts["documents"] == counts["acls"] == len(tenant.document_ids)
        assert counts["chunks"] == counts["visible"] == len(tenant.chunk_ids)
        assert counts["memberships"] == len(tenant.memberships)


async def test_the_trigger_copies_each_documents_acl_onto_its_chunks(loaded: Corpus) -> None:
    for tenant in loaded.tenants:
        rows = await _writer_rows(
            tenant.id, "SELECT id, acl_principals FROM chunks WHERE tenant_id = $1"
        )
        stored = {row["id"]: row["acl_principals"] for row in rows}
        expected = {c: [acl] for c, acl in zip(tenant.chunk_ids, tenant.chunk_acls, strict=True)}
        assert stored == expected


async def test_vectors_are_stored_unchanged(loaded: Corpus) -> None:
    tenant = loaded.tenants[-1]
    async with session(WRITER, tenant.id) as conn:
        await register_vector(conn)
        rows = await conn.fetch("SELECT id, embedding FROM chunks")
    stored = {row["id"]: row["embedding"].to_numpy() for row in rows}
    for i, chunk_id in enumerate(tenant.chunk_ids):
        assert (stored[chunk_id] == tenant.embeddings[i]).all()


@pytest.mark.leaks
async def test_each_user_reads_exactly_what_the_generator_says(loaded: Corpus) -> None:
    for tenant in loaded.tenants:
        for user in tenant.users:
            async with session(READER, tenant.id, user) as conn:
                visible = {row["id"] for row in await conn.fetch("SELECT id FROM chunks")}
            mask = tenant.readable(user)
            assert visible == {c for c, ok in zip(tenant.chunk_ids, mask, strict=True) if ok}


@pytest.mark.leaks
async def test_ground_truth_matches_an_exact_search_under_rls(loaded: Corpus) -> None:
    """The top-k an exact (no index) search returns to the user through app_rw."""
    for query in loaded.queries:
        async with session(READER, query.tenant_id, query.user_sub) as conn:
            await register_vector(conn)
            await conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
            rows = await conn.fetch(
                "SELECT id FROM chunks ORDER BY embedding <=> $1 LIMIT $2",
                query.vector,
                loaded.config.k,
            )
        assert tuple(row["id"] for row in rows) == query.expected


async def test_loading_again_replaces_the_tenants_contents(
    writer_engine: AsyncEngine, loaded: Corpus, config: Config
) -> None:
    other = generate(Config(**{**config.__dict__, "seed": config.seed + 1}))
    await load(writer_engine, other)
    tenant = other.tenants[2]
    rows = await _writer_rows(tenant.id, "SELECT id FROM chunks WHERE tenant_id = $1")
    assert {row["id"] for row in rows} == set(tenant.chunk_ids)


async def test_reset_empties_only_the_synthetic_tenants(
    writer_engine: AsyncEngine, loaded: Corpus, config: Config
) -> None:
    bystander = uuid4()
    async with session(WRITER, bystander) as conn:
        await conn.execute("INSERT INTO tenants (id, name) VALUES ($1, 'Bystander')", bystander)
        await conn.execute(
            "INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES ($1, 'u', 'hr')",
            bystander,
        )

    removed = await reset(writer_engine, config.namespace)

    assert removed == {t.key: (len(t.document_ids), len(t.chunk_ids)) for t in loaded.tenants}
    for tenant_id in tenant_ids(config.namespace).values():
        async with session(WRITER, tenant_id) as conn:
            left = await conn.fetchrow(
                "SELECT (SELECT count(*) FROM documents) AS documents,"
                " (SELECT count(*) FROM chunks) AS chunks,"
                " (SELECT count(*) FROM memberships) AS memberships,"
                " (SELECT count(*) FROM tenants) AS tenants"
            )
        assert left is not None
        assert (left["documents"], left["chunks"], left["memberships"]) == (0, 0, 0)
        assert left["tenants"] == 1
    rows = await _writer_rows(bystander, "SELECT 1 FROM memberships WHERE tenant_id = $1")
    assert len(rows) == 1
