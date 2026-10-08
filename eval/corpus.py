"""A synthetic corpus for the retrieval benchmarks, with exact ground truth.

`python -m eval.corpus --total 100000 --seed 42` generates the corpus and loads
it as app_ingest (INGEST_DATABASE_URL). `--reset` only removes it. No Ollama:
the vectors are made up here.

Shape of the corpus:

- **Tenants.** Four measured tenants hold 50%, 10%, 1% and 0.1% of `total`
  chunks. The rest is split across four filler tenants, so the chunks table (and
  its HNSW index) has `total` rows, as the planner's choices in ADR 0009 depend
  on table size, not only on the tenant's.
- **ACLs.** Each document has one principal, picked by quota so every tenant
  gets the same mix: `tenant:*`, one of a few groups, or a single user. Users get
  memberships by a fixed pattern: user 0 has no groups (reads little, the
  overfiltering case), user 1 has every group (reads almost everything), the
  others have one or two.
- **Vectors.** Topic centres are random unit vectors in `dim` dimensions, shared
  by all tenants, so a query's nearest neighbours in other tenants are close too.
  Each document belongs to one topic, and each of its chunks is
  `centre + spread * noise / sqrt(dim)`, L2-normalised. Its cosine to the centre
  is about `1 / sqrt(1 + spread**2)` (0.71 for the default spread of 1).
- **Queries.** Per tenant, a query vector near a topic centre for one of the
  tenant's users, and the exact top-k chunk ids that user may read, by cosine
  similarity, computed in numpy over the readable set. This is what an exact
  search under RLS must return; recall is measured against it.

Everything comes from `numpy.random.default_rng` seeded with `seed`, so the same
seed and parameters give the same ids, vectors and queries (for the same numpy
version). The tenant ids and user subs depend only on `namespace`, so `--reset`
finds the tenants whatever seed loaded them.

Loading writes documents, document_acl and chunks directly, one transaction per
tenant inside `tenant_session` as app_ingest: the RLS policies still confine
each write to its tenant, and the ACL trigger still fills `acl_principals`.
COPY FROM is not an option: PostgreSQL refuses it on tables with row-level
security. Rows go in with `unnest()` inserts and, for chunks, batched
`executemany`. app_ingest may not delete tenants, so `--reset` empties them
(documents, ACLs, chunks, memberships) and leaves the tenant rows and their
audit rows.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

import numpy as np
from numpy.typing import NDArray
from pgvector.asyncpg import register_vector
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from ragmt.tenancy import tenant_session

# Picks the synthetic tenants' ids and user subs. Fixed, so --reset can find
# them; tests pass their own to stay out of the way of a benchmark load.
NAMESPACE = UUID("5b0e9c1e-2f4a-4c7b-9e1d-7a3f0c8b6d42")

# The four measured tenants: key -> share of `total`.
SHARES: dict[str, float] = {"share-50": 0.5, "share-10": 0.1, "share-1": 0.01, "share-0.1": 0.001}
FILLERS = ("filler-1", "filler-2", "filler-3", "filler-4")
GROUPS = ("engineering", "sales", "hr", "finance")
# Smallest total that leaves every tenant at least one chunk.
MIN_TOTAL = 1000

ACTOR = "eval-corpus"
CHUNK_BATCH = 1000

# An asyncpg.Connection. asyncpg ships no type information, so it is Any to mypy.
Driver = Any


@dataclass(frozen=True)
class Config:
    total: int = 100_000
    seed: int = 42
    dim: int = 768
    topics: int = 50
    spread: float = 1.0
    chunks_per_document: int = 10
    users: int = 8
    # Share of each tenant's documents readable by tenant:*, by a group, by one user.
    acl_mix: tuple[float, float, float] = (0.2, 0.6, 0.2)
    queries: int = 20
    query_spread: float = 0.5
    k: int = 10
    namespace: UUID = NAMESPACE

    def __post_init__(self) -> None:
        if self.total < MIN_TOTAL:
            raise ValueError(f"total must be at least {MIN_TOTAL}")
        if self.users < 3:
            raise ValueError("users must be at least 3 (no groups, all groups, some)")
        if min(self.dim, self.topics, self.chunks_per_document, self.k) < 1:
            raise ValueError("dim, topics, chunks_per_document and k must be positive")
        if min(self.acl_mix) < 0 or not math.isclose(sum(self.acl_mix), 1.0):
            raise ValueError("acl_mix must be three non-negative shares that add up to 1")


@dataclass(frozen=True)
class Query:
    tenant_id: UUID
    user_sub: str
    topic: int
    vector: NDArray[np.float32]
    # Chunks the user may read in the tenant.
    readable: int
    # The exact top-k readable chunks, closest first, and their cosine similarity.
    # Fewer than k when the user can read fewer.
    expected: tuple[UUID, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class Tenant:
    key: str
    id: UUID
    name: str
    users: tuple[str, ...]
    # (user_sub, group_name)
    memberships: tuple[tuple[str, str], ...]
    document_ids: tuple[UUID, ...]
    # One principal per document.
    document_acls: tuple[str, ...]
    document_topics: NDArray[np.int64]
    chunk_ids: tuple[UUID, ...]
    # Index into document_ids, and position within the document.
    chunk_documents: NDArray[np.int64]
    chunk_ordinals: NDArray[np.int64]
    embeddings: NDArray[np.float32]
    queries: tuple[Query, ...] = field(default=())

    def principals(self, user_sub: str) -> frozenset[str]:
        """The user's principals, as the RLS policies compute them from memberships."""
        groups = {f"group:{g}" for u, g in self.memberships if u == user_sub}
        return frozenset({f"user:{user_sub}", "tenant:*", *groups})

    def readable(self, user_sub: str) -> NDArray[np.bool_]:
        """Mask over the chunks: True where the user may read the chunk."""
        principals = self.principals(user_sub)
        documents = np.fromiter((p in principals for p in self.document_acls), dtype=np.bool_)
        return documents[self.chunk_documents]

    @cached_property
    def chunk_acls(self) -> tuple[str, ...]:
        return tuple(self.document_acls[d] for d in self.chunk_documents)


@dataclass(frozen=True)
class Corpus:
    config: Config
    centres: NDArray[np.float32]
    tenants: tuple[Tenant, ...]

    @property
    def queries(self) -> Iterator[Query]:
        for tenant in self.tenants:
            yield from tenant.queries


def tenant_ids(namespace: UUID = NAMESPACE) -> dict[str, UUID]:
    """key -> id of every synthetic tenant: what --reset removes."""
    return {key: uuid5(namespace, f"tenant/{key}") for key in (*SHARES, *FILLERS)}


def chunk_counts(total: int) -> dict[str, int]:
    """key -> chunks per tenant. The measured shares are rounded; fillers take the rest."""
    counts = {key: max(1, round(total * share)) for key, share in SHARES.items()}
    rest, extra = divmod(total - sum(counts.values()), len(FILLERS))
    for i, key in enumerate(FILLERS):
        counts[key] = rest + (1 if i < extra else 0)
    return counts


def generate(config: Config) -> Corpus:
    """Build the whole corpus in memory. Deterministic in `config`."""
    rng = np.random.default_rng([config.seed, 0])
    centres = _normalise(rng.standard_normal((config.topics, config.dim), dtype=np.float32))
    ids = tenant_ids(config.namespace)
    tenants = tuple(
        _tenant(config, centres, index, key, ids[key], count)
        for index, (key, count) in enumerate(chunk_counts(config.total).items())
    )
    return Corpus(config, centres, tenants)


def _normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return cast(NDArray[np.float32], (vectors / norms).astype(np.float32))


def _uuids(rng: np.random.Generator, n: int) -> tuple[UUID, ...]:
    raw = rng.bytes(16 * n)
    return tuple(UUID(bytes=raw[i : i + 16], version=4) for i in range(0, 16 * n, 16))


def _quota(n: int, shares: Sequence[float]) -> list[int]:
    """Split n into len(shares) integer parts by largest remainder."""
    exact = [n * s for s in shares]
    parts = [math.floor(x) for x in exact]
    by_remainder = sorted(range(len(shares)), key=lambda i: parts[i] - exact[i])
    for i in by_remainder[: n - sum(parts)]:
        parts[i] += 1
    return parts


def _memberships(users: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """User 0: no groups. User 1: all groups. User i >= 2: one group, two when i is odd."""
    rows: list[tuple[str, str]] = [(users[1], g) for g in GROUPS]
    for i in range(2, len(users)):
        rows.append((users[i], GROUPS[(i - 2) % len(GROUPS)]))
        if i % 2:
            rows.append((users[i], GROUPS[(i - 1) % len(GROUPS)]))
    return tuple(rows)


def _document_acls(
    rng: np.random.Generator, documents: int, users: Sequence[str], mix: Sequence[float]
) -> tuple[str, ...]:
    tenant, group, user = _quota(documents, mix)
    acls = (
        ["tenant:*"] * tenant
        + [f"group:{GROUPS[i % len(GROUPS)]}" for i in range(group)]
        + [f"user:{users[i % len(users)]}" for i in range(user)]
    )
    return tuple(acls[i] for i in rng.permutation(documents))


def _tenant(
    config: Config,
    centres: NDArray[np.float32],
    index: int,
    key: str,
    tenant_id: UUID,
    chunks: int,
) -> Tenant:
    rng = np.random.default_rng([config.seed, 1, index])
    id_rng = np.random.default_rng([config.seed, 2, index])

    users = tuple(str(uuid5(config.namespace, f"{key}/user/{i}")) for i in range(config.users))
    documents = math.ceil(chunks / config.chunks_per_document)
    chunk_documents = np.arange(chunks, dtype=np.int64) // config.chunks_per_document
    chunk_ordinals = np.arange(chunks, dtype=np.int64) % config.chunks_per_document

    document_topics = rng.integers(config.topics, size=documents, dtype=np.int64)
    noise = rng.standard_normal((chunks, config.dim), dtype=np.float32)
    noise *= np.float32(config.spread / math.sqrt(config.dim))
    embeddings = _normalise(centres[document_topics[chunk_documents]] + noise)
    del noise

    tenant = Tenant(
        key=key,
        id=tenant_id,
        name=f"Eval {key}",
        users=users,
        memberships=_memberships(users),
        document_ids=_uuids(id_rng, documents),
        document_acls=_document_acls(rng, documents, users, config.acl_mix),
        document_topics=document_topics,
        chunk_ids=_uuids(id_rng, chunks),
        chunk_documents=chunk_documents,
        chunk_ordinals=chunk_ordinals,
        embeddings=embeddings,
    )
    queries = _queries(config, centres, tenant, np.random.default_rng([config.seed, 3, index]))
    return Tenant(**{**tenant.__dict__, "queries": queries})


def _queries(
    config: Config, centres: NDArray[np.float32], tenant: Tenant, rng: np.random.Generator
) -> tuple[Query, ...]:
    topics = rng.integers(config.topics, size=config.queries, dtype=np.int64)
    noise = rng.standard_normal((config.queries, config.dim), dtype=np.float32)
    noise *= np.float32(config.query_spread / math.sqrt(config.dim))
    vectors = _normalise(centres[topics] + noise)
    masks = {user: tenant.readable(user) for user in tenant.users}

    queries = []
    for q in range(config.queries):
        user = tenant.users[q % len(tenant.users)]
        positions, scores = exact_top_k(tenant.embeddings, vectors[q], masks[user], config.k)
        queries.append(
            Query(
                tenant_id=tenant.id,
                user_sub=user,
                topic=int(topics[q]),
                vector=vectors[q],
                readable=int(masks[user].sum()),
                expected=tuple(tenant.chunk_ids[p] for p in positions),
                scores=tuple(float(s) for s in scores),
            )
        )
    return tuple(queries)


def exact_top_k(
    embeddings: NDArray[np.float32],
    query: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Positions and cosine similarities of the k allowed rows closest to `query`.

    Rows and query are unit vectors, so cosine is the dot product. A float32 pass
    picks candidates, which are rescored in float64; ties go to the lower position.
    """
    candidates = np.flatnonzero(allowed)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    rough = embeddings[candidates] @ query
    keep = min(candidates.size, 4 * k + 16)
    if keep < candidates.size:
        candidates = candidates[np.argpartition(-rough, keep - 1)[:keep]]
    exact = embeddings[candidates].astype(np.float64) @ query.astype(np.float64)
    order = np.lexsort((candidates, -exact))[:k]
    return candidates[order].astype(np.int64), exact[order]


# --- Loading -----------------------------------------------------------------------


@dataclass(frozen=True)
class Loaded:
    key: str
    documents: int
    chunks: int
    seconds: float


async def _driver(conn: AsyncConnection) -> Driver:
    """The asyncpg connection under `conn`, in the same transaction."""
    raw = await conn.get_raw_connection()
    return raw.driver_connection


async def _empty(driver: Driver, tenant_id: UUID) -> tuple[int, int]:
    """Delete the tenant's documents (ACL rows and chunks go by cascade) and memberships."""
    chunks = await driver.fetchval("SELECT count(*) FROM chunks WHERE tenant_id = $1", tenant_id)
    documents = await driver.execute("DELETE FROM documents WHERE tenant_id = $1", tenant_id)
    await driver.execute("DELETE FROM memberships WHERE tenant_id = $1", tenant_id)
    return int(documents.split()[-1]), int(chunks)


async def _audit(driver: Driver, tenant_id: UUID, action: str, **details: Any) -> None:
    await driver.execute(
        "INSERT INTO audit_events (tenant_id, actor_sub, action, details)"
        " VALUES ($1, $2, $3, $4::text::jsonb)",
        tenant_id,
        ACTOR,
        action,
        json.dumps(details),
    )


async def load_tenant(engine: AsyncEngine, config: Config, tenant: Tenant) -> Loaded:
    """Replace the tenant's contents with `tenant`, in one transaction. `engine` is app_ingest."""
    started = time.perf_counter()
    async with tenant_session(engine, tenant.id) as conn:
        driver = await _driver(conn)
        await register_vector(driver)
        await driver.execute(
            "INSERT INTO tenants (id, name) VALUES ($1, $2)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            tenant.id,
            tenant.name,
        )
        await _empty(driver, tenant.id)
        await driver.execute(
            "INSERT INTO memberships (tenant_id, user_sub, group_name)"
            " SELECT $1, u, g FROM unnest($2::text[], $3::text[]) AS m(u, g)",
            tenant.id,
            [u for u, _ in tenant.memberships],
            [g for _, g in tenant.memberships],
        )
        await driver.execute(
            "INSERT INTO documents (id, tenant_id, title, source_uri, source_hash)"
            " SELECT d, $1, 'Synthetic ' || d, 'eval://' || d,"
            "  encode(sha256(convert_to('eval:' || $2 || ':' || d, 'UTF8')), 'hex')"
            " FROM unnest($3::uuid[]) AS t(d)",
            tenant.id,
            str(config.seed),
            list(tenant.document_ids),
        )
        # Before the chunks: their trigger copies these into acl_principals.
        await driver.execute(
            "INSERT INTO document_acl (tenant_id, document_id, principal)"
            " SELECT $1, d, p FROM unnest($2::uuid[], $3::text[]) AS a(d, p)",
            tenant.id,
            list(tenant.document_ids),
            list(tenant.document_acls),
        )
        for start in range(0, len(tenant.chunk_ids), CHUNK_BATCH):
            stop = min(start + CHUNK_BATCH, len(tenant.chunk_ids))
            await driver.executemany(
                "INSERT INTO chunks (id, tenant_id, document_id, ordinal, content, embedding)"
                " VALUES ($1, $2, $3, $4, $5, $6)",
                [
                    (
                        tenant.chunk_ids[i],
                        tenant.id,
                        tenant.document_ids[tenant.chunk_documents[i]],
                        int(tenant.chunk_ordinals[i]),
                        f"Synthetic chunk {i} of {tenant.key}",
                        tenant.embeddings[i],
                    )
                    for i in range(start, stop)
                ],
            )
        await _audit(
            driver,
            tenant.id,
            "eval_corpus_load",
            seed=config.seed,
            total=config.total,
            documents=len(tenant.document_ids),
            chunks=len(tenant.chunk_ids),
        )
    seconds = time.perf_counter() - started
    return Loaded(tenant.key, len(tenant.document_ids), len(tenant.chunk_ids), seconds)


async def load(engine: AsyncEngine, corpus: Corpus) -> list[Loaded]:
    """Load every synthetic tenant, replacing what was there."""
    return [await load_tenant(engine, corpus.config, tenant) for tenant in corpus.tenants]


async def reset(engine: AsyncEngine, namespace: UUID = NAMESPACE) -> dict[str, tuple[int, int]]:
    """Empty the synthetic tenants of `namespace`: key -> (documents, chunks) removed.

    Touches no other tenant: each delete runs in that tenant's own session. The
    tenant rows stay (app_ingest may not delete tenants), and so do audit rows.
    """
    removed: dict[str, tuple[int, int]] = {}
    for key, tenant_id in tenant_ids(namespace).items():
        async with tenant_session(engine, tenant_id) as conn:
            driver = await _driver(conn)
            documents, chunks = await _empty(driver, tenant_id)
            if documents or chunks:
                await _audit(
                    driver, tenant_id, "eval_corpus_reset", documents=documents, chunks=chunks
                )
        removed[key] = (documents, chunks)
    return removed


def queries_json(corpus: Corpus) -> dict[str, Any]:
    """The queries and their ground truth, for a benchmark run in another process."""
    config = corpus.config
    return {
        "config": {**config.__dict__, "namespace": str(config.namespace)},
        "tenants": {t.key: str(t.id) for t in corpus.tenants},
        "queries": [
            {
                "tenant_id": str(q.tenant_id),
                "user_sub": q.user_sub,
                "topic": q.topic,
                "readable": q.readable,
                "vector": q.vector.tolist(),
                "expected": [str(c) for c in q.expected],
                "scores": list(q.scores),
            }
            for q in corpus.queries
        ],
    }


# --- Command line ------------------------------------------------------------------


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(
        prog="python -m eval.corpus", description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    parser.add_argument("--total", type=int, default=defaults.total, help="chunks in all")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--topics", type=int, default=defaults.topics)
    parser.add_argument("--spread", type=float, default=defaults.spread)
    parser.add_argument("--chunks-per-document", type=int, default=defaults.chunks_per_document)
    parser.add_argument("--users", type=int, default=defaults.users, help="per tenant")
    parser.add_argument("--queries", type=int, default=defaults.queries, help="per tenant")
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument(
        "--queries-out", type=Path, help="write the queries and ground truth here (JSON)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="only remove the synthetic tenants' contents"
    )
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    from ragmt.settings import get_settings

    args = _arguments(argv)
    settings = get_settings()
    engine = create_async_engine(settings.ingest_database_url.get_secret_value())
    try:
        if args.reset:
            for key, (documents, chunks) in (await reset(engine)).items():
                print(f"{key}: removed {documents} documents, {chunks} chunks")
            return

        config = Config(
            total=args.total,
            seed=args.seed,
            dim=settings.embedding_dim,
            topics=args.topics,
            spread=args.spread,
            chunks_per_document=args.chunks_per_document,
            users=args.users,
            queries=args.queries,
            k=args.k,
        )
        started = time.perf_counter()
        corpus = generate(config)
        print(f"generated {config.total} chunks in {time.perf_counter() - started:.1f} s")
        if args.queries_out:
            args.queries_out.parent.mkdir(parents=True, exist_ok=True)
            args.queries_out.write_text(json.dumps(queries_json(corpus)), encoding="utf-8")
            print(f"wrote {sum(1 for _ in corpus.queries)} queries to {args.queries_out}")

        started = time.perf_counter()
        for loaded in await load(engine, corpus):
            print(
                f"{loaded.key}: {loaded.documents} documents, {loaded.chunks} chunks"
                f" in {loaded.seconds:.1f} s"
            )
        seconds = time.perf_counter() - started
        print(
            f"loaded {config.total} chunks in {seconds:.1f} s"
            f" ({config.total / seconds:.0f} chunks/s)."
            " Run VACUUM ANALYZE chunks as migrator before measuring plans."
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValueError as exc:
        sys.exit(f"eval.corpus: {exc}")
