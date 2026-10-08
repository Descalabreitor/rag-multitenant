"""Retrieval benchmarks on the synthetic corpus: recall and latency (Phase 6).

`python -m eval.bench` (or `make eval`). Local only, never in CI. Needs PostgreSQL
from compose.yaml and the superuser password (POSTGRES_PASSWORD, as in .env).
No Ollama: the vectors come from `eval.corpus`.

Steps, each skipped when already done:

1. Create the `ragmt_eval` database and migrate it to head (`eval.database`).
2. Generate the corpus (`--total`, `--seed`) and load it as app_ingest unless the
   database already holds exactly these chunks; then VACUUM ANALYZE.
3. Build the scratch variants in `eval_scratch` (partitions by tenant, the
   EXISTS policy) unless they were built from the same corpus.
4. Measure every variant below on the four measured tenants (50%, 10%, 1%,
   0.1% of the chunks) for three users each: no groups, one group, all groups.

Each variant runs the retriever's own statement (`ragmt.retrieval.pgvector`),
with only the table swapped, as app_rw inside `tenant_session`; the baseline
runs it as the superuser with the policy written out as WHERE clauses. Plans
are forced per transaction with `set_config(..., true)`:

- hnsw: enable_seqscan, enable_bitmapscan and enable_sort off, so ORDER BY
  distance can only come from the HNSW index; jit off (forcing inflates costs).
- exact: enable_indexscan off, so no HNSW scan: a bitmap or sequential scan and
  an exact top-N sort.
- planner: nothing forced, the server's defaults, as the service runs.

Every run sets plan_cache_mode = force_custom_plan, so asyncpg's prepared
statements don't fall back to a generic plan after five executions. The plan of
each (variant, tenant, user) is recorded with EXPLAIN (ANALYZE, BUFFERS) on the
first query, and a forced variant whose plan isn't the one it forces is flagged.

Per variant: one untimed pass over all cells (warm-up), one pass at k=10 for
recall@10, then `--reps` timed passes at k=5 (the service's RETRIEVAL_K) for
latency and recall@5. Latency is the client's wall time for the search
statement alone: no HTTP, no embedding, no LLM, not the BEGIN or set_config
round trips around it.

Ground truth is the exact top-k over the chunks the user may read, computed in
numpy (`eval.corpus.exact_top_k`). Output goes to docs/results/raw/eval-<UTC>/
(git-ignored): samples.csv, plans.csv, plans.jsonl and meta.json, which
`python -m eval.report` turns into docs/results/eval.md and the charts.
"""

import argparse
import asyncio
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from eval.corpus import SHARES, Config, Corpus, _driver, exact_top_k, generate, load
from eval.database import (
    NORMALIZED,
    PARTITIONED,
    ROOT,
    EvalSettings,
    Urls,
    analyze,
    build_scratch,
    connect,
    ensure_database,
    migrate,
    scratch_fingerprint,
)
from ragmt.retrieval.pgvector import _SEARCH, IterativeScan
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session

RAW_DIR = ROOT / "docs" / "results" / "raw"
K_RECALL = 10
K_LATENCY = 5

Plan = Literal["hnsw", "exact", "planner"]
Role = Literal["app_rw", "superuser"]

# user index in the tenant -> access level (eval.corpus._memberships).
ACCESS = {0: "no groups", 2: "one group", 1: "all groups"}

_FORCED: dict[Plan, dict[str, str]] = {
    "hnsw": {
        "enable_seqscan": "off",
        "enable_bitmapscan": "off",
        "enable_sort": "off",
        "jit": "off",
    },
    "exact": {"enable_indexscan": "off", "enable_indexonlyscan": "off", "jit": "off"},
    "planner": {},
}


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    table: str
    role: Role
    plan: Plan
    iterative_scan: IterativeScan = "relaxed_order"


VARIANTS = (
    # Recall: the four asked for, plus the two neighbours that explain them.
    Variant("hnsw_off", "(a) HNSW, iterative scan off", "chunks", "app_rw", "hnsw", "off"),
    Variant("hnsw_relaxed", "(b) HNSW, relaxed_order", "chunks", "app_rw", "hnsw"),
    Variant("hnsw_strict", "HNSW, strict_order", "chunks", "app_rw", "hnsw", "strict_order"),
    Variant("partition_relaxed", "(c) partitions, relaxed_order", PARTITIONED, "app_rw", "hnsw"),
    Variant(
        "partition_off", "(c) partitions, iterative scan off", PARTITIONED, "app_rw", "hnsw", "off"
    ),
    Variant("exact", "(d) exact (no index scan)", "chunks", "app_rw", "exact"),
    Variant("planner", "RLS, planner's choice", "chunks", "app_rw", "planner"),
    # Latency of the policy itself: no RLS, and the normalised EXISTS policy.
    Variant("baseline_hnsw", "no RLS + WHERE, HNSW relaxed", "chunks", "superuser", "hnsw"),
    Variant("baseline_exact", "no RLS + WHERE, exact", "chunks", "superuser", "exact"),
    Variant("baseline_planner", "no RLS + WHERE, planner", "chunks", "superuser", "planner"),
    Variant("exists_hnsw", "RLS EXISTS, HNSW relaxed", NORMALIZED, "app_rw", "hnsw"),
    Variant("exists_exact", "RLS EXISTS, exact", NORMALIZED, "app_rw", "exact"),
    Variant("exists_planner", "RLS EXISTS, planner", NORMALIZED, "app_rw", "planner"),
)


# --- Statements ----------------------------------------------------------------------


def _replace_once(sql: str, old: str, new: str) -> str:
    if sql.count(old) != 1:
        raise RuntimeError(f"expected exactly one {old!r} in the retriever's statement")
    return sql.replace(old, new)


def search_sql(variant: Variant) -> str:
    """The retriever's statement for `variant`, with asyncpg placeholders.

    $1 query vector (text), $2 k; the baseline adds $3 tenant id, $4 principals.
    """
    sql = _replace_once(_SEARCH.text, "CAST(:query AS vector)", "CAST($1 AS vector)")
    sql = _replace_once(sql, "LIMIT :k", "LIMIT $2")
    if variant.table != "chunks":
        sql = _replace_once(sql, "FROM chunks c", f"FROM {variant.table} c")
    if variant.role == "superuser":
        # The chunks and documents policies, written out (migrations aef343c73b03, 5c1e9a7d3b20).
        sql = _replace_once(
            sql,
            "FROM chunks c",
            "FROM chunks c WHERE c.tenant_id = $3 AND c.acl_principals && $4::text[]",
        )
        sql = _replace_once(
            sql,
            "JOIN documents d ON d.id = n.document_id",
            "JOIN documents d ON d.id = n.document_id AND d.tenant_id = $3"
            " AND d.deleted_at IS NULL AND EXISTS (SELECT 1 FROM document_acl a"
            " WHERE a.document_id = d.id AND a.tenant_id = $3 AND a.principal = ANY($4::text[]))",
        )
    return sql


def run_settings(variant: Variant, hnsw: "HnswSettings") -> dict[str, str]:
    return {
        "plan_cache_mode": "force_custom_plan",
        "hnsw.ef_search": str(hnsw.ef_search),
        "hnsw.iterative_scan": variant.iterative_scan,
        "hnsw.max_scan_tuples": str(hnsw.max_scan_tuples),
        **_FORCED[variant.plan],
    }


def vector_literal(vector: np.ndarray[Any, Any]) -> str:
    """As `PgVectorRetriever._vector_literal`: the query goes in as text."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


# --- Cells and ground truth ----------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One query vector for one user of one measured tenant, with its ground truth."""

    tenant_key: str
    tenant_id: UUID
    chunks: int
    user: int
    user_sub: str
    principals: tuple[str, ...]
    readable: int
    query: int
    vector: str
    expected: tuple[UUID, ...]


def cells(corpus: Corpus) -> list[Cell]:
    out = []
    for tenant in corpus.tenants:
        if tenant.key not in SHARES:
            continue
        for user in ACCESS:
            sub = tenant.users[user]
            mask = tenant.readable(sub)
            for q, query in enumerate(tenant.queries):
                positions, _ = exact_top_k(tenant.embeddings, query.vector, mask, K_RECALL)
                out.append(
                    Cell(
                        tenant_key=tenant.key,
                        tenant_id=tenant.id,
                        chunks=len(tenant.chunk_ids),
                        user=user,
                        user_sub=sub,
                        principals=tuple(sorted(tenant.principals(sub))),
                        readable=int(mask.sum()),
                        query=q,
                        vector=vector_literal(query.vector),
                        expected=tuple(tenant.chunk_ids[p] for p in positions),
                    )
                )
    return out


def recall(returned: Sequence[UUID], expected: Sequence[UUID], k: int) -> float | None:
    """|top-k returned ∩ top-k expected| / |top-k expected|; None when nothing is readable."""
    truth = set(expected[:k])
    if not truth:
        return None
    return len(truth & set(returned[:k])) / len(truth)


# --- Running -------------------------------------------------------------------------


@dataclass(frozen=True)
class HnswSettings:
    ef_search: int
    max_scan_tuples: int


class Runner:
    def __init__(self, reader: AsyncEngine, superuser: AsyncEngine, hnsw: HnswSettings) -> None:
        self._reader = reader
        self._superuser = superuser
        self._hnsw = hnsw

    async def _with_settings(
        self, variant: Variant, cell: Cell, statement: str, args: Sequence[Any]
    ) -> tuple[list[Any], float]:
        settings = run_settings(variant, self._hnsw)
        set_sql = text(
            "SELECT " + ", ".join(f"set_config(:n{i}, :v{i}, true)" for i in range(len(settings)))
        )
        params = {}
        for i, (name, value) in enumerate(settings.items()):
            params[f"n{i}"], params[f"v{i}"] = name, value

        session: AbstractAsyncContextManager[AsyncConnection]
        if variant.role == "app_rw":
            session = tenant_session(self._reader, cell.tenant_id, cell.user_sub)
        else:
            session = _plain_transaction(self._superuser)
        async with session as conn:
            # Through SQLAlchemy, whose lazy BEGIN this sends: on the bare driver,
            # set_config(..., true) would end with its own implicit transaction.
            await conn.execute(set_sql, params)
            driver = await _driver(conn)
            started = time.perf_counter()
            rows = await driver.fetch(statement, *args)
            elapsed = time.perf_counter() - started
        return rows, elapsed

    def _args(self, variant: Variant, cell: Cell, k: int) -> list[Any]:
        args: list[Any] = [cell.vector, k]
        if variant.role == "superuser":
            args += [cell.tenant_id, list(cell.principals)]
        return args

    async def search(self, variant: Variant, cell: Cell, k: int) -> tuple[list[UUID], float]:
        rows, elapsed = await self._with_settings(
            variant, cell, search_sql(variant), self._args(variant, cell, k)
        )
        return [row["id"] for row in rows], elapsed

    async def explain(self, variant: Variant, cell: Cell, k: int) -> dict[str, Any]:
        statement = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + search_sql(variant)
        rows, _ = await self._with_settings(variant, cell, statement, self._args(variant, cell, k))
        value = rows[0][0]
        plan: dict[str, Any] = (json.loads(value) if isinstance(value, str) else value)[0]
        return plan


@asynccontextmanager
async def _plain_transaction(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """A transaction with no tenant context, rolled back (the superuser baseline)."""
    async with engine.connect() as conn, conn.begin() as transaction:
        yield conn
        await transaction.rollback()


# --- Plans ---------------------------------------------------------------------------


def _walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The node and its descendants, without the policy's InitPlan (the memberships lookup)."""
    yield node
    for child in node.get("Plans", []):
        initplan = child.get("Parent Relationship") == "InitPlan"
        if not initplan or str(child.get("Subplan Name", "")).startswith("CTE "):
            yield from _walk(child)


def summarise_plan(explained: dict[str, Any], hnsw_indexes: frozenset[str]) -> dict[str, Any]:
    """The CTE's scan as one line, whether it used an HNSW index, and its counters."""
    root = explained["Plan"]
    cte = next(
        (n for n in _walk(root) if n.get("Subplan Name") == "CTE nearest"),
        root,
    )
    steps = []
    removed = 0
    pruned = 0
    used_hnsw = False
    for node in _walk(cte):
        kind = node["Node Type"]
        if kind in {"Limit", "Result"}:
            continue
        index = node.get("Index Name")
        relation = node.get("Relation Name")
        step = kind
        if index:
            step += f" {index}"
            used_hnsw |= index in hnsw_indexes
        elif relation and kind.endswith("Scan"):
            step += f" {relation}"
        if node.get("Sort Method"):
            step += f" ({node['Sort Method']})"
        steps.append(step)
        removed += int(node.get("Rows Removed by Filter", 0))
        pruned += int(node.get("Subplans Removed", 0))
    return {
        "kind": "hnsw" if used_hnsw else "exact",
        "scan": " > ".join(steps),
        "rows_removed": removed,
        "subplans_removed": pruned,
        "buffers": int(root.get("Shared Hit Blocks", 0)) + int(root.get("Shared Read Blocks", 0)),
        "execution_ms": float(explained.get("Execution Time", 0.0)),
    }


# --- Orchestration -------------------------------------------------------------------


def _fingerprint(config: Config) -> str:
    fields = asdict(config)
    for key in ("queries", "query_spread", "k"):
        fields.pop(key)
    return json.dumps({k: str(v) for k, v in fields.items()}, sort_keys=True)


async def corpus_loaded(writer: AsyncEngine, corpus: Corpus) -> bool:
    """True when every synthetic tenant holds exactly the generated chunks and memberships."""
    for tenant in corpus.tenants:
        async with tenant_session(writer, tenant.id) as conn:
            driver = await _driver(conn)
            total = await driver.fetchval("SELECT count(*) FROM chunks")
            known = await driver.fetchval(
                "SELECT count(*) FROM chunks WHERE id = ANY($1::uuid[])", list(tenant.chunk_ids)
            )
            members = await driver.fetchval("SELECT count(*) FROM memberships")
        if not total == known == len(tenant.chunk_ids) or members != len(tenant.memberships):
            return False
    return True


async def hnsw_index_names(urls: Urls) -> frozenset[str]:
    async with connect(urls.superuser) as conn:
        rows = await conn.fetch(
            "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid"
            " JOIN pg_am a ON a.oid = c.relam WHERE a.amname = 'hnsw'"
        )
    return frozenset(row["relname"] for row in rows)


async def server_info(urls: Urls) -> dict[str, Any]:
    names = (
        "server_version",
        "shared_buffers",
        "work_mem",
        "effective_cache_size",
        "max_parallel_workers_per_gather",
        "jit",
        "random_page_cost",
    )
    async with connect(urls.superuser) as conn:
        info = {n: await conn.fetchval(f"SELECT current_setting('{n}')") for n in names}
        info["pgvector"] = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        info["sizes"] = {
            row["relname"]: row["size"]
            for row in await conn.fetch(
                # Summed over partitions: a partitioned table has no storage of its own.
                "SELECT c.relname,"
                " pg_size_pretty(sum(pg_total_relation_size(coalesce(t.relid, c.oid)))) AS size"
                " FROM pg_class c LEFT JOIN LATERAL pg_partition_tree(c.oid) t ON true"
                " WHERE c.relname IN ('chunks', 'chunks_embedding_hnsw_idx',"
                " 'chunks_by_tenant', 'chunks_normalized') GROUP BY c.relname"
            )
        }
    return info


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform == "win32":
        import ctypes
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

        class _Memory(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("rest", ctypes.c_ulonglong * 6),
            ]

        memory = _Memory()
        memory.length = ctypes.sizeof(_Memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        info["memory_gib"] = round(memory.total / 2**30, 1)
    elif Path("/proc/meminfo").is_file():
        first = Path("/proc/meminfo").read_text().splitlines()[0]
        info["memory_gib"] = round(int(first.split()[1]) / 2**20, 1)
    try:
        docker = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],  # noqa: S607 - local tool
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        raw = json.loads(docker.stdout)
        info["docker"] = {
            "server": raw.get("ServerVersion"),
            "os": raw.get("OperatingSystem"),
            "cpus": raw.get("NCPU"),
            "memory_gib": round(raw.get("MemTotal", 0) / 2**30, 1),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        info["docker"] = None
    return info


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - local tool
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() + (" (dirty)" if _dirty() else "")


def _dirty() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607 - local tool
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(out.stdout.strip())


SAMPLE_FIELDS = (
    "variant",
    "tenant",
    "chunks",
    "user",
    "access",
    "readable",
    "query",
    "k",
    "rep",
    "ms",
    "returned",
    "expected",
    "recall",
)
PLAN_FIELDS = (
    "variant",
    "tenant",
    "user",
    "access",
    "forced",
    "kind",
    "scan",
    "rows_removed",
    "subplans_removed",
    "buffers",
    "execution_ms",
)


def _engine(url: URL) -> AsyncEngine:
    return create_async_engine(url, pool_size=1, max_overflow=0)


async def bench(args: argparse.Namespace) -> Path:
    steps: dict[str, float] = {}
    started_all = time.perf_counter()
    urls = Urls.from_settings(EvalSettings(), args.database)
    settings = Settings()

    t = time.perf_counter()
    created = await ensure_database(urls)
    migrate(urls)
    steps["database"] = time.perf_counter() - t
    print(f"database {urls.reader.database}: {'created, ' if created else ''}at head")

    config = Config(
        total=args.total, seed=args.seed, dim=settings.embedding_dim, queries=args.queries
    )
    t = time.perf_counter()
    corpus = generate(config)
    steps["generate"] = time.perf_counter() - t
    print(f"generated {config.total} chunks in {steps['generate']:.1f} s")

    writer = _engine(urls.writer)
    reader = _engine(urls.reader)
    superuser = _engine(urls.superuser)
    try:
        t = time.perf_counter()
        loaded = await corpus_loaded(writer, corpus)
        if not loaded:
            print("loading the corpus as app_ingest ...")
            await load(writer, corpus)
        await analyze(urls)
        steps["load"] = time.perf_counter() - t
        print(f"corpus {'already loaded' if loaded else 'loaded'} ({steps['load']:.0f} s)")

        fingerprint = _fingerprint(config)
        t = time.perf_counter()
        if args.rebuild_scratch or await scratch_fingerprint(urls) != fingerprint:
            print("building eval_scratch (partitions, EXISTS policy) ...")
            by_key = {tenant.key: tenant.id for tenant in corpus.tenants}
            scratch = await build_scratch(urls, by_key, config.dim, fingerprint)
            print("  " + ", ".join(f"{k} {v:.0f} s" for k, v in scratch.items()))
        steps["scratch"] = time.perf_counter() - t

        hnsw = HnswSettings(
            ef_search=args.ef_search,
            max_scan_tuples=args.max_scan_tuples,
        )
        runner = Runner(reader, superuser, hnsw)
        hnsw_indexes = await hnsw_index_names(urls)
        all_cells = cells(corpus)
        variants = [v for v in VARIANTS if not args.variants or v.name in args.variants]

        out = args.out or RAW_DIR / f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        out.mkdir(parents=True, exist_ok=True)
        mismatches: list[str] = []
        with (
            (out / "samples.csv").open("w", newline="", encoding="utf-8") as samples_file,
            (out / "plans.csv").open("w", newline="", encoding="utf-8") as plans_file,
            (out / "plans.jsonl").open("w", encoding="utf-8") as plans_json,
        ):
            samples = csv.DictWriter(samples_file, SAMPLE_FIELDS)
            plans = csv.DictWriter(plans_file, PLAN_FIELDS)
            samples.writeheader()
            plans.writeheader()

            def row(
                variant: Variant, cell: Cell, k: int, rep: int, ids: list[UUID], s: float
            ) -> None:
                r = recall(ids, cell.expected, k)
                samples.writerow(
                    {
                        "variant": variant.name,
                        "tenant": cell.tenant_key,
                        "chunks": cell.chunks,
                        "user": cell.user,
                        "access": ACCESS[cell.user],
                        "readable": cell.readable,
                        "query": cell.query,
                        "k": k,
                        "rep": rep,
                        "ms": f"{s * 1000:.4f}",
                        "returned": len(ids),
                        "expected": len(cell.expected[:k]),
                        "recall": "" if r is None else f"{r:.4f}",
                    }
                )

            for variant in variants:
                t = time.perf_counter()
                for cell in all_cells:  # warm-up
                    await runner.search(variant, cell, K_LATENCY)
                for cell in all_cells:
                    ids, s = await runner.search(variant, cell, K_RECALL)
                    row(variant, cell, K_RECALL, 0, ids, s)
                for rep in range(args.reps):
                    for cell in all_cells:
                        ids, s = await runner.search(variant, cell, K_LATENCY)
                        row(variant, cell, K_LATENCY, rep, ids, s)
                for cell in all_cells:
                    if cell.query != 0:
                        continue
                    explained = await runner.explain(variant, cell, K_LATENCY)
                    summary = summarise_plan(explained, hnsw_indexes)
                    if variant.plan != "planner" and summary["kind"] != variant.plan:
                        mismatches.append(f"{variant.name} {cell.tenant_key} user {cell.user}")
                    plans.writerow(
                        {
                            "variant": variant.name,
                            "tenant": cell.tenant_key,
                            "user": cell.user,
                            "access": ACCESS[cell.user],
                            "forced": variant.plan,
                            **summary,
                        }
                    )
                    plans_json.write(
                        json.dumps(
                            {
                                "variant": variant.name,
                                "tenant": cell.tenant_key,
                                "user": cell.user,
                                "plan": explained,
                            }
                        )
                        + "\n"
                    )
                samples_file.flush()
                steps[f"variant:{variant.name}"] = time.perf_counter() - t
                print(f"  {variant.name}: {steps[f'variant:{variant.name}']:.0f} s")

        steps["total"] = time.perf_counter() - started_all
        meta = {
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": git_commit(),
            "host": host_info(),
            "server": await server_info(urls),
            "corpus": {**json.loads(_fingerprint(config)), "queries_per_tenant": config.queries},
            "tenants": {
                t.key: {"chunks": len(t.chunk_ids), "documents": len(t.document_ids)}
                for t in corpus.tenants
            },
            "readable": {
                f"{c.tenant_key}/{ACCESS[c.user]}": c.readable for c in all_cells if c.query == 0
            },
            "hnsw": asdict(hnsw),
            "reps": args.reps,
            "k_recall": K_RECALL,
            "k_latency": K_LATENCY,
            "variants": [asdict(v) for v in variants],
            "forced_settings": {plan: dict(values) for plan, values in _FORCED.items()},
            "plan_mismatches": mismatches,
            "seconds": {k: round(v, 1) for k, v in steps.items()},
            "corpus_was_loaded": loaded,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    finally:
        for engine in (writer, reader, superuser):
            await engine.dispose()
    if mismatches:
        print("forced plan not used: " + "; ".join(mismatches), file=sys.stderr)
    print(f"wrote {out} in {steps['total']:.0f} s")
    return out


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = Config()
    fields = Settings.model_fields
    parser = argparse.ArgumentParser(prog="python -m eval.bench", description=__doc__)
    parser.add_argument("--total", type=int, default=defaults.total)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--queries", type=int, default=defaults.queries, help="per tenant")
    parser.add_argument("--reps", type=int, default=5, help="timed passes at k=5")
    parser.add_argument("--ef-search", type=int, default=fields["hnsw_ef_search"].default)
    parser.add_argument(
        "--max-scan-tuples", type=int, default=fields["hnsw_max_scan_tuples"].default
    )
    parser.add_argument("--database", default="ragmt_eval")
    parser.add_argument("--variants", nargs="*", help="only these (names in VARIANTS)")
    parser.add_argument("--rebuild-scratch", action="store_true")
    parser.add_argument("--out", type=Path, help="raw output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(bench(_arguments(argv)))


if __name__ == "__main__":
    main()
