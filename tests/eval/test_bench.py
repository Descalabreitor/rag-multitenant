"""The benchmark's pure parts: statements, recall, plan summaries, percentiles."""

from uuid import uuid4

import pytest

from eval.bench import VARIANTS, HnswSettings, recall, run_settings, search_sql, summarise_plan
from eval.database import NORMALIZED, PARTITIONED
from eval.report import percentile
from ragmt.retrieval.pgvector import _SEARCH

BY_NAME = {v.name: v for v in VARIANTS}


def test_rls_variants_run_the_retrievers_statement_on_their_table() -> None:
    plain = search_sql(BY_NAME["hnsw_relaxed"])
    assert plain == _SEARCH.text.replace(":query", "$1").replace(":k", "$2")
    assert f"FROM {PARTITIONED} c" in search_sql(BY_NAME["partition_relaxed"])
    assert f"FROM {NORMALIZED} c" in search_sql(BY_NAME["exists_hnsw"])
    for variant in VARIANTS:
        if variant.role == "app_rw":
            assert "$3" not in search_sql(variant), "RLS variants add no filter"


def test_baseline_writes_the_policies_out() -> None:
    sql = search_sql(BY_NAME["baseline_hnsw"])
    assert "c.tenant_id = $3 AND c.acl_principals && $4::text[]" in sql
    assert "d.deleted_at IS NULL" in sql
    assert "a.principal = ANY($4::text[])" in sql


def test_forced_settings() -> None:
    hnsw = HnswSettings(ef_search=40, max_scan_tuples=20_000)
    forced = run_settings(BY_NAME["hnsw_off"], hnsw)
    assert forced["hnsw.iterative_scan"] == "off"
    assert forced["enable_sort"] == "off"
    assert forced["plan_cache_mode"] == "force_custom_plan"
    assert run_settings(BY_NAME["exact"], hnsw)["enable_indexscan"] == "off"
    assert "enable_sort" not in run_settings(BY_NAME["planner"], hnsw)


def test_recall() -> None:
    a, b, c, d = (uuid4() for _ in range(4))
    assert recall([a, b], [a, b, c], 2) == 1.0
    assert recall([c, a], [a, b, c], 2) == 0.5
    assert recall([], [a], 5) == 0.0
    assert recall([d], [a, b], 5) == 0.0
    assert recall([a], [], 5) is None


def _node(kind: str, **extra: object) -> dict[str, object]:
    return {"Node Type": kind, **extra}


def test_summarise_plan_finds_the_hnsw_scan_in_the_cte() -> None:
    explained = {
        "Execution Time": 1.5,
        "Plan": _node(
            "Sort",
            **{"Shared Hit Blocks": 10, "Shared Read Blocks": 2},
            Plans=[
                _node(
                    "Limit",
                    **{"Subplan Name": "CTE nearest"},
                    Plans=[
                        _node(
                            "Index Scan",
                            **{"Index Name": "chunks_embedding_hnsw_idx"},
                            **{"Rows Removed by Filter": 7},
                        )
                    ],
                ),
                _node("Nested Loop"),
            ],
        ),
    }
    summary = summarise_plan(explained, frozenset({"chunks_embedding_hnsw_idx"}))
    assert summary["kind"] == "hnsw"
    assert summary["scan"] == "Index Scan chunks_embedding_hnsw_idx"
    assert summary["rows_removed"] == 7
    assert summary["buffers"] == 12


def test_summarise_plan_exact() -> None:
    explained = {
        "Plan": _node(
            "Limit",
            **{"Subplan Name": "CTE nearest"},
            Plans=[
                _node(
                    "Sort",
                    **{"Sort Method": "top-N heapsort"},
                    Plans=[_node("Bitmap Heap Scan", **{"Relation Name": "chunks"})],
                )
            ],
        )
    }
    summary = summarise_plan(explained, frozenset({"chunks_embedding_hnsw_idx"}))
    assert summary["kind"] == "exact"
    assert summary["scan"] == "Sort (top-N heapsort) > Bitmap Heap Scan chunks"


@pytest.mark.parametrize(("q", "expected"), [(0.0, 1.0), (0.5, 2.5), (0.95, 3.85), (1.0, 4.0)])
def test_percentile_matches_numpy_linear(q: float, expected: float) -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(expected)
