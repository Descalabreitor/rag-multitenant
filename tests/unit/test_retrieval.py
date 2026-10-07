"""PgVectorRetriever's input checks (ragmt.retrieval, ADR 0009). No database.

Every check runs before the first statement, so a connection that fails on use
proves nothing bad reaches PostgreSQL.
"""

from typing import Any

import pytest

from ragmt.retrieval import PgVectorRetriever
from ragmt.settings import Settings
from tests.unit.test_settings import VALID

DIM = 4


class NoConnection:
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the retriever reached the database")


def retriever(**overrides: Any) -> PgVectorRetriever:
    kwargs: dict[str, Any] = {
        "dim": DIM,
        "ef_search": 40,
        "iterative_scan": "relaxed_order",
        "max_scan_tuples": 20_000,
    }
    return PgVectorRetriever(**(kwargs | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dim": 0},
        {"ef_search": 0},
        {"ef_search": 1001},
        {"iterative_scan": "OFF"},
        {"iterative_scan": "off; RESET ALL"},
        {"max_scan_tuples": 0},
        {"max_scan_tuples": 2**31},
    ],
)
def test_rejects_invalid_hnsw_settings(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        retriever(**overrides)


@pytest.mark.parametrize("k", [0, -1, 41])
async def test_rejects_k_outside_one_to_ef_search(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        await retriever().search(NoConnection(), [0.5] * DIM, k)


@pytest.mark.parametrize(
    "vector",
    [[0.5] * (DIM - 1), [0.5] * (DIM + 1), [0.5, 0.5, 0.5, float("nan")], [float("inf")] * DIM],
)
async def test_rejects_a_query_vector_of_the_wrong_size_or_not_finite(
    vector: list[float],
) -> None:
    with pytest.raises(ValueError, match="query vector"):
        await retriever().search(NoConnection(), vector, 5)


def test_from_settings_takes_the_hnsw_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in VALID.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HNSW_EF_SEARCH", "64")
    monkeypatch.setenv("HNSW_ITERATIVE_SCAN", "strict_order")
    monkeypatch.setenv("HNSW_MAX_SCAN_TUPLES", "5000")
    built = PgVectorRetriever.from_settings(Settings(_env_file=None))
    assert (built._dim, built._ef_search, built._iterative_scan, built._max_scan_tuples) == (
        768,
        64,
        "strict_order",
        5000,
    )
