"""Vector search over chunks with pgvector's HNSW index, under RLS (ADR 0001, 0009).

`PgVectorRetriever` runs on the request's `tenant_session` connection as app_rw.
Its query has no tenant or ACL filter: the chunks and documents policies decide
which rows the scan may return. The HNSW settings are set for the current
transaction only (`set_config(..., true)`), from values checked here, and passed
as bound parameters.
"""

import math
from collections.abc import Sequence
from typing import Final, Literal, get_args

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from ragmt.domain import RetrievedChunk
from ragmt.settings import Settings

IterativeScan = Literal["off", "relaxed_order", "strict_order"]
_ITERATIVE_SCANS: Final = frozenset(get_args(IterativeScan))

# pgvector's limits for hnsw.ef_search and hnsw.max_scan_tuples.
_EF_SEARCH_MAX: Final = 1000
_INT_MAX: Final = 2**31 - 1

# No tenant_id or ACL condition, on purpose: RLS filters chunks (tenant and
# acl_principals, ADR 0003) and documents (tenant, ACL and deleted_at) for the
# session's tenant and user. Adding a filter here would only hide a policy bug.
#
# The CTE scans chunks alone, so the plan is an HNSW index scan under a Limit.
# It is MATERIALIZED so the planner can't fold the join into it, and the outer
# ORDER BY re-sorts its rows: with hnsw.iterative_scan = relaxed_order they can
# come out of the index slightly out of order (pgvector's documented pattern).
# A visible chunk always has a visible document (its acl_principals is a copy of
# the document's ACL, empty once the document is deleted), so the inner join
# drops nothing that the scan let through.
_SEARCH = text("""
    WITH nearest AS MATERIALIZED (
        SELECT c.id, c.document_id, c.ordinal, c.heading, c.content,
               c.embedding <=> CAST(:query AS vector) AS distance
        FROM chunks c
        ORDER BY distance
        LIMIT :k
    )
    SELECT n.id, n.document_id, n.ordinal, d.title, n.heading, n.content, n.distance
    FROM nearest n
    JOIN documents d ON d.id = n.document_id
    ORDER BY n.distance, n.id
""")


class PgVectorRetriever:
    """`Retriever[AsyncConnection]`: cosine-distance search with pgvector.

    The HNSW settings are checked once, here. `search` sets them in the caller's
    transaction before the query, so they end with it and never reach the next
    user of the pooled connection.
    """

    def __init__(
        self,
        *,
        dim: int,
        ef_search: int,
        iterative_scan: IterativeScan,
        max_scan_tuples: int,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not 1 <= ef_search <= _EF_SEARCH_MAX:
            raise ValueError(f"ef_search must be between 1 and {_EF_SEARCH_MAX}")
        if iterative_scan not in _ITERATIVE_SCANS:
            raise ValueError(f"iterative_scan must be one of {sorted(_ITERATIVE_SCANS)}")
        if not 1 <= max_scan_tuples <= _INT_MAX:
            raise ValueError(f"max_scan_tuples must be between 1 and {_INT_MAX}")
        self._dim = dim
        self._ef_search = ef_search
        self._iterative_scan: IterativeScan = iterative_scan
        self._max_scan_tuples = max_scan_tuples

    @classmethod
    def from_settings(cls, settings: Settings) -> "PgVectorRetriever":
        return cls(
            dim=settings.embedding_dim,
            ef_search=settings.hnsw_ef_search,
            iterative_scan=settings.hnsw_iterative_scan,
            max_scan_tuples=settings.hnsw_max_scan_tuples,
        )

    async def search(
        self, conn: AsyncConnection, query_vector: Sequence[float], k: int
    ) -> list[RetrievedChunk]:
        """At most `k` chunks the session's user may read, closest first.

        `conn` must be inside a `tenant_session` on app_rw: without a tenant and
        user set, RLS returns nothing.
        """
        if not 1 <= k <= self._ef_search:
            # One index scan returns at most ef_search rows (settings enforce the same rule).
            raise ValueError(f"k must be between 1 and ef_search ({self._ef_search})")
        query = self._vector_literal(query_vector)

        await conn.execute(
            select(
                func.set_config("hnsw.ef_search", str(self._ef_search), True),
                func.set_config("hnsw.iterative_scan", self._iterative_scan, True),
                func.set_config("hnsw.max_scan_tuples", str(self._max_scan_tuples), True),
            )
        )
        result = await conn.execute(_SEARCH, {"query": query, "k": k})
        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                ordinal=row.ordinal,
                title=row.title,
                heading=row.heading,
                content=row.content,
                score=1.0 - float(row.distance),
            )
            for row in result
        ]

    def _vector_literal(self, vector: Sequence[float]) -> str:
        if len(vector) != self._dim:
            raise ValueError(f"query vector has {len(vector)} dims, expected {self._dim}")
        values = [float(x) for x in vector]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("query vector must be finite")
        return "[" + ",".join(repr(x) for x in values) + "]"
