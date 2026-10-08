"""A small fictional world for the leak suite, written through app_ingest.

Each test gets two fresh tenants with random ids, so tests don't depend on each
other or on leftovers. Rows are not cleaned up afterwards: audit_events is
insert-only by design, and no application role may delete tenants. Run
`docker compose down -v` to start from an empty database.

Tenant B reuses the same user subs and group names as tenant A on purpose: a
leak through "same group name, other tenant" must show up here.
"""

import hashlib
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest

from tests.db.pg import session

READER = "DATABASE_URL"
INGEST = "INGEST_DATABASE_URL"

# Any vector of the right size will do: these tests check who sees a row, not ranking.
_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
EMBEDDING = "[" + ",".join(["0.1"] * _DIM) + "]"

# document name -> (owning tenant key, ACL)
DOCUMENTS: dict[str, tuple[str, list[str]]] = {
    "hr_a": ("a", ["group:hr"]),
    "public_a": ("a", ["tenant:*"]),
    "bob_a": ("a", ["user:bob"]),
    "hr_b": ("b", ["group:hr"]),
    "public_b": ("b", ["tenant:*"]),
}
MEMBERSHIPS = [("a", "alice", "hr"), ("b", "alice", "hr")]
# Tenant admins for the write routes (the `admins` fixture): group `admins` only,
# so they read nothing an ACL doesn't give them.
ADMINS = {"a": "ada", "b": "bea"}


@dataclass
class World:
    tenants: dict[str, UUID]
    documents: dict[str, UUID] = field(default_factory=dict)


def source_hash(document: str) -> str:
    """A valid documents.source_hash (SHA-256 hex), unique per document name."""
    return hashlib.sha256(document.encode()).hexdigest()


def canary(document: str) -> str:
    """Content of every chunk of `document`: a unique string a leak would expose."""
    return f"CANARY-{document}"


def canaries(*documents: str) -> set[str]:
    return {canary(d) for d in documents}


async def visible_chunks(tenant: UUID, user: str | None) -> set[str]:
    """Chunk contents `user` can read in `tenant` through app_rw, as a user request would."""
    async with session(READER, tenant, user) as conn:
        rows = await conn.fetch("SELECT content FROM chunks")
    return {row["content"] for row in rows}


async def insert_chunk(
    conn: asyncpg.Connection, tenant: UUID, document: UUID, text: str, ordinal: int
) -> UUID:
    chunk_id: UUID = await conn.fetchval(
        """
        INSERT INTO chunks (tenant_id, document_id, ordinal, content, embedding)
        VALUES ($1, $2, $3, $4, $5::text::vector)
        RETURNING id
        """,
        tenant,
        document,
        ordinal,
        text,
        EMBEDDING,
    )
    return chunk_id


@pytest.fixture
async def world() -> World:
    w = World(tenants={"a": uuid4(), "b": uuid4()})
    for key, tenant in w.tenants.items():
        async with session(INGEST, tenant) as conn:
            await conn.execute(
                "INSERT INTO tenants (id, name) VALUES ($1, $2)", tenant, f"Tenant {key}"
            )
            for t, sub, group in MEMBERSHIPS:
                if t == key:
                    await conn.execute(
                        "INSERT INTO memberships (tenant_id, user_sub, group_name)"
                        " VALUES ($1, $2, $3)",
                        tenant,
                        sub,
                        group,
                    )
            for name, (t, acl) in DOCUMENTS.items():
                if t != key:
                    continue
                doc: UUID = await conn.fetchval(
                    "INSERT INTO documents (tenant_id, title, source_hash)"
                    " VALUES ($1, $2, $3) RETURNING id",
                    tenant,
                    name,
                    source_hash(name),
                )
                w.documents[name] = doc
                await conn.executemany(
                    "INSERT INTO document_acl (tenant_id, document_id, principal)"
                    " VALUES ($1, $2, $3)",
                    [(tenant, doc, p) for p in acl],
                )
                for ordinal in range(2):
                    await insert_chunk(conn, tenant, doc, canary(name), ordinal)
    return w


@pytest.fixture
async def admins(world: World) -> World:
    """The world, plus ADMINS in each tenant's `admins` group."""
    for tenant, sub in ADMINS.items():
        async with session(INGEST, world.tenants[tenant]) as conn:
            await conn.execute(
                "INSERT INTO memberships (tenant_id, user_sub, group_name)"
                " VALUES ($1, $2, 'admins')",
                world.tenants[tenant],
                sub,
            )
    return world


# --- the property test's tally, and docs/results/leaks.md ------------------------------

# What each kind of check in test_properties.py compares, in report order.
CHECK_KINDS = {
    "documents": "`GET /documents` equals the oracle's readable documents (ids and titles)",
    "document_by_id": "`GET /documents/{id}` is 404 for a document the user may not read",
    "chunks": "`SELECT … FROM chunks` as app_rw equals the oracle's readable chunks",
    "retrieval": "`PgVectorRetriever.search` returns only readable chunks",
    "ask_citations": "`POST /ask` cites, and its prompt holds, only readable chunks",
    "ask_canaries": "`POST /ask` holds no canary of a document the user may not read",
    "foreign_write": "a write naming another tenant finds nothing and changes nothing",
}


@dataclass
class LeakTally:
    """Counted by test_properties.py as it runs; written out by `pytest_sessionfinish`."""

    checks: Counter[str] = field(default_factory=Counter)
    examples: int = 0
    steps: int = 0

    @property
    def total(self) -> int:
        return sum(self.checks.values())


TALLY = LeakTally()

# `make leaks` sets it to docs/results/leaks.md.
REPORT_ENV = "LEAKS_REPORT"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = os.environ.get(REPORT_ENV)
    if path:
        Path(path).write_text(_report(session, exitstatus), encoding="utf-8", newline="\n")


def _report(session: pytest.Session, exitstatus: int) -> str:
    profile = os.environ.get("HYPOTHESIS_PROFILE", "ci")
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    passed = exitstatus == pytest.ExitCode.OK and session.testsfailed == 0
    if not passed:
        headline = (
            f"**FAILED** after {TALLY.total} checks: a leak or an error. The pytest output"
            " has the failing example; Hypothesis prints the shortest one it found."
        )
    elif TALLY.examples == 0:
        headline = "**The property test did not run** (no database?): no leak count."
    else:
        headline = f"**0 leaks in {TALLY.total} attempts.**"
    lines = [
        "# Leak suite results",
        "",
        headline,
        "",
        f"Written by `make leaks` on {when}: `pytest -m leaks` with the Hypothesis profile"
        f" `{profile}`, {session.testscollected} tests collected, {session.testsfailed} failed.",
        "",
        "## Property test (`tests/leaks/test_properties.py`)",
        "",
        f"{TALLY.examples} random worlds of 2 or 3 tenants, {TALLY.steps} states checked"
        " (after setup and after every operation). In each state, every user of every"
        " tenant is compared with a plain-Python oracle (`tests/leaks/oracle.py`) that"
        " replays the same operations without SQL. An attempt is one comparison for one"
        " user in one state, or one write aimed at another tenant:",
        "",
        "| Check | Compares | Attempts |",
        "|---|---|---:|",
        *(f"| {kind} | {what} | {TALLY.checks[kind]} |" for kind, what in CHECK_KINDS.items()),
        f"| **Total** | | **{TALLY.total}** |",
        "",
        "Only the property test's checks are counted. The other leak tests are fixed"
        " cases (one per leak scenario) and count as tests above.",
        "",
    ]
    return "\n".join(lines)
