"""The seed corpus loads repeatably, and each canary reaches exactly its readers.

Later phases (retrieval, eval, the API leak tests) rely on this corpus, so who
may read what is written out by hand here rather than derived from corpus.py.
"""

import os
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from seed.corpus import ACME, ALICE, BOB, CAROL, DAVE, ERIN, TENANTS, UMBRA
from seed.load import load
from tests.db.pg import session

pytestmark = [pytest.mark.db, pytest.mark.leaks]

READER = "DATABASE_URL"
WRITER = "INGEST_DATABASE_URL"
DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

EXPECTED: dict[tuple[UUID, str], set[str]] = {
    (ACME, ALICE): {"CANARY-ACME-HANDBOOK", "CANARY-ACME-BUDGET", "CANARY-ACME-ALICE-REVIEW"},
    (ACME, BOB): {"CANARY-ACME-HANDBOOK", "CANARY-ACME-RUNBOOK"},
    (ACME, ERIN): {"CANARY-ACME-HANDBOOK"},
    # alice is in finance at Acme only: Umbra's finance budget stays hidden.
    (UMBRA, ALICE): {"CANARY-UMBRA-SAFETY"},
    (UMBRA, CAROL): {"CANARY-UMBRA-SAFETY", "CANARY-UMBRA-TRIAL"},
    (UMBRA, DAVE): {"CANARY-UMBRA-SAFETY", "CANARY-UMBRA-BUDGET"},
}

ALL_CANARIES = {doc.canary for tenant in TENANTS for doc in tenant.documents}


@pytest.fixture
async def seeded() -> None:
    """Load the corpus twice through app_ingest, as `make seed` run twice would."""
    url = os.environ.get(WRITER)
    if not url:
        pytest.skip(f"{WRITER} is not set")
    engine = create_async_engine(url)
    try:
        first = await load(engine, DIM)
        second = await load(engine, DIM)
    finally:
        await engine.dispose()
    assert first == second


async def _visible_canaries(tenant: UUID, user: str) -> set[str]:
    async with session(READER, tenant, user) as conn:
        rows = await conn.fetch("SELECT content FROM chunks")
    return {c for c in ALL_CANARIES if any(c in row["content"] for row in rows)}


def test_every_canary_is_unique() -> None:
    canaries = [doc.canary for tenant in TENANTS for doc in tenant.documents]
    assert len(canaries) == len(set(canaries))
    assert set().union(*EXPECTED.values()) == ALL_CANARIES


@pytest.mark.usefixtures("seeded")
@pytest.mark.parametrize(("tenant", "user"), list(EXPECTED))
async def test_canaries_reach_exactly_their_readers(tenant: UUID, user: str) -> None:
    assert await _visible_canaries(tenant, user) == EXPECTED[(tenant, user)]


@pytest.mark.usefixtures("seeded")
async def test_reloading_does_not_duplicate_chunks() -> None:
    expected = sum(len(doc.sections) for doc in TENANTS[0].documents)
    async with session(WRITER, ACME) as conn:
        count = await conn.fetchval("SELECT count(*) FROM chunks")
    assert count == expected
