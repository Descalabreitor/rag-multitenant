"""AskService and the ask audit row, without a database.

The flow against PostgreSQL (retrieval under RLS, the audit insert, the 503 and
the empty answer) is in tests/leaks/test_ask.py.
"""

import hashlib
from collections.abc import Sequence
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from ragmt.adapters.llm import FakeChat, FakeEmbeddings
from ragmt.ask import AskService, QuestionTooLongError
from ragmt.audit import ask_details
from ragmt.domain import Principal, RetrievedChunk
from tests.unit.test_api import SETTINGS


class RecordingEmbeddings(FakeEmbeddings):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


class UnusedRetriever:
    async def search(
        self, conn: AsyncConnection, query_vector: Sequence[float], k: int
    ) -> list[RetrievedChunk]:
        raise AssertionError("the retriever must not be called")


def chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=0,
        title="t",
        heading=None,
        content="c",
        score=score,
    )


async def test_question_over_the_limit_is_refused_before_embedding() -> None:
    settings = SETTINGS.model_copy(update={"ask_max_question_chars": 10})
    embedder = RecordingEmbeddings(settings.embedding_dim)
    chat = FakeChat()
    # Points at port 1: any connection attempt would fail loudly.
    engine = create_async_engine(settings.database_url.get_secret_value())
    service = AskService(engine, embedder, UnusedRetriever(), chat, settings)
    principal = Principal(sub="alice", tenant_id=uuid4())
    try:
        with pytest.raises(QuestionTooLongError):
            await service.ask(principal, "x" * 11)
    finally:
        await engine.dispose()
    assert embedder.queries == []
    assert chat.calls == []


def test_ask_details_hold_scores_model_and_hash_but_no_question_by_default() -> None:
    chunks = [chunk(0.9), chunk(0.5)]
    details = ask_details(chunks, model="m", question="secret question", store_text=False)
    assert details == {
        "scores": [0.9, 0.5],
        "model": "m",
        "question_sha256": hashlib.sha256(b"secret question").hexdigest(),
    }
    assert "secret question" not in str(details)


def test_ask_details_store_the_question_only_when_enabled() -> None:
    details = ask_details([], model=None, question="q", store_text=True)
    assert details["question"] == "q"
    assert details["model"] is None
    assert details["scores"] == []
