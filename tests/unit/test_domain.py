"""Domain value objects, errors and ports (ragmt.domain). No I/O."""

from collections.abc import Sequence
from dataclasses import fields
from uuid import UUID, uuid4

import pytest

from ragmt.domain import (
    NO_CONTEXT_ANSWER,
    Answer,
    ChatMessage,
    ChatProvider,
    ChunkDraft,
    Citation,
    ConvertedDocument,
    DocumentConverter,
    DocumentTooLargeError,
    EmbeddingProvider,
    IngestError,
    RetrievedChunk,
    Retriever,
    UnsupportedDocumentError,
    citation_marker,
)


def test_chunk_draft_rejects_negative_ordinals_and_blank_content() -> None:
    assert ChunkDraft(0, None, "text").heading is None
    with pytest.raises(ValueError, match="ordinal"):
        ChunkDraft(-1, "Heading", "text")
    with pytest.raises(ValueError, match="content"):
        ChunkDraft(0, "Heading", " \n")


def test_converted_document_needs_a_title() -> None:
    with pytest.raises(ValueError, match="title"):
        ConvertedDocument(title=" ", markdown="# x")


def test_value_objects_are_immutable() -> None:
    chunk = ChunkDraft(0, None, "text")
    with pytest.raises(AttributeError):
        chunk.content = "other"  # type: ignore[misc]


def test_errors_carry_their_details_and_share_a_base() -> None:
    too_large = DocumentTooLargeError(size=11, limit=10)
    unsupported = UnsupportedDocumentError("notes.pdf", "unsupported file type")
    assert isinstance(too_large, IngestError)
    assert isinstance(unsupported, IngestError)
    assert (too_large.size, too_large.limit) == (11, 10)
    assert str(unsupported) == "notes.pdf: unsupported file type"


class _Converter:
    def convert(self, data: bytes, filename: str) -> ConvertedDocument:
        return ConvertedDocument(title=filename, markdown=data.decode())


class _Embedder:
    dim = 2

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


async def test_adapters_satisfy_the_ports_by_shape() -> None:
    converter: DocumentConverter = _Converter()
    embedder: EmbeddingProvider = _Embedder()
    assert converter.convert(b"# Hi", "hi.md").markdown == "# Hi"
    assert await embedder.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert len(await embedder.embed_query("q")) == embedder.dim


# --- Retrieval and generation (ADR 0009) ----------------------------------------

DOCUMENT_ID = UUID("4d9a4c3e-2b1f-4e6a-9c7d-0f1e2d3c4b5a")


def _retrieved(ordinal: int = 2, heading: str | None = "Budget") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=DOCUMENT_ID,
        ordinal=ordinal,
        title="Q3 plan",
        heading=heading,
        content="text",
        score=0.83,
    )


def test_citation_marker_format() -> None:
    marker = citation_marker(DOCUMENT_ID, 3)
    assert marker == "[doc:4d9a4c3e-2b1f-4e6a-9c7d-0f1e2d3c4b5a#3]"
    assert _retrieved(ordinal=3).marker == marker


def test_retrieved_chunk_rejects_negative_ordinals() -> None:
    with pytest.raises(ValueError, match="ordinal"):
        _retrieved(ordinal=-1)


def test_citation_carries_only_document_title_and_heading() -> None:
    citation = _retrieved().citation()
    assert citation == Citation(document_id=DOCUMENT_ID, title="Q3 plan", heading="Budget")
    assert {f.name for f in fields(Citation)} == {"document_id", "title", "heading"}


def test_no_type_carries_an_embedding() -> None:
    for cls in (RetrievedChunk, Citation, Answer):
        assert not any("embedding" in f.name or "vector" in f.name for f in fields(cls))


def test_answer_without_context_names_no_model() -> None:
    answer = Answer(text=NO_CONTEXT_ANSWER, citations=(), model=None)
    assert answer.model is None
    with pytest.raises(AttributeError):
        answer.text = "other"  # type: ignore[misc]


class _Retriever:
    async def search(
        self, conn: str, query_vector: Sequence[float], k: int
    ) -> list[RetrievedChunk]:
        return [_retrieved()][:k]


class _Chat:
    model = "fake-chat"

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        return f"{len(messages)} {messages[-1].content}"


async def test_retriever_and_chat_satisfy_the_ports_by_shape() -> None:
    retriever: Retriever[str] = _Retriever()
    chat: ChatProvider = _Chat()
    assert len(await retriever.search("conn", [0.0, 1.0], 1)) == 1
    reply = await chat.complete("system", [ChatMessage(role="user", content="hi")])
    assert (chat.model, reply) == ("fake-chat", "1 hi")
