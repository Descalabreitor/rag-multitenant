"""Ingestion value objects and errors (ragmt.domain). No I/O."""

from collections.abc import Sequence

import pytest

from ragmt.domain import (
    ChunkDraft,
    ConvertedDocument,
    DocumentConverter,
    DocumentTooLargeError,
    EmbeddingProvider,
    IngestError,
    UnsupportedDocumentError,
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
