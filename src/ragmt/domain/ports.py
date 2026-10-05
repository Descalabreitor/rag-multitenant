"""Interfaces the domain needs from the outside world. Adapters implement them.

They are Protocols, so an adapter satisfies one by shape, without importing
anything from here.
"""

from collections.abc import Sequence
from typing import Protocol

from ragmt.domain.ingest import ConvertedDocument


class DocumentConverter(Protocol):
    """Turns uploaded bytes into Markdown (ADR 0008).

    Raises `UnsupportedDocumentError` for a file type it doesn't handle, or for
    bytes it can't parse as the type `filename` says. Pure CPU work: callers in
    async code run it in a thread.
    """

    def convert(self, data: bytes, filename: str) -> ConvertedDocument: ...


class EmbeddingProvider(Protocol):
    """Computes embeddings of `dim` floats, which must equal EMBEDDING_DIM.

    Documents and queries are separate calls because models such as
    nomic-embed-text expect a different task prefix for each
    (`search_document: ` and `search_query: `). The provider adds the prefix;
    callers pass plain text.
    """

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One embedding per text, in the same order."""
        ...

    async def embed_query(self, text: str) -> list[float]: ...
