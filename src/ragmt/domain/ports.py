"""Interfaces the domain needs from the outside world. Adapters implement them.

They are Protocols, so an adapter satisfies one by shape, without importing
anything from here.
"""

from collections.abc import Sequence
from typing import Protocol

from ragmt.domain.ingest import ConvertedDocument
from ragmt.domain.retrieval import ChatMessage, RetrievedChunk


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


class Retriever[Conn](Protocol):
    """Finds the chunks closest to a query vector (ADR 0009).

    `conn` must be the request's `tenant_session` connection as app_rw: the
    retriever adds no tenant or ACL filter of its own, RLS decides what it can
    see. `Conn` is the adapter's connection type, so the domain names no driver.
    Returns at most `k` chunks, closest first, never their embeddings.
    """

    async def search(
        self, conn: Conn, query_vector: Sequence[float], k: int
    ) -> list[RetrievedChunk]: ...


class ChatProvider(Protocol):
    """A chat model with no tools and no database access (ADR 0009).

    It sees only what the caller puts in `system` and `messages`: retrieved
    chunks arrive there already filtered by RLS and wrapped as untrusted data.
    """

    @property
    def model(self) -> str:
        """The model name, recorded in `Answer.model` and in the audit row."""
        ...

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        """The model's reply to `messages` under the `system` prompt."""
        ...
