"""Core entities (Principal, Document, Chunk) and ports. No I/O."""

from ragmt.domain.ingest import (
    ChunkDraft,
    ConvertedDocument,
    DocumentTooLargeError,
    IngestError,
    UnsupportedDocumentError,
)
from ragmt.domain.ports import ChatProvider, DocumentConverter, EmbeddingProvider, Retriever
from ragmt.domain.principal import TENANT_ADMIN_GROUP, Principal
from ragmt.domain.retrieval import (
    NO_CONTEXT_ANSWER,
    Answer,
    ChatMessage,
    Citation,
    RetrievedChunk,
    citation_marker,
)

__all__ = [
    "NO_CONTEXT_ANSWER",
    "TENANT_ADMIN_GROUP",
    "Answer",
    "ChatMessage",
    "ChatProvider",
    "ChunkDraft",
    "Citation",
    "ConvertedDocument",
    "DocumentConverter",
    "DocumentTooLargeError",
    "EmbeddingProvider",
    "IngestError",
    "Principal",
    "RetrievedChunk",
    "Retriever",
    "UnsupportedDocumentError",
    "citation_marker",
]
