"""Core entities (Principal, Document, Chunk) and ports. No I/O."""

from ragmt.domain.ingest import (
    ChunkDraft,
    ConvertedDocument,
    DocumentTooLargeError,
    IngestError,
    UnsupportedDocumentError,
)
from ragmt.domain.ports import DocumentConverter, EmbeddingProvider
from ragmt.domain.principal import TENANT_ADMIN_GROUP, Principal

__all__ = [
    "TENANT_ADMIN_GROUP",
    "ChunkDraft",
    "ConvertedDocument",
    "DocumentConverter",
    "DocumentTooLargeError",
    "EmbeddingProvider",
    "IngestError",
    "Principal",
    "UnsupportedDocumentError",
]
