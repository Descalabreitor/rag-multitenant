"""What ingestion produces and how it fails. No I/O.

The pipeline is: uploaded bytes → `ConvertedDocument` (a `DocumentConverter`)
→ `ChunkDraft`s (the chunker) → embeddings (an `EmbeddingProvider`) → rows
written as app_ingest (ADR 0008).
"""

from dataclasses import dataclass


class IngestError(Exception):
    """Base for errors the upload route turns into a client error.

    Messages are shown to the caller, so they never include document content.
    """


class UnsupportedDocumentError(IngestError):
    """The file type is not supported, or the file can't be read as that type."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(f"{filename}: {reason}")
        self.filename = filename
        self.reason = reason


class DocumentTooLargeError(IngestError):
    """The upload is larger than INGEST_MAX_BYTES."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"document is {size} bytes, the limit is {limit}")
        self.size = size
        self.limit = limit


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    """A document as Markdown. Headings (`#` lines) are what the chunker splits on."""

    title: str
    markdown: str

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty")


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """One chunk before it is embedded and stored.

    `ordinal` is its position in the document (0, 1, ...), `heading` the
    heading it falls under, if any.
    """

    ordinal: int
    heading: str | None
    content: str

    def __post_init__(self) -> None:
        # Same rule as the chunks_ordinal_non_negative constraint.
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if not self.content.strip():
            raise ValueError("content must not be empty")
