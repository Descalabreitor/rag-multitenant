"""The write path for documents: ingest, replace, ACL changes and deletes (ADR 0008).

Every method runs on the app_ingest engine (INGEST_DATABASE_URL), inside
`tenant_session(writer_engine, tenant_id)`, so RLS confines it to that one
tenant whatever the SQL says. `tenant_id` is always an argument, and the caller
must pass the one from the verified token; nothing here reads a tenant from
anywhere else. The admin check (ADR 0008) is the caller's job and comes first:
this module trusts that the caller may write.

Converting, chunking and embedding happen before the write transaction, so a
slow embedding model never holds row locks. A cheap read transaction runs
first, so re-uploading a file that is already there costs no embedding calls.

Audit rows hold ids, counts, hashes and principals, never document content or
titles.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.domain import (
    ChunkDraft,
    DocumentConverter,
    DocumentTooLargeError,
    EmbeddingProvider,
    IngestError,
    UnsupportedDocumentError,
)
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session

# The document_acl_principal_format constraint, minus what it lets through by
# accident: newlines, other control characters and blank names.
_PRINCIPAL = re.compile(r"(?:user|group):[^\x00-\x1f\x7f]*[^\s\x00-\x1f\x7f][^\x00-\x1f\x7f]*")
_TENANT_WIDE = "tenant:*"


class DocumentNotFoundError(IngestError):
    """No live document with that id in the tenant.

    The same error for an id that never existed, one that was deleted and one of
    another tenant, so a caller can't tell them apart (the API answers 404).
    """

    def __init__(self, document_id: UUID) -> None:
        super().__init__(f"document {document_id} not found")
        self.document_id = document_id


class InvalidAclError(IngestError):
    """An ACL that is empty or holds a principal that isn't user:, group: or tenant:*."""


class DuplicateDocumentError(IngestError):
    """A replace would give the document the same content as another live document."""

    def __init__(self, existing_id: UUID) -> None:
        super().__init__(f"the same file is already stored as document {existing_id}")
        self.existing_id = existing_id


class Chunker(Protocol):
    """Splits a document's Markdown into chunks (ragmt.ingest's chunker)."""

    def __call__(self, markdown: str, max_chars: int, overlap_chars: int) -> Sequence[ChunkDraft]:
        """Same signature as `ragmt.ingest.chunking.chunk`."""
        ...


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What an ingest or replace left in the database.

    With `unchanged`, the bytes were already stored and nothing was written:
    `document_id`, `title` and `chunks` describe the existing document.
    """

    document_id: UUID
    title: str
    chunks: int
    unchanged: bool


def normalize_acl(principals: Iterable[str]) -> tuple[str, ...]:
    """Check every principal and drop duplicates, keeping the first occurrence.

    Raises `InvalidAclError` for an empty ACL: a document nobody can read is
    what a delete is for.
    """
    acl = tuple(dict.fromkeys(principals))
    if not acl:
        raise InvalidAclError("the ACL must not be empty")
    for principal in acl:
        if principal != _TENANT_WIDE and not _PRINCIPAL.fullmatch(principal):
            raise InvalidAclError(
                f"invalid principal {principal!r}: expected user:<sub>, group:<name> or tenant:*"
            )
    return acl


def source_hash(data: bytes) -> str:
    """documents.source_hash: SHA-256 of the uploaded bytes, lower-case hex."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class _Prepared:
    title: str
    drafts: tuple[ChunkDraft, ...]
    embeddings: tuple[str, ...]  # pgvector literals, one per draft


class IngestService:
    """Document writes for one process, on an engine connected as app_ingest."""

    def __init__(
        self,
        writer_engine: AsyncEngine,
        converter: DocumentConverter,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        if embedder.dim != settings.embedding_dim:
            raise ValueError(
                f"embedder dim {embedder.dim} != EMBEDDING_DIM {settings.embedding_dim}"
            )
        self._engine = writer_engine
        self._converter = converter
        self._chunker = chunker
        self._embedder = embedder
        self._settings = settings

    @property
    def max_bytes(self) -> int:
        """The largest upload accepted (INGEST_MAX_BYTES)."""
        return self._settings.ingest_max_bytes

    # --- ingest and replace ------------------------------------------------------

    async def ingest(
        self,
        tenant_id: UUID,
        actor_sub: str,
        data: bytes,
        filename: str,
        acl: Sequence[str] | None = None,
    ) -> IngestResult:
        """Store a new document, or return the live one with the same bytes.

        Without `acl` only the uploader can read it (`user:<actor_sub>`); an
        explicit ACL replaces that default.
        """
        _require_actor(actor_sub)
        principals = normalize_acl([f"user:{actor_sub}"] if acl is None else acl)
        self._check_size(data)
        digest = source_hash(data)

        async with tenant_session(self._engine, tenant_id) as conn:
            existing = await _live_by_hash(conn, tenant_id, digest)
        if existing is not None:
            return existing

        prepared = await self._prepare(data, filename)

        async with tenant_session(self._engine, tenant_id) as conn:
            # Another upload of the same bytes may have committed since the check
            # above. ON CONFLICT waits for it and then inserts nothing.
            document_id: UUID | None = (
                await conn.execute(
                    text(
                        "INSERT INTO documents (tenant_id, title, source_hash, created_by)"
                        " VALUES (:t, :title, :hash, :actor)"
                        " ON CONFLICT (tenant_id, source_hash) WHERE deleted_at IS NULL"
                        " DO NOTHING RETURNING id"
                    ),
                    {"t": tenant_id, "title": prepared.title, "hash": digest, "actor": actor_sub},
                )
            ).scalar_one_or_none()
            if document_id is None:
                existing = await _live_by_hash(conn, tenant_id, digest)
                if existing is None:  # deleted again in between; rare enough to just fail
                    raise RuntimeError("concurrent upload of the same file; retry")
                return existing

            # The ACL goes in before the chunks, so the chunk trigger copies it.
            await _insert_acl(conn, tenant_id, document_id, principals)
            await _insert_chunks(conn, tenant_id, document_id, prepared)
            await _audit(
                conn,
                tenant_id,
                actor_sub,
                "ingest",
                document_id=document_id,
                chunks=len(prepared.drafts),
                source_hash=digest,
                principals=list(principals),
            )
        return IngestResult(document_id, prepared.title, len(prepared.drafts), unchanged=False)

    async def replace(
        self, tenant_id: UUID, actor_sub: str, document_id: UUID, data: bytes, filename: str
    ) -> IngestResult:
        """Give a live document new content: new chunks, title and hash, same ACL."""
        _require_actor(actor_sub)
        self._check_size(data)
        digest = source_hash(data)

        async with tenant_session(self._engine, tenant_id) as conn:
            current = await _lock_live(conn, tenant_id, document_id)
            if current.source_hash == digest:
                return IngestResult(document_id, current.title, current.chunks, unchanged=True)

        prepared = await self._prepare(data, filename)

        async with tenant_session(self._engine, tenant_id) as conn:
            current = await _lock_live(conn, tenant_id, document_id)
            if current.source_hash == digest:  # replaced with the same bytes meanwhile
                return IngestResult(document_id, current.title, current.chunks, unchanged=True)
            other = await _live_by_hash(conn, tenant_id, digest)
            if other is not None:
                raise DuplicateDocumentError(other.document_id)

            await conn.execute(
                text("DELETE FROM chunks WHERE tenant_id = :t AND document_id = :d"),
                {"t": tenant_id, "d": document_id},
            )
            await conn.execute(
                text(
                    "UPDATE documents SET title = :title, source_hash = :hash"
                    " WHERE tenant_id = :t AND id = :d"
                ),
                {"t": tenant_id, "d": document_id, "title": prepared.title, "hash": digest},
            )
            await _insert_chunks(conn, tenant_id, document_id, prepared)
            await _audit(
                conn,
                tenant_id,
                actor_sub,
                "replace",
                document_id=document_id,
                chunks=len(prepared.drafts),
                previous_chunks=current.chunks,
                source_hash=digest,
                previous_source_hash=current.source_hash,
            )
        return IngestResult(document_id, prepared.title, len(prepared.drafts), unchanged=False)

    # --- ACL and deletes -------------------------------------------------------------

    async def set_acl(
        self, tenant_id: UUID, actor_sub: str, document_id: UUID, principals: Sequence[str]
    ) -> tuple[str, ...]:
        """Replace a live document's ACL; the triggers update its chunks in the same
        transaction. Returns the ACL as stored (checked, without duplicates)."""
        _require_actor(actor_sub)
        new = normalize_acl(principals)
        async with tenant_session(self._engine, tenant_id) as conn:
            await _lock_live(conn, tenant_id, document_id)
            old = await _acl(conn, tenant_id, document_id)
            removed = [p for p in old if p not in new]
            added = [p for p in new if p not in old]
            if removed:
                await conn.execute(
                    text(
                        "DELETE FROM document_acl"
                        " WHERE tenant_id = :t AND document_id = :d AND principal = ANY(:p)"
                    ),
                    {"t": tenant_id, "d": document_id, "p": removed},
                )
            if added:
                await _insert_acl(conn, tenant_id, document_id, added)
            await _audit(
                conn,
                tenant_id,
                actor_sub,
                "acl_change",
                document_id=document_id,
                old_principals=sorted(old),
                new_principals=sorted(new),
            )
        return new

    async def soft_delete(self, tenant_id: UUID, actor_sub: str, document_id: UUID) -> None:
        """Hide a live document from every reader (ADR 0008). Its ACL rows are
        removed by a trigger, so the audit row keeps them."""
        _require_actor(actor_sub)
        async with tenant_session(self._engine, tenant_id) as conn:
            current = await _lock_live(conn, tenant_id, document_id)
            old = await _acl(conn, tenant_id, document_id)
            await conn.execute(
                text("UPDATE documents SET deleted_at = now() WHERE tenant_id = :t AND id = :d"),
                {"t": tenant_id, "d": document_id},
            )
            await _audit(
                conn,
                tenant_id,
                actor_sub,
                "soft_delete",
                document_id=document_id,
                chunks=current.chunks,
                source_hash=current.source_hash,
                old_principals=sorted(old),
            )

    async def hard_delete(self, tenant_id: UUID, actor_sub: str, document_id: UUID) -> None:
        """Remove a document, live or soft-deleted; its ACL and chunks go by cascade."""
        _require_actor(actor_sub)
        async with tenant_session(self._engine, tenant_id) as conn:
            current = await _lock(conn, tenant_id, document_id, live_only=False)
            old = await _acl(conn, tenant_id, document_id)
            await conn.execute(
                text("DELETE FROM documents WHERE tenant_id = :t AND id = :d"),
                {"t": tenant_id, "d": document_id},
            )
            await _audit(
                conn,
                tenant_id,
                actor_sub,
                "hard_delete",
                document_id=document_id,
                chunks=current.chunks,
                source_hash=current.source_hash,
                old_principals=sorted(old),
            )

    # --- outside the transaction -------------------------------------------------

    def _check_size(self, data: bytes) -> None:
        if len(data) > self._settings.ingest_max_bytes:
            raise DocumentTooLargeError(len(data), self._settings.ingest_max_bytes)

    async def _prepare(self, data: bytes, filename: str) -> _Prepared:
        # Conversion and chunking are CPU work: keep them off the event loop.
        title, drafts = await asyncio.to_thread(self._convert_and_chunk, data, filename)
        if not drafts:
            raise UnsupportedDocumentError(filename, "no text to index")
        embeddings: list[str] = []
        size = self._settings.embed_batch_size
        for start in range(0, len(drafts), size):
            batch = drafts[start : start + size]
            vectors = await self._embedder.embed_documents([d.content for d in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)}")
            embeddings.extend(self._vector_literal(v) for v in vectors)
        return _Prepared(title, drafts, tuple(embeddings))

    def _convert_and_chunk(self, data: bytes, filename: str) -> tuple[str, tuple[ChunkDraft, ...]]:
        converted = self._converter.convert(data, filename)
        drafts = self._chunker(
            converted.markdown,
            max_chars=self._settings.chunk_max_chars,
            overlap_chars=self._settings.chunk_overlap_chars,
        )
        return converted.title, tuple(drafts)

    def _vector_literal(self, vector: Sequence[float]) -> str:
        if len(vector) != self._settings.embedding_dim:
            raise RuntimeError(
                f"embedding has {len(vector)} dims, EMBEDDING_DIM is {self._settings.embedding_dim}"
            )
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"


# --- SQL helpers (all on a tenant_session connection as app_ingest) ------------------


@dataclass(frozen=True, slots=True)
class _Current:
    title: str
    source_hash: str
    chunks: int


def _require_actor(actor_sub: str) -> None:
    if not actor_sub:
        raise ValueError("actor_sub must not be empty")


async def _live_by_hash(conn: AsyncConnection, tenant_id: UUID, digest: str) -> IngestResult | None:
    row = (
        await conn.execute(
            text(
                "SELECT d.id, d.title,"
                " (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunks"
                " FROM documents d"
                " WHERE d.tenant_id = :t AND d.source_hash = :hash AND d.deleted_at IS NULL"
            ),
            {"t": tenant_id, "hash": digest},
        )
    ).one_or_none()
    if row is None:
        return None
    return IngestResult(row.id, row.title, row.chunks, unchanged=True)


async def _lock_live(conn: AsyncConnection, tenant_id: UUID, document_id: UUID) -> _Current:
    return await _lock(conn, tenant_id, document_id, live_only=True)


async def _lock(
    conn: AsyncConnection, tenant_id: UUID, document_id: UUID, *, live_only: bool
) -> _Current:
    """Lock the document row until commit, or raise DocumentNotFoundError.

    RLS already limits app_ingest to the session's tenant, so another tenant's id
    finds nothing; the tenant_id condition only makes that visible here.
    """
    row = (
        await conn.execute(
            text(
                "SELECT d.title, d.source_hash,"
                " (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunks"
                " FROM documents d WHERE d.tenant_id = :t AND d.id = :d"
                " AND (d.deleted_at IS NULL OR NOT CAST(:live_only AS boolean))"
                " FOR UPDATE OF d"
            ),
            {"t": tenant_id, "d": document_id, "live_only": live_only},
        )
    ).one_or_none()
    if row is None:
        raise DocumentNotFoundError(document_id)
    return _Current(row.title, row.source_hash, row.chunks)


async def _acl(conn: AsyncConnection, tenant_id: UUID, document_id: UUID) -> list[str]:
    result = await conn.execute(
        text(
            "SELECT principal FROM document_acl"
            " WHERE tenant_id = :t AND document_id = :d ORDER BY principal"
        ),
        {"t": tenant_id, "d": document_id},
    )
    return list(result.scalars())


async def _insert_acl(
    conn: AsyncConnection, tenant_id: UUID, document_id: UUID, principals: Iterable[str]
) -> None:
    await conn.execute(
        text("INSERT INTO document_acl (tenant_id, document_id, principal) VALUES (:t, :d, :p)"),
        [{"t": tenant_id, "d": document_id, "p": p} for p in principals],
    )


async def _insert_chunks(
    conn: AsyncConnection, tenant_id: UUID, document_id: UUID, prepared: _Prepared
) -> None:
    # acl_principals is left out: the trigger computes it (ADR 0003).
    await conn.execute(
        text(
            "INSERT INTO chunks (tenant_id, document_id, ordinal, heading, content, embedding)"
            " VALUES (:t, :d, :o, :h, :c, CAST(:e AS vector))"
        ),
        [
            {
                "t": tenant_id,
                "d": document_id,
                "o": d.ordinal,
                "h": d.heading,
                "c": d.content,
                "e": e,
            }
            for d, e in zip(prepared.drafts, prepared.embeddings, strict=True)
        ],
    )


async def _audit(
    conn: AsyncConnection, tenant_id: UUID, actor_sub: str, action: str, **details: Any
) -> None:
    await conn.execute(
        text(
            "INSERT INTO audit_events (tenant_id, actor_sub, action, details)"
            " VALUES (:t, :actor, :action, CAST(:details AS jsonb))"
        ),
        {
            "t": tenant_id,
            "actor": actor_sub,
            "action": action,
            "details": json.dumps(details, default=str),
        },
    )
