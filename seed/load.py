"""Load the corpus through the real ingestion pipeline, as app_ingest.

Each file in seed/files/ goes through `IngestService` as an upload by its
tenant's admin, with the ACL from corpus.py: the converter, the chunker and the
embedder the caller passes (`make seed` uses LLM_PROVIDER). Nothing here writes
documents, ACLs or chunks by hand.

Loading is a reset to what corpus.py says, and it is repeatable:

- Tenants are upserted (app_ingest may not delete them), and memberships are
  diffed against the corpus.
- A document whose bytes are already stored is left alone (the service
  deduplicates by SHA-256), so it keeps its id and chunks. Only its ACL is reset,
  and only if it differs.
- Every other document of a seed tenant is hard-deleted first: files that
  changed or left the corpus, soft-deleted documents, uploads made since the
  last load. The seed tenants belong to the seed.

A second run with the same files writes nothing at all, audit rows included.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.domain import EmbeddingProvider
from ragmt.ingest.chunking import chunk
from ragmt.ingest.convert import UploadConverter
from ragmt.ingest.service import IngestService, source_hash
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session
from seed.corpus import TENANTS, Document, Tenant
from seed.docx import build_docx

FILES = Path(__file__).resolve().parent / "files"


@dataclass(frozen=True)
class Loaded:
    tenant: str
    memberships: int
    # title -> id of each corpus document, as stored after the load.
    documents: Mapping[str, UUID]
    chunks: int
    # Writes this run made: memberships added or removed, documents ingested or
    # deleted, ACLs reset, the tenant's name. 0 when the tenant was already loaded.
    changes: int


def source_path(document: Document) -> Path:
    """The file in seed/files/ for `document`: a .docx's is its text source."""
    path = FILES / document.path
    return path.with_name(f"{path.name}.md") if path.suffix == ".docx" else path


def document_bytes(document: Document) -> bytes:
    """What the seed uploads for `document`: the file, or the .docx built from its source."""
    path = source_path(document)
    if PurePosixPath(document.path).suffix == ".docx":
        return build_docx(path.read_text(encoding="utf-8"))
    return path.read_bytes()


async def load(
    engine: AsyncEngine, embedder: EmbeddingProvider, settings: Settings
) -> list[Loaded]:
    """Load every seed tenant. `engine` must connect as app_ingest."""
    service = IngestService(
        engine, UploadConverter(settings.ingest_max_bytes), chunk, embedder, settings
    )
    return [await _load_tenant(engine, service, tenant) for tenant in TENANTS]


async def _load_tenant(engine: AsyncEngine, service: IngestService, tenant: Tenant) -> Loaded:
    files = [(document, document_bytes(document)) for document in tenant.documents]
    hashes = [source_hash(data) for _, data in files]

    async with tenant_session(engine, tenant.id) as conn:
        changes = await _upsert_tenant(conn, tenant)
        changes += await _reset_memberships(conn, tenant)
        # The policies confine this to the tenant; the WHERE says so too.
        stale_ids: list[UUID] = list(
            (
                await conn.execute(
                    text(
                        "SELECT id FROM documents WHERE tenant_id = :t"
                        " AND (deleted_at IS NOT NULL OR NOT source_hash = ANY(:hashes))"
                    ),
                    {"t": tenant.id, "hashes": hashes},
                )
            ).scalars()
        )

    for document_id in stale_ids:
        await service.hard_delete(tenant.id, tenant.uploader, document_id)
    changes += len(stale_ids)

    ids: dict[str, UUID] = {}
    chunks = 0
    for document, data in files:
        result = await service.ingest(
            tenant.id,
            tenant.uploader,
            data,
            PurePosixPath(document.path).name,
            acl=document.acl,
        )
        if result.unchanged:
            if await _acl(engine, tenant.id, result.document_id) != set(document.acl):
                await service.set_acl(tenant.id, tenant.uploader, result.document_id, document.acl)
                changes += 1
        else:
            changes += 1
        ids[result.title] = result.document_id
        chunks += result.chunks

    return Loaded(tenant.name, len(tenant.memberships), ids, chunks, changes)


async def _upsert_tenant(conn: AsyncConnection, tenant: Tenant) -> int:
    result = await conn.execute(
        text(
            "INSERT INTO tenants (id, name) VALUES (:id, :name)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name"
            " WHERE tenants.name IS DISTINCT FROM EXCLUDED.name"
        ),
        {"id": tenant.id, "name": tenant.name},
    )
    return result.rowcount


async def _reset_memberships(conn: AsyncConnection, tenant: Tenant) -> int:
    rows = await conn.execute(
        text("SELECT user_sub, group_name FROM memberships WHERE tenant_id = :t"),
        {"t": tenant.id},
    )
    current = {(row.user_sub, row.group_name) for row in rows}
    wanted = set(tenant.memberships)
    extra = [{"t": tenant.id, "u": u, "g": g} for u, g in sorted(current - wanted)]
    missing = [{"t": tenant.id, "u": u, "g": g} for u, g in sorted(wanted - current)]
    if extra:
        await conn.execute(
            text(
                "DELETE FROM memberships WHERE tenant_id = :t AND user_sub = :u AND group_name = :g"
            ),
            extra,
        )
    if missing:
        await conn.execute(
            text("INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES (:t, :u, :g)"),
            missing,
        )
    return len(extra) + len(missing)


async def _acl(engine: AsyncEngine, tenant_id: UUID, document_id: UUID) -> set[str]:
    async with tenant_session(engine, tenant_id) as conn:
        rows = await conn.execute(
            text("SELECT principal FROM document_acl WHERE tenant_id = :t AND document_id = :d"),
            {"t": tenant_id, "d": document_id},
        )
        return set(rows.scalars())
