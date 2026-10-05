"""Write the corpus to the database as app_ingest, one tenant per transaction.

Loading is a reset, not a merge: each seed tenant's memberships and documents are
replaced with what `corpus.py` says, so running it twice gives the same state.
Tenants themselves are upserted (app_ingest may not delete them).

The embeddings are placeholders until the embedding adapter exists: unit vectors
derived from a hash of the chunk text. They are stable across runs and valid for
cosine distance, but carry no meaning, so ranking on them is noise.
"""

import hashlib
import math
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.tenancy import tenant_session
from seed.corpus import TENANTS, Document, Tenant


@dataclass(frozen=True)
class Loaded:
    tenant: str
    memberships: int
    documents: int
    chunks: int


def chunk_content(document: Document, body: str) -> str:
    return f"{body}\n\nReference: {document.canary}"


def placeholder_embedding(content: str, dim: int) -> str:
    """A unit vector from SHAKE-256 of `content`, as a pgvector literal."""
    raw = hashlib.shake_256(content.encode()).digest(2 * dim)
    values = [int.from_bytes(raw[i : i + 2], "big") - 32767.5 for i in range(0, 2 * dim, 2)]
    norm = math.sqrt(sum(v * v for v in values))
    return "[" + ",".join(f"{v / norm:.6f}" for v in values) + "]"


async def _load_tenant(conn: AsyncConnection, tenant: Tenant, dim: int) -> Loaded:
    await conn.execute(
        text(
            "INSERT INTO tenants (id, name) VALUES (:id, :name)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name"
        ),
        {"id": tenant.id, "name": tenant.name},
    )
    # The policies already confine both deletes to this tenant; the WHERE says so too.
    # Deleting a document cascades to its ACL entries and chunks.
    await conn.execute(text("DELETE FROM memberships WHERE tenant_id = :t"), {"t": tenant.id})
    await conn.execute(text("DELETE FROM documents WHERE tenant_id = :t"), {"t": tenant.id})

    if tenant.memberships:
        await conn.execute(
            text("INSERT INTO memberships (tenant_id, user_sub, group_name) VALUES (:t, :u, :g)"),
            [{"t": tenant.id, "u": sub, "g": group} for sub, group in tenant.memberships],
        )

    chunks = 0
    for doc in tenant.documents:
        await conn.execute(
            text("INSERT INTO documents (id, tenant_id, title) VALUES (:id, :t, :title)"),
            {"id": doc.id, "t": tenant.id, "title": doc.title},
        )
        await conn.execute(
            text(
                "INSERT INTO document_acl (tenant_id, document_id, principal) VALUES (:t, :d, :p)"
            ),
            [{"t": tenant.id, "d": doc.id, "p": principal} for principal in doc.acl],
        )
        # acl_principals is filled in by the trigger from document_acl (ADR 0003).
        rows = []
        for ordinal, (heading, body) in enumerate(doc.sections):
            content = chunk_content(doc, body)
            rows.append(
                {
                    "t": tenant.id,
                    "d": doc.id,
                    "o": ordinal,
                    "h": heading,
                    "c": content,
                    "e": placeholder_embedding(content, dim),
                }
            )
        await conn.execute(
            text(
                "INSERT INTO chunks (tenant_id, document_id, ordinal, heading, content, embedding)"
                " VALUES (:t, :d, :o, :h, :c, CAST(:e AS vector))"
            ),
            rows,
        )
        chunks += len(rows)

    return Loaded(tenant.name, len(tenant.memberships), len(tenant.documents), chunks)


async def load(engine: AsyncEngine, dim: int) -> list[Loaded]:
    """Load every seed tenant. `engine` must connect as app_ingest."""
    loaded = []
    for tenant in TENANTS:
        async with tenant_session(engine, tenant.id) as conn:
            loaded.append(await _load_tenant(conn, tenant, dim))
    return loaded
