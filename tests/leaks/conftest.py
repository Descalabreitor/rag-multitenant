"""A small fictional world for the leak suite, written through app_ingest.

Each test gets two fresh tenants with random ids, so tests don't depend on each
other or on leftovers. Rows are not cleaned up afterwards: audit_events is
insert-only by design, and no application role may delete tenants. Run
`docker compose down -v` to start from an empty database.

Tenant B reuses the same user subs and group names as tenant A on purpose: a
leak through "same group name, other tenant" must show up here.
"""

import os
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import asyncpg
import pytest

from tests.db.pg import session

INGEST = "INGEST_DATABASE_URL"

# Any vector of the right size will do: these tests check who sees a row, not ranking.
_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
EMBEDDING = "[" + ",".join(["0.1"] * _DIM) + "]"

# document name -> (owning tenant key, ACL)
DOCUMENTS: dict[str, tuple[str, list[str]]] = {
    "hr_a": ("a", ["group:hr"]),
    "public_a": ("a", ["tenant:*"]),
    "bob_a": ("a", ["user:bob"]),
    "hr_b": ("b", ["group:hr"]),
    "public_b": ("b", ["tenant:*"]),
}
MEMBERSHIPS = [("a", "alice", "hr"), ("b", "alice", "hr")]


@dataclass
class World:
    tenants: dict[str, UUID]
    documents: dict[str, UUID] = field(default_factory=dict)


def canary(document: str) -> str:
    """Content of every chunk of `document`: a unique string a leak would expose."""
    return f"CANARY-{document}"


async def insert_chunk(
    conn: asyncpg.Connection, tenant: UUID, document: UUID, text: str, ordinal: int
) -> UUID:
    chunk_id: UUID = await conn.fetchval(
        """
        INSERT INTO chunks (tenant_id, document_id, ordinal, content, embedding)
        VALUES ($1, $2, $3, $4, $5::text::vector)
        RETURNING id
        """,
        tenant,
        document,
        ordinal,
        text,
        EMBEDDING,
    )
    return chunk_id


@pytest.fixture
async def world() -> World:
    w = World(tenants={"a": uuid4(), "b": uuid4()})
    for key, tenant in w.tenants.items():
        async with session(INGEST, tenant) as conn:
            await conn.execute(
                "INSERT INTO tenants (id, name) VALUES ($1, $2)", tenant, f"Tenant {key}"
            )
            for t, sub, group in MEMBERSHIPS:
                if t == key:
                    await conn.execute(
                        "INSERT INTO memberships (tenant_id, user_sub, group_name)"
                        " VALUES ($1, $2, $3)",
                        tenant,
                        sub,
                        group,
                    )
            for name, (t, acl) in DOCUMENTS.items():
                if t != key:
                    continue
                doc: UUID = await conn.fetchval(
                    "INSERT INTO documents (tenant_id, title) VALUES ($1, $2) RETURNING id",
                    tenant,
                    name,
                )
                w.documents[name] = doc
                await conn.executemany(
                    "INSERT INTO document_acl (tenant_id, document_id, principal)"
                    " VALUES ($1, $2, $3)",
                    [(tenant, doc, p) for p in acl],
                )
                for ordinal in range(2):
                    await insert_chunk(conn, tenant, doc, canary(name), ordinal)
    return w
