"""create core tables

Revision ID: 1ab7240d5bbc
Revises: <base>
Create Date: 2026-10-04 23:41:02.118734

Tables only. Every table gets ENABLE + FORCE ROW LEVEL SECURITY here, but no
policies and no grants to app_rw, so until the next migrations add them the
runtime role cannot read or write anything (fail closed). FORCE also applies to
the owner (migrator), so data migrations must set the tenant context too.

Child tables carry tenant_id and reference their parent through a composite
(tenant_id, id) foreign key, so a chunk or ACL entry can never point at a
document of another tenant.
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "1ab7240d5bbc"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Fixed at the time of this migration (nomic-embed-text). Deliberately not read
# from EMBEDDING_DIM: a migration must produce the same schema whenever it runs.
# Changing the model means a new migration and re-embedding everything.
EMBEDDING_DIM = 768

TABLES = ("tenants", "memberships", "documents", "document_acl", "chunks", "audit_events")


def _created_at() -> sa.Column[datetime]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def upgrade() -> None:
    # The primary key is the Keycloak Organization id, so the tenant from the JWT
    # maps to a row without a lookup (which would itself need to bypass RLS).
    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        _created_at(),
        sa.CheckConstraint("name <> ''", name="tenants_name_not_empty"),
    )

    # Group memberships synced from Keycloak by permsync (ADR 0002). Being in the
    # tenant at all comes from the JWT; this table only holds groups.
    op.create_table(
        "memberships",
        sa.Column("tenant_id", sa.Uuid, sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_sub", sa.Text, nullable=False),
        sa.Column("group_name", sa.Text, nullable=False),
        sa.Column(
            "synced_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("tenant_id", "user_sub", "group_name", name="memberships_pkey"),
        sa.CheckConstraint("user_sub <> ''", name="memberships_user_sub_not_empty"),
        sa.CheckConstraint("group_name <> ''", name="memberships_group_name_not_empty"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.Uuid, sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("source_uri", sa.Text),
        _created_at(),
        # Target of the composite foreign keys below.
        sa.UniqueConstraint("tenant_id", "id", name="documents_tenant_id_id_key"),
    )

    # Source of truth for document permissions (ADR 0003).
    op.create_table(
        "document_acl",
        sa.Column("tenant_id", sa.Uuid, nullable=False),
        sa.Column("document_id", sa.Uuid, nullable=False),
        sa.Column("principal", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("document_id", "principal", name="document_acl_pkey"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.id"],
            name="document_acl_document_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            r"principal ~ '^(user:.+|group:.+|tenant:\*)$'",
            name="document_acl_principal_format",
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.Uuid, nullable=False),
        sa.Column("document_id", sa.Uuid, nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("heading", sa.Text),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        # Copy of the document's ACL, maintained by a trigger (ADR 0003).
        sa.Column(
            "acl_principals",
            postgresql.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        _created_at(),
        sa.UniqueConstraint("document_id", "ordinal", name="chunks_document_id_ordinal_key"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.id"],
            name="chunks_document_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("ordinal >= 0", name="chunks_ordinal_non_negative"),
    )
    op.create_index("chunks_tenant_id_idx", "chunks", ["tenant_id"])
    op.create_index(
        "chunks_acl_principals_idx", "chunks", ["acl_principals"], postgresql_using="gin"
    )
    # nomic-embed-text vectors are compared by cosine distance (`<=>`).
    op.create_index(
        "chunks_embedding_hnsw_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # Insert-only for app_rw (UPDATE/DELETE revoked with the grants). chunk_ids has
    # no foreign key on purpose: the audit trail outlives deleted chunks.
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("tenant_id", sa.Uuid, sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("actor_sub", sa.Text, nullable=False),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column(
            "chunk_ids",
            postgresql.ARRAY(sa.Uuid),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "details",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.create_index(
        "audit_events_tenant_id_occurred_at_idx", "audit_events", ["tenant_id", "occurred_at"]
    )

    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_table(table)
