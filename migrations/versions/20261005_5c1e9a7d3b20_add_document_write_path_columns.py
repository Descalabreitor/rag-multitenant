"""add document write path columns

Revision ID: 5c1e9a7d3b20
Revises: aef343c73b03
Create Date: 2026-10-05 18:20:00.000000

Columns the upload, ACL and delete routes need (ADR 0008):

- source_hash: SHA-256 (lower-case hex) of the uploaded bytes. One live document
  per (tenant_id, source_hash), so uploading the same file twice is detected.
- created_by: `sub` of the uploader; NULL for rows that didn't come through the
  API (the seed, rows that existed before this migration).
- deleted_at: soft delete.

A soft-deleted document disappears for app_rw through three layers, none of
which adds a join to the chunks policy (ADR 0003 keeps it a per-row check):

1. Setting deleted_at deletes the document's ACL rows (trigger below). The
   existing statement triggers on document_acl then recompute its chunks to an
   empty acl_principals, which matches no principal.
2. The chunk ACL trigger yields an empty array for any chunk whose document is
   not live, so an ACL added to a deleted document never reaches its chunks.
3. The documents policy for app_rw also requires deleted_at IS NULL.

app_ingest still sees deleted documents and their chunks (whole tenant).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5c1e9a7d3b20"
down_revision: str | Sequence[str] | None = "aef343c73b03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Copied from aef343c73b03 rather than imported: a migration must keep producing
# the same SQL even if an earlier one is edited.
TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
USER_SUB = "NULLIF(current_setting('app.user_sub', true), '')"
PRINCIPALS = f"""(CASE WHEN {USER_SUB} IS NULL THEN NULL ELSE
    ARRAY['user:' || {USER_SUB}, 'tenant:*']
    || ARRAY(SELECT 'group:' || m.group_name FROM public.memberships m
             WHERE m.tenant_id = {TENANT} AND m.user_sub = {USER_SUB})
END)"""

DOCUMENTS_READER_BEFORE = f"""tenant_id = {TENANT} AND EXISTS (
        SELECT 1 FROM public.document_acl a
        WHERE a.document_id = documents.id AND a.principal = ANY({PRINCIPALS}))"""
DOCUMENTS_READER_AFTER = f"{DOCUMENTS_READER_BEFORE} AND deleted_at IS NULL"

CHUNK_ACL_BEFORE = """
    NEW.acl_principals := ARRAY(
        SELECT a.principal FROM public.document_acl a
        WHERE a.document_id = NEW.document_id
        ORDER BY a.principal
    );
    RETURN NEW;"""

# Fails closed: a document the writer can't see as live (deleted, or not there)
# gives an empty ACL, not the ACL rows.
CHUNK_ACL_AFTER = """
    IF EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = NEW.document_id AND d.deleted_at IS NULL
    ) THEN
        NEW.acl_principals := ARRAY(
            SELECT a.principal FROM public.document_acl a
            WHERE a.document_id = NEW.document_id
            ORDER BY a.principal
        );
    ELSE
        NEW.acl_principals := '{}';
    END IF;
    RETURN NEW;"""


def _chunk_acl_function(body: str) -> str:
    return f"""
        CREATE OR REPLACE FUNCTION chunks_set_acl_principals() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN{body}
        END
        $$
    """


def upgrade() -> None:
    # Backfill: every row of every tenant needs a value, but under FORCE RLS the
    # migrator can't read or UPDATE any row. A volatile default makes ADD COLUMN
    # rewrite the table and evaluate it once per row, which RLS does not filter.
    # 'legacy:<random uuid>' is unique (the unique index below can't fail on
    # documents that share a title) and can never equal a real SHA-256 hex. The
    # default is dropped right after, so new rows must supply the hash.
    op.add_column(
        "documents",
        sa.Column(
            "source_hash",
            sa.Text,
            nullable=False,
            server_default=sa.text("'legacy:' || gen_random_uuid()"),
        ),
    )
    op.alter_column("documents", "source_hash", server_default=None)
    op.create_check_constraint(
        "documents_source_hash_format",
        "documents",
        "source_hash ~ '^[0-9a-f]{64}$' OR source_hash LIKE 'legacy:%'",
    )
    op.add_column("documents", sa.Column("created_by", sa.Text))
    op.create_check_constraint("documents_created_by_not_empty", "documents", "created_by <> ''")
    op.add_column("documents", sa.Column("deleted_at", sa.DateTime(timezone=True)))
    op.create_index(
        "documents_tenant_id_source_hash_live_key",
        "documents",
        ["tenant_id", "source_hash"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # 3. The read policy hides deleted documents.
    op.execute(
        f"ALTER POLICY documents_app_rw_select ON documents USING ({DOCUMENTS_READER_AFTER})"
    )

    # 2. Chunks of a document that isn't live get an empty ACL.
    op.execute(_chunk_acl_function(CHUNK_ACL_AFTER))

    # 1. Deleting a document revokes its ACL. The DELETE fires the document_acl
    # statement triggers, which recompute the chunks. Restoring a document
    # (deleted_at back to NULL) touches its chunks too, but brings no ACL back:
    # it stays invisible until the writer sets one.
    op.execute("""
        CREATE FUNCTION documents_soft_delete() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.deleted_at IS NOT NULL THEN
                DELETE FROM public.document_acl WHERE document_id = NEW.id;
            END IF;
            UPDATE public.chunks SET acl_principals = acl_principals
            WHERE document_id = NEW.id;
            RETURN NULL;
        END
        $$
    """)
    op.execute("""
        CREATE TRIGGER documents_soft_delete
        AFTER UPDATE OF deleted_at ON documents
        FOR EACH ROW WHEN (OLD.deleted_at IS DISTINCT FROM NEW.deleted_at)
        EXECUTE FUNCTION documents_soft_delete()
    """)


def downgrade() -> None:
    # Deleted documents have no ACL rows (trigger 1), so once deleted_at is gone
    # they stay invisible to app_rw: the downgrade fails closed.
    op.execute("DROP TRIGGER documents_soft_delete ON documents")
    op.execute("DROP FUNCTION documents_soft_delete()")
    op.execute(_chunk_acl_function(CHUNK_ACL_BEFORE))
    op.execute(
        f"ALTER POLICY documents_app_rw_select ON documents USING ({DOCUMENTS_READER_BEFORE})"
    )
    op.drop_index("documents_tenant_id_source_hash_live_key", table_name="documents")
    op.drop_column("documents", "deleted_at")
    op.drop_constraint("documents_created_by_not_empty", "documents", type_="check")
    op.drop_column("documents", "created_by")
    op.drop_constraint("documents_source_hash_format", "documents", type_="check")
    op.drop_column("documents", "source_hash")
