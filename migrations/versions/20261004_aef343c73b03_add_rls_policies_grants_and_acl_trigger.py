"""add rls policies grants and acl trigger

Revision ID: aef343c73b03
Revises: 1ab7240d5bbc
Create Date: 2026-10-04 23:58:40.512301

The security layer on top of the tables from 1ab7240d5bbc, which already have
ENABLE + FORCE ROW LEVEL SECURITY and their indexes.

Two runtime roles (ADR 0004), both with policies keyed on the per-transaction
setting `app.tenant_id`:

- app_rw (user requests): reads only what the user's principals allow, and can
  only INSERT into audit_events. Also needs `app.user_sub`.
- app_ingest (ingest, ACL changes, permsync): reads and writes its whole tenant.

Every setting is read with current_setting(..., true) and NULLIF(..., ''): a
setting that was never set is NULL, and one that a previous SET LOCAL left
behind on a pooled connection is '', so both give zero rows instead of an error
or a match. migrator gets no policies, so under FORCE it sees no rows at all.

User principals are computed inside the policies from `app.user_sub` and the
memberships table (ADR 0002), so the application cannot grant itself groups.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "aef343c73b03"
down_revision: str | Sequence[str] | None = "1ab7240d5bbc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
USER_SUB = "NULLIF(current_setting('app.user_sub', true), '')"

# The current user's principals, or NULL when no user is set. NULL makes both
# `&&` and `= ANY` false, so a missing user can't fall back to `tenant:*`.
PRINCIPALS = f"""(CASE WHEN {USER_SUB} IS NULL THEN NULL ELSE
    ARRAY['user:' || {USER_SUB}, 'tenant:*']
    || ARRAY(SELECT 'group:' || m.group_name FROM public.memberships m
             WHERE m.tenant_id = {TENANT} AND m.user_sub = {USER_SUB})
END)"""

# table -> USING expression for app_rw SELECT.
READER_SELECT = {
    "tenants": f"id = {TENANT}",
    # Only the user's own memberships; the PRINCIPALS subquery reads through this.
    "memberships": f"tenant_id = {TENANT} AND user_sub = {USER_SUB}",
    "documents": f"""tenant_id = {TENANT} AND EXISTS (
        SELECT 1 FROM public.document_acl a
        WHERE a.document_id = documents.id AND a.principal = ANY({PRINCIPALS}))""",
    # Only the entries that match the user, not the document's full ACL.
    "document_acl": f"tenant_id = {TENANT} AND principal = ANY({PRINCIPALS})",
    # The per-row check from ADR 0003: no joins during the vector scan.
    "chunks": f"tenant_id = {TENANT} AND acl_principals && {PRINCIPALS}",
}

# table -> privileges for app_ingest (its policies are tenant-only).
WRITER_GRANTS = {
    "tenants": "SELECT, INSERT, UPDATE",
    "memberships": "SELECT, INSERT, UPDATE, DELETE",
    "documents": "SELECT, INSERT, UPDATE, DELETE",
    "document_acl": "SELECT, INSERT, UPDATE, DELETE",
    "chunks": "SELECT, INSERT, UPDATE, DELETE",
}


def _tenant_column(table: str) -> str:
    return "id" if table == "tenants" else "tenant_id"


def upgrade() -> None:
    # --- app_rw: ACL-filtered reads ------------------------------------------
    for table, using in READER_SELECT.items():
        op.execute(f"GRANT SELECT ON {table} TO app_rw")
        op.execute(
            f"CREATE POLICY {table}_app_rw_select ON {table} FOR SELECT TO app_rw USING ({using})"
        )

    # --- app_ingest: whole tenant --------------------------------------------
    for table, privileges in WRITER_GRANTS.items():
        tenant_match = f"{_tenant_column(table)} = {TENANT}"
        op.execute(f"GRANT {privileges} ON {table} TO app_ingest")
        op.execute(
            f"CREATE POLICY {table}_app_ingest_all ON {table} "
            f"FOR ALL TO app_ingest USING ({tenant_match}) WITH CHECK ({tenant_match})"
        )

    # --- audit_events: insert-only for both ----------------------------------
    op.execute("REVOKE ALL ON audit_events FROM app_rw, app_ingest")
    op.execute("GRANT INSERT ON audit_events TO app_rw, app_ingest")
    op.execute(
        "CREATE POLICY audit_events_app_rw_insert ON audit_events FOR INSERT TO app_rw "
        f"WITH CHECK (tenant_id = {TENANT} AND actor_sub = {USER_SUB})"
    )
    op.execute(
        "CREATE POLICY audit_events_app_ingest_insert ON audit_events FOR INSERT TO app_ingest "
        f"WITH CHECK (tenant_id = {TENANT})"
    )

    # --- document_acl -> chunks.acl_principals (ADR 0003) --------------------
    # Every write to a chunk recomputes its ACL from document_acl, so a caller
    # cannot store a different one, whatever it puts in the INSERT or UPDATE.
    op.execute("""
        CREATE FUNCTION chunks_set_acl_principals() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.acl_principals := ARRAY(
                SELECT a.principal FROM public.document_acl a
                WHERE a.document_id = NEW.document_id
                ORDER BY a.principal
            );
            RETURN NEW;
        END
        $$
    """)
    op.execute("""
        CREATE TRIGGER chunks_set_acl_principals
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION chunks_set_acl_principals()
    """)

    # Any ACL change touches the affected documents' chunks once per statement,
    # in the same transaction; the row trigger above does the recomputing.
    op.execute("""
        CREATE FUNCTION document_acl_sync_chunks() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            affected uuid[];
        BEGIN
            IF TG_OP = 'INSERT' THEN
                affected := ARRAY(SELECT DISTINCT document_id FROM new_rows);
            ELSIF TG_OP = 'DELETE' THEN
                affected := ARRAY(SELECT DISTINCT document_id FROM old_rows);
            ELSE
                affected := ARRAY(
                    SELECT document_id FROM new_rows
                    UNION SELECT document_id FROM old_rows
                );
            END IF;
            UPDATE public.chunks SET acl_principals = acl_principals
            WHERE document_id = ANY(affected);
            RETURN NULL;
        END
        $$
    """)
    # Transition tables need one trigger per event.
    op.execute("""
        CREATE TRIGGER document_acl_sync_chunks_insert
        AFTER INSERT ON document_acl REFERENCING NEW TABLE AS new_rows
        FOR EACH STATEMENT EXECUTE FUNCTION document_acl_sync_chunks()
    """)
    op.execute("""
        CREATE TRIGGER document_acl_sync_chunks_update
        AFTER UPDATE ON document_acl REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
        FOR EACH STATEMENT EXECUTE FUNCTION document_acl_sync_chunks()
    """)
    op.execute("""
        CREATE TRIGGER document_acl_sync_chunks_delete
        AFTER DELETE ON document_acl REFERENCING OLD TABLE AS old_rows
        FOR EACH STATEMENT EXECUTE FUNCTION document_acl_sync_chunks()
    """)


def downgrade() -> None:
    for event in ("delete", "update", "insert"):
        op.execute(f"DROP TRIGGER document_acl_sync_chunks_{event} ON document_acl")
    op.execute("DROP FUNCTION document_acl_sync_chunks()")
    op.execute("DROP TRIGGER chunks_set_acl_principals ON chunks")
    op.execute("DROP FUNCTION chunks_set_acl_principals()")

    op.execute("DROP POLICY audit_events_app_ingest_insert ON audit_events")
    op.execute("DROP POLICY audit_events_app_rw_insert ON audit_events")
    op.execute("REVOKE ALL ON audit_events FROM app_rw, app_ingest")

    for table in WRITER_GRANTS:
        op.execute(f"DROP POLICY {table}_app_ingest_all ON {table}")
        op.execute(f"REVOKE ALL ON {table} FROM app_ingest")
    for table in READER_SELECT:
        op.execute(f"DROP POLICY {table}_app_rw_select ON {table}")
        op.execute(f"REVOKE ALL ON {table} FROM app_rw")
