"""let tenant admins read audit_events

Revision ID: b7d41c2e9f05
Revises: 5c1e9a7d3b20
Create Date: 2026-10-06 12:00:00.000000

`GET /audit` (ADR 0009) lists a tenant's audit trail to its admins. app_rw gets
SELECT on audit_events, with one policy: the row is in the session's tenant AND
the session's user is in that tenant's `admins` group (ADR 0008), read from
memberships. Everyone else, including a user with no `app.user_sub`, sees zero
rows. UPDATE, DELETE and TRUNCATE stay revoked, and app_ingest still can't read
the table. An index on (tenant_id, id) serves the route's keyset pagination.

No policy cycle: this policy reads memberships, whose app_rw policy reads only
the session settings, and nothing reads audit_events.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b7d41c2e9f05"
down_revision: str | Sequence[str] | None = "5c1e9a7d3b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Copied from aef343c73b03 rather than imported: a migration must keep producing
# the same SQL even if an earlier one is edited.
TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
USER_SUB = "NULLIF(current_setting('app.user_sub', true), '')"
# ragmt.domain.TENANT_ADMIN_GROUP, written out for the same reason.
ADMIN_GROUP = "admins"

IS_TENANT_ADMIN = f"""EXISTS (
        SELECT 1 FROM public.memberships m
        WHERE m.tenant_id = {TENANT} AND m.user_sub = {USER_SUB}
          AND m.group_name = '{ADMIN_GROUP}')"""


def upgrade() -> None:
    op.execute("GRANT SELECT ON audit_events TO app_rw")
    op.execute(
        "CREATE POLICY audit_events_app_rw_select_admins ON audit_events FOR SELECT TO app_rw "
        f"USING (tenant_id = {TENANT} AND {IS_TENANT_ADMIN})"
    )
    # GET /audit pages newest first by id (keyset), within one tenant.
    op.create_index("audit_events_tenant_id_id_idx", "audit_events", ["tenant_id", "id"])


def downgrade() -> None:
    op.drop_index("audit_events_tenant_id_id_idx", table_name="audit_events")
    op.execute("DROP POLICY audit_events_app_rw_select_admins ON audit_events")
    op.execute("REVOKE SELECT ON audit_events FROM app_rw")
