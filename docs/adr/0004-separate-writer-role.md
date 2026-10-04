# 0004. A separate database role for writes

**Status:** Accepted (2026-10-04)

## Context

User requests must only read the chunks their principals allow (ADR 0001). The obvious way to get that is a SELECT policy on `chunks` that checks `acl_principals`. But PostgreSQL applies a table's SELECT policies whenever an UPDATE or DELETE has to read rows (a `WHERE` clause, `RETURNING`), so a role with that policy can only change rows it can read.

Several writers need to touch rows that no particular user can read:

- The trigger that copies `document_acl` into `chunks.acl_principals` (ADR 0003). If it runs as a user who cannot read some of the chunks, it silently skips them, and the denormalized ACL drifts from the source of truth.
- Ingest, which deletes and rewrites a document's chunks whatever their ACL.
- `permsync`, which writes every membership in a tenant (ADR 0002).

## Decision

Use two runtime login roles, each with its own policies:

- **`app_rw`** (`DATABASE_URL`) serves user requests. It needs `app.tenant_id` and `app.user_sub`. It can SELECT `chunks`, `documents` and `document_acl` only where the user's principals match, and only its own `memberships`. It can only INSERT into `audit_events`, and only rows whose tenant and actor match the context.
- **`app_ingest`** (`INGEST_DATABASE_URL`) runs ingest, ACL changes and `permsync`. Its policies only check `app.tenant_id`, so it sees and writes its whole tenant and nothing else. It can INSERT into `audit_events` too.

Both roles are `NOSUPERUSER NOBYPASSRLS NOINHERIT` and own no tables. `migrator` has no policies, so under `FORCE ROW LEVEL SECURITY` it sees no tenant rows.

## Alternatives considered

- **One role plus a mode flag.** A setting such as `app.mode = 'ingest'` lifts the ACL filter for writes. Fewer moving parts, but any bug that sets the flag on a user request widens access, which goes against failing closed.
- **One role, and every write runs as a user who can see the rows.** Ingest has to impersonate someone in each document's ACL, and changing an ACL you are not on leaves chunks out of sync.
- **A `SECURITY DEFINER` trigger function.** It only moves the problem: the owner (`migrator`) is also subject to forced RLS, and the project avoids definer functions.

## Consequences

- **Pro:** The read path has one job: what a user may see. The write path cannot leak across tenants, and it doesn't depend on who triggered it.
- **Pro:** The ACL trigger always sees every chunk of the tenant, so propagation is complete and atomic.
- **Con:** A second credential to manage, and a role that sees the whole tenant. Code that answers user queries must never use `INGEST_DATABASE_URL`; a leak test or a lint rule should eventually enforce that.
- **Con:** Adding the role changes `docker/postgres/initdb/01-roles.sh`, so existing local databases must be recreated once.
- **Note:** `app_rw` sees only the `document_acl` entries that match the user, not a document's full ACL. Showing "who else can read this" would need a separate, deliberate decision.
- **Note:** Unique and foreign-key checks ignore RLS. A writer that already knows another tenant's UUIDs can learn they exist from constraint errors. That needs a guessed UUIDv4 and the writer role, so it is accepted for now.
