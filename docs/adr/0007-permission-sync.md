# 0007. Permission sync: what it reads, and when it may revoke

**Status:** Accepted (2026-10-05)

## Context

`permsync` copies Keycloak group memberships into the `memberships` table (ADR 0002). The realm models each tenant as a top-level group named after the Organization alias, with an `organization_id` attribute, and its groups as children: `/acme/finance` (ADR 0005). The service account has only `view-users` and `query-groups`.

Two kinds of failure matter. A sync that reads Keycloak wrong can **grant** access, for example to someone in `/acme/finance` who is not an Acme member. It can also **revoke** access by accident: if a failed or half-finished listing were read as "this group has no members", one network error would delete every membership in a tenant.

The endpoints were checked against Keycloak 26.0 with exactly those two roles. `/groups`, `/groups/{id}/children` and `/groups/{id}/members` answer, and so does `/organizations/members/{user}/organizations`, which returns the user's Organizations with id, name and `enabled`. `/organizations` and `/organizations/{id}/members` return 403.

## Decision

- **Read the group tree, then check each member against their Organizations.** For each top-level group with an `organization_id`, list its children and their members. Each member counts only if the user is enabled, is a member of that Organization, and the Organization is enabled. Anyone else is skipped and logged by `sub`. The Organization name from that lookup becomes the tenant name. No role beyond ADR 0005's is needed.
- **A tenant is written only from a complete read.** All pages of every listing must load. The number of children must equal the parent's `subGroupCount`. Every response must parse. Any error (transport failure, non-200 status, unexpected body) skips that tenant for the cycle, and its rows stay as they were.
- **A tenant is purged only when a complete listing of the top-level groups no longer contains it.** If that listing fails, the cycle changes nothing anywhere. A realm misconfiguration (two groups claiming one tenant, a malformed id) leaves the tenant out, so its memberships are revoked rather than guessed.
- **One transaction per tenant**, through `tenant_session(ingest_engine, tenant_id)` as `app_ingest`. It upserts the tenant row, which also locks it, so concurrent syncs of one tenant run one after the other. Then it deletes and inserts only the difference, refreshes `synced_at`, and writes one `audit_events` row (`permsync`, with counts and the changed `(sub, group)` pairs).
- **Only direct members of `/<alias>/<group>` count.** Deeper subgroups are logged and ignored, because a `memberships` row holds one group name.
- **Each change is logged after its commit**, with a UTC millisecond timestamp, so the revocation window can be measured from the logs.

## Alternatives considered

- **Organization endpoints** (`/organizations`, `/organizations/{id}/members`). They give the tenant list directly, but return 403 to read-only roles in 26.0 (ADR 0005). The fix would be `manage-realm`, which would let a leaked permsync secret reconfigure the realm.
- **Walk the users** (`/users`, then `/users/{id}/groups` for each). That is one request per user in the realm, against one per group plus one per grouped user, and it still needs the group tree to map paths to tenants.
- **Treat an empty or failed listing as "no members"**. Simpler, but one network error would revoke a whole tenant.

## Consequences

- **Pro:** A grant needs the group, the Organization and an enabled user to agree. A Keycloak error never removes access.
- **Pro:** Running a sync twice changes no memberships. Only `synced_at` moves, which is the freshness signal ADR 0002 asks to monitor.
- **Con:** If Keycloak is down, revocations wait. The window is the sync interval plus the outage, and the logs show it (`cycle aborted ... nothing changed`).
- **Con:** `app_ingest` can't list tenants (its policies need a tenant id first), so permsync only notices a deleted Organization if it saw it earlier in the same process. If the Organization is deleted while permsync is stopped, its rows stay. They are inert: no valid token names that Organization any more, and the API takes the tenant only from the token. A later cleanup job would need a deliberate way to list tenants.
- **Con:** One audit row per tenant per cycle, even when nothing changed. At the default 60 s that is 1440 rows a day per tenant, which is fine at this scale. Batching them can come later if needed.
- **Note:** One request per grouped user per cycle (cached across that user's groups). Large realms would want a longer interval, or Keycloak's admin events to sync only what changed.
