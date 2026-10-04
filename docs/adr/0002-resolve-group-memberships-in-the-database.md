# 0002. Resolve group memberships in the database

**Status:** Accepted (2026-09-30)

## Context

Keycloak can put the user's groups in the JWT, but a token is a snapshot of the moment it was issued. If Ana is removed from the `hr` group at 10:00, a token issued at 9:58 still says `hr` until it expires. Revocation latency is a security property of this system and must be controllable and measurable.

## Decision

The token provides identity only: `sub` and organization (tenant). Group memberships are read on every request from a `memberships` table in PostgreSQL, which the `permsync` module keeps in sync with Keycloak. The `groups` claim in the token is ignored for authorization.

## Alternatives considered

- **A. Trust the `groups` claim with a short token lifetime** (for example, 5 minutes). Simple, with no extra queries, but the revocation window equals the token lifetime, and lowering it a lot forces constant token refreshes.
- **B. Query Keycloak on every request.** Always fresh, but adds network latency and makes Keycloak a single point of failure for every question.
- **C. Identity from the token, groups from a synced `memberships` table** (chosen).

## Consequences

- **Pro:** The database is the single source of truth for permissions, and RLS policies read it directly.
- **Pro:** Revocation latency depends on the sync interval, which is under our control and can be measured. It is one of the figures reported in the README.
- **Con:** Adds a component (`permsync`) and one extra query per request. The query is cheap: a small, indexed table.
- **Con:** If `permsync` stops, permissions freeze in their last state. The time since the last successful sync must be monitored.
- **Note:** The benchmarks should include a measured comparison of options A and C, so the decision is backed by measurements.
