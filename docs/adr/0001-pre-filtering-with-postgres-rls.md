# 0001. Pre-filtering with PostgreSQL Row-Level Security

**Status:** Accepted (2026-09-30)

## Context

A user must never retrieve a chunk they are not allowed to read, whether it belongs to another tenant or to a group they are not in. Permissions can be enforced in three places: in the application's queries, after retrieval, or inside the database.

## Decision

Enforce access with PostgreSQL Row-Level Security as a pre-filter. Every tenant table has `ENABLE` and `FORCE ROW LEVEL SECURITY`. The application connects as `app_rw`, a role that owns no tables and has neither `SUPERUSER` nor `BYPASSRLS`. The tenant and the user's principals are set per transaction with `set_config(..., true)` (equivalent to `SET LOCAL`), and policies read them with `current_setting(..., true)` so that a missing context returns zero rows.

## Alternatives considered

- **Filtering in the application** (`WHERE tenant_id = ...` in every query). Simple, but a single forgotten `WHERE` or a badly written new endpoint leaks data, and nothing prevents it.
- **Post-filtering.** Search everything, then drop what the user cannot see. Forbidden chunks still leave the database, and the top-k comes back short or empty when the user can see little of the corpus.
- **Pre-filtering with RLS.** The database applies the policy to every query, including a hurried `SELECT *`.

## Consequences

- **Pro:** Security does not depend on every line of application code being correct.
- **Con:** Filtering interacts with approximate vector search: in small tenants, HNSW may return fewer than k results (overfiltering). This must be measured and mitigated, with pgvector iterative index scans or per-tenant partitions.
- **Con:** Ties the design to PostgreSQL. Moving to Qdrant or another vector store would require rethinking isolation.
- **Con:** RLS has known pitfalls: plain `SET` leaking across pooled connections, and table owners bypassing policies. The phase 1 database tests must cover them.
