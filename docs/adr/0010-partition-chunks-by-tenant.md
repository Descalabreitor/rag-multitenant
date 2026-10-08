# 0010. Partition chunks by tenant

**Status:** Proposed (2026-10-08)

## Context

ADR 0001 left overfiltering open: RLS filters the HNSW scan, so a user who can read little of the table may get fewer than k chunks, to be mitigated "with pgvector iterative index scans or per-tenant partitions". ADR 0009 picked iterative scans (`relaxed_order`). The benchmarks (`docs/results/eval.md`, ADR 0009 Results, 100,000 chunks) show that this isn't enough on one shared index:

- On the HNSW plan, the 0.1% tenant's users get recall@5 of 0.28 to 0.60 and a third of their queries come back short, at 114 ms p50, because the scan gives up at `HNSW_MAX_SCAN_TUPLES` before it finds enough of their rows.
- The planner can't see the user's principals, underestimates what they can read, and picks an exact sort for every tenant at this size. Recall is perfect, but the 50% tenant pays 99 ms p50 / 260 ms p95, and its broadest user 238 ms p50. Past some table size the planner switches to HNSW, and then the tiny tenants get the first problem.

The same benchmark ran the same policy on a copy of `chunks` LIST-partitioned by tenant, with one HNSW index per partition: recall ≥ 0.99 for every tenant and user, p50 3 to 4 ms and p95 under 13 ms whatever the tenant's size.

## Decision (proposed)

Partition `chunks` by `LIST (tenant_id)`, one partition per tenant and an HNSW index on each (declared on the parent). The RLS policy, the retriever's statement and `tenant_session` stay as they are: `tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid` is stable, so the executor prunes every other tenant's partition at startup (EXPLAIN shows 8 of 9 removed). No tenant filter is added to the query.

Keep `relaxed_order`: within a partition only the ACL filters, and the benchmark's no-groups users still lost recall@10 with iterative scans off (0.72 to 0.92).

## Alternatives considered

- **Per-tenant partial HNSW indexes** (`WHERE tenant_id = '<id>'`). The planner only uses a partial index when the query's WHERE implies the predicate at plan time, and the policy compares with `current_setting(...)`. The retriever would have to add `tenant_id = <literal>`, the filter ADR 0009 keeps out of the query, and plans would be per tenant.
- **Raise `HNSW_MAX_SCAN_TUPLES` and `ef_search`.** Helps the tiny tenants on the HNSW plan at the cost of their latency, and does nothing for the large tenant's exact plan.
- **Force the HNSW plan in the retriever** (`enable_sort = off` per transaction). Fixes the large tenant's latency, makes the tiny tenants' recall worse, and fights the planner.
- **HASH partitions by tenant.** A fixed number of partitions needs no DDL per tenant, but a tiny tenant shares its partition with others and keeps part of the overfiltering.

## Consequences

- **Pro:** Every tenant searches a graph that holds only its own chunks. Recall no longer depends on the tenant's share of the table, and latency barely depends on its size.
- **Pro:** Erasing a tenant becomes `DROP TABLE` of a partition.
- **Con:** A new tenant needs a partition, which is DDL by the table owner (migrator). app_ingest creates tenants today and must not own tables (ADR 0004), and `SECURITY DEFINER` functions are ruled out without an ADR. Onboarding needs its own step, or a DEFAULT partition for new and small tenants, which brings back shared-index overfiltering for them, on a smaller graph.
- **Con:** Primary and unique keys must include the partition key: `chunks` becomes `PRIMARY KEY (tenant_id, id)` and `UNIQUE (tenant_id, document_id, ordinal)`. The ACL triggers (ADR 0003) work on partitioned tables, but the migration rebuilds the table and its indexes.
- **Con:** Thousands of tenants mean thousands of partitions and HNSW indexes. Planning and maintenance cost at that count was not measured (the benchmark had 9 partitions); it has to be before this is accepted.
- **Note:** To move to Accepted, this needs a migration, a decision on onboarding (per-tenant DDL vs a DEFAULT partition for tenants under a size threshold), leak tests against the partitioned table (`tests/leaks/`), and `make eval` re-run against head instead of the scratch copy.
