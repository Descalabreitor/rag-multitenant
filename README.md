# rag-multitenant

A retrieval-augmented generation service for multiple organizations, where **no user can ever retrieve, or see cited, a document they aren't allowed to read**. Isolation lives in PostgreSQL Row-Level Security, so a bug in the application or a prompt injection in a document can't widen access.

The goal is to show, with numbers, what it takes to make permission-aware RAG trustworthy: a leak-test suite (cross-tenant queries, JWT tampering, connection-pool reuse, revocation, prompt injection, property-based tests), recall under restrictive filters, and the latency cost of doing it right.

> This is a work in progress, built in the open. The scaffold, the local development stack and CI are in place; the database schema and RLS policies are next. Measured results will land in [`docs/results/`](docs/results/).

## Design decisions

Decisions are recorded as ADRs in [`docs/adr/`](docs/adr/):

- [0001. Pre-filtering with PostgreSQL Row-Level Security](docs/adr/0001-pre-filtering-with-postgres-rls.md)
- [0002. Resolve group memberships in the database](docs/adr/0002-resolve-group-memberships-in-the-database.md)
- [0003. Denormalize document ACLs onto chunks](docs/adr/0003-denormalize-acls-onto-chunks.md)

## How access control works

- There are two levels of permissions. Tenants (one Keycloak Organization each) are fully isolated from each other, and inside a tenant each document has an ACL of users, groups or the whole tenant.
- PostgreSQL enforces access. Every tenant table has `FORCE ROW LEVEL SECURITY`, and the API connects with a role that cannot bypass it. The tenant and the user's principals are set per transaction (`SET LOCAL`), so pooled connections can't leak context. With no context set, queries return nothing.
- The JWT proves who you are, but what you can read is resolved from memberships synced from Keycloak, so revoking access doesn't wait for tokens to expire.
- The LLM only sees chunks you could already read. It has no tools and no database access, and retrieved text is treated as untrusted data.
- Every retrieval is written to an insert-only audit log.

## What will be measured

| Question | Measurement |
|---|---|
| Does anything leak? | Leaks found across the adversarial and property-based suite (target: 0) |
| Does filtering hurt quality? | Recall@k for small vs. large tenants, with and without pgvector iterative scans / per-tenant partitions |
| What does security cost? | p50/p95 latency with RLS on vs. off |
| How fast is revocation? | Seconds from removing a user from a group to losing access |

## Stack

Python 3.12 · FastAPI · PostgreSQL + pgvector · SQLAlchemy (async) + Alembic · Keycloak (OIDC) · Ollama (local LLM and embeddings, no API key needed) · pytest + Hypothesis + Testcontainers

## Getting started

Requirements: Docker with Compose, and conda (or any Python 3.12 environment).

```bash
cp .env.example .env                  # local placeholders; change the passwords if you like
docker compose up -d                  # PostgreSQL + pgvector, Keycloak, Ollama
                                      # (first run also downloads the Ollama models)

conda env create -f environment.yml
conda activate ragmt
pytest
```

All ports are bound to `127.0.0.1`. On first start, PostgreSQL creates two separate roles: `migrator`, which owns the schema, and `app_rw`, which the API uses at runtime and which cannot bypass RLS. Keycloak gets its own database with no access to the application's.

Tests marked `db` need PostgreSQL running. The connection URLs are read from `.env` (variables already set in the environment take precedence); if the URLs are missing, those tests are skipped, and if the database is down, they fail.

## Roadmap

- [x] Scaffold: package layout, tooling, environment
- [x] Local stack with Docker Compose and CI (lint, types, tests against a real PostgreSQL)
- [x] Database roles: schema owner separate from a runtime role that cannot bypass RLS
- [ ] Database schema and RLS policies, with SQL-level isolation tests
- [ ] Keycloak realm, JWT validation, per-request tenant context, permission sync
- [ ] Ingestion: documents → Markdown → chunks → embeddings, with ACLs
- [ ] Retrieval and answer generation with citations, plus audit log
- [ ] Revocation and deletion (GDPR erasure) with measured propagation time
- [ ] Leak-test suite and benchmarks, published in `docs/results/`
- [ ] *(optional)* Relationship-based permissions with OpenFGA

## License

[Apache 2.0](LICENSE)
