# rag-multitenant

A retrieval-augmented generation service for multiple organizations, where **no user can ever retrieve, or see cited, a document they aren't allowed to read**. Isolation lives in PostgreSQL Row-Level Security, so a bug in the application or a prompt injection in a document can't widen access.

The goal is to show, with numbers, what it takes to make permission-aware RAG trustworthy: a leak-test suite (cross-tenant queries, JWT tampering, connection-pool reuse, revocation, prompt injection, property-based tests), recall under restrictive filters, and the latency cost of doing it right.

> This is a work in progress, built in the open. The database with its RLS policies, identity (Keycloak realm, JWT validation, permission sync), ingestion (upload `.md`, `.html` or `.docx`, chunked and embedded with Ollama), the documents API, and question answering with citations and an audit trail (`POST /ask`, `GET /audit`, the `ragctl` client) are in place, checked end to end against a real Keycloak and Ollama. Measured results will land in [`docs/results/`](docs/results/).

## Design decisions

Decisions are recorded as ADRs in [`docs/adr/`](docs/adr/):

- [0001. Pre-filtering with PostgreSQL Row-Level Security](docs/adr/0001-pre-filtering-with-postgres-rls.md)
- [0002. Resolve group memberships in the database](docs/adr/0002-resolve-group-memberships-in-the-database.md)
- [0003. Denormalize document ACLs onto chunks](docs/adr/0003-denormalize-acls-onto-chunks.md)
- [0004. A separate database role for writes](docs/adr/0004-separate-writer-role.md)
- [0005. Keycloak realm layout: organizations, groups and clients](docs/adr/0005-keycloak-realm-layout.md)
- [0006. Token validation, the tenant claim and the 401/403 split](docs/adr/0006-token-validation-and-tenant-claim.md)
- [0007. Permission sync: what it reads, and when it may revoke](docs/adr/0007-permission-sync.md)
- [0008. The write path: uploads, ACL changes and soft delete](docs/adr/0008-write-path.md)
- [0009. Retrieval and generation: ports, prompt, citations and audit](docs/adr/0009-retrieval-and-generation.md)

## How access control works

- There are two levels of permissions. Tenants (one Keycloak Organization each) are fully isolated from each other, and inside a tenant each document has an ACL of users, groups or the whole tenant.
- PostgreSQL enforces access. Every tenant table has `FORCE ROW LEVEL SECURITY`, and the API connects with roles that cannot bypass it: one for user requests, which only reads what the user's ACLs allow, and one for ingest and permission sync, which is confined to a single tenant. The tenant and the user are set per transaction (`SET LOCAL`), so pooled connections can't leak context, and the user's groups are resolved inside the RLS policies, so the application can't grant itself access. With no context set, queries return nothing.
- The JWT proves who you are, but what you can read is resolved from memberships synced from Keycloak, so revoking access doesn't wait for tokens to expire.
- The LLM only sees chunks you could already read. It has no tools and no database access, and retrieved text is treated as untrusted data.
- Every retrieval is written to an insert-only audit log.

## API

Every route but `/healthz` needs a bearer token from Keycloak. The tenant comes from the token only; a `tenant_id` anywhere in the request is ignored.

| Route | Who | What |
|---|---|---|
| `GET /documents` | any user | The documents the caller's ACLs allow: `[{id, title}]` |
| `GET /documents/{id}` | any user | One of them; 404 for anything else, including other tenants' ids |
| `POST /documents` | tenant admins | Multipart upload: `file` (`.md`, `.html` or `.docx`, up to `INGEST_MAX_BYTES`) and an optional `acl`, a JSON list such as `["group:hr"]`. Without one, only the uploader can read it. Returns 201 `{id, title, chunks, unchanged}`; uploading the same bytes again returns the stored document with `unchanged: true` |
| `PUT /documents/{id}/acl` | tenant admins | Replaces the ACL with `{"principals": [...]}`; the chunks follow in the same transaction |
| `DELETE /documents/{id}` | tenant admins | Soft delete: the document disappears for every reader. `?purge=true` removes it and its chunks |
| `POST /ask` | any user | `{"question": "..."}` → `{answer, citations: [{document_id, title, heading}], model}`, answered only from chunks the caller may read |
| `GET /audit` | tenant admins | The tenant's audit trail, newest first: who asked, when, the ids and scores of the chunks retrieved, the model. `?limit=` (up to 200), then `?before=<next_before>` for the next page |

Admins are the members of `/<organization>/admins` in Keycloak. Being an admin grants no read access: an admin reads only what the ACLs allow, like anyone else. A non-admin gets 403 on an upload, and the same 404 as a missing document on the routes with an id, so nobody learns that an id exists. Embeddings are never returned. Swagger is at `/docs`. See [ADR 0008](docs/adr/0008-write-path.md).

`POST /ask` embeds the question, searches with pgvector under RLS (so only chunks the caller may read can come back), records the retrieval in `audit_events`, and only then calls the chat model, which has no tools and sees the chunks as delimited, escaped reference data. Citations are built from the retrieved chunks, never taken on trust from the model's text: a marker that names no retrieved chunk is removed. When nothing readable matches, the answer is a fixed "I don't know" with no citations and `model: null`, and the model isn't called, whether nothing matched or nothing was visible. The audit row keeps a SHA-256 of the question, not its text (unless `AUDIT_STORE_QUERY_TEXT=true`). A non-admin gets 404 from `GET /audit`, as if the route didn't exist. See [ADR 0009](docs/adr/0009-retrieval-and-generation.md).

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
alembic upgrade head                  # create the schema (runs as the migrator role)
pytest
```

All ports are bound to `127.0.0.1`. On first start, PostgreSQL creates three separate roles: `migrator`, which owns the schema and is used only by Alembic, `app_rw`, which serves user requests, and `app_ingest`, which handles ingest and permission sync. Neither runtime role owns tables or can bypass RLS. Keycloak gets its own database with no access to the application's.

Tests marked `db` need PostgreSQL running. Tests marked `e2e` need Keycloak too, and are skipped when it isn't reachable; they also need Ollama with the embedding and chat models, because the API they call embeds uploads and questions and answers them for real. `make e2e` starts what they need, pulls both models, migrates, starts the API and runs them. The chat model (`llama3.1:8b`) is about 4.9 GB, a few minutes to download the first time, and on a CPU each answer takes from seconds to a couple of minutes; the whole e2e run takes about 6 minutes on a laptop. `make e2e FAKE_CHAT=1` skips the chat model and answers with a test double that echoes its prompt, with the same access and citation checks (about 3 minutes). The connection URLs are read from `.env` (variables already set in the environment take precedence); if the URLs are missing, those tests are skipped, and if the database is down, they fail.

## Try it

Two users of two different organizations call the same endpoint with real Keycloak tokens and get different documents. This assumes the stack from *Getting started* is up and migrated, with the placeholder passwords from `.env.example`.

```bash
python -m seed                        # fictional tenants, users' groups and documents (make seed)
python -m ragmt.permsync --once       # copy group memberships from Keycloak (make permsync)
uvicorn --factory ragmt.api.app:create_app &   # the API on http://127.0.0.1:8000

# A token from the dev-only password client (local development only, see ADR 0005).
token() {
  curl -s http://localhost:8080/realms/ragmt/protocol/openid-connect/token \
    -d grant_type=password -d client_id=ragmt-dev-password \
    -d username="$1" -d password=change-me-demo |
  python -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
}
ALICE=$(token alice)   # Acme Logistics, group finance
DAVE=$(token dave)     # Umbra Biotech, group finance

curl -s http://127.0.0.1:8000/documents -H "Authorization: Bearer $ALICE"
# [{"id":"e72cb086-…","title":"Employee handbook"},
#  {"id":"13f86cc8-…","title":"Performance review: Alice"},
#  {"id":"0e982739-…","title":"Q3 budget"}]

curl -s http://127.0.0.1:8000/documents -H "Authorization: Bearer $DAVE"
# [{"id":"193fedfc-…","title":"Annual budget"},
#  {"id":"34ba74ce-…","title":"Lab safety policy"}]

# Alice's budget, asked for by dave: the same answer as for a document that doesn't exist.
curl -s http://127.0.0.1:8000/documents/0e982739-b267-40bd-b180-2c57485c2521 \
  -H "Authorization: Bearer $DAVE"
# {"detail":"Document not found"}
```

Both are in a group called `finance`, but each sees only their own organization's budget. Remove alice from `/acme/finance` in the Keycloak admin console (http://localhost:8080, the `ragmt` realm), run `python -m ragmt.permsync --once`, and the same token no longer lists the Q3 budget.

Now add a document. Bob is Acme's admin, so he can upload, here for Acme's finance group only:

```bash
BOB=$(token bob)       # Acme Logistics, groups engineering and admins
ERIN=$(token erin)     # Acme Logistics, no groups

printf '# Depot night shift\n\nThe north depot moves its night shift in November.\n' > depot.md
curl -s http://127.0.0.1:8000/documents -H "Authorization: Bearer $BOB" \
  -F file=@depot.md -F 'acl=["group:finance"]'
# {"id":"5b0e…","title":"Depot night shift","chunks":1,"unchanged":false}

# Who gets it? Alice (Acme finance) does. Bob, erin and dave don't: bob is an admin
# but not in finance, and dave's finance group belongs to the other organization.
for t in "$ALICE" "$BOB" "$ERIN" "$DAVE"; do
  curl -s http://127.0.0.1:8000/documents -H "Authorization: Bearer $t" | grep -c 'Depot night shift'
done
# 1, 0, 0, 0

# Alice isn't an admin: her upload is refused.
curl -s http://127.0.0.1:8000/documents -H "Authorization: Bearer $ALICE" -F file=@depot.md
# {"detail":"Forbidden"}
```

`PUT /documents/{id}/acl` moves access for tokens already issued, and `DELETE /documents/{id}` hides the document from everyone (`?purge=true` removes it). The API embeds with `LLM_PROVIDER` from `.env`: `fake` in `.env.example`, so this works without Ollama; set it to `ollama` for real embeddings. `python -m seed` resets the seed tenants, uploads included.

### Ask questions

Answers need Ollama with both models, which `docker compose up -d` pulls (the chat model, `llama3.1:8b`, is about 4.9 GB). Set `LLM_PROVIDER=ollama` in `.env` *before* the first `python -m seed`, so the seed is embedded by the real model. Unchanged files are never re-embedded, so a corpus first seeded with `fake` keeps its meaningless vectors until `docker compose down -v`. With `fake`, `/ask` still works, but the answers come from a test double that echoes its prompt.

`ragctl` is the command-line client (installed by `pip install -e .`). `ragctl login` signs in through the browser; for the seed users, `--dev-user` gets a token from the same dev-only client as above and stores nothing:

```bash
uvicorn --factory ragmt.api.app:create_app &   # restarted, now with LLM_PROVIDER=ollama
export KC_DEMO_USER_PASSWORD=change-me-demo    # what --dev-user logs in with

ragctl --dev-user alice ask "Summarise the budget."
# The fleet budget for Q3 is **1.2 million credits**, of which 300,000 go to
# replacing the oldest delivery drones [doc:106d18d0-…#1].
#
# Sources:
#   [1] Q3 budget > Q3 budget > Fleet

ragctl --dev-user dave ask "How is the budget spent?"
# According to [doc:1f678851-…#1], the budget is spent as follows:
# * Research: 60%
# * Operations: 40%
#
# Sources:
#   [1] Annual budget > Annual budget > Research spend
```

The model's wording changes from run to run (and an 8B model sometimes says it doesn't know), but the sources can only be documents the user may read: alice's budget never reaches dave's prompt, and the other way round. Erin, in no group, can only be answered from the handbook. Bob, Acme's admin, sees every question asked in Acme. He sees the ids and scores of the chunks retrieved, not their text or the question (only its SHA-256). Alice gets the same 404 as for a route that doesn't exist:

```bash
curl -s 'http://127.0.0.1:8000/audit?limit=1' -H "Authorization: Bearer $BOB"
# {"events":[{"id":770,"occurred_at":"2026-10-06T21:51:58.939559Z",
#   "actor_sub":"4cb054f8-…","action":"ask","chunk_ids":["66de0724-…", …],
#   "details":{"model":"llama3.1:8b","scores":[0.0705, …],"question_sha256":"4430001d…"}}],
#  "next_before":770}

curl -s http://127.0.0.1:8000/audit -H "Authorization: Bearer $ALICE"
# {"detail":"Not Found"}
```

## Roadmap

- [x] Scaffold: package layout, tooling, environment
- [x] Local stack with Docker Compose and CI (lint, types, tests against a real PostgreSQL)
- [x] Database roles: schema owner separate from a runtime role that cannot bypass RLS
- [x] Database schema and RLS policies, with SQL-level isolation tests
- [x] Keycloak realm, JWT validation, per-request tenant context, permission sync
- [x] Ingestion: documents → Markdown → chunks → embeddings, with ACLs, through admin-only write routes
- [x] Retrieval and answer generation with citations, plus audit log
- [ ] Revocation and deletion (GDPR erasure) with measured propagation time
- [ ] Leak-test suite and benchmarks, published in `docs/results/`
- [ ] *(optional)* Relationship-based permissions with OpenFGA

## License

[Apache 2.0](LICENSE)
