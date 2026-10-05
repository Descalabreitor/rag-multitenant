# 0008. The write path: uploads, ACL changes and soft delete

**Status:** Accepted (2026-10-05)

## Context

Phase 3 adds writes through the API: upload a document, change its ACL, delete it. Until now only `permsync` and the seed wrote tenant data, both as `app_ingest` from their own processes (ADR 0004). The API has had a single engine on `app_rw`, which can't write anything. Writes bring new questions: who may make them, which role runs them, what a document's ACL is when the uploader doesn't give one, how a deleted document stops being readable, and where uploaded files become Markdown.

## Decision

**Converter.** `ragmt.ingest` has its own converter behind a `DocumentConverter` port (`ragmt.domain.ports`): `.md` is stored as is, `.html` goes to Markdown with markdownify, and `.docx` goes to HTML with mammoth and then to Markdown. The converter project's `engine/` package is not used. A dependency on another repository's internals, for three formats, was more coupling than it saved. Embeddings sit behind an `EmbeddingProvider` port with separate `embed_documents` and `embed_query` calls, because nomic-embed-text expects a different task prefix for each.

**Who may write.** Tenant admins are the members of the Keycloak group `/<alias>/admins`, which `permsync` stores as group `admins` (`ragmt.domain.TENANT_ADMIN_GROUP`). Only admins upload, change ACLs and delete. Being an admin grants no read access. An admin reads what the document ACLs allow, like anyone else. In the seed, bob (Acme) and carol (Umbra) are admins and neither is in `finance`.

**Which role writes.** API writes run as `app_ingest`, never as `app_rw`:

- One module, `ragmt.tenancy.writer`, owns the API's `app_ingest` engine. No other module under `src/` reads `INGEST_DATABASE_URL`, except `ragmt.permsync.__main__` (its own process). `tests/unit/test_writer_boundary.py` enforces this.
- The admin check runs before any write. It reads the caller's own `memberships` through `app_rw` (the `TenantConn` of the request), so the answer comes from the same policies as every read, and a failed check never touches the writer engine.
- No read route depends on the writer engine.

**Status codes.** `POST /documents` from a non-admin returns 403: the route exists for everyone, and refusing it reveals nothing. Routes with a document id (`PUT /documents/{id}/acl`, `DELETE /documents/{id}`) return 404 to non-admins, and for documents of other tenants, so a caller can't learn that an id exists (the same rule as the read routes).

**Default ACL.** An upload without an ACL gets `["user:<uploader sub>"]`: only the uploader can read it until someone widens it. An explicit ACL replaces the default; it is not added to it.

**Schema (migration `5c1e9a7d3b20`).**

- `documents.source_hash text NOT NULL`: SHA-256 of the uploaded bytes, lower-case hex. A unique index on `(tenant_id, source_hash) WHERE deleted_at IS NULL` allows one live copy of a file per tenant. Uploading a duplicate is detectable, and the same file can come back after a delete.
- `documents.created_by text`: the uploader's `sub`. NULL for the seed and for rows older than the migration.
- `documents.deleted_at timestamptz`: soft delete.

**Backfill.** Under `FORCE ROW LEVEL SECURITY` the migrator can't read or UPDATE any tenant row, so the backfill can't be an `UPDATE`. The column is added with a volatile default, `'legacy:' || gen_random_uuid()`. PostgreSQL then rewrites the table and evaluates the default once per row, and RLS doesn't apply to that rewrite. The default is dropped right after, so new rows must supply a hash. A hash of the title was rejected: two documents with the same title in a tenant would break the unique index. A check constraint allows only 64 hex characters or the `legacy:` prefix, so a legacy value never collides with a real hash.

**Soft delete without a join in the chunks policy.** ADR 0003 keeps the chunks policy a per-row check (`acl_principals && principals`), and it stays that way. A deleted document disappears through three layers:

1. A row trigger on `documents` (`AFTER UPDATE OF deleted_at`) deletes the document's `document_acl` rows. The existing statement triggers on `document_acl` then recompute its chunks to an empty `acl_principals`, which matches no principal.
2. The chunk trigger (`chunks_set_acl_principals`) yields `{}` for any chunk whose document isn't live. An ACL added to a deleted document, or a chunk written to one, therefore never becomes readable.
3. The `app_rw` policy on `documents` also requires `deleted_at IS NULL`.

The ACL rows go, not just the chunk copies, because the `document_acl` policy for `app_rw` can't check `documents.deleted_at`. The `documents` policy already reads `document_acl`, and a policy in the other direction is a cycle that PostgreSQL refuses ("infinite recursion detected in policy"). With the rows gone, `app_rw` sees nothing of a deleted document in any table. `app_ingest` still sees the document and its chunks, with empty ACLs. Restoring (setting `deleted_at` back to NULL) brings no access back: the document stays invisible until an admin sets an ACL. The delete route should record the ACL it removes in its audit row.

ADR 0003's drift check becomes: a live document's chunks carry its ACL, and a deleted document's chunks carry `{}`. With layer 1 both reduce to "matches `document_acl`", so the existing drift test still holds.

**Dependencies.** The project is Apache-2.0. All of these are permissive and compatible with it:

| Package | License | Pulled in by |
|---|---|---|
| mammoth (≥ 1.11) | BSD-2-Clause | direct |
| cobble | BSD-2-Clause | mammoth |
| markdownify | MIT | direct |
| beautifulsoup4, soupsieve, six | MIT | markdownify |
| python-multipart | Apache-2.0 | direct (FastAPI `UploadFile`) |

mammoth is at least 1.11 because that release turned off external file access by default: a `.docx` can link images by path or URL, and older versions would read them. mammoth ships no type information, so mypy ignores its missing imports.

**Settings.** `INGEST_MAX_BYTES` (10 MiB), `CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS` (must be smaller than the chunk), `EMBED_BATCH_SIZE`, and the LLM provider settings (`LLM_PROVIDER`, `OLLAMA_*`, `OPENAI_COMPAT_*`, with the API key as a `SecretStr`).

## Alternatives considered

- **The converter project's `engine/` package.** Richer formats, but it would add a dependency on another repository's internals and its release cycle, for a feature this project only needs three formats of.
- **Writes as `app_rw` with write policies.** The writer has to touch chunks the user can't read (ADR 0004), so this reopens the problem that ADR solved.
- **Hard delete only.** Simpler, but the audit trail points at chunk ids (`audit_events.chunk_ids`), and a delete that removes them loses what was cited.
- **Soft delete in the chunks policy** (`EXISTS (SELECT 1 FROM documents ...)`). Correct, but it is the per-candidate join that ADR 0003 rejected.
- **A `deleted` flag denormalized onto `document_acl`.** It would hide the ACL rows without deleting them, at the cost of another trigger-maintained copy. Keeping the old ACL for restore isn't needed yet; the audit row can hold it.
- **403 for non-admins on routes with a document id.** It would tell a caller that the id exists.

## Consequences

- **Pro:** A deleted document is invisible to `app_rw` in every table, enforced by the database, and the vector scan still has no joins.
- **Pro:** The downgrade fails closed: deleted documents have no ACL rows, so they stay invisible once `deleted_at` is gone.
- **Pro:** One module holds the writer engine, and a test enforces it, so a read route can't pick it up by accident.
- **Con:** Restoring a document needs a new ACL. The previous one only survives in the audit log.
- **Con:** The admin check reads `memberships`, so a new admin waits for the next `permsync` cycle, and so does a removed one (ADR 0002's window applies to writes too).
- **Con:** Each chunk write now does one extra primary-key lookup on `documents` in the trigger.
- **Note:** The unique index is per tenant. Like the other unique keys (ADR 0004), it can't reveal another tenant's documents.
