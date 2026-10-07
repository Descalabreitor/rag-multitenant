# 0009. Retrieval and generation: ports, prompt, citations and audit

**Status:** Accepted (2026-10-06)

## Context

Phase 4 answers questions: embed the question, find the closest chunks the caller may read, and have a chat model answer from them with citations. RLS already decides which chunks a user can see (ADR 0001, 0003). This ADR covers what happens around that: how search is abstracted, what the model sees, what clients get back, what happens when nothing is found, and what the audit trail keeps.

The model is the least trusted part of the path. Chunk text is written by whoever uploaded the document, so a chunk can say "ignore your instructions". The model must not be able to act on that beyond the text of its answer.

## Decision

**Ports** (`ragmt.domain`, no I/O):

- `Retriever[Conn].search(conn, query_vector, k) -> list[RetrievedChunk]`. `conn` is the request's `tenant_session` connection as `app_rw`. The retriever adds no tenant or ACL filter: RLS does. The port is generic over the connection type, so the domain names no driver.
- `ChatProvider` with `model` and `complete(system, messages) -> str`. `ChatMessage(role, content)` has roles `user` and `assistant`; the system prompt is a separate argument.
- `RetrievedChunk(chunk_id, document_id, ordinal, title, heading, content, score)`. `score` is cosine similarity (1 − cosine distance), higher is closer. There is no embedding field.
- `Citation(document_id, title, heading)` and `Answer(text, citations, model)`. `model` is None when no model was called.

**Vector search only in v1**, behind `Retriever`. Hybrid search (BM25 plus vectors) can be a second adapter later without touching callers. The HNSW scan is tuned per transaction with `set_config(..., true)` from `HNSW_EF_SEARCH`, `HNSW_ITERATIVE_SCAN` and `HNSW_MAX_SCAN_TUPLES`, never with a plain `SET`. Iterative scans default to `relaxed_order`: RLS filters rows during the scan, so with scans off a small tenant can get fewer than k chunks back (ADR 0001). Because relaxed order can return rows slightly out of order, the query re-sorts them by distance. `HNSW_EF_SEARCH` must be at least `RETRIEVAL_K`. The plan is checked with `EXPLAIN (ANALYZE, BUFFERS)` under RLS before the query lands.

**Retrieved chunks are untrusted data.** The prompt wraps each chunk in a delimiter that carries its citation marker, and escapes the chunk's content, title and heading so that no text inside can close the delimiter or open a new one. The system prompt says the delimited text is reference material, not instructions. The model has no tools, no function calling and no database access: it gets one string in and returns one string out. Chunks go into the prompt in score order until `ASK_MAX_CONTEXT_CHARS` is reached. A question longer than `ASK_MAX_QUESTION_CHARS` is refused before anything is embedded. A completion that takes more than `CHAT_TIMEOUT_SECONDS` fails.

**Citations.** A chunk is cited as `[doc:<document_id>#<ordinal>]` (`citation_marker`). The citations returned to clients are built from the retrieved chunks whose markers appear in the reply, never from whatever the model writes: a marker that names no retrieved chunk is dropped. A `Citation` holds the document id, title and heading only, never content, scores or embeddings.

**No chunks, no model.** When retrieval returns nothing, the answer is the fixed `NO_CONTEXT_ANSWER`, with no citations and `model = None`, and the chat model is not called. The text is the same whether nothing matched or nothing was visible, so it reveals nothing about other tenants or other ACLs.

**Audit.** Retrieval and its `audit_events` row are written in the same `app_rw` transaction, so a retrieval that wasn't recorded didn't happen. The row has `chunk_ids` (the column) and, in `details`, the scores in the same order, the chat model that will be called (null when there are no chunks) and the SHA-256 of the question. The question text is stored only if `AUDIT_STORE_QUERY_TEXT=true` (default false, for GDPR). The transaction commits before the model is called, so a slow model never holds a pool connection or a snapshot open.

**The ask route.** `POST /ask {question}` returns `{answer, citations: [{document_id, title, heading}], model}`. `ragmt.ask.AskService` does the work, built once in the lifespan from the embedding and chat providers (also built once there) and the app_rw engine. A question that is empty or longer than `ASK_MAX_QUESTION_CHARS` gets 422 before it is embedded. The question is embedded before any connection is taken. Search and the audit insert then run in one `tenant_session(engine, principal.tenant_id, principal.sub)` on app_rw, the same session the request's `TenantConn` would open, and it commits before generation. The route doesn't take `TenantConn` itself: that dependency keeps its connection until the route returns, which would hold it during the completion. The service opens its own session instead and closes it before calling the model. A failed embedding is 503 with nothing audited. A failed or timed-out completion is 503 too, and the retrieval stays audited. Error bodies are fixed strings that never quote the question.

**Reading the audit trail.** `GET /audit` lists the tenant's rows to its admins (group `admins`, ADR 0008), newest first, with keyset pagination (`limit`, then `before` = the previous page's `next_before`; ids only grow, so inserts during paging don't shift pages). Migration `b7d41c2e9f05` grants app_rw SELECT on `audit_events` with one policy: `tenant_id` is the session's tenant AND the session's user has an `admins` row in `memberships` for that tenant. Everyone else, and a session with no user, reads zero rows, so a non-admin's SELECT now returns nothing where it used to fail with "permission denied". UPDATE, DELETE and TRUNCATE stay revoked, and app_ingest still can't SELECT. There is no policy cycle: the new policy reads `memberships`, whose app_rw policy reads only the session settings, and no policy reads `audit_events`. The route also runs `is_tenant_admin` first, on the request's `TenantConn`, and a non-admin (another tenant's admin included, since the tenant comes from the token) gets FastAPI's own 404 body for an unknown path. An index on `(tenant_id, id)` serves the pages.

**Dependencies.** None added: escaping uses the standard library, and the chat adapters use httpx like the embedding adapters.

**Settings.** `RETRIEVAL_K` (5), `HNSW_EF_SEARCH` (40, at least `RETRIEVAL_K`), `HNSW_ITERATIVE_SCAN` (`off` / `relaxed_order` / `strict_order`, default `relaxed_order`), `HNSW_MAX_SCAN_TUPLES` (20000), `ASK_MAX_QUESTION_CHARS` (2000), `ASK_MAX_CONTEXT_CHARS` (12000, at least `CHUNK_MAX_CHARS`), `CHAT_TIMEOUT_SECONDS` (120), `AUDIT_STORE_QUERY_TEXT` (false). The chat model comes from the existing `LLM_PROVIDER` settings: `OLLAMA_CHAT_MODEL` or `OPENAI_COMPAT_CHAT_MODEL`. `CHAT_PROVIDER` (unset by default) picks a different chat adapter than `LLM_PROVIDER`; the end-to-end check sets it to `fake` to run without the chat model (`make e2e FAKE_CHAT=1`).

## Alternatives considered

- **Hybrid search from the start.** Better recall on exact terms (names, codes), but it needs a full-text column and index, a fusion rule and its own measurements. The port leaves room for it.
- **Let the model call a search tool.** More flexible, but a model that can run queries is a model that can be talked into running other ones. Retrieval stays outside the model.
- **Trust the citations the model writes.** Simpler, but a model can invent ids or copy markers out of chunk text. Building citations from the retrieved set means a client never sees a document the user didn't retrieve.
- **Call the model with an empty context and let it say it doesn't know.** Small models answer from their training data anyway, and the call costs time. A fixed answer is cheaper and predictable.
- **Store the question text by default.** Useful for debugging, but questions can hold personal data, and the audit table is insert-only, so it can't be cleaned up later.
- **The audit trail for every user of the tenant, or through a separate auditor role.** Every user would see who asked what, and a third runtime role would be another credential to manage. Admins already run the tenant's writes, so the existing group is the smallest grant that works.
- **A `SECURITY DEFINER` function that returns the rows.** The project avoids definer functions (ADR 0004), and a policy keeps the rule in the database next to the others.
- **Keep the transaction open across the model call**, and record the reply in the same row. The audit row would hold more, but a 120 s completion would hold a connection from a small pool.

## Consequences

- **Pro:** What the model can see is exactly what RLS let through, and its reply can't reach the database or add citations to documents that weren't retrieved.
- **Pro:** The audit trail records every retrieval with what was returned, without storing the question unless configured.
- **Con:** Escaping and delimiters reduce prompt injection, they don't remove it. A chunk can still steer the answer the user reads. The leak suite should include injected chunks, and check that answers cite nothing outside the retrieved set.
- **Con:** A plain SHA-256 of a short question can be found by hashing guesses. It correlates repeated questions but isn't anonymisation. A keyed hash would need a secret to manage.
- **Con:** The audit row says what was retrieved, not whether the model answered: a failed or timed-out completion still leaves the row.
- **Con:** Admins read every user's rows: who asked, when, the ids and scores of the chunks retrieved (including chunks of documents the admin can't read; ids only, never content), and the question text when `AUDIT_STORE_QUERY_TEXT` is on. Admins can already change any document's ACL, so this adds metadata, not access. It does matter for GDPR when question text is stored.
- **Con:** Admin rights follow `memberships`, so a removed admin can read the trail until the next `permsync` cycle (ADR 0002, 0008).
- **Note:** `relaxed_order` and `HNSW_MAX_SCAN_TUPLES` trade completeness for latency. Recall and latency per setting belong in `docs/results/`.

## Query plan (measured 2026-10-06)

`PgVectorRetriever` (`ragmt.retrieval.pgvector`) sets the three `hnsw.*` settings with `set_config(..., true)`, then runs one statement: a `MATERIALIZED` CTE over `chunks` (`ORDER BY embedding <=> :query LIMIT :k`, no tenant or ACL condition), joined to `documents` for the title and re-sorted by distance. Checked with `EXPLAIN (ANALYZE, BUFFERS)` as app_rw through `tenant_session`, with the retriever's own settings (`ef_search` 40, `relaxed_order`, `max_scan_tuples` 20000, k = 5). The database was a throwaway pgvector 0.8.0 / PostgreSQL 16 container, migrated to head, with `ANALYZE` run. Vectors were clustered: 50 topics, about 0.7 cosine to their centre, and the query near one centre. One large tenant had 1% of its chunks `tenant:*` (carol, no groups, reads only those) and 40% `group:eng` (alice); nine small tenants had 2,000 chunks each.

| Chunks in table | Tenant, user (readable chunks) | Plan for the CTE | Execution |
|---|---|---|---|
| 48,000 | big (30k), carol (340) | Bitmap scan on the `acl_principals` GIN index, exact top-N heapsort | 5.9 ms, 1,417 buffers |
| 48,000 | big (30k), alice (12,320) | same | 61.9 ms, 37,899 buffers |
| 48,000 | small (2k), carol (980) | BitmapAnd of the tenant btree and GIN, exact sort | 4.0 ms |
| 148,000 | big (130k), carol (~1,300) | **HNSW index scan** (`chunks_embedding_hnsw_idx`), RLS as a filter, 2,978 rows removed before 5 passed | 12.5 ms, 9,246 buffers |
| 148,000 | big (130k), carol, `iterative_scan = off` | HNSW, 63 candidates removed, **0 rows returned** | 0.8 ms |
| 148,000 | big (130k), alice | HNSW, 0 rows removed | 1.0 ms, 913 buffers |
| 148,000 | small (2k), carol (980) | BitmapAnd of tenant btree and GIN, exact sort | 15.4 ms |

What this shows:

- **RLS is applied in every plan.** The policy's `tenant_id = …` and `acl_principals && <principals>` are either the index condition (btree and GIN, both leakproof operators) or a filter on the HNSW scan. Nothing reaches the CTE that the policy rejects, and the `documents` join only adds 5 primary-key lookups.
- **HNSW isn't always the plan, and that's fine.** The planner can't know the user's principals at plan time, so it estimates `acl_principals && …` at a flat ~1% of the table. While that 1% is cheap to sort (48k chunks), it pre-filters with GIN or the tenant btree and sorts exactly: there's no overfiltering, but the cost grows with what the user can read (62 ms for alice). Once the table is larger (148k), the HNSW scan wins for the large tenant, and small tenants keep the exact plan. The crossover depends on table size and statistics, not on the tenant.
- **Overfiltering is real on the HNSW plan.** With `iterative_scan = off`, carol got 0 of 5 rows. With `relaxed_order`, she got 5 after scanning about 3,000 tuples (12.5 ms). This confirms `relaxed_order` as the default.
- **`strict_order` can come back short.** In the leak test's overfiltering setup it sometimes returned fewer than k, or farther chunks, because it drops tuples that leave the graph out of order. `relaxed_order` plus the re-sort doesn't.
- Forcing HNSW on the 48k table (`enable_sort = off`) gave an 18 ms index scan for carol and 1 ms for alice, but the inflated cost estimate also turned on JIT (235 ms). Plans aren't forced in the application. If broad-access users on mid-sized tables turn out slow, the options (per-tenant partial HNSW indexes, or partitions, ADR 0001) belong in `docs/results/` with measurements.

The leak test for overfiltering (`tests/leaks/test_retriever.py`) forces the HNSW plan with `enable_sort = off` in its own transaction, because CI's table is small enough to get the exact plan.
