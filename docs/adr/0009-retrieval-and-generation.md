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

**What the trail shows (reviewed 2026-10-07).** Each row has `id`, `occurred_at`, `actor_sub`, `action`, `chunk_ids` and `details`. `details` holds ids, counts, hashes, scores, model names, principals and subs (permsync rows: the changed `[sub, group]` pairs), never chunk content, titles or file names. While `AUDIT_STORE_QUERY_TEXT` is off, `GET /audit` also drops the `question` key from ask rows written while it was on: the table is insert-only, so those rows keep the text, and turning the setting off must stop it being served (`ragmt.audit.list_events(with_question_text=...)`). The `question_sha256` stays, so repeated questions can still be correlated. `tests/leaks/test_ask.py` walks every key and value of the response and fails if any holds the question text.

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
- **Con:** A plain SHA-256 of a short question can be found by hashing guesses. It correlates repeated questions but isn't anonymisation, and an admin who reads the trail can confirm a guess at what a user asked. A keyed hash would need a secret to manage.
- **Con:** Hiding stored question text when the setting is off is a read filter, not erasure: the text is still in the table, readable by a database superuser (no runtime role or the migrator can bypass RLS to read it).
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

## Results (measured 2026-10-08)

`make eval` (`eval/bench.py`, tables and charts in `docs/results/eval.md`): 100,000 synthetic chunks (seed 42, 768 dimensions) in a separate `ragmt_eval` database, tenants holding 50%, 10%, 1% and 0.1% of the table, three users each (no groups, one group, all groups), the retriever's own statement as app_rw, plans forced per transaction and checked with EXPLAIN. k = 5, the service's HNSW settings.

| | recall@5, all cells | p50 / p95 ms, 50% tenant | p50 / p95 ms, 0.1% tenant |
|---|---|---|---|
| HNSW, iterative scan off | 0.33 | 3.8 / 5.7 | 3.3 / 5.1 |
| HNSW, `relaxed_order` (the default) | 0.79 | 7.0 / 14.8 | 114 / 268 |
| HNSW, `strict_order` | 0.70 | 4.1 / 7.1 | 98 / 338 |
| Exact (no index scan) | 1.00 | 102 / 253 | 8.9 / 19 |
| Planner's choice under RLS (what the service runs) | 1.00 | 99 / 260 | 4.9 / 12 |
| Partitions by tenant, `relaxed_order` (scratch) | 1.00 | 3.9 / 12 | 3.0 / 8.9 |

What this confirms and what it changes:

- **`relaxed_order` stays the default.** With iterative scans off, recall@5 falls to 0.20 to 0.66 at the 10% tenant and to 0.06 or less at 1% and below, where every query comes back short. `strict_order` is below `relaxed_order` at every size.
- **`HNSW_MAX_SCAN_TUPLES` (20,000) is too low for tiny tenants on the shared index.** The 0.1% tenant's users hit it: 21 of 60 queries came back short, recall 0.28 to 0.60, 114 ms. Raising it trades latency for recall on exactly these tenants; partitions remove the problem instead.
- **The planner misjudges the ACL filter.** It folds the tenant setting into its estimate (501 rows for the 50% tenant, 1 for the 0.1% one) but can't see the user's principals, so it guesses about 1% for `acl_principals && …` against 22% to 83% actually readable. The exact plan then looks cheap, and the service gets it for every tenant at this table size: correct (recall 1.00), but 99 ms p50 and 260 ms p95 for the large tenant, 238 ms p50 for its all-groups user. The no-RLS baseline, with the principals as literals, estimates better and picks HNSW there (4.8 ms). This is the "broad-access users on mid-sized tables" case above, now measured.
- **The denormalised ACL stays (ADR 0003).** Same forced HNSW plan, same recall budget: the EXISTS policy is 2.6× slower at the 50% tenant (18 vs 7 ms p50), 1.3× at 10%, 1.8× at 1%; under exact plans they are within 17% of each other. That is not the negligible difference that would bring back the normalised design.
- **RLS itself costs nothing measurable** on the same plan: 7.02 vs 7.03 ms p50 at the 50% tenant against the superuser with the policy written as WHERE. On small tenants the baseline was slower (202 vs 114 ms), because its plan checked the array overlap before `tenant_id` on every scanned tuple, where the policy checks `tenant_id` first.
- **Partitions by tenant fix both problems at once**: recall ≥ 0.99 for every tenant and user, p50 3 to 4 ms and p95 under 13 ms whatever the tenant's size, with the same policy text, no tenant filter in the query (the executor prunes from the policy's `current_setting`, 8 of 9 partitions removed) and the same plan for everyone. ADR 0010 proposes adopting them.

Caveats: one run on one laptop (Docker Desktop, default PostgreSQL memory settings), synthetic vectors, and the scratch indexes were built in one pass where `public.chunks`' was built row by row (under the same filter, that difference was worth about 0.03 of recall). A full run takes about 15 minutes with the corpus loaded, 25 from an empty database.
