# 0003. Denormalize document ACLs onto chunks

**Status:** Accepted (2026-09-30)

## Context

ACLs are defined per document ("this document is visible to the `hr` group"), but search runs over chunks. The RLS policy must decide, for every candidate chunk produced by the vector search, whether the user may see it.

## Decision

Each chunk stores a copy of its document's ACL in an `acl_principals text[]` column, indexed with GIN. The RLS policy compares it with the user's principals using the array overlap operator (`&&`) on the row itself. `document_acl` remains the source of truth, and a trigger copies the ACL to all of the document's chunks within the same transaction.

## Alternatives considered

- **Normalized.** The policy checks `document_acl` with `EXISTS (SELECT 1 FROM document_acl WHERE ...)`. This is the textbook design, with no duplicated data, but the subquery is evaluated for every candidate returned by the HNSW index, which can be hundreds per question.
- **Denormalized** (chosen). A per-row array comparison, with no joins during the vector scan.

## Consequences

- **Pro:** The permission check is cheap and needs no joins while the vector index is being scanned.
- **Pro:** Because the trigger runs in the same transaction, an ACL change is atomic: there is no moment when the document says one thing and its chunks another.
- **Con:** Changing the ACL of a document with 500 chunks updates 500 rows. This is acceptable because permission changes are rare compared with searches.
- **Con:** Duplicated data can drift if something writes to `chunks` without going through the trigger. A test must check that every chunk's `acl_principals` matches its document's ACL.
- **Note:** The benchmarks will compare the latency of the normalized and denormalized policies. If the difference turns out to be negligible, a new ADR will supersede this one and return to the normalized design.
