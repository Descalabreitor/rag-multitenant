# Leak suite results

**0 leaks in 393256 attempts.**

Written by `make leaks` on 2026-10-07 23:04 UTC: `pytest -m leaks` with the Hypothesis profile `nightly`, 246 tests collected, 0 failed.

## Property test (`tests/leaks/test_properties.py`)

2153 random worlds of 2 or 3 tenants, 9043 states checked (after setup and after every operation). In each state, every user of every tenant is compared with a plain-Python oracle (`tests/leaks/oracle.py`) that replays the same operations without SQL. An attempt is one comparison for one user in one state, or one write aimed at another tenant:

| Check | Compares | Attempts |
|---|---|---:|
| documents | `GET /documents` equals the oracle's readable documents (ids and titles) | 65829 |
| document_by_id | `GET /documents/{id}` is 404 for a document the user may not read | 63623 |
| chunks | `SELECT … FROM chunks` as app_rw equals the oracle's readable chunks | 65829 |
| retrieval | `PgVectorRetriever.search` returns only readable chunks | 65829 |
| ask_citations | `POST /ask` cites, and its prompt holds, only readable chunks | 65829 |
| ask_canaries | `POST /ask` holds no canary of a document the user may not read | 65829 |
| foreign_write | a write naming another tenant finds nothing and changes nothing | 488 |
| **Total** | | **393256** |

Only the property test's checks are counted. The other leak tests are fixed cases (one per leak scenario) and count as tests above.
