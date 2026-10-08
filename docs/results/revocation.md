# Revocation window

How long a user keeps reading a group's documents after being removed from the group in Keycloak. Summary of run `20261007T214838Z` (raw runs and permsync logs in `docs/results/raw/revocation-20261007T214838Z/`, git-ignored). `make revocation` reproduces it. Method: `eval/revocation.py`.

## Setup

- Same host as [`eval.md`](eval.md): AMD Ryzen 7 260, Windows 11, Docker Desktop. PostgreSQL 16 + pgvector 0.8.0 and Keycloak 26.0 from `compose.yaml`; the API and permsync run on the host (the compose `permsync` service stopped). No Ollama: `GET /documents` embeds nothing.
- **Option C (this service, ADR 0002/0007):** permissions come from `memberships`, synced by permsync. Each run waits a random delay in `[0, interval)` (so the revocation lands at a random point of the sync cycle), removes alice from `/acme/finance` (t0, when the admin API call returns), then polls `GET /documents` with **the same token** every 100 ms until the Q3 budget is gone (t1), then puts her back. 20 runs per `PERMSYNC_INTERVAL_SECONDS`.
- **Option A (groups from the token, not implemented in the API):** a temporary dev-only client adds a `groups` claim to dave's tokens (5 min lifetime). Each run revokes `/umbra/finance` at a random point of the token's life and measures how long the old token, which still lists the group, keeps being accepted. That is how long a resource server trusting the claim would keep granting the group. 5 runs.

## Results

Option C, seconds from t0 to the first `GET /documents` without the document:

| `PERMSYNC_INTERVAL_SECONDS` | runs | min | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| 60 | 20 | 1.83 | 40.01 | 54.96 | 58.60 |
| 10 | 20 | 0.64 | 5.77 | 9.59 | 9.63 |
| 5 | 20 | 0.11 | 2.36 | 4.39 | 4.73 |

The window is bounded by the interval: every run lost access within one cycle (max 58.6 s at 60 s, 9.6 s at 10 s, 4.7 s at 5 s), and the median is about half the interval, as expected for a revocation at a random point of the cycle. Restoring access took one cycle too (median 55.9 s, 5.9 s and 5.0 s).

Option A, seconds the pre-revocation token was still accepted after t0 (5-minute tokens):

| run | token age at t0 (s) | window (s) |
|---:|---:|---:|
| 1 | 224.9 | 85.1 |
| 2 | 267.0 | 43.2 |
| 3 | 144.7 | 166.2 |
| 4 | 63.1 | 247.2 |
| 5 | 272.0 | 38.2 |

With groups in the token, the window is the token's remaining lifetime: up to the full token lifetime (5 min here) plus the validator's clock-skew leeway, whatever permsync does. A token issued after t0 no longer listed the group. In the same runs, option C (permsync running alongside) hid the document from the same user after 2.4 to 10.1 s.

## Conclusion

Resolving groups in the database (ADR 0002) makes the revocation window a setting rather than a property of the token: with the default 60 s it is at most one interval (median ~40 s in this run), and 5 to 10 s intervals bring it under 10 s at the cost of more Keycloak admin requests per minute (one listing per group and per grouped user each cycle, ADR 0007). Trusting the token's `groups` claim would leave it at up to the token lifetime.

## Caveats

- One machine, one run, a two-tenant realm. With many groups and users a cycle takes longer and adds to the window.
- The cross-check against permsync's `revoked` log line agrees with t1 in 58 of 60 runs. In two runs at 60 s the matched log line is later than t1 (76.8 s and 157.2 s): the script matches the first `revoked` line after t0 for that user and group, so a run whose own line wasn't matched picks up a later cycle's. The API measurement (t1) is the number reported.
- Keycloak outages aren't covered: during one, revocations wait (ADR 0007).
