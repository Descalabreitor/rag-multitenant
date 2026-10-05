# 0006. Token validation, the tenant claim and the 401/403 split

**Status:** Accepted (2026-10-05)

## Context

Every request carries a Keycloak access token, and the tenant must come from that token only (never from the path, query or body). ADR 0005 sets up the realm so the token names the user's Organization. This ADR decides how the API reads it, which tokens it refuses, and with what status code.

Keycloak 26's *Organization Membership* mapper emits the `organization` claim only when the `organization` scope is in the token's scope (with no scope it adds nothing). Its shape depends on the mapper's options, checked in the Keycloak docs ("Mapping organization claims") and in `OrganizationMembershipMapper`:

- default: a list of aliases, `"organization": ["acme"]`;
- *Add organization id* on (our realm): an object keyed by alias, `"organization": {"acme": {"id": "7f66d4f3-…"}}`;
- not multivalued: a single alias string.

With the plain `organization` scope, a user in several Organizations is asked to pick one at login. With `organization:*` or several `organization:<alias>` scopes, the claim can hold more than one.

## Decision

**Validation** (`ragmt.auth.tokens`, PyJWT):

- The header `alg` must be in `OIDC_ALGORITHMS`, which allows asymmetric algorithms only. Settings refuse `none` and `HS*` at startup, and `TokenValidator` checks again. This closes RS256→HS256 confusion, and PyJWT also requires the header `alg` to match the JWK's own algorithm.
- The key is looked up by `kid` in the cached JWKS. `oct` keys and `use: enc` keys are never loaded.
- `exp`, `iss`, `aud` and `sub` are required. `iss` must equal `OIDC_ISSUER` exactly and `aud` must contain `OIDC_AUDIENCE`. Leeway is 10 s for clock skew.
- `sub` must be a non-empty string. `groups` and every other claim are ignored (ADR 0002).

**Tenant:** the `organization` claim must be an object with exactly one entry, whose `id` is a UUID in canonical form (lower-case, hyphenated, as Keycloak writes it). Anything else is refused: a missing claim, a list of aliases, an empty object, two Organizations, or a malformed id. The API never maps an alias to an id. With several Organizations it doesn't pick one, because that guess would decide which tenant's data the user sees.

**Status codes:**

| Case | Status |
|---|---|
| No `Authorization: Bearer`, malformed token, bad signature, disallowed `alg`, unknown `kid`, expired, wrong `iss`/`aud`, missing or empty `sub` | **401**, `WWW-Authenticate: Bearer` |
| Genuine token, but not exactly one well-formed Organization | **403** |
| The JWKS cannot be fetched (Keycloak down, bad response) | **503**, no token accepted |

401 means "we don't know who you are": retrying with a fresh token may help. 403 means the token is genuine but gives no tenant, so a new token with the same scopes won't help. The fix is in Keycloak (membership or requested scope). A malformed Organization id is 403 too: it can't be forged without breaking the signature, so it points to misconfiguration, not an attack.

Response bodies are fixed strings (`Not authenticated`, `Forbidden`). The reason (e.g. `ExpiredSignatureError`, `2 organizations, expected exactly 1`) is logged at INFO. The token, the `kid` and other caller-controlled values are never logged.

**JWKS cache** (`ragmt.auth.jwks`): fetched with httpx (async; PyJWKClient uses blocking urllib) from `{OIDC_ISSUER}/protocol/openid-connect/certs`. Keys are cached for `JWKS_CACHE_TTL_SECONDS`. An unknown `kid` triggers a refetch, to pick up key rotation, but there is at most one fetch every 30 s whatever the reason. Forged `kid`s therefore cost Keycloak at most one request every 30 s. After the TTL, expired keys are not used, even if the refresh fails.

## Alternatives considered

- **Tenant from the alias list** (default mapper), resolved through `tenants`. Needs a lookup before any tenant context exists, and aliases can be renamed. Rejected in ADR 0005 too.
- **Pick the first Organization when there are several.** Silent, order-dependent tenant choice. Rejected.
- **A custom header naming the tenant, checked against the claim.** Adds a client-controlled input that has to be validated anyway, and the invariant says the tenant comes from the JWT only.
- **401 for a missing Organization.** Clients would retry with a new token and loop, since the new token has the same problem.
- **Keep serving expired JWKS keys while Keycloak is down.** More available, but a key Keycloak removed would keep working. Failing closed (503) is the project's default.

## Consequences

- **Pro:** One place decides the tenant, and the leak suite (`tests/leaks/test_jwt.py`) covers each refusal: `alg: none`, HS256 with the public key, stripped signature, other key with the same `kid`, tampered payload, wrong `iss`/`aud`, expired, unknown `kid`, and every bad `organization` shape.
- **Pro:** Unknown-`kid` floods are bounded. Key rotation is picked up within 30 s without a restart.
- **Con:** Users in several Organizations must request a single one (plain `organization` scope, or `organization:<alias>`). `organization:*` tokens are refused.
- **Con:** A Keycloak outage longer than the TTL makes the API return 503. Raising `JWKS_CACHE_TTL_SECONDS` trades that for slower key revocation.
- **Note:** The JWKS path is Keycloak-specific. Another IdP would need OIDC discovery (`.well-known/openid-configuration`) instead.
