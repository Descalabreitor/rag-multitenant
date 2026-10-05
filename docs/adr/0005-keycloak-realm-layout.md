# 0005. Keycloak realm layout: organizations, groups and clients

**Status:** Accepted (2026-10-05)

## Context

The `ragmt` realm (`keycloak/realm-export.json`, imported with `--import-realm`) has to provide three things: a verified tenant in every user token, the group memberships that `permsync` copies into the `memberships` table (ADR 0002), and a way for the CLI to log in. It is committed to git, so it must hold no secrets. Keycloak 26.0 shaped several choices, and they were checked against a running instance.

## Decision

- **One Organization per tenant, and the Organization id is the tenant id.** The realm file sets the ids explicitly (the same ones as `seed/corpus.py`), and each user is a member of exactly one Organization.
- **The tenant claim carries ids, not aliases.** The realm declares its own client scopes, so Keycloak does not create its defaults. The `organization` scope's mapper has *Add organization id* on, and tokens carry `"organization": {"acme": {"id": "<tenant id>"}}`. With Keycloak's default scope, the claim holds only aliases (`["acme"]`). The mapper only runs when the `organization` scope is in the token's scope, so adding a second mapper on the client merged both shapes into one list. The declared scopes are minimal (`basic`, `profile` with just the username, `roles`, `organization`), so tokens carry no email or names.
- **Groups are realm groups under one top-level group per Organization**: `/acme/finance`, `/umbra/research`. The top-level group's name is the Organization alias, and its `organization_id` attribute holds the tenant id. `permsync` maps `/<alias>/<group>` to `(organization_id, user sub, group)`. Keycloak 26.0 has no groups scoped to an Organization.
- **`ragmt-permsync` is a confidential client with only a service account, and its roles are `view-users` and `query-groups`.** That is enough to list groups with their attributes, group members and a user's groups. With read-only roles, including `view-realm`, the Organization endpoints (`/organizations`, `/organizations/{id}/members`) returned 403, so `permsync` does not use them. Granting `manage-realm` would let a leaked permsync secret reconfigure the realm.
- **`ragmt-cli` is a public client using authorization code with PKCE (S256 required).** Redirects are allowed only to `http://127.0.0.1/*` (loopback, RFC 8252). It never allows the password grant. An audience mapper adds `ragmt-api` (`OIDC_AUDIENCE`).
- **The password grant lives in a separate, dev-only client, `ragmt-dev-password`.** It is public, allows only the password grant, and its `enabled` is `${KC_DEV_PASSWORD_CLIENT:false}`, so it doesn't work unless the environment turns it on at import. `.env.example` turns it on for local development, the end-to-end tests and the README walkthrough. A separate client keeps that switch away from the real login client: turning it on can't weaken `ragmt-cli`.
- **Secrets are placeholders** (`${PERMSYNC_CLIENT_SECRET}`, `${KC_DEMO_USER_PASSWORD}`). Keycloak resolves them from the container's environment at import time, and `compose.yaml` passes them from `.env`. Signing keys are not exported; Keycloak generates them on import.

## Consequences

- **Pro:** The API gets the tenant id straight from the verified token, with no alias lookup (which would need to read `tenants` before any tenant context exists).
- **Pro:** A leaked permsync secret can read users and groups, but cannot change anything.
- **Pro:** `tests/unit/test_realm_export.py` checks that the file holds only placeholders, that the clients keep these settings, and that Organizations, users and groups agree with the seed.
- **Con:** Organization membership and the `/<alias>/` group tree are two things to keep consistent. A user in `/acme/finance` who is not an Acme member would get an Acme membership row. That row is inert, because the tenant comes from the token's Organization, but `permsync` should skip such users and log them.
- **Con:** Because the realm declares its own client scopes, Keycloak's built-in ones (`email`, `web-origins`, `acr`, …) don't exist in this realm. Anything that needs them has to add them here.
- **Note:** `--import-realm` skips a realm that already exists. After editing the file, delete the `ragmt` realm (or run `docker compose down -v`) to import it again.
