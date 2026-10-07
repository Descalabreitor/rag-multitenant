"""Getting and renewing tokens from Keycloak (ADR 0005).

- `login_interactive`: authorization code + PKCE (S256) against the public
  `ragmt-cli` client, with the browser coming back to a loopback port.
- `current_tokens`: the stored tokens for `ragctl ask`, refreshed when the
  access token has expired.
- `password_grant`: DEV ONLY, the `ragmt-dev-password` client.

Errors carry a message safe to print: never a token, code, password or
response body, only the HTTP status and the OAuth `error` code.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from ragmt.cli.config import CliConfig, ConfigError, check_url
from ragmt.cli.loopback import CallbackError, LoopbackCallback
from ragmt.cli.pkce import challenge_s256, make_state, make_verifier
from ragmt.cli.tokens import TokenFileError, TokenSet, TokenStore

_ERROR_CODE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
# The scopes the client gets come from its default client scopes; "openid" makes it OIDC.
_SCOPE = "openid"


class AuthError(RuntimeError):
    """Getting a token failed. The message is safe to print."""


class TokenRejected(AuthError):
    """The token endpoint refused the grant (bad code, expired refresh token, bad password)."""


class NotLoggedIn(AuthError):
    """There is no usable token: the user has to run `ragctl login`."""


@dataclass(frozen=True, slots=True)
class Endpoints:
    authorization: str
    token: str


def discover(client: httpx.Client, issuer: str) -> Endpoints:
    """The issuer's endpoints, from its OpenID configuration."""
    try:
        response = client.get(f"{issuer}/.well-known/openid-configuration")
    except httpx.HTTPError as exc:
        raise AuthError(f"could not reach the identity provider at {issuer}") from exc
    if response.status_code != httpx.codes.OK:
        raise AuthError(f"OpenID discovery failed: HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise AuthError("OpenID discovery returned invalid JSON") from exc
    if not isinstance(body, dict) or body.get("issuer") != issuer:
        # A configuration for another issuer would give tokens the API rejects,
        # or send credentials somewhere else.
        raise AuthError("OpenID discovery returned a different issuer")
    try:
        return Endpoints(
            authorization=check_url("authorization endpoint", str(body["authorization_endpoint"])),
            token=check_url("token endpoint", str(body["token_endpoint"])),
        )
    except (KeyError, ConfigError) as exc:
        raise AuthError("OpenID discovery returned unusable endpoints") from exc


def authorization_url(
    endpoints: Endpoints, *, client_id: str, redirect_uri: str, state: str, challenge: str
) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": _SCOPE,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{endpoints.authorization}?{query}"


def _oauth_error(response: httpx.Response) -> str | None:
    try:
        body: Any = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, str) and _ERROR_CODE.fullmatch(error) else None


def _token_request(
    client: httpx.Client, token_url: str, data: dict[str, str], *, issuer: str, client_id: str
) -> TokenSet:
    try:
        response = client.post(token_url, data=data)
    except httpx.HTTPError as exc:
        raise AuthError("could not reach the identity provider") from exc
    if response.status_code != httpx.codes.OK:
        error = _oauth_error(response)
        detail = f"HTTP {response.status_code}" + (f", {error}" if error else "")
        if response.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.UNAUTHORIZED):
            raise TokenRejected(f"the identity provider refused the request ({detail})")
        raise AuthError(f"token request failed ({detail})")
    try:
        return TokenSet.from_response(response.json(), issuer=issuer, client_id=client_id)
    except (ValueError, TokenFileError) as exc:
        raise AuthError("the identity provider returned an unusable token response") from exc


def exchange_code(
    client: httpx.Client,
    endpoints: Endpoints,
    config: CliConfig,
    *,
    code: str,
    redirect_uri: str,
    verifier: str,
) -> TokenSet:
    return _token_request(
        client,
        endpoints.token,
        {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
        issuer=config.issuer,
        client_id=config.client_id,
    )


def refresh(
    client: httpx.Client, endpoints: Endpoints, config: CliConfig, tokens: TokenSet
) -> TokenSet:
    if tokens.refresh_token is None:
        raise NotLoggedIn("no refresh token")
    renewed = _token_request(
        client,
        endpoints.token,
        {
            "grant_type": "refresh_token",
            "client_id": config.client_id,
            "refresh_token": tokens.refresh_token,
        },
        issuer=config.issuer,
        client_id=config.client_id,
    )
    if renewed.refresh_token is None:
        # Without rotation the old refresh token stays valid.
        return TokenSet(
            issuer=renewed.issuer,
            client_id=renewed.client_id,
            access_token=renewed.access_token,
            expires_at=renewed.expires_at,
            refresh_token=tokens.refresh_token,
            refresh_expires_at=tokens.refresh_expires_at,
        )
    return renewed


def login_interactive(
    client: httpx.Client,
    config: CliConfig,
    *,
    open_browser: Callable[[str], object],
    notify: Callable[[str], None],
    timeout: float = 300.0,
) -> TokenSet:
    """Browser login: authorization code + PKCE (S256) with a loopback redirect."""
    endpoints = discover(client, config.issuer)
    verifier = make_verifier()
    state = make_state()
    try:
        with LoopbackCallback(state) as callback:
            url = authorization_url(
                endpoints,
                client_id=config.client_id,
                redirect_uri=callback.redirect_uri,
                state=state,
                challenge=challenge_s256(verifier),
            )
            notify(f"Opening the browser to log in. If it doesn't open, visit:\n  {url}")
            open_browser(url)
            code = callback.wait(timeout)
    except CallbackError as exc:
        raise AuthError(str(exc)) from exc
    except OSError as exc:
        raise AuthError("could not listen on a loopback port for the login") from exc
    return exchange_code(
        client,
        endpoints,
        config,
        code=code,
        redirect_uri=callback.redirect_uri,
        verifier=verifier,
    )


def current_tokens(client: httpx.Client, config: CliConfig, store: TokenStore) -> TokenSet:
    """Stored tokens with a valid access token, refreshing (and saving) them if needed."""
    try:
        tokens = store.load()
    except TokenFileError as exc:
        raise NotLoggedIn("the token file is unreadable; run `ragctl login` again") from exc
    if tokens is None:
        raise NotLoggedIn("not logged in; run `ragctl login`")
    if tokens.issuer != config.issuer or tokens.client_id != config.client_id:
        raise NotLoggedIn(f"not logged in to {config.issuer}; run `ragctl login`")
    if tokens.access_valid():
        return tokens
    if not tokens.can_refresh():
        raise NotLoggedIn("the session has expired; run `ragctl login`")
    try:
        renewed = refresh(client, discover(client, config.issuer), config, tokens)
    except TokenRejected as exc:
        raise NotLoggedIn("the session has expired; run `ragctl login`") from exc
    store.save(renewed)
    return renewed


def password_grant(
    client: httpx.Client, config: CliConfig, *, client_id: str, username: str, password: str
) -> TokenSet:
    """DEV ONLY: a token from the password grant of `client_id` (`ragmt-dev-password`)."""
    endpoints = discover(client, config.issuer)
    return _token_request(
        client,
        endpoints.token,
        {
            "grant_type": "password",
            "client_id": client_id,
            "username": username,
            "password": password,
            "scope": _SCOPE,
        },
        issuer=config.issuer,
        client_id=client_id,
    )
