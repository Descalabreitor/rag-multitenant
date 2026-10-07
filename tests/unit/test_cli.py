"""ragctl against httpx.MockTransport (Keycloak and the API), plus its real loopback server.

No Keycloak or API is needed. The login tests run the real loopback callback
on 127.0.0.1 and play the browser from a thread.
"""

import http.client
import io
import json
import logging
import os
import stat
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID

import httpx
import pytest

from ragmt.cli.api import AskResponse, CitationOut, format_answer
from ragmt.cli.config import CliConfig, ConfigError, config_dir
from ragmt.cli.loopback import CallbackError, LoopbackCallback
from ragmt.cli.main import EXIT_ERROR, EXIT_NOT_LOGGED_IN, EXIT_OK, build_parser, main
from ragmt.cli.pkce import challenge_s256, make_state, make_verifier
from ragmt.cli.tokens import TokenSet, TokenStore

ISSUER = "http://localhost:8080/realms/ragmt"
API = "http://127.0.0.1:8000"
AUTH_URL = f"{ISSUER}/protocol/openid-connect/auth"
TOKEN_URL = f"{ISSUER}/protocol/openid-connect/token"
# Secrets that must never show up in output or logs.
ACCESS = "canary-access-token"
REFRESH = "canary-refresh-token"
NEW_ACCESS = "canary-new-access-token"
NEW_REFRESH = "canary-new-refresh-token"
CODE = "canary-auth-code"
DEMO_CREDENTIAL = "canary-dev-password"
SECRETS = (ACCESS, REFRESH, NEW_ACCESS, NEW_REFRESH, CODE, DEMO_CREDENTIAL)

DOC = UUID("11111111-2222-3333-4444-555555555555")
ANSWER = {
    "answer": f"Expenses are reimbursed monthly [doc:{DOC}#0].",
    "citations": [
        {"document_id": str(DOC), "title": "Expense policy", "heading": "Reimbursement"},
        {"document_id": str(DOC), "title": "Travel guide", "heading": None},
    ],
    "model": "llama3.1:8b",
}

Handler = Callable[[httpx.Request], httpx.Response]
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="no POSIX modes on Windows")


def _form(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}


def _token_body(access: str = ACCESS, refresh: str | None = REFRESH) -> dict[str, Any]:
    body: dict[str, Any] = {"access_token": access, "expires_in": 300, "token_type": "Bearer"}
    if refresh is not None:
        body |= {"refresh_token": refresh, "refresh_expires_in": 1800}
    return body


class FakeServers:
    """Keycloak (discovery + token endpoint) and the API, behind one MockTransport."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.token: Handler = lambda r: httpx.Response(200, json=_token_body())
        self.ask: Handler = lambda r: httpx.Response(200, json=ANSWER)
        self.discovery: Handler = lambda r: httpx.Response(
            200,
            json={
                "issuer": ISSUER,
                "authorization_endpoint": AUTH_URL,
                "token_endpoint": TOKEN_URL,
            },
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return self.discovery(request)
        if url == TOKEN_URL and request.method == "POST":
            return self.token(request)
        if url == f"{API}/ask" and request.method == "POST":
            return self.ask(request)
        return httpx.Response(404)

    def to(self, url: str) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url) == url]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@pytest.fixture
def servers() -> FakeServers:
    return FakeServers()


@pytest.fixture
def environ(tmp_path: Path) -> dict[str, str]:
    # Explicit, so the variables tests/conftest.py loads from .env play no part.
    return {"RAGCTL_CONFIG_DIR": str(tmp_path / "ragmt")}


@pytest.fixture
def store(environ: dict[str, str]) -> TokenStore:
    return TokenStore(config_dir(environ))


class Run:
    def __init__(self, code: int, out: str, err: str) -> None:
        self.code, self.out, self.err = code, out, err


def run_cli(
    servers: FakeServers,
    environ: dict[str, str],
    *argv: str,
    open_browser: Callable[[str], object] = lambda url: None,
) -> Run:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        list(argv),
        environ=environ,
        transport=servers.transport,
        open_browser=open_browser,
        stdout=out,
        stderr=err,
    )
    return Run(code, out.getvalue(), err.getvalue())


def assert_no_secrets(*texts: str) -> None:
    for text in texts:
        for secret in SECRETS:
            assert secret not in text


def stored(
    access: str = ACCESS, *, expires_in: float = 300, refresh_in: float | None = 1800
) -> TokenSet:
    now = time.time()
    return TokenSet(
        issuer=ISSUER,
        client_id="ragmt-cli",
        access_token=access,
        expires_at=now + expires_in,
        refresh_token=REFRESH,
        refresh_expires_at=None if refresh_in is None else now + refresh_in,
    )


# --- PKCE ------------------------------------------------------------------------


def test_challenge_matches_rfc_7636_appendix_b() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert challenge_s256(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_verifier_is_fresh_and_in_the_unreserved_set() -> None:
    verifiers = {make_verifier() for _ in range(50)}
    assert len(verifiers) == 50
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
    for verifier in verifiers:
        assert 43 <= len(verifier) <= 128
        assert set(verifier) <= allowed
        assert "=" not in challenge_s256(verifier)


@pytest.mark.parametrize("length", [42, 129])
def test_challenge_rejects_verifiers_of_the_wrong_length(length: int) -> None:
    with pytest.raises(ValueError):
        challenge_s256("a" * length)


def test_state_is_unguessable() -> None:
    states = {make_state() for _ in range(50)}
    assert len(states) == 50
    assert all(len(s) >= 43 for s in states)


# --- Loopback callback -------------------------------------------------------------


class Hits:
    """GETs each path on the callback server from another thread, like a browser."""

    def __init__(self, callback: LoopbackCallback, *paths: str) -> None:
        self.statuses: list[int] = []
        base = callback.redirect_uri.rstrip("/")
        self._thread = threading.Thread(target=self._go, args=(base, paths))
        self._thread.start()

    def _go(self, base: str, paths: tuple[str, ...]) -> None:
        for path in paths:
            self.statuses.append(httpx.get(base + path, timeout=5).status_code)

    def join(self) -> list[int]:
        self._thread.join(timeout=5)
        return self.statuses


def test_redirect_uri_is_loopback_with_a_random_port() -> None:
    with LoopbackCallback("s") as callback:
        parts = urlsplit(callback.redirect_uri)
        assert (parts.scheme, parts.hostname, parts.path) == ("http", "127.0.0.1", "/")
        assert parts.port is not None and parts.port > 0


def test_callback_with_the_right_state_returns_the_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = make_state()
    with LoopbackCallback(state) as callback:
        hits = Hits(callback, "/favicon.ico", "/?" + urlencode({"code": CODE, "state": state}))
        assert callback.wait(timeout=5) == CODE
        assert hits.join() == [404, 200]
    captured = capsys.readouterr()
    # http.server logs request lines (with the code) to stderr unless silenced.
    assert_no_secrets(captured.out, captured.err)


@pytest.mark.parametrize(
    "query",
    [
        {"code": CODE, "state": "attacker-state"},
        {"code": CODE},
        {"code": CODE, "state": ["right", "right"]},
    ],
    ids=["wrong state", "no state", "repeated state"],
)
def test_callback_without_our_state_ends_the_login(query: dict[str, Any]) -> None:
    with LoopbackCallback("right") as callback:
        hits = Hits(callback, "/?" + urlencode(query, doseq=True))
        with pytest.raises(CallbackError, match="state") as excinfo:
            callback.wait(timeout=5)
        assert hits.join() == [400]
    assert CODE not in str(excinfo.value)


def test_callback_reports_the_providers_error_code_only() -> None:
    query = {"state": "s", "error": "access_denied", "error_description": "<script>x</script>"}
    with LoopbackCallback("s") as callback:
        hits = Hits(callback, "/?" + urlencode(query))
        with pytest.raises(CallbackError, match="access_denied") as excinfo:
            callback.wait(timeout=5)
        hits.join()
    assert "script" not in str(excinfo.value)


def test_callback_times_out() -> None:
    with LoopbackCallback("s") as callback, pytest.raises(CallbackError, match="timed out"):
        callback.wait(timeout=0.2)


# --- Login ---------------------------------------------------------------------


class Browser:
    """Plays the user's browser: checks the authorization URL, then hits the redirect."""

    def __init__(self, *, state_override: str | None = None) -> None:
        self.state_override = state_override
        self.params: dict[str, str] = {}
        self.thread: threading.Thread | None = None

    def __call__(self, url: str) -> bool:
        assert url.startswith(AUTH_URL + "?")
        self.params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        state = self.state_override or self.params["state"]
        redirect = urlsplit(self.params["redirect_uri"])
        query = urlencode({"code": CODE, "state": state})
        # http.client, not httpx: httpx would log this URL, code included, into caplog.
        self.thread = threading.Thread(target=self._visit, args=(redirect.port, f"/?{query}"))
        self.thread.start()
        return True

    @staticmethod
    def _visit(port: int | None, path: str) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            conn.getresponse().read()
        finally:
            conn.close()

    def join(self) -> None:
        assert self.thread is not None
        self.thread.join(timeout=5)


def test_login_runs_pkce_and_stores_the_tokens(
    servers: FakeServers,
    environ: dict[str, str],
    store: TokenStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    browser = Browser()
    result = run_cli(servers, environ, "login", open_browser=browser)
    browser.join()

    assert result.code == EXIT_OK, result.err
    params = browser.params
    assert params["response_type"] == "code"
    assert params["client_id"] == "ragmt-cli"
    assert params["code_challenge_method"] == "S256"
    assert urlsplit(params["redirect_uri"]).hostname == "127.0.0.1"

    (token_request,) = servers.to(TOKEN_URL)
    form = _form(token_request)
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == CODE
    assert form["client_id"] == "ragmt-cli"
    assert form["redirect_uri"] == params["redirect_uri"]
    assert challenge_s256(form["code_verifier"]) == params["code_challenge"]
    assert "client_secret" not in form

    tokens = store.load()
    assert tokens is not None
    assert (tokens.access_token, tokens.refresh_token, tokens.issuer) == (ACCESS, REFRESH, ISSUER)
    assert_no_secrets(result.out, result.err, caplog.text)


def test_login_with_a_forged_callback_stores_nothing(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    browser = Browser(state_override="forged")
    result = run_cli(servers, environ, "login", open_browser=browser)
    browser.join()

    assert result.code == EXIT_ERROR
    assert "state" in result.err
    assert servers.to(TOKEN_URL) == []
    assert store.load() is None


def test_login_refuses_a_discovery_document_for_another_issuer(
    servers: FakeServers, environ: dict[str, str]
) -> None:
    servers.discovery = lambda r: httpx.Response(
        200, json={"issuer": "https://evil.example", "token_endpoint": "https://evil.example/t"}
    )
    result = run_cli(servers, environ, "login")
    assert result.code == EXIT_ERROR
    assert "different issuer" in result.err
    assert servers.to(TOKEN_URL) == []


def test_logout_deletes_the_token_file(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored())
    assert run_cli(servers, environ, "logout").out.strip() == "Logged out."
    assert not store.path.exists()
    assert run_cli(servers, environ, "logout").out.strip() == "Not logged in."


# --- Token file -------------------------------------------------------------------


def test_token_file_round_trips(store: TokenStore) -> None:
    tokens = stored()
    store.save(tokens)
    assert store.load() == tokens
    assert list(store.path.parent.iterdir()) == [store.path]  # no temp file left behind


@posix_only
def test_token_file_is_readable_by_the_user_only(store: TokenStore) -> None:
    store.save(stored())
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


@posix_only
def test_token_file_stays_0600_under_a_permissive_umask(store: TokenStore) -> None:
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{}")
    os.chmod(store.path, 0o644)
    old = os.umask(0)
    try:
        store.save(stored())
    finally:
        os.umask(old)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_token_set_repr_hides_the_tokens() -> None:
    assert_no_secrets(repr(stored()))


# --- Using and refreshing stored tokens -----------------------------------------------


def test_ask_without_login_exits_2_without_calling_the_api(
    servers: FakeServers, environ: dict[str, str]
) -> None:
    result = run_cli(servers, environ, "ask", "What is the expense policy?")
    assert result.code == EXIT_NOT_LOGGED_IN
    assert "ragctl login" in result.err
    assert servers.requests == []


def test_ask_with_a_valid_token_sends_it(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored())
    result = run_cli(servers, environ, "ask", "What is the expense policy?")
    assert result.code == EXIT_OK, result.err
    (request,) = servers.requests  # no discovery, no refresh
    assert request.headers["Authorization"] == f"Bearer {ACCESS}"
    assert json.loads(request.content) == {"question": "What is the expense policy?"}
    assert_no_secrets(result.out, result.err)


def test_expired_access_token_is_refreshed_and_saved(
    servers: FakeServers,
    environ: dict[str, str],
    store: TokenStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    store.save(stored(expires_in=-10))
    servers.token = lambda r: httpx.Response(200, json=_token_body(NEW_ACCESS, NEW_REFRESH))

    result = run_cli(servers, environ, "ask", "q")

    assert result.code == EXIT_OK, result.err
    (refresh_request,) = servers.to(TOKEN_URL)
    assert _form(refresh_request) == {
        "grant_type": "refresh_token",
        "client_id": "ragmt-cli",
        "refresh_token": REFRESH,
    }
    (ask_request,) = servers.to(f"{API}/ask")
    assert ask_request.headers["Authorization"] == f"Bearer {NEW_ACCESS}"
    saved = store.load()
    assert saved is not None
    assert (saved.access_token, saved.refresh_token) == (NEW_ACCESS, NEW_REFRESH)
    assert_no_secrets(result.out, result.err, caplog.text)


def test_refresh_without_rotation_keeps_the_old_refresh_token(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored(expires_in=-10))
    servers.token = lambda r: httpx.Response(200, json=_token_body(NEW_ACCESS, refresh=None))
    assert run_cli(servers, environ, "ask", "q").code == EXIT_OK
    saved = store.load()
    assert saved is not None
    assert (saved.access_token, saved.refresh_token) == (NEW_ACCESS, REFRESH)


def test_rejected_refresh_means_not_logged_in(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored(expires_in=-10))
    servers.token = lambda r: httpx.Response(
        400, json={"error": "invalid_grant", "error_description": f"bad {REFRESH}"}
    )
    result = run_cli(servers, environ, "ask", "q")
    assert result.code == EXIT_NOT_LOGGED_IN
    assert "expired" in result.err
    assert servers.to(f"{API}/ask") == []
    assert_no_secrets(result.err)


def test_expired_refresh_token_is_not_sent(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored(expires_in=-10, refresh_in=-10))
    assert run_cli(servers, environ, "ask", "q").code == EXIT_NOT_LOGGED_IN
    assert servers.requests == []


def test_identity_provider_down_during_refresh_is_an_error_not_a_logout(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored(expires_in=-10))
    servers.token = lambda r: httpx.Response(503)
    assert run_cli(servers, environ, "ask", "q").code == EXIT_ERROR
    assert store.load() is not None


def test_tokens_for_another_issuer_are_not_used(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored())
    result = run_cli(servers, environ, "--issuer", "https://sso.example/realms/x", "ask", "q")
    assert result.code == EXIT_NOT_LOGGED_IN
    assert servers.requests == []


def test_malformed_token_file_means_not_logged_in(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.path.parent.mkdir(parents=True)
    store.path.write_text("not json")
    assert run_cli(servers, environ, "ask", "q").code == EXIT_NOT_LOGGED_IN


# --- ask: API responses and output ------------------------------------------------------


def test_ask_prints_the_answer_and_its_citations(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored())
    result = run_cli(servers, environ, "ask", "q")
    assert result.code == EXIT_OK
    assert result.out == (
        f"Expenses are reimbursed monthly [doc:{DOC}#0].\n"
        "\n"
        "Sources:\n"
        "  [1] Expense policy > Reimbursement\n"
        "  [2] Travel guide\n"
    )


def test_the_cli_reads_what_the_api_returns() -> None:
    """The API's own response model, serialized, parses as the CLI's. The CLI
    doesn't import the server side; this test does, so the two can't drift."""
    from ragmt.api.ask import AskResponse as ApiAskResponse
    from ragmt.api.ask import CitationOut as ApiCitationOut

    sent = ApiAskResponse(
        answer=f"Yes [doc:{DOC}#0].",
        citations=[ApiCitationOut(document_id=DOC, title="Expense policy", heading=None)],
        model="llama3.1:8b",
    )
    received = AskResponse.model_validate_json(sent.model_dump_json())
    assert received.answer == sent.answer
    assert received.citations == [CitationOut(document_id=DOC, title="Expense policy")]
    assert received.model == "llama3.1:8b"


def test_format_answer_without_citations() -> None:
    answer = AskResponse(answer="I don't know.", citations=[], model=None)
    assert format_answer(answer) == "I don't know."


def test_format_answer_strips_terminal_control_characters() -> None:
    answer = AskResponse(
        answer="ok\x1b[2J\x1b]0;pwned\x07\r\nline two‮",
        citations=[CitationOut(document_id=DOC, title="T\x1b[31mitle\nx", heading="H\tead")],
    )
    text = format_answer(answer)
    assert "\x1b" not in text and "\x07" not in text and "‮" not in text and "\r" not in text
    assert text.startswith("ok[2J]0;pwned\nline two")
    assert text.endswith("  [1] T[31mitle x > H ead")


@pytest.mark.parametrize(
    ("response", "code", "message"),
    [
        (httpx.Response(401, json={"detail": "Not authenticated"}), EXIT_NOT_LOGGED_IN, "login"),
        (
            httpx.Response(422, json={"detail": "question too long"}),
            EXIT_ERROR,
            "question too long",
        ),
        (httpx.Response(500, text="Internal Server Error"), EXIT_ERROR, "HTTP 500"),
        (httpx.Response(503, json={"detail": [{"loc": "x"}]}), EXIT_ERROR, "HTTP 503"),
        (httpx.Response(200, json={"text": "wrong shape"}), EXIT_ERROR, "unexpected"),
    ],
)
def test_ask_api_errors(
    servers: FakeServers,
    environ: dict[str, str],
    store: TokenStore,
    response: httpx.Response,
    code: int,
    message: str,
) -> None:
    store.save(stored())
    servers.ask = lambda r: response
    result = run_cli(servers, environ, "ask", "q")
    assert result.code == code
    assert message in result.err
    assert result.out == ""
    assert_no_secrets(result.err)


def test_ask_api_unreachable(
    servers: FakeServers, environ: dict[str, str], store: TokenStore
) -> None:
    store.save(stored())

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    servers.ask = down
    result = run_cli(servers, environ, "ask", "q")
    assert result.code == EXIT_ERROR
    assert "could not reach the API" in result.err


def test_ask_rejects_an_empty_question(servers: FakeServers, environ: dict[str, str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(servers, environ, "ask", "   ")
    assert excinfo.value.code == 2


# --- --dev-user (DEV ONLY) ----------------------------------------------------------------


def test_dev_user_uses_the_password_grant_and_stores_nothing(
    servers: FakeServers,
    environ: dict[str, str],
    store: TokenStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    environ["KC_DEMO_USER_PASSWORD"] = DEMO_CREDENTIAL
    result = run_cli(servers, environ, "--dev-user", "alice", "ask", "q")

    assert result.code == EXIT_OK, result.err
    (token_request,) = servers.to(TOKEN_URL)
    form = _form(token_request)
    assert form["grant_type"] == "password"
    assert form["client_id"] == "ragmt-dev-password"
    assert (form["username"], form["password"]) == ("alice", DEMO_CREDENTIAL)
    (ask_request,) = servers.to(f"{API}/ask")
    assert ask_request.headers["Authorization"] == f"Bearer {ACCESS}"
    assert store.load() is None
    assert_no_secrets(result.out, result.err, caplog.text)


def test_dev_user_without_a_password_exits_2(servers: FakeServers, environ: dict[str, str]) -> None:
    result = run_cli(servers, environ, "--dev-user", "alice", "ask", "q")
    assert result.code == EXIT_NOT_LOGGED_IN
    assert "KC_DEMO_USER_PASSWORD" in result.err
    assert servers.requests == []


def test_dev_user_rejected_exits_2(servers: FakeServers, environ: dict[str, str]) -> None:
    environ["KC_DEMO_USER_PASSWORD"] = DEMO_CREDENTIAL
    servers.token = lambda r: httpx.Response(401, json={"error": "unauthorized_client"})
    result = run_cli(servers, environ, "--dev-user", "alice", "ask", "q")
    assert result.code == EXIT_NOT_LOGGED_IN
    assert "unauthorized_client" in result.err
    assert "KC_DEV_PASSWORD_CLIENT" in result.err
    assert_no_secrets(result.err)


def test_dev_user_is_only_for_ask(servers: FakeServers, environ: dict[str, str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(servers, environ, "--dev-user", "alice", "login")
    assert excinfo.value.code == 2


def test_help_marks_dev_user_as_dev_only() -> None:
    # argparse wraps lines, also at hyphens: compare without whitespace.
    help_text = "".join(build_parser().format_help().split())
    assert "--dev-userUSERDEVONLY" in help_text
    assert "ragmt-dev-password" in help_text


# --- Settings ------------------------------------------------------------------------


def test_flags_beat_the_environment_which_beats_the_defaults() -> None:
    env = {"RAGMT_API_URL": "https://api.example/", "OIDC_ISSUER": "https://sso.example/realms/r"}
    assert CliConfig.resolve(api_url=None, issuer=None, environ={}) == CliConfig(
        api_url="http://127.0.0.1:8000", issuer="http://localhost:8080/realms/ragmt"
    )
    assert CliConfig.resolve(api_url=None, issuer=None, environ=env) == CliConfig(
        api_url="https://api.example", issuer="https://sso.example/realms/r"
    )
    assert CliConfig.resolve(
        api_url="http://localhost:9000", issuer="https://other.example/realms/r", environ=env
    ) == CliConfig(api_url="http://localhost:9000", issuer="https://other.example/realms/r")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example",  # plain http off this machine
        "ftp://127.0.0.1",
        "127.0.0.1:8000",
        "https://user:pw@api.example",
        "https://api.example/?x=1",
    ],
)
def test_unsafe_urls_are_refused(url: str) -> None:
    with pytest.raises(ConfigError):
        CliConfig.resolve(api_url=url, issuer=None, environ={})


def test_unsafe_url_flag_is_a_usage_error(servers: FakeServers, environ: dict[str, str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(servers, environ, "--api-url", "http://api.example", "ask", "q")
    assert excinfo.value.code == 2


def test_config_dir_override(tmp_path: Path) -> None:
    assert config_dir({"RAGCTL_CONFIG_DIR": str(tmp_path)}) == tmp_path
