"""The loopback redirect target for `ragctl login` (RFC 8252, section 7.3).

A one-shot HTTP server on 127.0.0.1 and a port the OS picks. It waits for the
browser to come back with `?code=...&state=...`, checks `state` in constant
time and returns the code. A callback with the wrong state, or with an
`error` from the authorization server, ends the login. Requests to other paths
(a browser asking for /favicon.ico) are ignored.

The server never logs request lines: they carry the authorization code.
"""

import re
import secrets
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

_DEFAULT_TIMEOUT_SECONDS = 300.0
# An OAuth error code is printable ASCII (RFC 6749, 4.1.2.1); show only a safe subset.
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ragctl</title></head>
<body style="font-family: sans-serif"><p>{message}</p></body></html>"""


class CallbackError(RuntimeError):
    """The login can't finish from what came back to the loopback address."""


@dataclass(frozen=True, slots=True)
class _Outcome:
    code: str | None = None
    error: str | None = None


class _Server(HTTPServer):
    expected_state: str
    outcome: _Outcome | None = None


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        if parts.path != "/":
            self._reply(HTTPStatus.NOT_FOUND, "Not found.")
            return
        params = parse_qs(parts.query)
        state = _single(params, "state")
        if state is None or not secrets.compare_digest(
            state.encode("utf-8"), self.server.expected_state.encode("utf-8")
        ):
            # Checked before anything else in the query is looked at: a request
            # without our state is not an answer to our authorization request.
            self.server.outcome = _Outcome(error="the login callback had the wrong state")
            self._reply(HTTPStatus.BAD_REQUEST, "Login failed: unexpected callback.")
            return
        error = _single(params, "error")
        if error is not None:
            code = error if _ERROR_CODE.fullmatch(error) else "invalid error code"
            self.server.outcome = _Outcome(error=f"the identity provider refused: {code}")
            self._reply(HTTPStatus.OK, "Login failed. You can close this tab.")
            return
        auth_code = _single(params, "code")
        if not auth_code:
            self.server.outcome = _Outcome(error="the login callback had no code")
            self._reply(HTTPStatus.BAD_REQUEST, "Login failed: no code.")
            return
        self.server.outcome = _Outcome(code=auth_code)
        self._reply(HTTPStatus.OK, "Logged in. You can close this tab and return to ragctl.")

    def _reply(self, status: HTTPStatus, message: str) -> None:
        body = _PAGE.format(message=message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # The default writes the request line, with the code, to stderr.
        return


def _single(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values or len(values) != 1:
        return None
    return values[0]


class LoopbackCallback:
    """Listens on 127.0.0.1:<random port> for one authorization response."""

    def __init__(self, state: str, *, host: str = "127.0.0.1") -> None:
        self._server = _Server((host, 0), _Handler)
        self._server.expected_state = state

    @property
    def redirect_uri(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/"

    def wait(self, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> str:
        """The authorization code. Raises `CallbackError` on a bad callback or timeout."""
        deadline = time.monotonic() + timeout
        while self._server.outcome is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CallbackError("timed out waiting for the browser login")
            self._server.timeout = remaining
            self._server.handle_request()
        outcome = self._server.outcome
        if outcome.code is None:
            raise CallbackError(outcome.error or "login failed")
        return outcome.code

    def close(self) -> None:
        self._server.server_close()

    def __enter__(self) -> "LoopbackCallback":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
