"""The `ragctl` command (console script in pyproject.toml).

Exit codes: 0 ok, 1 API or identity provider error, 2 not logged in (also
argparse's code for usage errors).
"""

import argparse
import io
import os
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO

import httpx

from ragmt.cli.api import ApiError, Unauthorized, ask, format_answer
from ragmt.cli.config import (
    DEFAULT_API_URL,
    DEFAULT_ISSUER,
    DEV_CLIENT_ID,
    DEV_CREDENTIAL_ENV,
    CliConfig,
    ConfigError,
    config_dir,
)
from ragmt.cli.oidc import (
    AuthError,
    NotLoggedIn,
    TokenRejected,
    current_tokens,
    login_interactive,
    password_grant,
)
from ragmt.cli.tokens import TokenStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_LOGGED_IN = 2

# Discovery and token requests; `ask` sets its own, longer timeout.
_HTTP_TIMEOUT = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragctl",
        description="Ask the ragmt API questions about the documents you can read.",
        epilog=(
            "exit codes: 0 ok, 1 API or identity provider error, 2 not logged in (or bad usage)"
        ),
    )
    parser.add_argument(
        "--api-url",
        help=f"API base URL (env RAGMT_API_URL, default {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--issuer",
        help=f"OIDC issuer URL (env OIDC_ISSUER, default {DEFAULT_ISSUER})",
    )
    parser.add_argument(
        "--dev-user",
        metavar="USER",
        help=(
            "DEV ONLY, never against a real deployment: ask as this seed user with a "
            f"password-grant token from the {DEV_CLIENT_ID} client (password from "
            f"{DEV_CREDENTIAL_ENV}; the client works only if the realm was imported with "
            "KC_DEV_PASSWORD_CLIENT=true). Nothing is stored. Only for `ask`."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("login", help="log in through the browser (authorization code + PKCE)")
    commands.add_parser("logout", help="delete the stored tokens")
    ask_parser = commands.add_parser("ask", help="ask a question; prints the answer and sources")
    ask_parser.add_argument("question")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    open_browser: Callable[[str], object] = webbrowser.open,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    environ = os.environ if environ is None else environ
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    def fail(message: str, code: int) -> int:
        print(f"ragctl: {message}", file=err)
        return code

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dev_user is not None and args.command != "ask":
        parser.error("--dev-user only works with `ask`")
    if args.command == "ask" and not args.question.strip():
        parser.error("the question is empty")
    try:
        config = CliConfig.resolve(api_url=args.api_url, issuer=args.issuer, environ=environ)
    except ConfigError as exc:
        parser.error(str(exc))
    store = TokenStore(config_dir(environ))

    if args.command == "logout":
        print("Logged out." if store.clear() else "Not logged in.", file=out)
        return EXIT_OK

    with httpx.Client(transport=transport, timeout=_HTTP_TIMEOUT) as client:
        if args.command == "login":
            try:
                tokens = login_interactive(
                    client, config, open_browser=open_browser, notify=lambda m: print(m, file=err)
                )
            except AuthError as exc:
                return fail(f"login failed: {exc}", EXIT_ERROR)
            store.save(tokens)
            print(f"Logged in to {config.issuer}.", file=out)
            return EXIT_OK

        # ask
        try:
            if args.dev_user is not None:
                password = environ.get(DEV_CREDENTIAL_ENV)
                if not password:
                    return fail(f"--dev-user needs {DEV_CREDENTIAL_ENV} set", EXIT_NOT_LOGGED_IN)
                try:
                    tokens = password_grant(
                        client,
                        config,
                        client_id=DEV_CLIENT_ID,
                        username=args.dev_user,
                        password=password,
                    )
                except TokenRejected as exc:
                    return fail(
                        f"dev login failed: {exc}. Is the user right, and was the realm "
                        "imported with KC_DEV_PASSWORD_CLIENT=true?",
                        EXIT_NOT_LOGGED_IN,
                    )
            else:
                tokens = current_tokens(client, config, store)
        except NotLoggedIn as exc:
            return fail(str(exc), EXIT_NOT_LOGGED_IN)
        except AuthError as exc:
            return fail(str(exc), EXIT_ERROR)

        try:
            answer = ask(client, config.api_url, tokens.access_token, args.question)
        except Unauthorized as exc:
            return fail(str(exc), EXIT_NOT_LOGGED_IN)
        except ApiError as exc:
            return fail(str(exc), EXIT_ERROR)
    print(format_answer(answer), file=out)
    return EXIT_OK


def run() -> None:
    """Console script entry point."""
    # Answers and titles can hold any character; never crash on a narrow console.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    sys.exit(main())


if __name__ == "__main__":
    run()
