"""Measure the revocation window: how long a user keeps reading a group's
document after leaving the group in Keycloak (ADR 0002).

`python -m eval.revocation` (or `make revocation`). Needs the real stack, as the
e2e tests do: Keycloak with KC_DEV_PASSWORD_CLIENT=true, PostgreSQL migrated,
and the bootstrap admin's credentials in the environment or .env. Ollama is not
needed: the API is started with fake providers, since GET /documents embeds
nothing. The compose `permsync` service must be stopped, or it would revoke
first.

Option C (ours: identity from the token, groups from `memberships`). For each
interval in --intervals, permsync runs as a background process with that
PERMSYNC_INTERVAL_SECONDS. Each run:

1. waits a random delay in [0, interval), so the revocation lands at a random
   point of the sync cycle (otherwise it would always follow the cycle that
   restored access, and every window would be close to the maximum);
2. gets a token for alice and checks GET /documents lists the Q3 budget;
3. removes alice from /acme/finance through the admin API; t0 is when that
   call returns, so Keycloak has committed the change;
4. polls GET /documents with the same token every 100 ms until the budget is
   gone (t1);
5. puts alice back in the group and polls until the budget is listed again.

Each window is cross-checked against permsync's `revoked ... sub=<alice>
group=finance` log line (UTC ms, written after the commit).

Option A (groups from the token, not implemented in the API). A temporary
DEV-only client adds a `groups` claim to dave's tokens. Each run gets a token,
waits a random part of its lifetime, removes dave from /umbra/finance (t0), and
polls the API with the old token every second until it is rejected: until then,
a resource server that trusted the claim would still grant the group. The claim
itself never changes (a JWT is immutable); a token issued after t0 no longer
lists the group, which is also checked.

Raw runs and permsync logs go to docs/results/raw/ (git-ignored). The summary
tables are printed; docs/results/revocation.md is written from them by hand.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.adapters.llm import FakeEmbeddings
from ragmt.auth.tokens import LEEWAY_SECONDS
from ragmt.permsync.__main__ import run as run_permsync
from ragmt.settings import Settings
from seed.corpus import ALICE, DAVE
from seed.load import load as load_seed

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "docs" / "results" / "raw"
REALM = "ragmt"
DEV_CLIENT = "ragmt-dev-password"
GROUPS_CLIENT = "ragmt-eval-groups-claim"
_TIMEOUT = 10.0

POLL_C_SECONDS = 0.1
POLL_A_SECONDS = 1.0
# permsync runs during option A too, so the same old token shows option C's window.
A_PERMSYNC_INTERVAL = 10


@dataclass(frozen=True)
class Subject:
    username: str
    sub: str
    group_path: str
    group: str
    title: str


C_SUBJECT = Subject("alice", ALICE, "/acme/finance", "finance", "Q3 budget")
# Another tenant, so option A can run alongside option C without touching alice.
A_SUBJECT = Subject("dave", DAVE, "/umbra/finance", "finance", "Annual budget")


def load_dotenv(path: Path) -> None:
    """KEY=VALUE lines from .env, without overriding what is already set (as tests/conftest.py)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set (see .env.example)")
    return value


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds")


def claims(token: str) -> dict[str, Any]:
    part = token.split(".")[1]
    payload: dict[str, Any] = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    return payload


# --- Keycloak --------------------------------------------------------------------


class Keycloak:
    """User tokens from the dev password grant, and the admin API as the bootstrap admin."""

    def __init__(self, issuer: str) -> None:
        self.issuer = issuer
        self.base_url = issuer.rsplit("/realms/", 1)[0]
        self._admin = httpx.Client(
            base_url=f"{self.base_url}/admin/realms/{REALM}/", timeout=_TIMEOUT
        )
        self._admin_token = ""
        self._admin_token_expires = 0.0
        self._group_ids: dict[str, str] = {}

    def user_token(self, username: str, client_id: str = DEV_CLIENT) -> str:
        response = httpx.post(
            f"{self.issuer}/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": client_id,
                "username": username,
                "password": env("KC_DEMO_USER_PASSWORD"),
            },
            timeout=_TIMEOUT,
        )
        if response.status_code != httpx.codes.OK:
            sys.exit(
                f"no token for {username} from {client_id}: HTTP {response.status_code}. "
                "Is KC_DEV_PASSWORD_CLIENT=true, and was the realm imported after setting it?"
            )
        token: str = response.json()["access_token"]
        return token

    def _headers(self) -> dict[str, str]:
        # admin-cli tokens last 60 s; a run at a 60 s interval outlives one.
        if time.monotonic() > self._admin_token_expires:
            response = httpx.post(
                f"{self.base_url}/realms/master/protocol/openid-connect/token",
                data={
                    "grant_type": "password",
                    "client_id": "admin-cli",
                    "username": env("KC_BOOTSTRAP_ADMIN_USERNAME"),
                    "password": env("KC_BOOTSTRAP_ADMIN_PASSWORD"),
                },
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            body = response.json()
            self._admin_token = body["access_token"]
            self._admin_token_expires = time.monotonic() + body["expires_in"] - 15
        return {"Authorization": f"Bearer {self._admin_token}"}

    def _group_id(self, path: str) -> str:
        if path not in self._group_ids:
            response = self._admin.get(f"group-by-path/{path.strip('/')}", headers=self._headers())
            response.raise_for_status()
            self._group_ids[path] = response.json()["id"]
        return self._group_ids[path]

    def leave_group(self, user_id: str, path: str) -> None:
        gid = self._group_id(path)
        headers = self._headers()
        self._admin.delete(f"users/{user_id}/groups/{gid}", headers=headers).raise_for_status()

    def join_group(self, user_id: str, path: str) -> None:
        gid = self._group_id(path)
        headers = self._headers()
        self._admin.put(f"users/{user_id}/groups/{gid}", headers=headers).raise_for_status()

    @contextmanager
    def groups_claim_client(self) -> Iterator[str]:
        """A DEV-only client whose tokens carry `groups` (full paths): option A's token.

        Same scopes and audience as ragmt-dev-password, plus the group mapper the
        real realm leaves out on purpose (ADR 0002). Deleted afterwards.
        """
        representation = {
            "clientId": GROUPS_CLIENT,
            "name": "DEV ONLY - eval/revocation.py, deleted afterwards",
            "publicClient": True,
            "standardFlowEnabled": False,
            "directAccessGrantsEnabled": True,
            "fullScopeAllowed": False,
            "defaultClientScopes": ["basic", "profile", "organization"],
            "protocolMappers": [
                {
                    "name": "ragmt-api audience",
                    "protocol": "openid-connect",
                    "protocolMapper": "oidc-audience-mapper",
                    "config": {
                        "included.custom.audience": "ragmt-api",
                        "access.token.claim": "true",
                    },
                },
                {
                    "name": "groups",
                    "protocol": "openid-connect",
                    "protocolMapper": "oidc-group-membership-mapper",
                    "config": {
                        "claim.name": "groups",
                        "full.path": "true",
                        "access.token.claim": "true",
                    },
                },
            ],
        }
        stale = self._admin.get(
            "clients", params={"clientId": GROUPS_CLIENT}, headers=self._headers()
        ).json()
        for client in stale:  # left over from an interrupted run
            self._admin.delete(f"clients/{client['id']}", headers=self._headers())
        response = self._admin.post("clients", json=representation, headers=self._headers())
        response.raise_for_status()
        internal_id = response.headers["Location"].rsplit("/", 1)[1]
        try:
            yield GROUPS_CLIENT
        finally:
            self._admin.delete(f"clients/{internal_id}", headers=self._headers())

    def close(self) -> None:
        self._admin.close()


# --- the stack ----------------------------------------------------------------------


def check_compose_permsync_stopped() -> None:
    if shutil.which("docker") is None:
        print("warning: docker not found; make sure no other permsync is running")
        return
    result = subprocess.run(
        ["docker", "compose", "ps", "--status", "running", "-q", "permsync"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout.strip():
        sys.exit(
            "the compose permsync service is running and would revoke before ours: "
            "`docker compose stop permsync` first"
        )


def seed_and_sync() -> None:
    """The seed corpus (idempotent) and one permsync cycle, as the e2e fixtures do."""

    async def go() -> None:
        settings = Settings()
        engine = create_async_engine(env("INGEST_DATABASE_URL"))
        try:
            await load_seed(engine, FakeEmbeddings(settings.embedding_dim), settings)
        finally:
            await engine.dispose()
        if await run_permsync(settings, once=True) != 0:
            sys.exit("permsync cycle failed")

    asyncio.run(go())


def sync_once() -> None:
    subprocess.run([sys.executable, "-m", "ragmt.permsync", "--once"], cwd=ROOT, check=False)


# GET /documents embeds nothing, so the revocation runs need no Ollama.
FAKE_PROVIDERS = {"LLM_PROVIDER": "fake", "CHAT_PROVIDER": "fake"}


@contextmanager
def api_server(
    log_path: Path, overrides: Mapping[str, str] = FAKE_PROVIDERS, startup_seconds: float = 60
) -> Iterator[str]:
    """The API at EVAL_API_URL, or a uvicorn process on a free port with `overrides`
    on top of the environment (fake providers by default)."""
    url = os.environ.get("EVAL_API_URL")
    if url:
        yield url.rstrip("/")
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with log_path.open("wb") as out:
        process = subprocess.Popen(  # noqa: S603 -- fixed command, this interpreter
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--factory",
                "ragmt.api.app:create_app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**os.environ, **overrides},
        )
    try:
        deadline = time.monotonic() + startup_seconds
        while True:
            if process.poll() is not None:
                sys.exit(f"the API exited with status {process.returncode}; see {log_path}")
            try:
                if httpx.get(f"{url}/healthz", timeout=1).status_code == httpx.codes.OK:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                sys.exit(f"the API did not become healthy; see {log_path}")
            time.sleep(0.2)
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


@contextmanager
def permsync_process(interval: int, log_path: Path) -> Iterator[Path]:
    """`python -m ragmt.permsync` with this interval, logging to log_path.

    Yields once the first cycle has finished, so memberships match the realm.
    """
    out = log_path.open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "ragmt.permsync"],
        cwd=ROOT,
        stdout=out,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PERMSYNC_INTERVAL_SECONDS": str(interval), "PYTHONUNBUFFERED": "1"},
    )
    try:
        deadline = time.monotonic() + 60
        while "cycle finished" not in log_path.read_text(errors="replace"):
            if process.poll() is not None or time.monotonic() > deadline:
                sys.exit(f"permsync did not complete a cycle; see {log_path}")
            time.sleep(0.2)
        yield log_path
    finally:
        process.terminate()
        process.wait(timeout=10)
        out.close()


def lists(http: httpx.Client, url: str, token: str, title: str) -> bool:
    response = http.get(f"{url}/documents", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return any(item["title"] == title for item in response.json())


def hidden(http: httpx.Client, url: str, token: str, title: str) -> bool:
    return not lists(http, url, token, title)


def delay(upper: float) -> float:
    """A random wait in [0, upper): timing jitter, not security."""
    return random.uniform(0, upper)  # noqa: S311 -- not a secret


def poll_until(predicate: Callable[[], bool], period: float, timeout: float, what: str) -> float:
    """Call predicate every `period` s until it is true; return the wall time it was."""
    deadline = time.monotonic() + timeout
    while True:
        started = time.monotonic()
        if predicate():
            return time.time()
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} did not happen within {timeout:.0f} s")
        time.sleep(max(0.0, period - (time.monotonic() - started)))


# --- option C -----------------------------------------------------------------------

_REVOKED = re.compile(
    r"^(?P<ts>\S+)Z INFO ragmt\.permsync\.sync revoked tenant=\S+ sub=(?P<sub>\S+) "
    r"group=(?P<group>\S+)$"
)


def revoked_log_times(log_path: Path, sub: str, group: str) -> list[float]:
    times = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = _REVOKED.match(line.strip())
        if match and match["sub"] == sub and match["group"] == group:
            ts = datetime.fromisoformat(match["ts"]).replace(tzinfo=UTC)
            times.append(ts.timestamp())
    return times


@dataclass
class CRun:
    interval: int
    run: int
    jitter_s: float
    t0: str
    t1: str
    window_s: float
    log_revoked: str | None = None
    log_window_s: float | None = None
    restored_after_s: float = 0.0


def measure_c(
    keycloak: Keycloak, http: httpx.Client, url: str, interval: int, runs: int, raw: Path
) -> list[CRun]:
    subject = C_SUBJECT
    timeout = 3 * interval + 30
    log_path = raw / f"permsync-{interval}s.log"
    results: list[CRun] = []
    with permsync_process(interval, log_path):
        token = keycloak.user_token(subject.username)
        if not lists(http, url, token, subject.title):
            sys.exit(f"{subject.username} cannot see {subject.title!r} before the first run")
        for n in range(1, runs + 1):
            jitter = delay(interval)
            time.sleep(jitter)
            token = keycloak.user_token(subject.username)
            if not lists(http, url, token, subject.title):
                sys.exit(f"run {n}: {subject.username} lost {subject.title!r} before revoking")

            keycloak.leave_group(subject.sub, subject.group_path)
            t0 = time.time()
            try:
                t1 = poll_until(
                    partial(hidden, http, url, token, subject.title),
                    POLL_C_SECONDS,
                    timeout,
                    "revocation",
                )
            finally:
                keycloak.join_group(subject.sub, subject.group_path)
            rejoined = time.time()
            fresh = keycloak.user_token(subject.username)
            restored = poll_until(
                partial(lists, http, url, fresh, subject.title),
                POLL_C_SECONDS,
                timeout,
                "restoring access",
            )
            result = CRun(
                interval=interval,
                run=n,
                jitter_s=round(jitter, 3),
                t0=utc(t0),
                t1=utc(t1),
                window_s=round(t1 - t0, 3),
                restored_after_s=round(restored - rejoined, 3),
            )
            results.append(result)
            print(f"  C {interval:>3}s run {n:>2}: t1-t0 = {result.window_s:7.3f} s", flush=True)

    # Each run's revocation is the first `revoked` line after its t0.
    log_times = revoked_log_times(log_path, subject.sub, subject.group)
    for result in results:
        t0 = datetime.fromisoformat(result.t0).timestamp()
        after = [ts for ts in log_times if ts >= t0]
        if after:
            result.log_revoked = utc(after[0])
            result.log_window_s = round(after[0] - t0, 3)
    return results


# --- option A -----------------------------------------------------------------------


@dataclass
class ARun:
    run: int
    token_age_at_t0_s: float
    t0: str
    exp: str
    rejected: str
    window_s: float
    old_token_groups: list[str] = field(default_factory=list)
    fresh_token_groups: list[str] = field(default_factory=list)
    # When our API (option C) stopped listing the document for the same old token.
    c_hidden_after_s: float | None = None


def measure_a(keycloak: Keycloak, http: httpx.Client, url: str, runs: int, raw: Path) -> list[ARun]:
    subject = A_SUBJECT
    results: list[ARun] = []
    with (
        permsync_process(A_PERMSYNC_INTERVAL, raw / "permsync-a.log"),
        keycloak.groups_claim_client() as client_id,
    ):
        for n in range(1, runs + 1):
            token = keycloak.user_token(subject.username, client_id)
            issued = claims(token)
            lifetime = issued["exp"] - issued["iat"]
            if subject.group_path not in issued.get("groups", []):
                sys.exit(f"run {n}: the token's groups claim lacks {subject.group_path}")
            # Revoke at a random point of the token's life.
            time.sleep(delay(lifetime))
            keycloak.leave_group(subject.sub, subject.group_path)
            t0 = time.time()
            c_hidden: float | None = None
            try:
                fresh = keycloak.user_token(subject.username, client_id)
                headers = {"Authorization": f"Bearer {token}"}
                deadline = time.monotonic() + lifetime + LEEWAY_SECONDS + 30
                while True:
                    started = time.monotonic()
                    response = http.get(f"{url}/documents", headers=headers)
                    if response.status_code == httpx.codes.UNAUTHORIZED:
                        rejected = time.time()
                        break
                    response.raise_for_status()
                    titles = {item["title"] for item in response.json()}
                    if c_hidden is None and subject.title not in titles:
                        c_hidden = time.time() - t0
                    if time.monotonic() > deadline:
                        sys.exit(f"run {n}: the old token was never rejected")
                    time.sleep(max(0.0, POLL_A_SECONDS - (time.monotonic() - started)))
            finally:
                keycloak.join_group(subject.sub, subject.group_path)
            result = ARun(
                run=n,
                token_age_at_t0_s=round(t0 - issued["iat"], 3),
                t0=utc(t0),
                exp=utc(issued["exp"]),
                rejected=utc(rejected),
                window_s=round(rejected - t0, 3),
                old_token_groups=claims(token).get("groups", []),
                fresh_token_groups=claims(fresh).get("groups", []),
                c_hidden_after_s=None if c_hidden is None else round(c_hidden, 3),
            )
            results.append(result)
            print(
                f"  A run {n}: token age {result.token_age_at_t0_s:6.1f} s at t0, "
                f"old token accepted for {result.window_s:6.1f} s after t0",
                flush=True,
            )
    return results


# --- summary ------------------------------------------------------------------------


def nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def stats(values: list[float]) -> str:
    if not values:
        return "| - | - | - | - |"
    cols = (min(values), nearest_rank(values, 0.5), nearest_rank(values, 0.95), max(values))
    return "| " + " | ".join(f"{v:.2f}" for v in cols) + " |"


def summary_c(results: list[CRun]) -> str:
    lines = [
        "| Interval (s) | Runs | min | median | p95 | max | log: median | log: max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for interval in sorted({r.interval for r in results}, reverse=True):
        rows = [r for r in results if r.interval == interval]
        observed = [r.window_s for r in rows]
        logged = [r.log_window_s for r in rows if r.log_window_s is not None]
        log_cols = (
            f"| {nearest_rank(logged, 0.5):.2f} | {max(logged):.2f} |" if logged else "| - | - |"
        )
        lines.append(f"| {interval} | {len(rows)} {stats(observed)}{log_cols[1:]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.revocation", description=__doc__)
    parser.add_argument("--option", choices=["c", "a", "both"], default="both")
    parser.add_argument("--runs", type=int, default=20, help="runs per interval (option C)")
    parser.add_argument(
        "--intervals", default="60,10,5", help="PERMSYNC_INTERVAL_SECONDS values, comma-separated"
    )
    parser.add_argument("--a-runs", type=int, default=5, help="runs of option A (~5 min each)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the delays")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    os.chdir(ROOT)  # Settings reads .env from the working directory
    random.seed(args.seed)
    check_compose_permsync_stopped()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw = RAW_DIR / f"revocation-{stamp}"
    raw.mkdir(parents=True, exist_ok=True)
    keycloak = Keycloak(env("OIDC_ISSUER"))
    print(f"seeding and syncing once; raw output in {raw}", flush=True)
    seed_and_sync()

    c_results: list[CRun] = []
    a_results: list[ARun] = []
    try:
        with api_server(raw / "api.log") as url, httpx.Client(timeout=_TIMEOUT) as http:
            if args.option in ("c", "both"):
                for interval in (int(i) for i in args.intervals.split(",")):
                    print(f"option C, PERMSYNC_INTERVAL_SECONDS={interval}", flush=True)
                    c_results += measure_c(keycloak, http, url, interval, args.runs, raw)
            if args.option in ("a", "both"):
                print("option A, groups claim in the token", flush=True)
                a_results = measure_a(keycloak, http, url, args.a_runs, raw)
    finally:
        # Whatever happened, the users measured end up back in their groups, and
        # memberships agree with the realm.
        if args.option in ("c", "both"):
            keycloak.join_group(C_SUBJECT.sub, C_SUBJECT.group_path)
        if args.option in ("a", "both"):
            keycloak.join_group(A_SUBJECT.sub, A_SUBJECT.group_path)
        keycloak.close()
        sync_once()
        (raw / "runs.json").write_text(
            json.dumps(
                {
                    "started": stamp,
                    "c": [asdict(r) for r in c_results],
                    "a": [asdict(r) for r in a_results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if c_results:
        print("\nOption C: t1 - t0 (s), observed through the API; log = permsync `revoked` - t0")
        print(summary_c(c_results))
    if a_results:
        print("\nOption A: seconds the pre-revocation token was still accepted after t0")
        print("| Runs | min | median | p95 | max |\n|---:|---:|---:|---:|---:|")
        print(f"| {len(a_results)} {stats([r.window_s for r in a_results])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
