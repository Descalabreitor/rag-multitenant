"""The real stack for the end-to-end tests: Keycloak, PostgreSQL and the API over HTTP.

Every test here is skipped unless Keycloak answers at OIDC_ISSUER. Then:

- User tokens come from Keycloak's password grant on the dev-only
  `ragmt-dev-password` client (KC_DEV_PASSWORD_CLIENT=true at realm import).
- The realm is changed through the admin API as the bootstrap admin
  (KC_BOOTSTRAP_ADMIN_*), never through the database.
- The API is the one at E2E_API_URL (`make e2e` starts it), or, when that is
  unset, a uvicorn process started here on a free port. Both run with
  LLM_PROVIDER=ollama, so uploads are embedded by the real model: Ollama must
  be up at OLLAMA_BASE_URL with OLLAMA_EMBED_MODEL pulled, or the tests fail
  saying what to run.
- Before the tests, the seed corpus is loaded as app_ingest (`make seed`) and
  permsync runs once, so the database agrees with the realm. The seed itself
  uses fake embeddings: these tests check access, not ranking.

The fixtures are synchronous: the tests talk to separate processes over HTTP,
and the few async steps (seed, permsync) run with asyncio.run.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.adapters.llm import FakeEmbeddings
from ragmt.permsync.__main__ import run as run_permsync
from ragmt.settings import Settings
from seed.load import load as load_seed

DEV_CLIENT = "ragmt-dev-password"
REALM = "ragmt"
_TIMEOUT = 10.0


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is not set (see .env.example)")
    return value


@dataclass(frozen=True)
class Keycloak:
    base_url: str
    issuer: str

    @property
    def token_url(self) -> str:
        return f"{self.issuer}/protocol/openid-connect/token"

    def user_token(self, username: str, client_id: str = DEV_CLIENT) -> str:
        """An access token for a seed user, from the password grant."""
        response = httpx.post(
            self.token_url,
            data={
                "grant_type": "password",
                "client_id": client_id,
                "username": username,
                "password": _env("KC_DEMO_USER_PASSWORD"),
            },
            timeout=_TIMEOUT,
        )
        if response.status_code != httpx.codes.OK:
            pytest.fail(
                f"no token for {username} from {client_id}: HTTP {response.status_code} "
                f"{response.text}. Is KC_DEV_PASSWORD_CLIENT=true, and was the realm imported "
                "after setting it? (delete the ragmt realm or `docker compose down -v`)"
            )
        token: str = response.json()["access_token"]
        return token

    def service_token(self) -> str:
        """permsync's own client-credentials token: genuine, but not for the API."""
        response = httpx.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": _env("PERMSYNC_CLIENT_ID"),
                "client_secret": _env("PERMSYNC_CLIENT_SECRET"),
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        token: str = response.json()["access_token"]
        return token


class RealmAdmin:
    """The admin API as the bootstrap admin of the master realm. Test setup only."""

    def __init__(self, keycloak: Keycloak) -> None:
        response = httpx.post(
            f"{keycloak.base_url}/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": _env("KC_BOOTSTRAP_ADMIN_USERNAME"),
                "password": _env("KC_BOOTSTRAP_ADMIN_PASSWORD"),
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        self._http = httpx.Client(
            base_url=f"{keycloak.base_url}/admin/realms/{REALM}/",
            headers={"Authorization": f"Bearer {response.json()['access_token']}"},
            timeout=_TIMEOUT,
        )

    def group_id(self, path: str) -> str:
        response = self._http.get(f"group-by-path/{path.strip('/')}")
        response.raise_for_status()
        group_id: str = response.json()["id"]
        return group_id

    def leave_group(self, user_id: str, group_path: str) -> None:
        gid = self.group_id(group_path)
        self._http.delete(f"users/{user_id}/groups/{gid}").raise_for_status()

    def join_group(self, user_id: str, group_path: str) -> None:
        gid = self.group_id(group_path)
        self._http.put(f"users/{user_id}/groups/{gid}").raise_for_status()

    @contextmanager
    def temporary_client(self, representation: dict[str, Any]) -> Iterator[str]:
        """Create a client for the duration of the block; yields its clientId."""
        client_id: str = representation["clientId"]
        stale = self._http.get("clients", params={"clientId": client_id}).json()
        for client in stale:  # left over from an interrupted run
            self._http.delete(f"clients/{client['id']}").raise_for_status()
        response = self._http.post("clients", json=representation)
        response.raise_for_status()
        internal_id = response.headers["Location"].rsplit("/", 1)[1]
        try:
            yield client_id
        finally:
            self._http.delete(f"clients/{internal_id}").raise_for_status()

    def close(self) -> None:
        self._http.close()


def sync_permissions() -> None:
    """One permsync cycle, exactly as `python -m ragmt.permsync --once` runs it."""
    assert asyncio.run(run_permsync(Settings(), once=True)) == 0, "permsync cycle failed"


# --- fixtures -------------------------------------------------------------------


@pytest.fixture(scope="session")
def keycloak() -> Keycloak:
    issuer = os.environ.get("OIDC_ISSUER")
    if not issuer:
        pytest.skip("OIDC_ISSUER is not set")
    try:
        httpx.get(f"{issuer}/.well-known/openid-configuration", timeout=3).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"Keycloak is not reachable at {issuer}")
    return Keycloak(base_url=issuer.rsplit("/realms/", 1)[0], issuer=issuer)


@pytest.fixture(scope="session")
def admin(keycloak: Keycloak) -> Iterator[RealmAdmin]:
    realm_admin = RealmAdmin(keycloak)
    try:
        yield realm_admin
    finally:
        realm_admin.close()


@pytest.fixture(scope="session")
def seeded(keycloak: Keycloak) -> dict[str, UUID]:
    """The seed corpus, ingested as app_ingest with fake embeddings (these tests
    check access, not ranking), then memberships from the realm.

    Returns each seed document's id by title: the ingestion pipeline assigns them.
    """

    async def seed() -> dict[str, UUID]:
        settings = Settings()
        engine = create_async_engine(_env("INGEST_DATABASE_URL"))
        try:
            loaded = await load_seed(engine, FakeEmbeddings(settings.embedding_dim), settings)
        finally:
            await engine.dispose()
        return {title: id_ for tenant in loaded for title, id_ in tenant.documents.items()}

    document_ids = asyncio.run(seed())
    sync_permissions()
    return document_ids


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@pytest.fixture(scope="session")
def ollama(keycloak: Keycloak) -> None:
    """Fails, saying what to run, unless Ollama serves the embedding model."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    hint = f"run `docker compose up -d --wait ollama` and pull {model} (`make e2e` does both)"
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
    except httpx.HTTPError:
        pytest.fail(f"Ollama is not reachable at {base_url}: {hint}")
    pulled = {entry["name"] for entry in response.json().get("models", [])}
    if model not in pulled and f"{model}:latest" not in pulled:
        pytest.fail(f"{model} is not pulled in Ollama: {hint}")


# The API embeds a probe text at startup (the provider's check), which loads the
# model into Ollama first: on a CPU that can take a while.
_STARTUP_SECONDS = 120


def _wait_until_healthy(
    url: str, process: subprocess.Popen[bytes] | None, log: Path | None
) -> None:
    deadline = time.monotonic() + _STARTUP_SECONDS
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            tail = log.read_text(errors="replace")[-2000:] if log else ""
            pytest.fail(f"the API exited with status {process.returncode}:\n{tail}")
        try:
            if httpx.get(f"{url}/healthz", timeout=1).status_code == httpx.codes.OK:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    pytest.fail(f"the API at {url} did not become healthy within {_STARTUP_SECONDS} s")


@pytest.fixture(scope="session")
def api(
    seeded: dict[str, UUID], ollama: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[httpx.Client]:
    """An HTTP client for the running API."""
    url = os.environ.get("E2E_API_URL")
    process = None
    log = None
    if not url:
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        log = tmp_path_factory.mktemp("api") / "uvicorn.log"
        command = [sys.executable, "-m", "uvicorn", "--factory", "ragmt.api.app:create_app"]
        with log.open("wb") as out:
            process = subprocess.Popen(  # noqa: S603 -- fixed command, this interpreter
                [*command, "--host", "127.0.0.1", "--port", str(port)],
                stdout=out,
                stderr=subprocess.STDOUT,
                env={**os.environ, "LLM_PROVIDER": "ollama"},
            )
    try:
        _wait_until_healthy(url, process, log)
        with httpx.Client(base_url=url, timeout=_TIMEOUT) as client:
            yield client
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)
