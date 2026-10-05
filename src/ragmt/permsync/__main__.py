"""`python -m ragmt.permsync [--once]` (or `make permsync`): sync Keycloak into memberships.

Without `--once` it runs a cycle every PERMSYNC_INTERVAL_SECONDS, measured from
the start of one cycle to the start of the next, until stopped. A failed cycle
is logged and the next one tries again. With `--once` it runs one cycle and
exits with status 1 if any tenant could not be synced.

It connects only as app_ingest (INGEST_DATABASE_URL). Logs are one line per
event with a UTC timestamp in milliseconds.
"""

import argparse
import asyncio
import logging
import sys
import time

import httpx
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.permsync.keycloak import KeycloakAdmin, keycloak_location
from ragmt.permsync.store import SqlMembershipStore
from ragmt.permsync.sync import PermissionSync
from ragmt.settings import Settings, get_settings

log = logging.getLogger("ragmt.permsync")

_HTTP_TIMEOUT_SECONDS = 10.0


def _configure_logging(verbose: bool) -> None:
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler])
    # httpx logs every request at INFO, URLs (with user ids) included.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def run(settings: Settings, *, once: bool) -> int:
    base_url, realm = keycloak_location(settings.oidc_issuer, settings.permsync_keycloak_url)
    # The writer role: memberships must be written whatever any user can read (ADR 0004).
    engine = create_async_engine(
        settings.ingest_database_url.get_secret_value(), pool_size=1, max_overflow=0
    )
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=_HTTP_TIMEOUT_SECONDS) as http:
            keycloak = KeycloakAdmin(
                http,
                realm=realm,
                client_id=settings.permsync_client_id,
                client_secret=settings.permsync_client_secret,
            )
            store = SqlMembershipStore(engine, actor_sub=f"service:{settings.permsync_client_id}")
            sync = PermissionSync(keycloak, store)
            if once:
                return 0 if (await sync.run_cycle()).ok else 1

            interval = settings.permsync_interval_seconds
            log.info("permsync started realm=%s interval_s=%d", realm, interval)
            while True:
                started = time.monotonic()
                try:
                    await sync.run_cycle()
                except Exception:
                    # Anything unexpected ends this cycle, not the process.
                    log.exception("cycle failed unexpectedly")
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ragmt.permsync", description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return asyncio.run(run(get_settings(), once=args.once))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
