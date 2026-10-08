"""The benchmark database and the scratch variants it compares (Phase 6).

The benchmarks run in their own database, `ragmt_eval` by default, on the same
server and with the same roles as the development one, migrated to head. The
development database collects test tenants and earlier experiments, and table
size changes the planner's choices (ADR 0009), so measuring there would measure
whatever happened to be in it.

Connections, by role:

- `postgres` (superuser): creates the database as `01-roles.sh` does, copies
  rows into the scratch tables (migrator can't read them: FORCE RLS and no
  policy), and runs the no-RLS baseline. Its URL is `EVAL_SUPERUSER_DATABASE_URL`,
  or `postgres` with `POSTGRES_PASSWORD` on the host of DATABASE_URL. Local only:
  nothing in the service uses it.
- `migrator`: `alembic upgrade head`, `VACUUM ANALYZE`, and the scratch DDL, so
  the scratch tables are owned the way the real ones are.
- `app_ingest` and `app_rw`: the corpus load and the measured queries.

The scratch schema `eval_scratch` is not a migration and never reaches head:

- `chunks_by_tenant`: the chunks, LIST-partitioned by tenant, one HNSW index per
  partition, and the same app_rw policy text as `public.chunks`. Partitions
  rather than per-tenant partial indexes: a partial index is only used when the
  query's WHERE implies its predicate at plan time, and the RLS predicate
  compares tenant_id with `current_setting(...)`, which the planner can't fold.
  The retriever would have to add `tenant_id = <literal>` to its query, the
  filter ADR 0009 keeps out of it. Partition pruning works on the policy as it
  is: `current_setting` is stable, so the executor prunes at startup.
- `chunks_normalized`: the chunks without `acl_principals`, with the textbook
  policy ADR 0003 rejected: `EXISTS (SELECT 1 FROM document_acl ...)` per row.

Both are static copies of `public.chunks` (no ACL trigger): enough to read.
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parent.parent
EVAL_DATABASE = "ragmt_eval"
SCRATCH = "eval_scratch"
PARTITIONED = f"{SCRATCH}.chunks_by_tenant"
NORMALIZED = f"{SCRATCH}.chunks_normalized"
_RLS_MIGRATION = "20261004_aef343c73b03_add_rls_policies_grants_and_acl_trigger.py"

# An asyncpg.Connection (no type information, as in eval.corpus).
Driver = Any


class EvalSettings(BaseSettings):
    """The URLs the benchmark needs beyond `ragmt.settings`. Environment first, then .env."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: SecretStr
    ingest_database_url: SecretStr
    migrator_database_url: SecretStr
    eval_superuser_database_url: SecretStr | None = None
    postgres_user: str = "postgres"
    postgres_password: SecretStr | None = None


@dataclass(frozen=True)
class Urls:
    """SQLAlchemy URLs (postgresql+asyncpg) for each role, on the eval database."""

    reader: URL
    writer: URL
    migrator: URL
    superuser: URL

    @classmethod
    def from_settings(cls, settings: EvalSettings, database: str = EVAL_DATABASE) -> "Urls":
        reader = make_url(settings.database_url.get_secret_value())
        if settings.eval_superuser_database_url is not None:
            superuser = make_url(settings.eval_superuser_database_url.get_secret_value())
        elif settings.postgres_password is not None:
            superuser = reader.set(
                username=settings.postgres_user,
                password=settings.postgres_password.get_secret_value(),
            )
        else:
            raise ValueError("set EVAL_SUPERUSER_DATABASE_URL or POSTGRES_PASSWORD")
        return cls(
            reader=reader.set(database=database),
            writer=make_url(settings.ingest_database_url.get_secret_value()).set(database=database),
            migrator=make_url(settings.migrator_database_url.get_secret_value()).set(
                database=database
            ),
            superuser=superuser.set(database=database),
        )


@asynccontextmanager
async def connect(url: URL) -> AsyncIterator[Driver]:
    """An asyncpg connection in autocommit mode (DDL, CREATE DATABASE, VACUUM)."""
    engine = create_async_engine(url, poolclass=NullPool, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            raw = await conn.get_raw_connection()
            yield raw.driver_connection
    finally:
        await engine.dispose()


# --- The database --------------------------------------------------------------------


async def ensure_database(urls: Urls) -> bool:
    """Create the eval database as 01-roles.sh sets up the main one. True if it was created."""
    name = urls.superuser.database
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"unexpected database name {name!r}")
    async with connect(urls.superuser.set(database="postgres")) as conn:
        if await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            return False
        await conn.execute(f'CREATE DATABASE "{name}" OWNER migrator')
        await conn.execute(f'REVOKE ALL ON DATABASE "{name}" FROM PUBLIC')
        await conn.execute(f'GRANT CONNECT ON DATABASE "{name}" TO app_rw, app_ingest')
    async with connect(urls.superuser) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    return True


def migrate(urls: Urls) -> None:
    """`alembic upgrade head` on the eval database, as migrator."""
    env = {**os.environ, "MIGRATOR_DATABASE_URL": urls.migrator.render_as_string(False)}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )


async def analyze(urls: Urls) -> None:
    async with connect(urls.migrator) as conn:
        for table in ("memberships", "documents", "document_acl", "chunks"):
            await conn.execute(f"VACUUM ANALYZE {table}")


# --- Scratch variants ----------------------------------------------------------------


def _rls_migration() -> ModuleType:
    """The migration that defines the app_rw policies, for its TENANT and PRINCIPALS text."""
    path = ROOT / "migrations" / "versions" / _RLS_MIGRATION
    spec = importlib.util.spec_from_file_location("_eval_rls_migration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policies() -> dict[str, str]:
    """table -> app_rw USING expression for the scratch tables."""
    rls = _rls_migration()
    tenant: str = rls.TENANT
    principals: str = rls.PRINCIPALS
    chunks: str = rls.READER_SELECT["chunks"]
    return {
        PARTITIONED: chunks,
        # Text from the migration module, no input: S608 doesn't apply.
        NORMALIZED: f"""tenant_id = {tenant} AND EXISTS (
            SELECT 1 FROM public.document_acl a
            WHERE a.document_id = chunks_normalized.document_id
              AND a.principal = ANY({principals}))""",  # noqa: S608
    }


async def scratch_fingerprint(urls: Urls) -> str | None:
    async with connect(urls.superuser) as conn:
        value: str | None = await conn.fetchval(
            "SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace WHERE nspname = $1",
            SCRATCH,
        )
        return value


async def build_scratch(
    urls: Urls, tenants: dict[str, UUID], dim: int, fingerprint: str
) -> dict[str, float]:
    """(Re)create eval_scratch from public.chunks. Returns seconds per step."""
    timings: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    using = policies()
    async with connect(urls.migrator) as mig, connect(urls.superuser) as su:
        started = loop.time()
        await mig.execute(f"DROP SCHEMA IF EXISTS {SCRATCH} CASCADE")
        await mig.execute(f"CREATE SCHEMA {SCRATCH}")
        await mig.execute(f"GRANT USAGE ON SCHEMA {SCRATCH} TO app_rw")
        columns = f"""
            id uuid NOT NULL,
            tenant_id uuid NOT NULL,
            document_id uuid NOT NULL,
            ordinal integer NOT NULL,
            heading text,
            content text NOT NULL,
            embedding vector({dim}) NOT NULL"""
        await mig.execute(
            f"CREATE TABLE {PARTITIONED} ({columns}, acl_principals text[] NOT NULL)"
            " PARTITION BY LIST (tenant_id)"
        )
        for key, tenant_id in tenants.items():
            name = "p_" + key.replace("-", "_").replace(".", "_")
            await mig.execute(
                f"CREATE TABLE {SCRATCH}.{name} PARTITION OF {PARTITIONED}"
                f" FOR VALUES IN ('{tenant_id}')"
            )
        await mig.execute(f"CREATE TABLE {SCRATCH}.p_default PARTITION OF {PARTITIONED} DEFAULT")
        await mig.execute(f"CREATE TABLE {NORMALIZED} ({columns})")
        timings["ddl"] = loop.time() - started

        # migrator sees no rows in public.chunks (FORCE RLS, no policy), so the
        # superuser copies them. Table names are module constants (S608).
        started = loop.time()
        await su.execute(
            f"INSERT INTO {PARTITIONED} SELECT id, tenant_id, document_id, ordinal, heading,"  # noqa: S608
            " content, embedding, acl_principals FROM public.chunks"
        )
        await su.execute(
            f"INSERT INTO {NORMALIZED} SELECT id, tenant_id, document_id, ordinal, heading,"  # noqa: S608
            " content, embedding FROM public.chunks"
        )
        timings["copy"] = loop.time() - started

        # Same indexes as public.chunks (minus the GIN where there's no array).
        # Serial builds: Docker's 64 MB /dev/shm is too small for parallel HNSW builds.
        started = loop.time()
        await mig.execute("SET maintenance_work_mem = '2GB'")
        await mig.execute("SET max_parallel_maintenance_workers = 0")
        for table in (PARTITIONED, NORMALIZED):
            short = table.split(".")[1]
            await mig.execute(f"ALTER TABLE {table} ADD PRIMARY KEY (tenant_id, id)")
            await mig.execute(f"CREATE INDEX {short}_tenant_id_idx ON {table} (tenant_id)")
            await mig.execute(
                f"CREATE INDEX {short}_embedding_hnsw_idx ON {table}"
                " USING hnsw (embedding vector_cosine_ops)"
            )
        await mig.execute(
            f"CREATE INDEX chunks_by_tenant_acl_principals_idx ON {PARTITIONED}"
            " USING gin (acl_principals)"
        )
        timings["indexes"] = loop.time() - started

        for table, expression in using.items():
            short = table.split(".")[1]
            await mig.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            await mig.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            await mig.execute(f"GRANT SELECT ON {table} TO app_rw")
            await mig.execute(
                f"CREATE POLICY {short}_app_rw_select ON {table}"
                f" FOR SELECT TO app_rw USING ({expression})"
            )
        started = loop.time()
        for table in (PARTITIONED, NORMALIZED):
            await mig.execute(f"VACUUM ANALYZE {table}")
        timings["analyze"] = loop.time() - started
        await mig.execute(f"COMMENT ON SCHEMA {SCRATCH} IS '{fingerprint}'")

        await _check_same_policy(su)
    return timings


async def _check_same_policy(conn: Driver) -> None:
    """The partitioned table's policy must be the one public.chunks has at head."""
    rows = await conn.fetch(
        "SELECT c.relname, pg_get_expr(p.polqual, p.polrelid) AS qual"
        " FROM pg_policy p JOIN pg_class c ON c.oid = p.polrelid"
        " WHERE p.polname IN ('chunks_app_rw_select', 'chunks_by_tenant_app_rw_select')"
    )
    quals = {row["relname"]: row["qual"] for row in rows}
    if len(quals) != 2 or quals["chunks"] != quals["chunks_by_tenant"]:
        raise RuntimeError(f"scratch policy differs from public.chunks: {quals}")
