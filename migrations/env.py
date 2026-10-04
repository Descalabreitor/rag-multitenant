"""Alembic environment: runs every migration as the `migrator` role.

The URL comes only from MIGRATOR_DATABASE_URL. There is deliberately no fallback
to DATABASE_URL: `app_rw` must never own schema objects, because table owners are
the role RLS is easiest to get wrong for.
"""

import asyncio
from pathlib import Path

from alembic import context
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class MigrationSettings(BaseSettings):
    """Settings Alembic needs. Kept here, out of `src/`, so the app never sees them."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    migrator_database_url: str = Field(min_length=1)


config = context.config

# Migrations are hand-written (tables, RLS policies, grants, triggers), so there is
# no ORM metadata to autogenerate from.
target_metadata = None


def _database_url() -> str:
    # pydantic-settings reads the environment first, then .env (CI exports its own).
    return MigrationSettings().migrator_database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout (`alembic upgrade head --sql`) without connecting."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect as `migrator` and apply migrations in a transaction."""
    # NullPool: a migration run is a one-shot process, nothing to pool.
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
