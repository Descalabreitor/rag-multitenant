#!/usr/bin/env bash
# Runs once, on first start of an empty data volume, as the postgres superuser.
# Creates only what must exist before `alembic upgrade head` can run:
# login roles, database ownership and the vector extension.
# Tables, RLS policies and table-level GRANTs belong to the migrations.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v db="$POSTGRES_DB" \
  -v migrator_pw="$MIGRATOR_PASSWORD" \
  -v app_pw="$APP_RW_PASSWORD" \
  -v ingest_pw="$APP_INGEST_PASSWORD" \
  -v kc_pw="$KEYCLOAK_DB_PASSWORD" <<-'EOSQL'
	-- Owns the schema; used only by Alembic.
	CREATE ROLE migrator LOGIN PASSWORD :'migrator_pw'
	  NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;

	-- Runtime role: not the table owner, cannot bypass RLS.
	CREATE ROLE app_rw LOGIN PASSWORD :'app_pw'
	  NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;

	-- Writer role for ingest, ACL changes and permsync (ADR 0004).
	-- Same restrictions as app_rw; its policies are tenant-scoped, not ACL-filtered.
	CREATE ROLE app_ingest LOGIN PASSWORD :'ingest_pw'
	  NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;

	-- On PG15+ the public schema is owned by pg_database_owner,
	-- so migrator owns it through database ownership.
	ALTER DATABASE :"db" OWNER TO migrator;
	REVOKE ALL ON DATABASE :"db" FROM PUBLIC;
	GRANT CONNECT ON DATABASE :"db" TO app_rw, app_ingest;

	CREATE EXTENSION IF NOT EXISTS vector;

	-- Keycloak gets its own role and database, with no access to the app database.
	CREATE ROLE keycloak LOGIN PASSWORD :'kc_pw'
	  NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
	CREATE DATABASE keycloak OWNER keycloak;
EOSQL
