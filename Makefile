# Shortcuts for local development. Each target wraps one command, so CI can keep
# calling the commands directly. Both read their database URL from the environment
# or .env.

PYTHON ?= python
# Port of the API that `make e2e` starts (8000 stays free for a dev server).
E2E_PORT ?= 8001

.DEFAULT_GOAL := help
.PHONY: help migrate seed permsync e2e

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

migrate: ## Apply migrations as the migrator role (MIGRATOR_DATABASE_URL)
	alembic upgrade head

seed: ## Reset the fictional seed tenants and corpus, as app_ingest (INGEST_DATABASE_URL)
	$(PYTHON) -m seed

permsync: ## Sync Keycloak groups into memberships, as app_ingest (ARGS=--once for one cycle)
	$(PYTHON) -m ragmt.permsync $(ARGS)

# Starts the API in the background, runs the tests against it over HTTP, and
# stops it when they finish, pass or fail. The tests seed the corpus and run
# permsync once themselves.
e2e: ## End-to-end check against the real stack: Keycloak + PostgreSQL + the API
	docker compose up -d --wait --wait-timeout 300 postgres keycloak
	alembic upgrade head
	$(PYTHON) -m uvicorn --factory ragmt.api.app:create_app --host 127.0.0.1 --port $(E2E_PORT) & \
	api=$$!; trap 'kill $$api' EXIT; \
	E2E_API_URL=http://127.0.0.1:$(E2E_PORT) $(PYTHON) -m pytest -m e2e
