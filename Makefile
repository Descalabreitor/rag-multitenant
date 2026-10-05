# Shortcuts for local development. Each target wraps one command, so CI can keep
# calling the commands directly. Both read their database URL from the environment
# or .env.

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help migrate seed permsync

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

migrate: ## Apply migrations as the migrator role (MIGRATOR_DATABASE_URL)
	alembic upgrade head

seed: ## Reset the fictional seed tenants and corpus, as app_ingest (INGEST_DATABASE_URL)
	$(PYTHON) -m seed

permsync: ## Sync Keycloak groups into memberships, as app_ingest (ARGS=--once for one cycle)
	$(PYTHON) -m ragmt.permsync $(ARGS)
