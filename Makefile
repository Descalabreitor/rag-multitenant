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

# Ingests seed/files/ through the real pipeline with the LLM_PROVIDER embedder
# (from the environment or .env; `make seed LLM_PROVIDER=ollama` overrides it).
seed: ## Reset the seed tenants and ingest seed/files/, as app_ingest (INGEST_DATABASE_URL)
	$(PYTHON) -m seed

permsync: ## Sync Keycloak groups into memberships, as app_ingest (ARGS=--once for one cycle)
	$(PYTHON) -m ragmt.permsync $(ARGS)

# Starts the stack and Ollama, pulls only the embedding model (not the chat
# model, which the tests don't use), starts the API in the background, runs the
# tests against it over HTTP, and stops it when they finish, pass or fail. The
# tests seed the corpus and run permsync once themselves. The API embeds uploads
# with Ollama; the pull reads OLLAMA_EMBED_MODEL from .env, as the API does.
e2e: ## End-to-end check against the real stack: Keycloak + PostgreSQL + Ollama + the API
	docker compose up -d --wait --wait-timeout 300 postgres keycloak ollama
	docker compose run --rm --entrypoint sh ollama-pull -c 'ollama pull "$$OLLAMA_EMBED_MODEL"'
	alembic upgrade head
	LLM_PROVIDER=ollama $(PYTHON) -m uvicorn --factory ragmt.api.app:create_app --host 127.0.0.1 --port $(E2E_PORT) & 	api=$$!; trap 'kill $$api' EXIT; 	E2E_API_URL=http://127.0.0.1:$(E2E_PORT) $(PYTHON) -m pytest -m e2e
