# Shortcuts for local development. Each target wraps one command, so CI can keep
# calling the commands directly. Both read their database URL from the environment
# or .env.

PYTHON ?= python
# Port of the API that `make e2e` starts (8000 stays free for a dev server).
E2E_PORT ?= 8001
# `make e2e FAKE_CHAT=1`: answer /ask with FakeChat (CHAT_PROVIDER=fake) and skip
# pulling the chat model. Embeddings still come from Ollama.
FAKE_CHAT ?=
E2E_CHAT_PROVIDER := $(if $(FAKE_CHAT),fake,ollama)
# A completion from an 8B model on a CPU can pass the 120 s default.
E2E_CHAT_TIMEOUT ?= 300
# More chunks than any seed user can read, so every search returns all of them
# (E2E_RETRIEVAL_K in tests/e2e/conftest.py: keep the two equal).
E2E_RETRIEVAL_K ?= 10

.DEFAULT_GOAL := help
.PHONY: help migrate seed permsync e2e

help: ## List the targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

migrate: ## Apply migrations as the migrator role (MIGRATOR_DATABASE_URL)
	alembic upgrade head

# Ingests seed/files/ through the real pipeline with the LLM_PROVIDER embedder
# (from the environment or .env; `make seed LLM_PROVIDER=ollama` overrides it).
seed: ## Reset the seed tenants and ingest seed/files/, as app_ingest (INGEST_DATABASE_URL)
	$(PYTHON) -m seed

permsync: ## Sync Keycloak groups into memberships, as app_ingest (ARGS=--once for one cycle)
	$(PYTHON) -m ragmt.permsync $(ARGS)

# Starts the stack and Ollama, pulls the embedding model and the chat model
# (llama3.1:8b is about 4.9 GB: several minutes on the first run, nothing after;
# FAKE_CHAT=1 skips it), starts the API in the background, runs the tests against
# it over HTTP, and stops it when they finish, pass or fail. The tests seed the
# corpus and run permsync once themselves. The pull reads OLLAMA_EMBED_MODEL and
# OLLAMA_CHAT_MODEL from .env, as the API does.
e2e: ## End-to-end check against the real stack: Keycloak + PostgreSQL + Ollama + the API (FAKE_CHAT=1: no chat model)
	docker compose up -d --wait --wait-timeout 300 postgres keycloak ollama
	docker compose run --rm --entrypoint sh ollama-pull -c 'ollama pull "$$OLLAMA_EMBED_MODEL"$(if $(FAKE_CHAT),, && ollama pull "$$OLLAMA_CHAT_MODEL")'
	alembic upgrade head
	export LLM_PROVIDER=ollama CHAT_PROVIDER=$(E2E_CHAT_PROVIDER) \
	  CHAT_TIMEOUT_SECONDS=$(E2E_CHAT_TIMEOUT) RETRIEVAL_K=$(E2E_RETRIEVAL_K); \
	$(PYTHON) -m uvicorn --factory ragmt.api.app:create_app --host 127.0.0.1 --port $(E2E_PORT) & \
	api=$$!; trap 'kill $$api' EXIT; \
	E2E_API_URL=http://127.0.0.1:$(E2E_PORT) $(PYTHON) -m pytest -m e2e
