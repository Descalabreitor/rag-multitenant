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
# Port of the API that `make demo` starts.
DEMO_PORT ?= 8002
# The shell `make demo` records in. On Windows, VHS's "bash" resolves to System32's
# WSL bash.exe before Git Bash, so the default there is Windows PowerShell.
DEMO_SHELL ?= $(if $(filter Windows_NT,$(OS)),powershell,bash)
# VHS 0.12 on Windows records the frames but silently writes no GIF, so `make demo`
# has it write PNG frames here (Set Framerate in the tape) and encodes them with ffmpeg.
DEMO_FRAMES := .demo-frames
DEMO_FRAMERATE := 20
# `make e2e GPU=1` (also quality, demo): run Ollama on an NVIDIA GPU through the
# compose.gpu.yaml override. Keep passing GPU=1 while Ollama runs on the GPU: a
# compose call without the override recreates the container on the CPU.
GPU ?=
COMPOSE := docker compose$(if $(GPU), -f compose.yaml -f compose.gpu.yaml)

.DEFAULT_GOAL := help
.PHONY: help migrate seed permsync e2e revocation eval quality leaks demo

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
	$(COMPOSE) up -d --wait --wait-timeout 300 postgres keycloak ollama
	$(COMPOSE) run --rm --entrypoint sh ollama-pull -c 'ollama pull "$$OLLAMA_EMBED_MODEL"$(if $(FAKE_CHAT),, && ollama pull "$$OLLAMA_CHAT_MODEL")'
	alembic upgrade head
	export LLM_PROVIDER=ollama CHAT_PROVIDER=$(E2E_CHAT_PROVIDER) \
	  CHAT_TIMEOUT_SECONDS=$(E2E_CHAT_TIMEOUT) RETRIEVAL_K=$(E2E_RETRIEVAL_K); \
	$(PYTHON) -m uvicorn --factory ragmt.api.app:create_app --host 127.0.0.1 --port $(E2E_PORT) & \
	api=$$!; trap 'kill $$api' EXIT; \
	E2E_API_URL=http://127.0.0.1:$(E2E_PORT) $(PYTHON) -m pytest -m e2e

# Measures how long a user keeps a group's documents after leaving the group in
# Keycloak (ADR 0002): permsync at each interval, and option A's token lifetime.
# About an hour with the defaults; raw runs go to docs/results/raw/ (git-ignored),
# the summary to docs/results/revocation.md. Not run in CI. The compose permsync
# service must be stopped. ARGS passes options, e.g. ARGS="--runs 5 --intervals 10".
revocation: ## Measure the revocation window against the real stack (Keycloak + PostgreSQL; no Ollama)
	$(COMPOSE) up -d --wait --wait-timeout 300 postgres keycloak
	alembic upgrade head
	$(PYTHON) -m eval.revocation $(ARGS)

# Recall and latency of the retrieval query on the synthetic corpus (eval.corpus),
# in its own database (ragmt_eval) on the compose PostgreSQL: creates and migrates
# it, loads the corpus if it isn't there (about 7 min for 100k chunks: the HNSW
# index is built row by row), builds the scratch variants (partitions, EXISTS
# policy), measures every variant, then writes docs/results/eval.md and its
# charts. Raw runs go to docs/results/raw/ (git-ignored). Needs POSTGRES_PASSWORD
# (or EVAL_SUPERUSER_DATABASE_URL) for the no-RLS baseline. Not run in CI.
# ARGS passes options, e.g. ARGS="--reps 3 --variants hnsw_off hnsw_relaxed".
eval: ## Retrieval benchmarks (recall, latency) on the synthetic corpus; no Ollama
	$(COMPOSE) up -d --wait postgres
	$(PYTHON) -m eval.bench $(ARGS)
	$(PYTHON) -m eval.report

# Answer quality on the seed corpus (eval/quality/): the questions through POST /ask
# with the real models, scored for citations, "I don't know" and, with an LLM judge,
# faithfulness and correctness. Re-embeds the seed tenants with Ollama first (their
# documents are deleted and loaded again), then starts the API with LLM_PROVIDER and
# CHAT_PROVIDER set to ollama. The judge is OLLAMA_CHAT_MODEL unless QUALITY_JUDGE_*
# says otherwise. Raw runs go to docs/results/raw/ (git-ignored); the summary to
# docs/results/quality.md. Not run in CI. ARGS passes options, e.g. ARGS="--users erin".
quality: ## Answer-quality baseline on the seed corpus (Keycloak + PostgreSQL + Ollama)
	$(COMPOSE) up -d --wait --wait-timeout 300 postgres keycloak ollama
	$(COMPOSE) run --rm --entrypoint sh ollama-pull -c 'ollama pull "$$OLLAMA_EMBED_MODEL" && ollama pull "$$OLLAMA_CHAT_MODEL"'
	alembic upgrade head
	$(PYTHON) -m eval.quality $(ARGS)

# The leak suite with the "nightly" Hypothesis profile: the property-based test
# (tests/leaks/test_properties.py) runs thousands of random worlds instead of
# CI's 200, and counts every check it makes. docs/results/leaks.md gets the
# count ("0 leaks in N attempts"), or says the run failed. Rows are left behind,
# as in the rest of the leak suite: `docker compose down -v` starts afresh.
leaks: ## Leak suite with the nightly Hypothesis profile, written to docs/results/leaks.md (PostgreSQL only)
	$(COMPOSE) up -d --wait --wait-timeout 120 postgres
	alembic upgrade head
	HYPOTHESIS_PROFILE=nightly LEAKS_REPORT=docs/results/leaks.md $(PYTHON) -m pytest -m leaks

# Records docs/demo.gif from docs/demo/demo.tape with VHS (vhs, ttyd and ffmpeg on
# PATH): alice, dave and erin ask the same question through ragctl --dev-user.
# Starts the stack, pulls both models, migrates, seeds, runs permsync once, starts
# the API on DEMO_PORT with real models and stops it afterwards. Like `make e2e`,
# the API gets RETRIEVAL_K=$(E2E_RETRIEVAL_K), more chunks than any seed user can
# read, so every answer sees all of the user's readable chunks whatever the seed was
# embedded with, and the sources shown follow from the ACLs alone.
demo: ## Record docs/demo.gif with VHS (the whole stack; vhs, ttyd and ffmpeg installed)
	$(COMPOSE) up -d --wait --wait-timeout 300 postgres keycloak ollama
	$(COMPOSE) run --rm --entrypoint sh ollama-pull -c 'ollama pull "$$OLLAMA_EMBED_MODEL" && ollama pull "$$OLLAMA_CHAT_MODEL"'
	alembic upgrade head
	LLM_PROVIDER=ollama $(PYTHON) -m seed
	$(PYTHON) -m ragmt.permsync --once
	export LLM_PROVIDER=ollama CHAT_PROVIDER=ollama \
	  CHAT_TIMEOUT_SECONDS=$(E2E_CHAT_TIMEOUT) RETRIEVAL_K=$(E2E_RETRIEVAL_K); \
	$(PYTHON) -m uvicorn --factory ragmt.api.app:create_app --host 127.0.0.1 --port $(DEMO_PORT) & \
	api=$$!; trap 'kill $$api' EXIT; \
	until curl -sf http://127.0.0.1:$(DEMO_PORT)/healthz >/dev/null; do \
	  kill -0 $$api 2>/dev/null || exit 1; sleep 1; done; \
	tape=$$(mktemp --suffix=.tape); trap 'kill $$api; rm -rf "$$tape" $(DEMO_FRAMES)' EXIT; \
	rm -rf $(DEMO_FRAMES); \
	sed -e 's/^Set Shell .*/Set Shell "$(DEMO_SHELL)"/' -e 's|^Output .*|Output $(DEMO_FRAMES)/|' \
	  docs/demo/demo.tape > "$$tape"; \
	KC_DEMO_USER_PASSWORD=$${KC_DEMO_USER_PASSWORD:-$$(grep '^KC_DEMO_USER_PASSWORD=' .env | cut -d= -f2- | tr -d '\r')} \
	RAGMT_API_URL=http://127.0.0.1:$(DEMO_PORT) vhs "$$tape" && \
	test -n "$$(ls $(DEMO_FRAMES))" && \
	ffmpeg -hide_banner -loglevel error -y \
	  -framerate $(DEMO_FRAMERATE) -start_number 1 -i $(DEMO_FRAMES)/frame-text-%05d.png \
	  -framerate $(DEMO_FRAMERATE) -start_number 1 -i $(DEMO_FRAMES)/frame-cursor-%05d.png \
	  -filter_complex "[0][1]overlay,pad=iw+40:ih+40:20:20:color=0x171717,fps=10,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none" \
	  docs/demo.gif && \
	ls -l docs/demo.gif
