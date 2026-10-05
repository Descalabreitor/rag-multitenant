"""The embedder `make seed` uses: the one LLM_PROVIDER names, checked before any write.

With "ollama", the embedding model must already be pulled (`docker compose up -d`
runs the one-shot `ollama-pull` service). The seed never pulls it: a pull is
gigabytes over the network, which a reset of local data should not start
silently. Without the model, or without Ollama, the seed stops with a message
saying what to run.
"""

from typing import Any

import httpx

from ragmt.adapters.llm import EmbeddingAdapter, EmbeddingError, build_embedding_provider
from ragmt.settings import Settings

_TAGS_TIMEOUT_SECONDS = 10.0


class SeedError(Exception):
    """The seed can't run as configured. The message says what to do."""


async def open_embedder(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> EmbeddingAdapter:
    """Build the LLM_PROVIDER embedder and check that it answers with EMBEDDING_DIM floats.

    `transport` replaces the network for the Ollama model check (tests). The
    caller closes the returned adapter with `aclose()`.
    """
    if settings.llm_provider == "ollama":
        await require_ollama_model(settings, transport)
    embedder = build_embedding_provider(settings)
    try:
        await embedder.check()
    except EmbeddingError as exc:
        await embedder.aclose()
        raise SeedError(
            f"LLM_PROVIDER={settings.llm_provider}: the embedding check failed ({exc}). "
            "Check the provider settings and EMBEDDING_DIM, or set LLM_PROVIDER=fake."
        ) from exc
    except BaseException:
        await embedder.aclose()
        raise
    return embedder


async def require_ollama_model(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> None:
    """Raise `SeedError` unless Ollama answers and lists OLLAMA_EMBED_MODEL as pulled."""
    url, model = settings.ollama_base_url, settings.ollama_embed_model
    try:
        async with httpx.AsyncClient(
            base_url=url, transport=transport, timeout=_TAGS_TIMEOUT_SECONDS
        ) as http:
            response = await http.get("/api/tags")
            response.raise_for_status()
            pulled = _model_names(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise SeedError(
            f"Ollama does not answer at {url} (OLLAMA_BASE_URL): {type(exc).__name__}. "
            "Start it with `docker compose up -d ollama`, or set LLM_PROVIDER=fake."
        ) from exc
    if not _is_pulled(model, pulled):
        raise SeedError(
            f"Ollama at {url} has no model {model!r} (OLLAMA_EMBED_MODEL). Pull it first: "
            f"`docker compose up ollama-pull` or `docker compose exec ollama ollama pull {model}`."
        )


def _model_names(body: Any) -> set[str]:
    """Names in an `/api/tags` body: `{"models": [{"name": "nomic-embed-text:latest"}, ...]}`."""
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        raise ValueError("unexpected /api/tags response")
    return {m["name"] for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)}


def _is_pulled(model: str, pulled: set[str]) -> bool:
    # Ollama lists models with their tag; a name without one means ":latest".
    return model in pulled or (":" not in model and f"{model}:latest" in pulled)
