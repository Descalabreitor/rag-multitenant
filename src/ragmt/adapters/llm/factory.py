"""Pick the embedding adapter named by LLM_PROVIDER."""

from typing import Protocol

import httpx

from ragmt.adapters.llm.fake import FakeEmbeddings
from ragmt.adapters.llm.ollama import OllamaEmbeddings
from ragmt.adapters.llm.openai_compat import OpenAICompatEmbeddings
from ragmt.domain import EmbeddingProvider
from ragmt.settings import Settings


class EmbeddingAdapter(EmbeddingProvider, Protocol):
    """An `EmbeddingProvider` with the lifecycle calls the application needs."""

    async def check(self) -> None:
        """Raise unless the provider answers with vectors of `dim` floats (call at startup)."""
        ...

    async def aclose(self) -> None: ...


def build_embedding_provider(settings: Settings) -> EmbeddingAdapter:
    """Build the adapter for `settings.llm_provider`; an unknown value raises `ValueError`.

    The adapter owns its HTTP client: close it with `aclose()`. Nothing is
    contacted here; call `check()` for that.
    """
    provider = settings.llm_provider
    match provider:
        case "ollama":
            return OllamaEmbeddings(
                httpx.AsyncClient(base_url=settings.ollama_base_url),
                model=settings.ollama_embed_model,
                dim=settings.embedding_dim,
                batch_size=settings.embed_batch_size,
            )
        case "openai_compat":
            base_url, model = settings.openai_compat_base_url, settings.openai_compat_embed_model
            if base_url is None or model is None:  # Settings already rejects this
                raise ValueError("LLM_PROVIDER=openai_compat needs a base URL and an embed model")
            return OpenAICompatEmbeddings(
                httpx.AsyncClient(base_url=base_url),
                model=model,
                dim=settings.embedding_dim,
                batch_size=settings.embed_batch_size,
                api_key=settings.openai_compat_api_key,
            )
        case "fake":
            return FakeEmbeddings(settings.embedding_dim)
        case _:
            # Settings rejects other values; this covers settings built without validation.
            raise ValueError(f"unknown LLM_PROVIDER {provider!r}")
