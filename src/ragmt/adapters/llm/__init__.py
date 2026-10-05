"""EmbeddingProvider and ChatProvider implementations (Ollama, OpenAI-compatible, fake)."""

from ragmt.adapters.llm._http import EmbeddingDimensionError, EmbeddingError
from ragmt.adapters.llm.factory import EmbeddingAdapter, build_embedding_provider
from ragmt.adapters.llm.fake import FakeEmbeddings
from ragmt.adapters.llm.ollama import OllamaEmbeddings
from ragmt.adapters.llm.openai_compat import OpenAICompatEmbeddings

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "FakeEmbeddings",
    "OllamaEmbeddings",
    "OpenAICompatEmbeddings",
    "build_embedding_provider",
]
