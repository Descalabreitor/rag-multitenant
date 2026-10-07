"""EmbeddingProvider and ChatProvider implementations (Ollama, OpenAI-compatible, fake)."""

from ragmt.adapters.llm._http import ChatUnavailableError, EmbeddingDimensionError, EmbeddingError
from ragmt.adapters.llm.factory import (
    ChatAdapter,
    EmbeddingAdapter,
    build_chat_provider,
    build_embedding_provider,
)
from ragmt.adapters.llm.fake import FakeChat, FakeEmbeddings
from ragmt.adapters.llm.ollama import OllamaChat, OllamaEmbeddings
from ragmt.adapters.llm.openai_compat import OpenAICompatChat, OpenAICompatEmbeddings

__all__ = [
    "ChatAdapter",
    "ChatUnavailableError",
    "EmbeddingAdapter",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "FakeChat",
    "FakeEmbeddings",
    "OllamaChat",
    "OllamaEmbeddings",
    "OpenAICompatChat",
    "OpenAICompatEmbeddings",
    "build_chat_provider",
    "build_embedding_provider",
]
