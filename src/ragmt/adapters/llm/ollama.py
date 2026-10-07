"""Embeddings and chat from a local Ollama server.

Checked against Ollama 0.35.1 (the image in compose.yaml): `POST /api/embed`
takes `{"model", "input": [...]}` and answers `{"embeddings": [[...], ...]}`,
one vector per input, in order. Inputs longer than the model's context are
truncated by the server (`truncate` defaults to true).

`POST /api/chat` with `"stream": false` answers one JSON object whose
`message.content` is the whole reply. The request never has a `tools` field.
"""

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from ragmt.adapters.llm._http import (
    ChatUnavailableError,
    HttpChat,
    HttpEmbeddings,
    chat_messages,
    parse_response,
)
from ragmt.domain import ChatMessage

# nomic-embed-text was trained with these task prefixes; without them, queries
# and documents land in less comparable regions of the space.
NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "

_EMBED_PATH = "/api/embed"
_CHAT_PATH = "/api/chat"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _EmbedResponse(_Model):
    embeddings: list[list[float]]


class _ChatReply(_Model):
    content: str


class _ChatResponse(_Model):
    message: _ChatReply


class OllamaEmbeddings(HttpEmbeddings):
    """`EmbeddingProvider` on Ollama's `/api/embed`. `http` has Ollama's URL as `base_url`.

    The prefixes default to nomic-embed-text's. Another model needs its own.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        dim: int,
        batch_size: int,
        document_prefix: str = NOMIC_DOCUMENT_PREFIX,
        query_prefix: str = NOMIC_QUERY_PREFIX,
        **options: Any,
    ) -> None:
        super().__init__(
            http,
            dim=dim,
            batch_size=batch_size,
            document_prefix=document_prefix,
            query_prefix=query_prefix,
            **options,
        )
        self._model = model

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = await self._post_json(_EMBED_PATH, {"model": self._model, "input": texts})
        return parse_response(_EmbedResponse, body, _EMBED_PATH).embeddings


class OllamaChat(HttpChat):
    """`ChatProvider` on Ollama's `/api/chat`. `http` has Ollama's URL as `base_url`.

    No streaming, temperature 0, no tools (ADR 0009).
    """

    async def _complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": chat_messages(system, messages),
            "stream": False,
            "options": {"temperature": 0},
        }
        body = await self._post_json(_CHAT_PATH, payload)
        reply = parse_response(_ChatResponse, body, _CHAT_PATH, ChatUnavailableError)
        return reply.message.content
