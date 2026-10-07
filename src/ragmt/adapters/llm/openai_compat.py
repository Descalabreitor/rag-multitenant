"""Embeddings and chat from an OpenAI-compatible API.

`POST /v1/embeddings`: the request is `{"model", "input": [...],
"encoding_format": "float"}`, and the response `{"data": [{"index",
"embedding"}, ...]}`. Entries are put back in input order by `index`, since the
spec doesn't promise the order.

`POST /v1/chat/completions` with `"stream": false` answers
`{"choices": [{"message": {"content": ...}}, ...]}`; the first choice is the
reply. The request never has a `tools` field. A null `content` (a refusal, or a
tool call the server made up) is an error.

The API key goes only into the Authorization header. It stays a `SecretStr`
until then, and nothing here logs headers.
"""

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ragmt.adapters.llm._http import (
    ChatUnavailableError,
    EmbeddingError,
    HttpChat,
    HttpEmbeddings,
    chat_messages,
    parse_response,
)
from ragmt.domain import ChatMessage

_EMBED_PATH = "/v1/embeddings"
_CHAT_PATH = "/v1/chat/completions"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _Embedding(_Model):
    index: int
    embedding: list[float]


class _EmbeddingsResponse(_Model):
    data: list[_Embedding]


class _ChatReply(_Model):
    content: str


class _Choice(_Model):
    message: _ChatReply


class _ChatResponse(_Model):
    choices: list[_Choice] = Field(min_length=1)


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    if api_key is None:
        return {}
    return {"Authorization": f"Bearer {api_key.get_secret_value()}"}


class OpenAICompatEmbeddings(HttpEmbeddings):
    """`EmbeddingProvider` on `/v1/embeddings`. `http` has the server root as `base_url`.

    No task prefixes by default: OpenAI's models don't use them. Pass them for a
    model that does (nomic-embed-text served by vLLM, for instance).
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        dim: int,
        batch_size: int,
        api_key: SecretStr | None = None,
        **options: Any,
    ) -> None:
        super().__init__(http, dim=dim, batch_size=batch_size, **options)
        self._model = model
        self._api_key = api_key

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = await self._post_json(
            _EMBED_PATH,
            {"model": self._model, "input": texts, "encoding_format": "float"},
            _auth_headers(self._api_key),
        )
        data = parse_response(_EmbeddingsResponse, body, _EMBED_PATH).data
        if sorted(item.index for item in data) != list(range(len(data))):
            raise EmbeddingError(f"POST {_EMBED_PATH}: indices are not 0..n-1")
        return [item.embedding for item in sorted(data, key=lambda item: item.index)]


class OpenAICompatChat(HttpChat):
    """`ChatProvider` on `/v1/chat/completions`. `http` has the server root as `base_url`.

    No streaming, temperature 0, no tools (ADR 0009).
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        api_key: SecretStr | None = None,
        **options: Any,
    ) -> None:
        super().__init__(http, model=model, **options)
        self._api_key = api_key

    async def _complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": chat_messages(system, messages),
            "stream": False,
            "temperature": 0,
        }
        body = await self._post_json(_CHAT_PATH, payload, _auth_headers(self._api_key))
        reply = parse_response(_ChatResponse, body, _CHAT_PATH, ChatUnavailableError)
        return reply.choices[0].message.content
