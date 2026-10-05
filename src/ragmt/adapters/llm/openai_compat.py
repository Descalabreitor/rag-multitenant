"""Embeddings from an OpenAI-compatible API (`POST /v1/embeddings`).

The request is `{"model", "input": [...], "encoding_format": "float"}`, and the
response `{"data": [{"index", "embedding"}, ...]}`. Entries are put back in
input order by `index`, since the spec doesn't promise the order.

The API key goes only into the Authorization header. It stays a `SecretStr`
until then, and nothing here logs headers.
"""

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

from ragmt.adapters.llm._http import EmbeddingError, HttpEmbeddings, parse_response

_PATH = "/v1/embeddings"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _Embedding(_Model):
    index: int
    embedding: list[float]


class _EmbeddingsResponse(_Model):
    data: list[_Embedding]


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
        headers = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        body = await self._post_json(
            _PATH,
            {"model": self._model, "input": texts, "encoding_format": "float"},
            headers,
        )
        data = parse_response(_EmbeddingsResponse, body, _PATH).data
        if sorted(item.index for item in data) != list(range(len(data))):
            raise EmbeddingError(f"POST {_PATH}: indices are not 0..n-1")
        return [item.embedding for item in sorted(data, key=lambda item: item.index)]
