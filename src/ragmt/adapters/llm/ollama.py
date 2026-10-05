"""Embeddings from a local Ollama server.

Checked against Ollama 0.35.1 (the image in compose.yaml): `POST /api/embed`
takes `{"model", "input": [...]}` and answers `{"embeddings": [[...], ...]}`,
one vector per input, in order. Inputs longer than the model's context are
truncated by the server (`truncate` defaults to true).
"""

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from ragmt.adapters.llm._http import HttpEmbeddings, parse_response

# nomic-embed-text was trained with these task prefixes; without them, queries
# and documents land in less comparable regions of the space.
NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "

_PATH = "/api/embed"


class _EmbedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    embeddings: list[list[float]]


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
        body = await self._post_json(_PATH, {"model": self._model, "input": texts})
        return parse_response(_EmbedResponse, body, _PATH).embeddings
