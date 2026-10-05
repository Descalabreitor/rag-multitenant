"""What the HTTP embedding adapters share: batching, retries, response checks.

Texts sent for embedding are tenant data, as sensitive as the chunks they come
from, and so are the vectors (CLAUDE.md). Nothing here logs them or puts them in
an exception: messages carry the path, the HTTP status or the exception type,
and response bodies are never quoted (a server error may echo its input).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Embedding a batch on a CPU-only Ollama, or the first call that loads the model,
# can take tens of seconds.
DEFAULT_TIMEOUT_SECONDS = 60.0
# Retries after the first attempt, for transport errors and 5xx only.
DEFAULT_MAX_RETRIES = 2
# The n-th retry waits backoff * 2**(n-1) seconds.
DEFAULT_BACKOFF_SECONDS = 0.5

_DIMENSION_CHECK_TEXT = "dimension check"


class EmbeddingError(Exception):
    """The embedding service could not be reached, refused the request or answered garbage."""


class EmbeddingDimensionError(EmbeddingError):
    """The service returned vectors of another length than EMBEDDING_DIM."""


class HttpEmbeddings:
    """Base for adapters that call an embedding API over HTTP.

    Subclasses implement `_embed_batch`. The adapter owns `http` and closes it in
    `aclose()`. Every response is checked: one vector per input, each of `dim`
    floats, so a misconfigured model fails instead of writing wrong-sized vectors.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        dim: int,
        batch_size: int,
        document_prefix: str = "",
        query_prefix: str = "",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if dim <= 0 or batch_size <= 0:
            raise ValueError("dim and batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._http = http
        self._dim = dim
        self._batch_size = batch_size
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._sleep = sleep

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [
                self._document_prefix + text for text in texts[start : start + self._batch_size]
            ]
            vectors.extend(await self._checked_batch(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self._checked_batch([self._query_prefix + text])
        return vector

    async def check(self) -> None:
        """Embed one string and raise `EmbeddingDimensionError` unless it has `dim` floats.

        Call it at startup, so a wrong model or EMBEDDING_DIM fails before any ingest.
        Also raises `EmbeddingError` if the service is unreachable.
        """
        vector = await self.embed_query(_DIMENSION_CHECK_TEXT)
        if len(vector) != self._dim:  # already enforced per response; kept explicit
            raise EmbeddingDimensionError(f"got {len(vector)} dimensions, expected {self._dim}")

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- for subclasses -----------------------------------------------------------

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One request for `texts` (already prefixed). Return one vector per text, in order."""
        raise NotImplementedError

    async def _post_json(
        self, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        """POST and return the parsed JSON body, retrying transport errors and 5xx.

        4xx is never retried: the same request would fail the same way.
        """
        for attempt in range(self._max_retries + 1):
            retry = attempt < self._max_retries
            try:
                response = await self._http.post(
                    path, json=payload, headers=headers, timeout=self._timeout
                )
            except httpx.TransportError as exc:
                problem = type(exc).__name__
                if not retry:
                    raise EmbeddingError(f"POST {path}: {problem}") from exc
            else:
                status = response.status_code
                if status == httpx.codes.OK:
                    return _json(response, path)
                problem = f"HTTP {status}"
                if status < httpx.codes.INTERNAL_SERVER_ERROR or not retry:
                    raise EmbeddingError(f"POST {path}: {problem}")
            delay = self._backoff * 2**attempt
            logger.warning(
                "POST %s failed (%s), retry %d of %d in %.1f s",
                path,
                problem,
                attempt + 1,
                self._max_retries,
                delay,
            )
            await self._sleep(delay)
        raise AssertionError("unreachable")  # the last attempt returns or raises

    # --- internals ----------------------------------------------------------------

    async def _checked_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = await self._embed_batch(texts)
        if len(vectors) != len(texts):
            raise EmbeddingError(f"got {len(vectors)} embeddings for {len(texts)} inputs")
        for vector in vectors:
            if len(vector) != self._dim:
                raise EmbeddingDimensionError(
                    f"got {len(vector)} dimensions, expected {self._dim} (EMBEDDING_DIM)"
                )
        return vectors


def _json(response: httpx.Response, path: str) -> Any:
    # Raised outside the `except`, so the error has no __context__ either: a
    # JSONDecodeError keeps the whole body in `.doc`.
    try:
        return response.json()
    except ValueError:
        pass
    raise EmbeddingError(f"POST {path}: body is not JSON")


def parse_response[M: BaseModel](model: type[M], body: Any, path: str) -> M:
    """Validate a response body, raising `EmbeddingError` with nothing of the body in it."""
    try:
        return model.model_validate(body)
    except ValidationError:
        # A ValidationError quotes the input, so it is neither chained nor left as
        # __context__: the error is raised outside this block.
        pass
    raise EmbeddingError(f"POST {path}: unexpected response shape")
