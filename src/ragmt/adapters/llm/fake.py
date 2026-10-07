"""Test doubles for CI and tests: hash-derived embeddings (MEANINGLESS FOR RANKING)
and a chat model that echoes its input (NOT A MODEL)."""

import hashlib
import math
import re
import struct
from collections.abc import Sequence

from ragmt.domain import ChatMessage

_UINT32 = struct.Struct("<I")


class FakeEmbeddings:
    """Deterministic fake `EmbeddingProvider`: NOT a model, MEANINGLESS FOR RANKING.

    Each vector is derived from a hash of the text alone, so the same text always
    gives the same vector and similar texts give unrelated ones. Vectors have
    `dim` floats and unit L2 norm, like Ollama's. No network, no prefixes:
    `embed_query(t)` equals `embed_documents([t])[0]`, so a query can find a
    chunk with exactly its text, and nothing else is meaningful.
    Use it where a real model is unavailable (CI, unit tests), never to measure recall.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def check(self) -> None:
        """Nothing to reach; vectors have `dim` floats by construction."""

    async def aclose(self) -> None:
        """Nothing to close."""

    def _vector(self, text: str) -> list[float]:
        raw = hashlib.shake_256(text.encode()).digest(_UINT32.size * self._dim)
        # Uniform in [-1, 1).
        values = [n / 2**31 - 1.0 for (n,) in _UINT32.iter_unpack(raw)]
        norm = math.sqrt(math.fsum(v * v for v in values))
        if norm == 0.0:  # every value exactly 0: astronomically unlikely
            values[0], norm = 1.0, 1.0
        return [v / norm for v in values]


# What `citation_marker` produces: [doc:<uuid>#<ordinal>].
_MARKER = re.compile(r"\[doc:[0-9a-fA-F-]{36}#\d+\]")


class FakeChat:
    """TEST DOUBLE for `ChatProvider`: NOT A MODEL. It answers nothing, it echoes.

    The reply lists every citation marker found in the input (system prompt and
    messages, in order of first appearance, without repeats), then repeats the
    whole input verbatim. Tests can see exactly which chunks, and which bytes of
    them, reached the "LLM", and a reply cites every chunk it was given.
    Deterministic, no network. Each call is kept in `calls`, so a test can check
    that the model was not called at all. Use it in CI and tests, never to
    judge answers.
    """

    def __init__(self, model: str = "fake-chat") -> None:
        self._model = model
        self.calls: list[tuple[str, tuple[ChatMessage, ...]]] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        self.calls.append((system, tuple(messages)))
        texts = [system, *(message.content for message in messages)]
        markers = dict.fromkeys(m for text in texts for m in _MARKER.findall(text))
        lines = [
            "FakeChat test double, not a model.",
            "Cited: " + (" ".join(markers) or "(none)"),
            "--- system ---",
            system,
        ]
        for message in messages:
            lines += [f"--- {message.role} ---", message.content]
        return "\n".join(lines)

    async def aclose(self) -> None:
        """Nothing to close."""
