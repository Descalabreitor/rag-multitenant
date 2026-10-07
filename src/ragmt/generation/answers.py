"""From a model reply to an `Answer` whose citations come from the retrieved set (ADR 0009).

Citations are never taken from what the model writes: a marker counts only if
it names a chunk that was in the prompt. Other markers (invented, or copied
from chunk text) are removed from the reply, so the client never sees an id the
user didn't retrieve.

Pure functions, no I/O.
"""

import re
from collections.abc import Sequence
from uuid import UUID

from ragmt.domain import NO_CONTEXT_ANSWER, Answer, Citation, RetrievedChunk

# Anything shaped like a marker; whether it names a provided chunk is checked after.
_MARKER = re.compile(r"\[doc:([^\]#\s]+)#(\d+)\]")


def _key(match: re.Match[str]) -> tuple[UUID, int] | None:
    try:
        return UUID(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def parse_answer(text: str, chunks: Sequence[RetrievedChunk], *, model: str) -> Answer:
    """The reply with its valid citations, in order of first mention, without duplicates.

    `chunks` must be the chunks that were in the prompt, and `model` the chat
    model that wrote `text`. Markers naming any other chunk are dropped from the
    text and from the citations.
    """
    provided = {(c.document_id, c.ordinal): c for c in chunks}
    citations: list[Citation] = []

    def keep_or_drop(match: re.Match[str]) -> str:
        key = _key(match)
        chunk = provided.get(key) if key is not None else None
        if chunk is None:
            return ""
        citation = chunk.citation()
        if citation not in citations:
            citations.append(citation)
        return chunk.marker

    cleaned = _MARKER.sub(keep_or_drop, text)
    return Answer(text=cleaned.strip(), citations=tuple(citations), model=model)


def answer_without_context() -> Answer:
    """The fixed answer when no chunk can be shown: no citations, no model called."""
    return Answer(text=NO_CONTEXT_ANSWER, citations=(), model=None)
