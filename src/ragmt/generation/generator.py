"""Turns retrieved chunks and a question into an `Answer` (ADR 0009).

No chunks, no model: when nothing reaches the prompt, the answer is
`NO_CONTEXT_ANSWER` and the chat model is not called. Retrieval, the audit row
and the commit happen before this, in the caller.
"""

from collections.abc import Sequence

from ragmt.domain import Answer, ChatProvider, RetrievedChunk
from ragmt.generation.answers import answer_without_context, parse_answer
from ragmt.generation.prompt import build_prompt, select_context


class Generator:
    """Builds the prompt, calls the chat model once and keeps only valid citations."""

    def __init__(self, chat: ChatProvider, *, max_context_chars: int) -> None:
        self._chat = chat
        self._max_context_chars = max_context_chars

    async def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> Answer:
        # Citations are checked against what the model saw, not everything retrieved.
        context = select_context(chunks, self._max_context_chars)
        if not context:
            return answer_without_context()
        system, messages = build_prompt(question, context, self._max_context_chars)
        reply = await self._chat.complete(system, messages)
        return parse_answer(reply, context, model=self._chat.model)
