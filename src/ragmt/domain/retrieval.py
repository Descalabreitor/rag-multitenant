"""What retrieval returns and what an answer carries (ADR 0009). No I/O.

The flow is: question → query embedding → `RetrievedChunk`s (a `Retriever`,
under RLS) → prompt → model reply (a `ChatProvider`) → `Answer` with
`Citation`s. With no chunks, the answer is `NO_CONTEXT_ANSWER` and no model
is called.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

# The fixed reply when retrieval finds nothing the caller may read. It is the
# same whether nothing matched or nothing was visible, so it reveals nothing.
NO_CONTEXT_ANSWER = "I don't know: none of the documents you can read answer this question."


def citation_marker(document_id: UUID, ordinal: int) -> str:
    """The marker that cites one chunk in a prompt and in the model's reply."""
    return f"[doc:{document_id}#{ordinal}]"


@dataclass(frozen=True, slots=True)
class Citation:
    """A cited source as clients see it: which document and section, never content or vectors."""

    document_id: UUID
    title: str
    heading: str | None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk a `Retriever` found, already filtered by RLS.

    `title` is the document's. `score` is cosine similarity (1 - cosine
    distance): higher is closer. The embedding is deliberately absent.
    """

    chunk_id: UUID
    document_id: UUID
    ordinal: int
    title: str
    heading: str | None
    content: str
    score: float

    def __post_init__(self) -> None:
        # Same rule as the chunks_ordinal_non_negative constraint.
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")

    @property
    def marker(self) -> str:
        return citation_marker(self.document_id, self.ordinal)

    def citation(self) -> Citation:
        return Citation(document_id=self.document_id, title=self.title, heading=self.heading)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn of a conversation. The system prompt is passed apart from these."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Answer:
    """What the API returns for a question.

    `model` names the chat model that wrote `text`, or is None when no model was
    called (the `NO_CONTEXT_ANSWER` case).
    """

    text: str
    citations: tuple[Citation, ...]
    model: str | None
