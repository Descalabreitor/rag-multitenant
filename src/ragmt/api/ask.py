"""`POST /ask`: answer a question from the documents the caller may read (ADR 0009).

The route only maps the service's outcome to HTTP (`ragmt.ask.AskService` does
the work). The tenant and user come from the Principal; a `tenant_id` in the
body is not declared, so it is ignored. The response holds the answer, the
citations built from the retrieved set (document id, title, heading) and the
model name: never scores, chunk ids, chunk contents or embeddings.

Status codes: 422 for an empty question or one over ASK_MAX_QUESTION_CHARS
(refused before it is embedded), 503 when the embedding or chat service fails.
A chat failure happens after the retrieval was audited, and the row stays.
Error bodies are fixed strings that never quote the question.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, StringConstraints

from ragmt.adapters.llm import ChatUnavailableError, EmbeddingError
from ragmt.ask import AskService, QuestionTooLongError
from ragmt.auth.dependencies import get_principal
from ragmt.domain import Principal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

QUESTION_TOO_LONG = "Question too long"
EMBEDDINGS_UNAVAILABLE = "Embedding service unavailable"
CHAT_UNAVAILABLE = "Chat service unavailable"


def get_ask_service(request: Request) -> AskService:
    """The AskService built in the app's lifespan (`ragmt.api.app`)."""
    service = getattr(request.app.state, "ask_service", None)
    if not isinstance(service, AskService):
        raise RuntimeError("no ask service: was the app started through its lifespan?")
    return service


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CitationOut(BaseModel):
    document_id: UUID
    title: str
    heading: str | None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    # None when no model was called (nothing the caller may read matched).
    model: str | None


@router.post(
    "/ask",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": QUESTION_TOO_LONG},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Embedding or chat service down"},
    },
)
async def ask(
    body: AskRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[AskService, Depends(get_ask_service)],
) -> AskResponse:
    try:
        answer = await service.ask(principal, body.question)
    except QuestionTooLongError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, QUESTION_TOO_LONG) from None
    except EmbeddingError as exc:
        logger.warning("embedding failed during ask: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, EMBEDDINGS_UNAVAILABLE) from None
    except ChatUnavailableError as exc:
        logger.warning("chat failed during ask: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, CHAT_UNAVAILABLE) from None
    return AskResponse(
        answer=answer.text,
        citations=[
            CitationOut(document_id=c.document_id, title=c.title, heading=c.heading)
            for c in answer.citations
        ],
        model=answer.model,
    )
