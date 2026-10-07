"""Answering a question from the chunks the caller may read (ADR 0009).

The flow, for one verified Principal:

1. Refuse a question over ASK_MAX_QUESTION_CHARS, before anything is embedded.
2. Embed it (no database connection held while the embedder works).
3. In one app_rw `tenant_session` for the caller's tenant and `sub`: search with
   the `Retriever` (RLS decides what it can see), then insert the audit row.
   The transaction commits and the connection goes back to the pool here.
4. Generate: no chunk fits in the prompt → `NO_CONTEXT_ANSWER` without calling
   the chat model; otherwise one completion, with citations taken from the
   chunks the model saw.

A chat failure after step 3 leaves the retrieval audited: the row says what was
retrieved, not whether the model answered.
"""

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ragmt.audit import record_ask
from ragmt.domain import Answer, ChatProvider, EmbeddingProvider, Principal, Retriever
from ragmt.generation import Generator, select_context
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session


class QuestionTooLongError(ValueError):
    """The question is longer than ASK_MAX_QUESTION_CHARS."""


class AskService:
    """Embeds, retrieves and audits under RLS, then generates. Built once per app.

    `engine` must be the app_rw engine (DATABASE_URL), never the writer one.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        embedder: EmbeddingProvider,
        retriever: Retriever[AsyncConnection],
        chat: ChatProvider,
        settings: Settings,
    ) -> None:
        self._engine = engine
        self._embedder = embedder
        self._retriever = retriever
        self._chat = chat
        self._generator = Generator(chat, max_context_chars=settings.ask_max_context_chars)
        self._k = settings.retrieval_k
        self._max_question_chars = settings.ask_max_question_chars
        self._max_context_chars = settings.ask_max_context_chars
        self._store_question_text = settings.audit_store_query_text

    async def ask(self, principal: Principal, question: str) -> Answer:
        """Raises `QuestionTooLongError`, the embedder's errors (nothing audited
        yet) and the chat model's (the retrieval is already audited)."""
        if len(question) > self._max_question_chars:
            raise QuestionTooLongError(f"question over {self._max_question_chars} characters")
        vector = await self._embedder.embed_query(question)

        async with tenant_session(self._engine, principal.tenant_id, principal.sub) as conn:
            chunks = await self._retriever.search(conn, vector, self._k)
            # The model the generator will call, or None if nothing fits in the prompt.
            will_call = bool(select_context(chunks, self._max_context_chars))
            await record_ask(
                conn,
                tenant_id=principal.tenant_id,
                actor_sub=principal.sub,
                chunks=chunks,
                model=self._chat.model if will_call else None,
                question=question,
                store_text=self._store_question_text,
            )
        # Committed: a slow model holds no connection and no snapshot.
        return await self._generator.answer(question, chunks)
