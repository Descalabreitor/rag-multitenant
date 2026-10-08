"""Writes and reads of audit_events, always on an app_rw `tenant_session` connection.

The table is insert-only for every runtime role (ADR 0004). app_rw may also
SELECT it, but its policy shows rows only to the tenant's admins (ADR 0009,
migration b7d41c2e9f05): `list_events` adds no tenant or admin filter of its own.

An "ask" row stores what was retrieved, not what was asked: the chunk ids (the
column), and in `details` their scores in the same order, the chat model that
will answer (null when none will be called) and the SHA-256 of the question.
The question text goes in only when AUDIT_STORE_QUERY_TEXT is on, and
`list_events` hands it out only while the setting is on: the table is
insert-only, so rows written while it was on keep the text, and turning it off
must stop them being served.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, Text, column, select, table, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.types import Uuid

from ragmt.domain import RetrievedChunk

ASK_ACTION: Final = "ask"
# The key of an ask row's `details` that holds the question text, when stored.
QUESTION_KEY: Final = "question"

_audit_events = table(
    "audit_events",
    column("id", BigInteger()),
    column("occurred_at", DateTime(timezone=True)),
    column("actor_sub", Text()),
    column("action", Text()),
    column("chunk_ids", ARRAY(Uuid())),
    column("details", JSONB()),
)

_INSERT = text(
    "INSERT INTO audit_events (tenant_id, actor_sub, action, chunk_ids, details)"
    " VALUES (:tenant, :actor, :action, CAST(:chunk_ids AS uuid[]), CAST(:details AS jsonb))"
)


def question_sha256(question: str) -> str:
    """Lower-case hex SHA-256 of the question's UTF-8 bytes."""
    return hashlib.sha256(question.encode()).hexdigest()


def ask_details(
    chunks: Sequence[RetrievedChunk], *, model: str | None, question: str, store_text: bool
) -> dict[str, Any]:
    """The `details` of an ask row. Scores follow the order of `chunk_ids`."""
    details: dict[str, Any] = {
        "scores": [c.score for c in chunks],
        "model": model,
        "question_sha256": question_sha256(question),
    }
    if store_text:
        details[QUESTION_KEY] = question
    return details


async def record_ask(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    actor_sub: str,
    chunks: Sequence[RetrievedChunk],
    model: str | None,
    question: str,
    store_text: bool,
) -> None:
    """Insert the ask row in the caller's transaction, the one that ran the search.

    The insert policy only accepts the session's own tenant and user, so
    `tenant_id` and `actor_sub` must be the ones `conn` was opened with.
    """
    await conn.execute(
        _INSERT,
        {
            "tenant": tenant_id,
            "actor": actor_sub,
            "action": ASK_ACTION,
            "chunk_ids": [c.chunk_id for c in chunks],
            "details": json.dumps(
                ask_details(chunks, model=model, question=question, store_text=store_text)
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class AuditEvent:
    id: int
    occurred_at: datetime
    actor_sub: str
    action: str
    chunk_ids: tuple[UUID, ...]
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AuditPage:
    events: tuple[AuditEvent, ...]
    # Pass as `before` to get the next (older) page; None on the last page.
    next_before: int | None


async def list_events(
    conn: AsyncConnection, *, limit: int, before: int | None, with_question_text: bool
) -> AuditPage:
    """Newest first (by id, the insert order), at most `limit` rows with ids below `before`.

    Keyset pagination: the table only grows, so a page never repeats or skips a
    row because of inserts made while a client pages through it.

    Without `with_question_text` (pass AUDIT_STORE_QUERY_TEXT), ask rows come
    back without their question text, even rows stored while it was on.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    query = (
        select(_audit_events)
        .order_by(_audit_events.c.id.desc())
        # One extra row tells whether there is a next page.
        .limit(limit + 1)
    )
    if before is not None:
        query = query.where(_audit_events.c.id < before)
    rows = (await conn.execute(query)).all()
    events = tuple(
        AuditEvent(
            id=row.id,
            occurred_at=row.occurred_at,
            actor_sub=row.actor_sub,
            action=row.action,
            chunk_ids=tuple(row.chunk_ids),
            details=_readable_details(row.action, row.details, with_question_text),
        )
        for row in rows[:limit]
    )
    next_before = events[-1].id if len(rows) > limit else None
    return AuditPage(events=events, next_before=next_before)


def _readable_details(
    action: str, details: dict[str, Any], with_question_text: bool
) -> dict[str, Any]:
    if action != ASK_ACTION or with_question_text:
        return details
    return {k: v for k, v in details.items() if k != QUESTION_KEY}
