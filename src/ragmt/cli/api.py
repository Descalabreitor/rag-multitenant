"""`POST /ask` from the client side, and how its answer is printed.

The response is the API's `Answer` (ADR 0009): `answer`, `citations` (document
id, title, heading) and `model`. Everything in it is untrusted text (the model
wrote the answer, uploaders wrote the titles), so control characters are
removed before printing: no escape sequence from a document reaches the
terminal.
"""

import re
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

# Fail rather than wait forever, but leave room for CHAT_TIMEOUT_SECONDS (120 s).
ASK_TIMEOUT = httpx.Timeout(10.0, read=180.0)
# C0 and C1 controls and DEL (except newline and tab), and the bidi controls
# that can make printed text read differently from what it is.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


class ApiError(RuntimeError):
    """The API call failed. The message is safe to print."""


class Unauthorized(ApiError):
    """The API rejected the token (401)."""


class CitationOut(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    document_id: UUID
    title: str
    heading: str | None = None


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    answer: str
    citations: list[CitationOut] = []
    model: str | None = None


def _detail(response: httpx.Response) -> str | None:
    """The API's own `detail`, when it is a short string (FastAPI's error shape)."""
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and len(detail) <= 200:
        return clean(detail)
    return None


def ask(client: httpx.Client, api_url: str, access_token: str, question: str) -> AskResponse:
    try:
        response = client.post(
            f"{api_url}/ask",
            json={"question": question},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=ASK_TIMEOUT,
        )
    except httpx.TimeoutException as exc:
        raise ApiError("the API did not answer in time") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"could not reach the API at {api_url}") from exc
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise Unauthorized("the API rejected the token; run `ragctl login`")
    if response.status_code != httpx.codes.OK:
        detail = _detail(response)
        raise ApiError(
            f"the API returned HTTP {response.status_code}" + (f": {detail}" if detail else "")
        )
    try:
        return AskResponse.model_validate_json(response.content)
    except ValidationError as exc:
        raise ApiError("the API returned an unexpected response") from exc


def clean(text: str) -> str:
    """`text` without terminal control characters (newlines and tabs are kept)."""
    return _CONTROL.sub("", text.replace("\r\n", "\n"))


def _one_line(text: str) -> str:
    return " ".join(clean(text).split())


def format_answer(answer: AskResponse) -> str:
    """The answer, then one numbered line per citation: `title > heading`."""
    lines = [clean(answer.answer).strip()]
    if answer.citations:
        lines += ["", "Sources:"]
        for number, citation in enumerate(answer.citations, start=1):
            source = _one_line(citation.title)
            if citation.heading:
                source += f" > {_one_line(citation.heading)}"
            lines.append(f"  [{number}] {source}")
    return "\n".join(lines)
