"""The LLM-as-judge of the quality baseline: faithfulness, correctness and abstention.

The judge is a chat model reached through the service's own adapters
(`OllamaChat`, `OpenAICompatChat`), configured apart from the service:

- QUALITY_JUDGE_PROVIDER: `ollama` (default) or `openai_compat`.
- QUALITY_JUDGE_MODEL: defaults to OLLAMA_CHAT_MODEL with Ollama, so by default
  the model grades its own answers (a known bias; docs/results/quality.md says so).
- QUALITY_JUDGE_BASE_URL: defaults to OLLAMA_BASE_URL; required for openai_compat.
- QUALITY_JUDGE_API_KEY: optional, for openai_compat.

Two prompts per answer, both asking for JSON only:

- **Faithfulness** (skipped for abstentions): the claims in the answer, each
  marked supported or not by the retrieved context, which is the exact chunk
  text the service put in the prompt. Score = supported / claims.
- **Verdict**: whether the answer abstains (says it doesn't know), and whether
  it gives the facts of the reference answer. A reply that is only one short
  "I don't know" sentence counts as abstaining without asking the judge.

Replies are parsed leniently (the first JSON object in the text); a reply that
still doesn't parse is retried once, then recorded as a judge error.
"""

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from ragmt.adapters.llm import ChatAdapter, OllamaChat, OpenAICompatChat
from ragmt.domain import ChatMessage

JUDGE_TIMEOUT_SECONDS = 300.0

# Citation markers say nothing the judge needs to check.
_MARKER = re.compile(r"\s*\[doc:[^\]]*\]")
# One sentence saying it doesn't know, and nothing after it.
_ABSTENTION = re.compile(r"(i don'?t know|i do not know)[^.?!]*[.!]?", re.IGNORECASE)
_PLAIN_ABSTENTION_MAX_CHARS = 120


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    model: str
    base_url: str
    api_key: SecretStr | None = None

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        provider = os.environ.get("QUALITY_JUDGE_PROVIDER") or "ollama"
        if provider == "ollama":
            model = os.environ.get("QUALITY_JUDGE_MODEL") or (
                os.environ.get("OLLAMA_CHAT_MODEL") or "llama3.1:8b"
            )
            base_url = os.environ.get("QUALITY_JUDGE_BASE_URL") or (
                os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
            )
            return cls(provider, model, base_url)
        if provider == "openai_compat":
            model = os.environ.get("QUALITY_JUDGE_MODEL", "")
            base_url = os.environ.get("QUALITY_JUDGE_BASE_URL", "")
            if not model or not base_url:
                raise ValueError(
                    "QUALITY_JUDGE_PROVIDER=openai_compat needs QUALITY_JUDGE_MODEL "
                    "and QUALITY_JUDGE_BASE_URL"
                )
            key = os.environ.get("QUALITY_JUDGE_API_KEY")
            return cls(provider, model, base_url, SecretStr(key) if key else None)
        raise ValueError(f"unknown QUALITY_JUDGE_PROVIDER {provider!r}")

    def build(self) -> ChatAdapter:
        http = httpx.AsyncClient(base_url=self.base_url)
        if self.provider == "openai_compat":
            return OpenAICompatChat(
                http, model=self.model, api_key=self.api_key, timeout_seconds=JUDGE_TIMEOUT_SECONDS
            )
        return OllamaChat(http, model=self.model, timeout_seconds=JUDGE_TIMEOUT_SECONDS)

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


FAITHFULNESS_SYSTEM = """\
You check answers written by a question-answering system against the context it \
was given. You reply with JSON only, no other text."""

FAITHFULNESS_PROMPT = """\
<context>
{context}
</context>

<answer>
{answer}
</answer>

List the factual claims made in the <answer> text, quoting or closely \
paraphrasing the answer. Take claims from the answer only, never from the \
context. Statements that something is unknown, or is not in the documents, are \
not claims. For each claim, decide whether the context states it or directly \
implies it. Do not use any knowledge outside the context: a claim that is true \
in the world but not in the context is not supported.

Reply with JSON in this shape, one entry per claim:
{{"claims": [{{"claim": "<claim from the answer>", "supported": <true or false>}}]}}
If the answer makes no factual claims (for example it only says it doesn't \
know), reply {{"claims": []}}."""

VERDICT_SYSTEM = """\
You grade answers written by a question-answering system against a reference \
answer. You reply with JSON only, no other text."""

VERDICT_PROMPT = """\
<question>
{question}
</question>

<reference>
{reference}
</reference>

<answer>
{answer}
</answer>

Decide two things:
- "abstains": true if the answer says it doesn't know, or that the documents \
don't contain the answer, instead of answering the question. Partly answering \
and then hedging is not abstaining.
- "correct": true if the answer gives the facts of the reference answer. \
Different wording is fine, and so is extra detail that doesn't contradict it. \
If the reference says the question can't be answered, "correct" is true only \
if the answer abstains.

Reply with JSON in this shape:
{{"abstains": <true or false>, "correct": <true or false>}}"""

UNANSWERABLE_REFERENCE = "The question cannot be answered from the documents available."


class JudgeError(Exception):
    """The judge's reply wasn't the JSON asked for, twice."""


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text).strip()


def plain_abstention(answer: str) -> bool:
    """A short reply that only says it doesn't know, such as "I don't know.".

    These skip the judge: in trials, llama3.1:8b as judge called "I don't
    know." a non-abstention and listed claims from the context as the answer's.
    Anything longer is judged.
    """
    text = strip_markers(answer)
    return len(text) <= _PLAIN_ABSTENTION_MAX_CHARS and bool(_ABSTENTION.fullmatch(text))


def first_json_object(text: str) -> dict[str, Any]:
    """The first JSON object in `text` (models wrap JSON in prose or code fences)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the reply")


@dataclass(frozen=True)
class Faithfulness:
    supported: int
    claims: int
    unsupported: tuple[str, ...]

    @property
    def score(self) -> float | None:
        return None if self.claims == 0 else self.supported / self.claims


@dataclass(frozen=True)
class Verdict:
    abstains: bool
    correct: bool


def parse_faithfulness(body: dict[str, Any]) -> Faithfulness:
    claims = body.get("claims")
    if not isinstance(claims, list):
        raise ValueError("`claims` is not a list")
    marks = [c for c in claims if isinstance(c, dict) and isinstance(c.get("supported"), bool)]
    if len(marks) != len(claims):
        raise ValueError("a claim has no boolean `supported`")
    unsupported = tuple(str(c.get("claim", "")) for c in marks if not c["supported"])
    return Faithfulness(len(marks) - len(unsupported), len(marks), unsupported)


def parse_verdict(body: dict[str, Any]) -> Verdict:
    abstains, correct = body.get("abstains"), body.get("correct")
    if not isinstance(abstains, bool) or not isinstance(correct, bool):
        raise ValueError("`abstains` and `correct` must be booleans")
    return Verdict(abstains, correct)


class Judge:
    def __init__(self, chat: ChatAdapter) -> None:
        self._chat = chat

    @property
    def model(self) -> str:
        return self._chat.model

    async def _ask(self, system: str, prompt: str) -> dict[str, Any]:
        messages: list[ChatMessage] = [ChatMessage(role="user", content=prompt)]
        for _ in range(2):
            reply = await self._chat.complete(system, messages)
            try:
                return first_json_object(reply)
            except ValueError:
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=reply),
                    ChatMessage(role="user", content="Reply with the JSON object only."),
                ]
        raise JudgeError("the judge did not reply with JSON")

    async def faithfulness(self, answer: str, context: Sequence[str]) -> Faithfulness:
        prompt = FAITHFULNESS_PROMPT.format(
            context="\n\n---\n\n".join(context) or "(empty)", answer=strip_markers(answer)
        )
        for attempt in range(2):
            body = await self._ask(FAITHFULNESS_SYSTEM, prompt)
            try:
                return parse_faithfulness(body)
            except ValueError:
                if attempt:
                    raise JudgeError("faithfulness reply has the wrong shape") from None
        raise AssertionError("unreachable")

    async def verdict(self, question: str, reference: str, answer: str) -> Verdict:
        prompt = VERDICT_PROMPT.format(
            question=question,
            reference=reference or UNANSWERABLE_REFERENCE,
            answer=strip_markers(answer),
        )
        for attempt in range(2):
            body = await self._ask(VERDICT_SYSTEM, prompt)
            try:
                return parse_verdict(body)
            except ValueError:
                if attempt:
                    raise JudgeError("verdict reply has the wrong shape") from None
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._chat.aclose()
