"""Chat adapters against httpx.MockTransport (no Ollama), and the FakeChat test double."""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from ragmt.adapters.llm import (
    ChatUnavailableError,
    FakeChat,
    OllamaChat,
    OpenAICompatChat,
    build_chat_provider,
)
from ragmt.adapters.llm._http import HttpChat
from ragmt.domain import ChatMessage, ChatProvider, citation_marker
from tests.unit.test_embeddings import (
    API_KEY,
    FakeServer,
    Handler,
    Sleeps,
    client,
    connect_error,
    make_settings,
    status,
)

# Stand in for retrieved chunks and the model's reply: never in a log line or an error.
CANARY_PROMPT = "canary-confidential-chunk-text"
CANARY_ANSWER = "canary-confidential-answer"
SYSTEM = "Answer from the delimited sources only."
MESSAGES = [ChatMessage("user", f"What does it say? {CANARY_PROMPT}")]

DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")


def ollama_ok(request: httpx.Request) -> httpx.Response:
    body = {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": CANARY_ANSWER},
        "done": True,
        "done_reason": "stop",
    }
    return httpx.Response(200, json=body)


def openai_ok(request: httpx.Request) -> httpx.Response:
    choice = {"index": 0, "message": {"role": "assistant", "content": CANARY_ANSWER}}
    return httpx.Response(200, json={"object": "chat.completion", "choices": [choice]})


def read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def connect_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("timed out", request=request)


def ollama(server: FakeServer, **options: Any) -> OllamaChat:
    return OllamaChat(client(server), model="llama3.1:8b", **{"sleep": Sleeps(), **options})


def openai(server: FakeServer, **options: Any) -> OpenAICompatChat:
    return OpenAICompatChat(
        client(server, "https://llm.test"), model="chat-large", **{"sleep": Sleeps(), **options}
    )


# Each adapter with a handler that answers successfully.
ADAPTERS: list[tuple[Callable[..., HttpChat], Handler]] = [
    (ollama, ollama_ok),
    (openai, openai_ok),
]
adapters = pytest.mark.parametrize(("make", "ok"), ADAPTERS, ids=["ollama", "openai_compat"])


def payload(request: httpx.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.content)
    return body


# --- request shape ------------------------------------------------------------------


async def test_ollama_request_shape() -> None:
    server = FakeServer(ollama_ok)
    reply = await ollama(server).complete(SYSTEM, MESSAGES)

    assert reply == CANARY_ANSWER
    [request] = server.requests
    assert request.method == "POST"
    assert request.url == "http://ollama.test/api/chat"
    assert payload(request) == {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": MESSAGES[0].content},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }


async def test_openai_compat_request_shape() -> None:
    server = FakeServer(openai_ok)
    turns = [*MESSAGES, ChatMessage("assistant", "earlier"), ChatMessage("user", "again")]
    reply = await openai(server, api_key=SecretStr(API_KEY)).complete(SYSTEM, turns)

    assert reply == CANARY_ANSWER
    [request] = server.requests
    assert request.method == "POST"
    assert request.url == "https://llm.test/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert payload(request) == {
        "model": "chat-large",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": MESSAGES[0].content},
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "again"},
        ],
        "stream": False,
        "temperature": 0,
    }


@adapters
async def test_no_tools_are_offered(make: Callable[..., HttpChat], ok: Handler) -> None:
    server = FakeServer(ok)
    await make(server).complete(SYSTEM, MESSAGES)
    body = payload(server.requests[0])
    assert not {"tools", "tool_choice", "functions", "function_call"} & body.keys()


async def test_openai_compat_without_key_sends_no_authorization() -> None:
    server = FakeServer(openai_ok)
    await openai(server).complete(SYSTEM, MESSAGES)
    assert "Authorization" not in server.requests[0].headers


@adapters
async def test_timeout_is_set_per_request(make: Callable[..., HttpChat], ok: Handler) -> None:
    seen: list[object] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return ok(request)

    await make(FakeServer(handle), timeout_seconds=7.5).complete(SYSTEM, MESSAGES)
    assert seen == [{"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}]


# --- retries ------------------------------------------------------------------------


@adapters
@pytest.mark.parametrize("failure", [status(500), status(503), connect_error, connect_timeout])
async def test_retries_transient_failures_with_backoff(
    make: Callable[..., HttpChat], ok: Handler, failure: Handler
) -> None:
    server, sleeps = FakeServer(failure, failure, ok), Sleeps()
    assert await make(server, sleep=sleeps).complete(SYSTEM, MESSAGES) == CANARY_ANSWER
    assert len(server.requests) == 3
    assert sleeps.delays == [0.5, 1.0]


@adapters
@pytest.mark.parametrize("failure", [status(502), connect_error])
async def test_gives_up_after_two_retries(
    make: Callable[..., HttpChat], ok: Handler, failure: Handler
) -> None:
    server, sleeps = FakeServer(failure), Sleeps()
    with pytest.raises(ChatUnavailableError):
        await make(server, sleep=sleeps).complete(SYSTEM, MESSAGES)
    assert len(server.requests) == 3
    assert len(sleeps.delays) == 2


@adapters
@pytest.mark.parametrize("code", [400, 401, 403, 404, 413, 422, 429])
async def test_never_retries_4xx(make: Callable[..., HttpChat], ok: Handler, code: int) -> None:
    server, sleeps = FakeServer(status(code), ok), Sleeps()
    with pytest.raises(ChatUnavailableError, match=f"HTTP {code}"):
        await make(server, sleep=sleeps).complete(SYSTEM, MESSAGES)
    assert len(server.requests) == 1
    assert sleeps.delays == []


# --- timeouts -----------------------------------------------------------------------


@adapters
async def test_read_timeout_fails_without_retry(make: Callable[..., HttpChat], ok: Handler) -> None:
    server, sleeps = FakeServer(read_timeout, ok), Sleeps()
    with pytest.raises(ChatUnavailableError, match="ReadTimeout"):
        await make(server, sleep=sleeps).complete(SYSTEM, MESSAGES)
    assert len(server.requests) == 1
    assert sleeps.delays == []


async def test_whole_completion_is_bounded_by_the_timeout() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return ollama_ok(request)

    chat = OllamaChat(
        httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(slow)),
        model="llama3.1:8b",
        timeout_seconds=0.05,
    )
    with pytest.raises(ChatUnavailableError, match=r"took over 0\.05 s"):
        await chat.complete(SYSTEM, MESSAGES)


async def test_retries_count_against_the_timeout() -> None:
    # Real sleeps: 0.5 s of backoff can't fit in a 0.05 s budget.
    chat = OllamaChat(
        client(FakeServer(status(503), ollama_ok)), model="llama3.1:8b", timeout_seconds=0.05
    )
    with pytest.raises(ChatUnavailableError, match="took over"):
        await chat.complete(SYSTEM, MESSAGES)


# --- unusable responses -------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"done": True}),
        httpx.Response(200, json={"message": {"role": "assistant"}}),
        httpx.Response(200, json={"message": {"content": None}}),
    ],
)
async def test_ollama_unusable_responses_raise(response: httpx.Response) -> None:
    with pytest.raises(ChatUnavailableError):
        await ollama(FakeServer(lambda request: response)).complete(SYSTEM, MESSAGES)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
        # A refusal or a tool call: no text to return.
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, json={"choices": [{"message": {"tool_calls": []}}]}),
    ],
)
async def test_openai_compat_unusable_responses_raise(response: httpx.Response) -> None:
    with pytest.raises(ChatUnavailableError):
        await openai(FakeServer(lambda request: response)).complete(SYSTEM, MESSAGES)


# --- nothing sensitive in logs or errors -------------------------------------------


async def test_logs_and_errors_hold_no_prompt_answer_or_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    errors: list[str] = []

    def with_key(server: FakeServer, **kwargs: Any) -> OpenAICompatChat:
        return openai(server, api_key=SecretStr(API_KEY), **kwargs)

    # The 200 bodies carry the answer where the adapter doesn't expect it.
    failing: list[tuple[Callable[..., HttpChat], Handler]] = [
        (ollama, status(500)),
        (ollama, connect_error),
        (ollama, read_timeout),
        (ollama, status(400)),
        (ollama, lambda r: httpx.Response(200, json={"message": CANARY_ANSWER})),
        (ollama, lambda r: httpx.Response(200, text=CANARY_ANSWER)),
        (with_key, status(503)),
        (with_key, status(401)),
        (with_key, lambda r: httpx.Response(200, json={"choices": CANARY_ANSWER})),
    ]
    for make, handler in failing:
        with pytest.raises(ChatUnavailableError) as excinfo:
            await make(FakeServer(handler)).complete(SYSTEM, MESSAGES)
        error: BaseException | None = excinfo.value
        while error is not None:  # the whole chain, as a traceback would print it
            errors.append(repr(error))
            error = error.__cause__ or error.__context__
    # And a success after a retry: the answer is returned, not logged.
    for make, ok in ADAPTERS:
        await make(FakeServer(status(500), ok)).complete(SYSTEM, MESSAGES)

    assert "retry" in caplog.text  # the retries were logged...
    for leaked in (CANARY_PROMPT, CANARY_ANSWER, API_KEY):  # ...without anything sensitive
        assert leaked not in caplog.text
        assert not [e for e in errors if leaked in e]


# --- FakeChat -----------------------------------------------------------------------


def context_prompt() -> str:
    # The real delimiter format belongs to the prompt builder; any text with markers will do.
    return (
        f"<source id={citation_marker(DOC_B, 3)}>B three</source>\n"
        f"<source id={citation_marker(DOC_A, 0)}>A zero, cites {citation_marker(DOC_B, 3)}</source>"
    )


async def test_fake_echoes_markers_and_content() -> None:
    reply = await FakeChat().complete(SYSTEM, [ChatMessage("user", context_prompt())])
    lines = reply.splitlines()
    assert "test double" in lines[0]
    # In order of first appearance, without repeats.
    assert lines[1] == f"Cited: {citation_marker(DOC_B, 3)} {citation_marker(DOC_A, 0)}"
    assert SYSTEM in reply
    assert context_prompt() in reply


async def test_fake_without_markers_cites_none() -> None:
    reply = await FakeChat().complete(SYSTEM, MESSAGES)
    assert reply.splitlines()[1] == "Cited: (none)"
    assert CANARY_PROMPT in reply


async def test_fake_is_deterministic() -> None:
    messages = [ChatMessage("user", context_prompt()), ChatMessage("assistant", "x")]
    first, second = FakeChat(), FakeChat()
    reply = await first.complete(SYSTEM, messages)
    assert reply == await first.complete(SYSTEM, messages)
    assert reply == await second.complete(SYSTEM, messages)
    assert reply != await first.complete(SYSTEM, messages[:1])


async def test_fake_records_calls() -> None:
    fake = FakeChat()
    assert fake.calls == []
    await fake.complete(SYSTEM, MESSAGES)
    assert fake.calls == [(SYSTEM, tuple(MESSAGES))]
    assert fake.model == "fake-chat"
    assert "test double" in (FakeChat.__doc__ or "").lower()


async def test_fake_makes_no_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("FakeChat opened an HTTP client")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_network)
    chat = build_chat_provider(make_settings(LLM_PROVIDER="fake"))
    await chat.complete(SYSTEM, MESSAGES)
    await chat.aclose()


def test_adapters_satisfy_the_port() -> None:
    providers: list[ChatProvider] = [
        FakeChat(),
        ollama(FakeServer(ollama_ok)),
        openai(FakeServer(openai_ok)),
    ]
    assert [p.model for p in providers] == ["fake-chat", "llama3.1:8b", "chat-large"]


# --- build_chat_provider -------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("ollama", OllamaChat), ("openai_compat", OpenAICompatChat), ("fake", FakeChat)],
)
async def test_factory_picks_the_adapter(provider: str, expected: type) -> None:
    built = build_chat_provider(make_settings(LLM_PROVIDER=provider))
    try:
        assert isinstance(built, expected)
    finally:
        await built.aclose()


async def test_factory_wires_ollama_settings() -> None:
    settings = make_settings(
        LLM_PROVIDER="ollama", OLLAMA_CHAT_MODEL="llama3.1:8b", CHAT_TIMEOUT_SECONDS="45"
    )
    built = build_chat_provider(settings)
    assert isinstance(built, OllamaChat)
    try:
        assert built.model == "llama3.1:8b"
        assert str(built._http.base_url) == "http://ollama.test:11434"
        assert built._timeout_seconds == 45.0
    finally:
        await built.aclose()


async def test_factory_wires_openai_compat_settings() -> None:
    built = build_chat_provider(make_settings(LLM_PROVIDER="openai_compat"))
    assert isinstance(built, OpenAICompatChat)
    try:
        assert built.model == "chat-large"
        assert str(built._http.base_url) == "https://llm.test"
        assert built._api_key is not None
        assert built._api_key.get_secret_value() == API_KEY
        assert built._timeout_seconds == 120.0
    finally:
        await built.aclose()


@pytest.mark.parametrize(
    ("llm_provider", "chat_provider", "expected"),
    [("ollama", "fake", FakeChat), ("fake", "ollama", OllamaChat), ("ollama", "", OllamaChat)],
)
async def test_chat_provider_overrides_llm_provider_for_chat_only(
    llm_provider: str, chat_provider: str, expected: type
) -> None:
    # `make e2e FAKE_CHAT=1`: Ollama embeddings, FakeChat answers.
    settings = make_settings(LLM_PROVIDER=llm_provider, CHAT_PROVIDER=chat_provider)
    built = build_chat_provider(settings)
    try:
        assert isinstance(built, expected)
    finally:
        await built.aclose()


def test_unknown_provider_has_no_chat_adapter() -> None:
    unchecked = make_settings().model_copy(update={"llm_provider": "anthropic"})
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        build_chat_provider(unchecked)
