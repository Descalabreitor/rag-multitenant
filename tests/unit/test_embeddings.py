"""Embedding adapters against httpx.MockTransport (no Ollama), and the fake provider."""

import json
import logging
import math
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from ragmt.adapters.llm import (
    EmbeddingDimensionError,
    EmbeddingError,
    FakeEmbeddings,
    OllamaEmbeddings,
    OpenAICompatEmbeddings,
    build_embedding_provider,
)
from ragmt.adapters.llm._http import HttpEmbeddings
from ragmt.domain import EmbeddingProvider
from ragmt.settings import Settings

DIM = 4
# Stands in for tenant data: it must never reach a log line or an exception message.
CANARY_TEXT = "canary-confidential-payroll-figures"
API_KEY = "sk-canary-api-key"

Handler = Callable[[httpx.Request], httpx.Response]


class FakeServer:
    """Records requests and answers each one with the next handler (the last one repeats)."""

    def __init__(self, *handlers: Handler) -> None:
        self.handlers = list(handlers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.handlers[min(len(self.requests), len(self.handlers)) - 1]
        return handler(request)

    def inputs(self) -> list[list[str]]:
        return [json.loads(request.content)["input"] for request in self.requests]


def vector_for(text: str, dim: int = DIM) -> list[float]:
    """A vector the test can trace back to its input: its length in the first slot."""
    return [float(len(text))] + [0.0] * (dim - 1)


def ollama_ok(dim: int = DIM) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [vector_for(t, dim) for t in texts]})

    return handle


def openai_ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": vector_for(t)} for i, t in enumerate(texts)]
    # Reversed: the adapter must restore input order from `index`.
    return httpx.Response(200, json={"object": "list", "data": data[::-1]})


def status(code: int) -> Handler:
    # The body echoes the input, as some servers do in error messages.
    return lambda request: httpx.Response(code, text=f"error for {request.content!r}")


def connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def client(server: FakeServer, base_url: str = "http://ollama.test") -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(server))


def ollama(
    server: FakeServer, *, batch_size: int = 3, sleeps: Sleeps | None = None
) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        client(server),
        model="nomic-embed-text",
        dim=DIM,
        batch_size=batch_size,
        sleep=sleeps or Sleeps(),
    )


def openai(server: FakeServer, *, api_key: SecretStr | None = None) -> OpenAICompatEmbeddings:
    return OpenAICompatEmbeddings(
        client(server, "https://llm.test"),
        model="embed-small",
        dim=DIM,
        batch_size=2,
        api_key=api_key,
        sleep=Sleeps(),
    )


# --- Ollama -----------------------------------------------------------------------


async def test_ollama_request_shape() -> None:
    server = FakeServer(ollama_ok())
    await ollama(server).embed_documents(["a"])
    [request] = server.requests
    assert request.method == "POST"
    assert request.url == "http://ollama.test/api/embed"
    assert json.loads(request.content) == {
        "model": "nomic-embed-text",
        "input": ["search_document: a"],
    }


async def test_ollama_batches_and_keeps_order() -> None:
    server = FakeServer(ollama_ok())
    texts = [f"text {'x' * i}" for i in range(7)]
    vectors = await ollama(server, batch_size=3).embed_documents(texts)
    assert [len(batch) for batch in server.inputs()] == [3, 3, 1]
    assert vectors == [vector_for("search_document: " + t) for t in texts]


async def test_no_texts_no_request() -> None:
    server = FakeServer(ollama_ok())
    assert await ollama(server).embed_documents([]) == []
    assert server.requests == []


async def test_ollama_prefixes_documents_and_queries() -> None:
    server = FakeServer(ollama_ok())
    embeddings = ollama(server)
    await embeddings.embed_documents(["passage"])
    await embeddings.embed_query("question")
    assert server.inputs() == [["search_document: passage"], ["search_query: question"]]


@pytest.mark.parametrize("failure", [status(500), status(503), connect_error])
async def test_retries_transient_failures_with_backoff(failure: Handler) -> None:
    server, sleeps = FakeServer(failure, failure, ollama_ok()), Sleeps()
    vector = await ollama(server, sleeps=sleeps).embed_query("q")
    assert vector == vector_for("search_query: q")
    assert len(server.requests) == 3
    assert sleeps.delays == [0.5, 1.0]


@pytest.mark.parametrize("failure", [status(502), connect_error])
async def test_gives_up_after_two_retries(failure: Handler) -> None:
    server, sleeps = FakeServer(failure), Sleeps()
    with pytest.raises(EmbeddingError):
        await ollama(server, sleeps=sleeps).embed_query("q")
    assert len(server.requests) == 3
    assert len(sleeps.delays) == 2


@pytest.mark.parametrize("code", [400, 401, 404, 413, 429])
async def test_never_retries_4xx(code: int) -> None:
    server, sleeps = FakeServer(status(code), ollama_ok()), Sleeps()
    with pytest.raises(EmbeddingError, match=f"HTTP {code}"):
        await ollama(server, sleeps=sleeps).embed_query("q")
    assert len(server.requests) == 1
    assert sleeps.delays == []


async def test_timeout_is_set_per_request() -> None:
    seen: list[object] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return ollama_ok()(request)

    embeddings = OllamaEmbeddings(
        client(FakeServer(handle)), model="m", dim=DIM, batch_size=8, timeout_seconds=7.5
    )
    await embeddings.embed_query("q")
    assert seen == [{"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}]


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [[0.1, 0.2]]},  # too short
        {"embeddings": [[0.0] * (DIM + 1)]},  # too long
    ],
)
async def test_wrong_dimension_is_rejected(body: object) -> None:
    server = FakeServer(lambda request: httpx.Response(200, json=body))
    with pytest.raises(EmbeddingDimensionError):
        await ollama(server).embed_documents(["a"])


async def test_check_passes_on_the_right_dimension() -> None:
    server = FakeServer(ollama_ok())
    await ollama(server).check()
    assert len(server.requests) == 1


async def test_check_fails_on_the_wrong_dimension() -> None:
    server = FakeServer(ollama_ok(dim=DIM * 2))
    with pytest.raises(EmbeddingDimensionError, match=f"expected {DIM}"):
        await ollama(server).check()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"embedding": [0.0] * DIM}),  # old /api/embeddings shape
        httpx.Response(200, json={"embeddings": [[0.0] * DIM] * 2}),  # 2 vectors for 1 input
    ],
)
async def test_unusable_responses_raise(response: httpx.Response) -> None:
    with pytest.raises(EmbeddingError):
        await ollama(FakeServer(lambda request: response)).embed_query("q")


# --- OpenAI-compatible ------------------------------------------------------------


async def test_openai_compat_request_and_order() -> None:
    server = FakeServer(openai_ok)
    texts = ["a", "bb", "ccc"]
    vectors = await openai(server, api_key=SecretStr(API_KEY)).embed_documents(texts)
    assert vectors == [vector_for(t) for t in texts]
    assert [len(batch) for batch in server.inputs()] == [2, 1]
    request = server.requests[0]
    assert request.url == "https://llm.test/v1/embeddings"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.content)
    assert body == {"model": "embed-small", "input": ["a", "bb"], "encoding_format": "float"}


async def test_openai_compat_without_key_sends_no_authorization() -> None:
    server = FakeServer(openai_ok)
    await openai(server).embed_query("q")
    assert "Authorization" not in server.requests[0].headers


async def test_openai_compat_does_not_retry_auth_errors() -> None:
    server = FakeServer(status(401), openai_ok)
    with pytest.raises(EmbeddingError, match="HTTP 401"):
        await openai(server, api_key=SecretStr(API_KEY)).embed_query("q")
    assert len(server.requests) == 1


async def test_openai_compat_checks_dimension() -> None:
    def short(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    with pytest.raises(EmbeddingDimensionError):
        await openai(FakeServer(short)).check()


async def test_openai_compat_rejects_bad_indices() -> None:
    def duplicated(request: httpx.Request) -> httpx.Response:
        item = {"index": 0, "embedding": [0.0] * DIM}
        return httpx.Response(200, json={"data": [item, item]})

    with pytest.raises(EmbeddingError, match="indices"):
        await openai(FakeServer(duplicated)).embed_documents(["a", "b"])


# --- nothing sensitive in logs or errors -------------------------------------------


async def test_logs_and_errors_hold_no_input_text_or_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    errors: list[str] = []

    def with_key(server: FakeServer) -> OpenAICompatEmbeddings:
        return openai(server, api_key=SecretStr(API_KEY))

    cases: list[tuple[Callable[[FakeServer], HttpEmbeddings], Handler]] = [
        (ollama, status(500)),
        (ollama, connect_error),
        (ollama, status(400)),
        (ollama, lambda r: httpx.Response(200, json={"embeddings": CANARY_TEXT})),
        (ollama, lambda r: httpx.Response(200, text=CANARY_TEXT)),
        (with_key, status(503)),
        (with_key, lambda r: httpx.Response(200, json={"data": CANARY_TEXT})),
    ]
    for make, handler in cases:
        embeddings = make(FakeServer(handler))
        with pytest.raises(EmbeddingError) as excinfo:
            await embeddings.embed_documents([CANARY_TEXT])
        error: BaseException | None = excinfo.value
        while error is not None:  # the whole chain, as a traceback would print it
            errors.append(repr(error))
            error = error.__cause__ or error.__context__

    assert "retry" in caplog.text  # the retries were logged...
    for leaked in (CANARY_TEXT, API_KEY):  # ...without the inputs or the key
        assert leaked not in caplog.text
        assert not [e for e in errors if leaked in e]


# --- fake -------------------------------------------------------------------------


async def test_fake_is_deterministic() -> None:
    first, second = FakeEmbeddings(768), FakeEmbeddings(768)
    texts = ["alpha", "beta", "alpha"]
    vectors = await first.embed_documents(texts)
    assert vectors == await second.embed_documents(texts)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]
    assert await first.embed_query("alpha") == vectors[0]


@pytest.mark.parametrize("dim", [1, 3, 768])
async def test_fake_has_dim_floats_and_unit_norm(dim: int) -> None:
    fake = FakeEmbeddings(dim)
    assert fake.dim == dim
    for vector in await fake.embed_documents(["", "a", "ü" * 1000]):
        assert len(vector) == dim
        assert math.isclose(math.fsum(v * v for v in vector), 1.0, rel_tol=1e-9)
    await fake.check()


async def test_fake_makes_no_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("FakeEmbeddings opened an HTTP client")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_network)
    settings = make_settings(LLM_PROVIDER="fake")
    provider = build_embedding_provider(settings)
    await provider.embed_documents(["a"])
    await provider.aclose()


def test_adapters_satisfy_the_port() -> None:
    providers: list[EmbeddingProvider] = [
        FakeEmbeddings(DIM),
        ollama(FakeServer(ollama_ok())),
        openai(FakeServer(openai_ok)),
    ]
    assert all(p.dim == DIM for p in providers)


# --- build_embedding_provider ------------------------------------------------------

BASE_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://app_rw:x@127.0.0.1:5432/ragmt",
    "INGEST_DATABASE_URL": "postgresql+asyncpg://app_ingest:x@127.0.0.1:5432/ragmt",
    "OIDC_ISSUER": "http://localhost:8080/realms/ragmt",
    "OIDC_AUDIENCE": "ragmt-api",
    "PERMSYNC_CLIENT_ID": "ragmt-permsync",
    "PERMSYNC_CLIENT_SECRET": "x",
    "EMBEDDING_DIM": "768",
    "EMBED_BATCH_SIZE": "16",
    "OLLAMA_BASE_URL": "http://ollama.test:11434",
    "OLLAMA_EMBED_MODEL": "nomic-embed-text",
    "OPENAI_COMPAT_BASE_URL": "https://llm.test",
    "OPENAI_COMPAT_API_KEY": API_KEY,
    "OPENAI_COMPAT_EMBED_MODEL": "embed-small",
    "OPENAI_COMPAT_CHAT_MODEL": "chat-large",
}


def make_settings(**env: str) -> Settings:
    # Passed as init values, so neither .env nor os.environ can change them.
    values = {key.lower(): value for key, value in {**BASE_ENV, **env}.items()}
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("ollama", OllamaEmbeddings),
        ("openai_compat", OpenAICompatEmbeddings),
        ("fake", FakeEmbeddings),
    ],
)
async def test_factory_picks_the_adapter(provider: str, expected: type) -> None:
    built = build_embedding_provider(make_settings(LLM_PROVIDER=provider))
    try:
        assert isinstance(built, expected)
        assert built.dim == 768
    finally:
        await built.aclose()


async def test_factory_wires_ollama_settings() -> None:
    built = build_embedding_provider(make_settings(LLM_PROVIDER="ollama"))
    assert isinstance(built, OllamaEmbeddings)
    try:
        assert str(built._http.base_url) == "http://ollama.test:11434"
        assert built._batch_size == 16
    finally:
        await built.aclose()


def test_unknown_provider_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="llm_provider"):
        make_settings(LLM_PROVIDER="anthropic")
    # Settings built without validation still can't produce an adapter.
    unchecked = make_settings().model_copy(update={"llm_provider": "anthropic"})
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        build_embedding_provider(unchecked)
