"""The seed's files and its embedder check, without a database or Ollama."""

import json
from pathlib import PurePosixPath

import httpx
import pytest

from ragmt.ingest.chunking import chunk
from ragmt.ingest.convert import UploadConverter
from ragmt.settings import Settings
from seed.corpus import TENANTS, Document
from seed.docx import build_docx
from seed.embedder import SeedError, open_embedder, require_ollama_model
from seed.load import FILES, document_bytes, source_path

DOCUMENTS = [doc for tenant in TENANTS for doc in tenant.documents]


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variables Settings requires; .env is never read (`_env_file=None`)."""
    for key, value in {
        "DATABASE_URL": "postgresql+asyncpg://app_rw@127.0.0.1:5432/ragmt",
        "INGEST_DATABASE_URL": "postgresql+asyncpg://app_ingest@127.0.0.1:5432/ragmt",
        "OIDC_ISSUER": "http://localhost:8080/realms/ragmt",
        "OIDC_AUDIENCE": "ragmt-api",
        "PERMSYNC_CLIENT_SECRET": "unused",
        "EMBEDDING_DIM": "768",
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://ollama.test",
        "OLLAMA_EMBED_MODEL": "nomic-embed-text",
    }.items():
        monkeypatch.setenv(key, value)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --- files ------------------------------------------------------------------------------


def test_the_corpus_mixes_markdown_html_and_docx() -> None:
    suffixes = {PurePosixPath(doc.path).suffix for doc in DOCUMENTS}
    assert suffixes == {".md", ".html", ".docx"}


def test_seed_files_are_exactly_the_corpus() -> None:
    on_disk = {p for p in FILES.rglob("*") if p.is_file()}
    assert on_disk == {source_path(doc) for doc in DOCUMENTS}


@pytest.mark.parametrize("doc", DOCUMENTS, ids=lambda d: d.path)
def test_each_file_converts_to_its_title_and_canary(doc: Document) -> None:
    data = document_bytes(doc)
    converted = UploadConverter(max_bytes=10 * 1024 * 1024).convert(
        data, PurePosixPath(doc.path).name
    )
    assert converted.title == doc.title
    drafts = chunk(converted.markdown, max_chars=2000, overlap_chars=200)
    assert any(doc.canary in d.content for d in drafts)


@pytest.mark.parametrize("doc", DOCUMENTS, ids=lambda d: d.path)
def test_seed_files_have_lf_line_endings(doc: Document) -> None:
    """The seed is deduplicated by hash, so a CRLF checkout would re-ingest everything
    (.gitattributes keeps seed/files/ at LF)."""
    assert b"\r" not in source_path(doc).read_bytes()


def test_built_docx_does_not_depend_on_the_platform() -> None:
    data = build_docx("# Title\n\nText")
    # Version made by: (create_system << 8) | create_version; 0 is MS-DOS everywhere.
    central = data.find(b"PK\x01\x02")
    assert central > 0
    assert data[central + 5] == 0
    assert build_docx("# Title\r\n\r\nText") == data


# --- the Ollama check -------------------------------------------------------------------


def _tags(*names: str) -> httpx.MockTransport:
    body = json.dumps({"models": [{"name": n} for n in names]}).encode()
    return httpx.MockTransport(lambda request: httpx.Response(200, content=body))


@pytest.mark.parametrize("pulled", ["nomic-embed-text:latest", "nomic-embed-text"])
async def test_a_pulled_model_passes(pulled: str) -> None:
    await require_ollama_model(_settings(), _tags("llama3.1:8b", pulled))


async def test_a_missing_model_says_how_to_pull_it() -> None:
    with pytest.raises(SeedError, match=r"no model 'nomic-embed-text'.*ollama pull"):
        await require_ollama_model(_settings(), _tags("llama3.1:8b"))


async def test_another_tag_of_the_model_does_not_count() -> None:
    with pytest.raises(SeedError, match="no model"):
        await require_ollama_model(
            _settings(ollama_embed_model="nomic-embed-text:v1.5"),
            _tags("nomic-embed-text:latest"),
        )


async def test_an_unreachable_ollama_says_how_to_start_it() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(SeedError, match=r"does not answer.*docker compose up -d ollama"):
        await require_ollama_model(_settings(), httpx.MockTransport(refuse))


async def test_a_garbled_tags_response_is_a_seed_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"[]"))
    with pytest.raises(SeedError, match="does not answer"):
        await require_ollama_model(_settings(), transport)


async def test_the_fake_provider_needs_nothing() -> None:
    embedder = await open_embedder(_settings(llm_provider="fake"))
    try:
        assert embedder.dim == _settings().embedding_dim
    finally:
        await embedder.aclose()
