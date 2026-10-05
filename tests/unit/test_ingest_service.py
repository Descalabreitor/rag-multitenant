"""IngestService checks that run before any SQL: ACLs, size, empty documents, dims.

The engine points at a port nobody listens on: every case here must fail (or
succeed) before the service opens a connection.
"""

from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.domain import ConvertedDocument, DocumentTooLargeError, UnsupportedDocumentError
from ragmt.ingest.service import IngestService, InvalidAclError, normalize_acl, source_hash
from ragmt.settings import Settings
from tests.ingest_helpers import CountingEmbeddings, FakeConverter, chunk_paragraphs

DIM = 4
UNREACHABLE = "postgresql+asyncpg://app_ingest:x@127.0.0.1:1/ragmt"


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "postgresql+asyncpg://app_rw:x@127.0.0.1:1/ragmt",
        "ingest_database_url": UNREACHABLE,
        "embedding_dim": DIM,
        "oidc_issuer": "http://localhost:8080/realms/ragmt",
        "oidc_audience": "ragmt-api",
        "permsync_client_id": "ragmt-permsync",
        "permsync_client_secret": "x",
        "ingest_max_bytes": 100,
        **overrides,
    }
    return Settings(_env_file=None, **values)


class _CountingConverter(FakeConverter):
    calls = 0

    def convert(self, data: bytes, filename: str) -> ConvertedDocument:
        self.calls += 1
        return super().convert(data, filename)


def service(converter: FakeConverter | None = None, dim: int = DIM) -> IngestService:
    return IngestService(
        create_async_engine(UNREACHABLE),
        converter or FakeConverter(),
        chunk_paragraphs,
        CountingEmbeddings(dim),
        settings(),
    )


def test_normalize_acl_deduplicates_in_order() -> None:
    assert normalize_acl(["group:hr", "user:a b", "group:hr", "tenant:*"]) == (
        "group:hr",
        "user:a b",
        "tenant:*",
    )


@pytest.mark.parametrize(
    "principal",
    ["", "hr", "user:", "group:", "group:  ", "tenant:", "tenant:other", "user:a\nb", "User:a"],
)
def test_normalize_acl_rejects_malformed_principals(principal: str) -> None:
    with pytest.raises(InvalidAclError):
        normalize_acl([principal])


def test_normalize_acl_rejects_an_empty_acl() -> None:
    with pytest.raises(InvalidAclError, match="empty"):
        normalize_acl([])


def test_source_hash_is_lowercase_sha256_hex() -> None:
    digest = source_hash(b"abc")
    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


async def test_too_large_is_refused_before_conversion() -> None:
    converter = _CountingConverter()
    with pytest.raises(DocumentTooLargeError):
        await service(converter).ingest(uuid4(), "admin", b"x" * 101, "big.md")
    assert converter.calls == 0


async def test_invalid_acl_is_refused_before_anything_else() -> None:
    converter = _CountingConverter()
    with pytest.raises(InvalidAclError):
        await service(converter).ingest(uuid4(), "admin", b"# t\n\nx", "a.md", acl=["nobody"])
    assert converter.calls == 0


async def test_empty_actor_is_refused() -> None:
    with pytest.raises(ValueError, match="actor_sub"):
        await service().ingest(uuid4(), "", b"# t\n\nx", "a.md")


async def test_a_document_without_text_is_unsupported() -> None:
    svc = service()
    with pytest.raises(UnsupportedDocumentError, match="no text"):
        await svc._prepare(b"# Only a heading", "empty.md")


def test_embedder_dim_must_match_the_setting() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        service(dim=DIM + 1)
