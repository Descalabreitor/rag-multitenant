"""The streaming multipart reader of POST /documents: what it accepts, and where it stops."""

from collections.abc import AsyncIterator

import pytest

from ragmt.api.uploads import (
    MAX_FIELD_BYTES,
    MAX_PARTS,
    MalformedUploadError,
    NotMultipartError,
    UploadTooLargeError,
    read_upload_form,
)
from tests.upload_helpers import STREAM_CONTENT_TYPE, CountingStream, multipart

LIMIT = 4096


async def pieces(body: bytes, size: int = 100) -> AsyncIterator[bytes]:
    for start in range(0, len(body), size):
        yield body[start : start + size]


async def read(content_type: str, body: bytes, limit: int = LIMIT) -> object:
    return await read_upload_form(content_type, str(len(body)), pieces(body), limit)


async def test_reads_the_file_and_the_acl_and_drops_other_fields() -> None:
    content_type, body = multipart(
        [("file", ("notes.md", b"# Notes\n\nhello"))],
        {"acl": '["group:hr"]', "tenant_id": "00000000-0000-0000-0000-000000000000"},
    )
    form = await read_upload_form(content_type, None, pieces(body, 7), LIMIT)
    assert (form.filename, form.data, form.acl) == (
        "notes.md",
        b"# Notes\n\nhello",
        b'["group:hr"]',
    )


async def test_no_acl_field_is_none() -> None:
    form = await read_upload_form(*_without_length(multipart([("file", ("a.md", b"a"))])), LIMIT)
    assert form.acl is None


async def test_a_file_of_exactly_the_limit_is_accepted() -> None:
    form = await read(*multipart([("file", ("a.md", b"x" * LIMIT))]))
    assert len(form.data) == LIMIT  # type: ignore[attr-defined]


@pytest.mark.parametrize("content_type", [None, "application/json", "multipart/form-data"])
async def test_anything_but_multipart_with_a_boundary_is_refused(content_type: str | None) -> None:
    with pytest.raises(NotMultipartError):
        await read_upload_form(content_type, None, pieces(b""), LIMIT)


async def test_a_file_one_byte_over_the_limit_is_refused() -> None:
    with pytest.raises(UploadTooLargeError):
        await read(*multipart([("file", ("a.md", b"x" * (LIMIT + 1)))]))


async def test_reading_stops_soon_after_the_file_passes_the_limit() -> None:
    stream = CountingStream(total=100 * LIMIT, piece=512)
    with pytest.raises(UploadTooLargeError):
        await read_upload_form(STREAM_CONTENT_TYPE, None, stream.__aiter__(), LIMIT)
    assert stream.sent <= LIMIT + 512


async def test_an_oversized_content_length_is_refused_before_reading() -> None:
    stream = CountingStream(total=10)
    with pytest.raises(UploadTooLargeError):
        await read_upload_form(
            STREAM_CONTENT_TYPE, str(10 * LIMIT * 1024), stream.__aiter__(), LIMIT
        )
    assert stream.sent == 0


async def test_oversized_fields_are_refused() -> None:
    content_type, body = multipart(
        [("file", ("a.md", b"a"))], {"tenant_id": "x" * (MAX_FIELD_BYTES + 1)}
    )
    with pytest.raises(UploadTooLargeError):
        await read_upload_form(content_type, None, pieces(body, 4096), LIMIT)


@pytest.mark.parametrize(
    ("files", "data"),
    [
        ([("other", ("a.md", b"a"))], {"acl": '["tenant:*"]'}),  # no file part
        ([("file", ("a.md", b"a")), ("file", ("b.md", b"b"))], {}),  # two files
        ([("acl", ("acl.json", b"[]"))], {}),  # the acl as a file, no file part
        ([("file", ("a.md", b"a"))], {f"f{i}": "v" for i in range(MAX_PARTS)}),
    ],
)
async def test_malformed_forms_are_refused(
    files: list[tuple[str, tuple[str, bytes]]], data: dict[str, str]
) -> None:
    with pytest.raises(MalformedUploadError):
        await read(*multipart(files, data))


async def test_a_truncated_body_is_refused() -> None:
    content_type, body = multipart([("file", ("a.md", b"hello world"))])
    with pytest.raises(MalformedUploadError):
        await read(content_type, body[:-20])


def _without_length(form: tuple[str, bytes]) -> tuple[str, None, AsyncIterator[bytes]]:
    content_type, body = form
    return content_type, None, pieces(body)
