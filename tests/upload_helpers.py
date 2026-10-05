"""Multipart bodies for upload tests, built by httpx or streamed piece by piece."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

BOUNDARY = "ragmt-test-boundary"
STREAM_CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"


def multipart(
    files: list[tuple[str, tuple[str, bytes]]], data: dict[str, str] | None = None
) -> tuple[str, bytes]:
    """(Content-Type, body) of a form as httpx would send it."""
    request = httpx.Request("POST", "http://test", files=files, data=data)
    return request.headers["content-type"], request.read()


@dataclass
class CountingStream:
    """A file part of `total` bytes sent in `piece`-byte pieces, counting what was read.

    Iterating it yields the whole multipart body; `sent` says how many file bytes
    the reader actually pulled before it stopped.
    """

    total: int
    piece: int = 1024
    filename: str = "big.md"
    sent: int = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield (
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{self.filename}"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
        ).encode()
        while self.sent < self.total:
            size = min(self.piece, self.total - self.sent)
            self.sent += size
            yield b"x" * size
        yield f"\r\n--{BOUNDARY}--\r\n".encode()
