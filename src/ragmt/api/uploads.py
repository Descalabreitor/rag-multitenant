"""Reads the multipart body of `POST /documents` from the stream, with hard caps.

FastAPI's `UploadFile` parses the whole form (spooling the file to disk) before
the route runs, so a size check there comes after the bytes are read. This
parser stops reading as soon as the file passes the limit (at max + 1 bytes),
and caps everything else in the body too: the other fields, the number of
parts and the total size.

The form has one file part named `file` and an optional text field `acl`. Other
fields (a `tenant_id`, say) are read and dropped: the tenant comes only from the
token. Nothing here logs or echoes what it reads.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

if TYPE_CHECKING:  # a TypedDict that python-multipart defines for type checkers only
    from python_multipart.multipart import MultipartCallbacks

# All non-file fields together, `acl` included.
MAX_FIELD_BYTES = 64 * 1024
MAX_PARTS = 16
# What the body may carry besides the file: fields, part headers, boundaries.
FORM_OVERHEAD_BYTES = 2 * MAX_FIELD_BYTES


class UploadError(Exception):
    """Base for upload bodies the route refuses. Messages are for logs, never for callers."""


class NotMultipartError(UploadError):
    """The request is not multipart/form-data with a boundary."""


class UploadTooLargeError(UploadError):
    """The file, the other fields or the whole body passed its limit."""


class MalformedUploadError(UploadError):
    """A broken multipart body, no file, two files, or too many parts."""


@dataclass(frozen=True, slots=True)
class UploadForm:
    filename: str
    data: bytes
    # The raw `acl` field, None when the form has none. The route parses it.
    acl: bytes | None


async def read_upload_form(
    content_type: str | None,
    content_length: str | None,
    stream: AsyncIterator[bytes],
    max_file_bytes: int,
) -> UploadForm:
    """Parse the form from `stream`, reading no more than the limits allow."""
    mime, options = parse_options_header(content_type)
    boundary = options.get(b"boundary")
    if mime != b"multipart/form-data" or not boundary:
        raise NotMultipartError("expected multipart/form-data with a boundary")

    max_body = max_file_bytes + FORM_OVERHEAD_BYTES
    if content_length is not None and content_length.isdigit() and int(content_length) > max_body:
        raise UploadTooLargeError("Content-Length is over the limit")

    form = _FormCollector(max_file_bytes)
    parser = MultipartParser(boundary, form.callbacks())
    received = 0
    try:
        async for data in stream:
            received += len(data)
            if received > max_body:
                raise UploadTooLargeError("body is over the limit")
            parser.write(data)
            if form.error is not None:
                raise form.error
        parser.finalize()
    except MultipartParseError:
        raise MalformedUploadError("unparseable multipart body") from None
    if form.error is not None:
        raise form.error
    if not form.ended or form.file is None:
        raise MalformedUploadError("incomplete body or no file part")
    return UploadForm(
        filename=form.filename,
        data=bytes(form.file),
        acl=None if form.acl is None else bytes(form.acl),
    )


class _FormCollector:
    """Callbacks for python-multipart. They never raise; the first error is kept
    in `error` and everything after it is ignored."""

    def __init__(self, max_file_bytes: int) -> None:
        self._max_file_bytes = max_file_bytes
        self.error: UploadError | None = None
        self.ended = False
        self.file: bytearray | None = None
        self.filename = ""
        self.acl: bytearray | None = None
        self._parts = 0
        self._field_bytes = 0
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        # Where the current part's data goes: "file", "acl" or None (dropped).
        self._target: str | None = None

    def callbacks(self) -> "MultipartCallbacks":
        return {
            "on_part_begin": self._on_part_begin,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
            "on_part_data": self._on_part_data,
            "on_end": self._on_end,
        }

    def _fail(self, error: UploadError) -> None:
        if self.error is None:
            self.error = error

    def _on_part_begin(self) -> None:
        self._parts += 1
        if self._parts > MAX_PARTS:
            self._fail(MalformedUploadError("too many parts"))
        self._headers = {}
        self._target = None

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        self._headers[bytes(self._header_field).lower()] = bytes(self._header_value)
        self._header_field.clear()
        self._header_value.clear()

    def _on_headers_finished(self) -> None:
        disposition, options = parse_options_header(
            self._headers.get(b"content-disposition", b"").decode("latin-1")
        )
        name, filename = options.get(b"name"), options.get(b"filename")
        if disposition != b"form-data" or name is None:
            self._fail(MalformedUploadError("part without a form-data name"))
        elif name == b"file":
            if filename is None or self.file is not None:
                self._fail(MalformedUploadError("expected exactly one file part"))
                return
            self.file = bytearray()
            self.filename = filename.decode("utf-8", errors="replace")
            self._target = "file"
        elif name == b"acl" and filename is None:
            if self.acl is not None:
                self._fail(MalformedUploadError("more than one acl field"))
                return
            self.acl = bytearray()
            self._target = "acl"

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self.error is not None:
            return
        size = end - start
        if self._target == "file" and self.file is not None:
            if len(self.file) + size > self._max_file_bytes:
                self._fail(UploadTooLargeError("file is over the limit"))
                return
            self.file += data[start:end]
            return
        self._field_bytes += size
        if self._field_bytes > MAX_FIELD_BYTES:
            self._fail(UploadTooLargeError("form fields are over the limit"))
        elif self._target == "acl" and self.acl is not None:
            self.acl += data[start:end]

    def _on_end(self) -> None:
        self.ended = True
