"""Uploaded bytes → Markdown (ADR 0008). Implements `DocumentConverter`.

Everything here treats the upload as hostile:

- The type comes from the content, and the filename's extension must agree
  with it. A `.docx` is a ZIP with `[Content_Types].xml` and
  `word/document.xml`; HTML is UTF-8 text that starts with an HTML tag; anything
  else that is UTF-8 text is Markdown. PDFs, images, other archives and binary
  data are refused.
- Limits are checked before any parsing: the upload size, then, for a `.docx`,
  the ZIP's entry count, total uncompressed size and per-entry compression
  ratio (zip bombs), encrypted entries, and XML parts that declare a DOCTYPE
  (entity expansion). `zipfile` never yields more bytes than an entry declares,
  so checking the declared sizes is enough.
- Nothing is fetched. HTML loses scripts, styles, frames, embedded objects and
  comments; images become their alt text; links keep their URL only for
  http(s) and mailto. mammoth runs with external file access off (its default
  since 1.11) and images are dropped before they are read.

Markdown that starts with an HTML element (a `<div>` banner, say) is still
Markdown, since Markdown allows raw HTML, so a `.md` file may start with a tag.
A full HTML document (doctype, `<html>`, `<head>` or `<body>`) is never Markdown.
"""

import io
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import mammoth
from bs4 import BeautifulSoup, Comment, Declaration, Doctype, ProcessingInstruction, Tag
from markdownify import ATX, MarkdownConverter

from ragmt.domain.ingest import ConvertedDocument, DocumentTooLargeError, UnsupportedDocumentError
from ragmt.ingest.chunking import headings

MIB = 1024 * 1024
MAX_TITLE_CHARS = 200
UNTITLED = "Untitled"


class _Kind(Enum):
    DOCX = "docx"
    HTML_DOCUMENT = "html document"
    HTML_FRAGMENT = "html fragment"
    MARKDOWN = "markdown"


# Which detected kinds each extension accepts.
_EXTENSIONS: dict[str, frozenset[_Kind]] = {
    ".docx": frozenset({_Kind.DOCX}),
    ".html": frozenset({_Kind.HTML_DOCUMENT, _Kind.HTML_FRAGMENT}),
    ".htm": frozenset({_Kind.HTML_DOCUMENT, _Kind.HTML_FRAGMENT}),
    ".md": frozenset({_Kind.MARKDOWN, _Kind.HTML_FRAGMENT}),
    ".markdown": frozenset({_Kind.MARKDOWN, _Kind.HTML_FRAGMENT}),
}

_DOCUMENT_TAGS = frozenset({"!doctype", "html", "head", "body"})
# Elements an HTML file may plausibly start with.
_HTML_TAGS = _DOCUMENT_TAGS | frozenset(
    "a abbr address article aside b blockquote br caption code dd div dl dt em figure "
    "footer form h1 h2 h3 h4 h5 h6 header hr i img li link main meta nav ol p pre "
    "section span strong style script table tbody td th thead title tr u ul".split()
)
_TAG_START = re.compile(r"<(!doctype|[a-z][a-z0-9]*)(?=[\s/>])", re.IGNORECASE)
# Control characters that don't occur in text files (tab, LF, FF and CR do).
_BINARY_CHARS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_BINARY_SIGNATURES = (b"%PDF-", b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"\x7fELF", b"MZ")

# Dropped from HTML with everything inside them.
_DROPPED_ELEMENTS = (
    "applet audio base canvas embed frame frameset head iframe link meta noscript "
    "object script source style svg template track video"
).split()
_LINK_SCHEMES = frozenset({"http", "https", "mailto"})

_XML_PARTS = re.compile(r"\.(xml|rels)$", re.IGNORECASE)
_DC_TITLE = "{http://purl.org/dc/elements/1.1/}title"
_MARKDOWN_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_EMPHASIS = re.compile(r"(?<![\\\w])([*_]{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class ZipLimits:
    """Bounds on a `.docx` archive, checked from its directory before reading it."""

    max_entries: int = 1000
    max_uncompressed_bytes: int = 100 * MIB
    # Deflated XML is usually 5-20x smaller; zip bombs are 1000x and more.
    max_compression_ratio: int = 100


class UploadConverter:
    """`DocumentConverter` for `.md`, `.html` and `.docx` uploads."""

    def __init__(self, max_bytes: int, zip_limits: ZipLimits | None = None) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes
        self._zip_limits = zip_limits or ZipLimits()

    def convert(self, data: bytes, filename: str) -> ConvertedDocument:
        name = _safe_name(filename)
        if len(data) > self._max_bytes:
            raise DocumentTooLargeError(size=len(data), limit=self._max_bytes)
        extension = PurePosixPath(name.lower()).suffix
        accepted = _EXTENSIONS.get(extension)
        if accepted is None:
            raise UnsupportedDocumentError(name, "unsupported file type")

        if _is_zip(data):
            archive = self._open_docx(data, name)
            kind = _Kind.DOCX
        else:
            text = _decode_text(data, name)
            kind = _detect_text_kind(text)
        if kind not in accepted:
            raise UnsupportedDocumentError(
                name, f"content does not match the {extension} extension"
            )

        fallback = _stem(name)
        if kind is _Kind.DOCX:
            with archive:
                markdown = _docx_to_markdown(data, name)
                core_title = _core_title(archive)
            return ConvertedDocument(_title(markdown, core_title, fallback), markdown)
        # A Markdown file that starts with raw HTML is still Markdown.
        if _Kind.MARKDOWN in accepted:
            markdown = text
        else:
            markdown = _html_to_markdown(text)
        return ConvertedDocument(_title(markdown, None, fallback), markdown)

    def _open_docx(self, data: bytes, name: str) -> zipfile.ZipFile:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise UnsupportedDocumentError(name, "not a readable ZIP archive") from exc
        try:
            names = set(archive.namelist())
            if not {"[Content_Types].xml", "word/document.xml"} <= names:
                raise UnsupportedDocumentError(
                    name, "ZIP archives other than .docx are not supported"
                )
            self._check_zip(archive, name)
            _check_xml_parts(archive, name)
        except BaseException:
            archive.close()
            raise
        return archive

    def _check_zip(self, archive: zipfile.ZipFile, name: str) -> None:
        limits = self._zip_limits
        entries = archive.infolist()
        if len(entries) > limits.max_entries:
            raise UnsupportedDocumentError(name, "the archive has too many entries")
        total = 0
        for entry in entries:
            if entry.flag_bits & 0x1:
                raise UnsupportedDocumentError(name, "encrypted documents are not supported")
            total += entry.file_size
            if total > limits.max_uncompressed_bytes:
                raise UnsupportedDocumentError(name, "the archive is too large when uncompressed")
            if entry.file_size > limits.max_compression_ratio * max(entry.compress_size, 1):
                raise UnsupportedDocumentError(name, "the archive is compressed suspiciously well")


# --- detection -------------------------------------------------------


def _is_zip(data: bytes) -> bool:
    return data.startswith(b"PK") or zipfile.is_zipfile(io.BytesIO(data))


def _decode_text(data: bytes, name: str) -> str:
    if data.startswith(_BINARY_SIGNATURES):
        raise UnsupportedDocumentError(name, "unsupported file type")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedDocumentError(name, "not UTF-8 text") from exc
    if _BINARY_CHARS.search(text):
        raise UnsupportedDocumentError(name, "binary content in a text file")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _detect_text_kind(text: str) -> _Kind:
    position = 0
    while True:  # skip leading whitespace and comments
        while position < len(text) and text[position].isspace():
            position += 1
        if not text.startswith("<!--", position):
            break
        end = text.find("-->", position + 4)
        if end < 0:
            return _Kind.MARKDOWN
        position = end + 3
    match = _TAG_START.match(text, position)
    if match is None:
        return _Kind.MARKDOWN
    tag = match.group(1).lower()
    if tag in _DOCUMENT_TAGS:
        return _Kind.HTML_DOCUMENT
    return _Kind.HTML_FRAGMENT if tag in _HTML_TAGS else _Kind.MARKDOWN


# --- .docx -----------------------------------------------------------


def _check_xml_parts(archive: zipfile.ZipFile, name: str) -> None:
    """Refuse XML parts that could expand entities: OOXML never needs a DOCTYPE.

    The scan is on bytes, so it also refuses parts not encoded as UTF-8 (UTF-16
    would hide the keyword).
    """
    for entry in archive.infolist():
        if not _XML_PARTS.search(entry.filename):
            continue
        with archive.open(entry) as part:
            head = part.read(4)
            if head.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in head:
                raise UnsupportedDocumentError(name, "XML parts must be UTF-8")
            tail = head
            while block := part.read(64 * 1024):
                window = tail + block
                if b"<!DOCTYPE" in window or b"<!ENTITY" in window:
                    raise UnsupportedDocumentError(name, "XML with a DOCTYPE is not supported")
                tail = window[-8:]
            if b"<!DOCTYPE" in tail or b"<!ENTITY" in tail:
                raise UnsupportedDocumentError(name, "XML with a DOCTYPE is not supported")


def _no_images(image: Any) -> list[Any]:
    """mammoth image handler: drop the image without reading it."""
    return []


def _docx_to_markdown(data: bytes, name: str) -> str:
    try:
        result = mammoth.convert_to_html(
            io.BytesIO(data),
            convert_image=_no_images,
            external_file_access=False,
            include_embedded_style_map=False,
        )
    except Exception as exc:  # mammoth raises many types for malformed input
        raise UnsupportedDocumentError(name, "not a readable .docx document") from exc
    return _html_to_markdown(str(result.value))


def _core_title(archive: zipfile.ZipFile) -> str | None:
    """dc:title from docProps/core.xml, or None. Size and DOCTYPE are checked already."""
    try:
        data = archive.read("docProps/core.xml")
    except KeyError:
        return None
    try:
        root = ET.fromstring(data)  # noqa: S314 - no DOCTYPE (checked), size bounded by ZipLimits
    except ET.ParseError:
        return None
    element = root.find(_DC_TITLE)
    return element.text if element is not None and element.text else None


# --- HTML ------------------------------------------------------------


def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(_DROPPED_ELEMENTS):
        element.decompose()
    for node in soup.find_all(
        string=lambda s: isinstance(s, Comment | Declaration | Doctype | ProcessingInstruction)
    ):
        node.extract()
    for image in soup.find_all("img"):
        alt = image.get("alt")
        image.replace_with(f" {alt} " if isinstance(alt, str) and alt.strip() else "")
    for link in soup.find_all("a"):
        if isinstance(link, Tag) and not _allowed_link(link.get("href")):
            del link["href"]
    converter = MarkdownConverter(heading_style=ATX, bullets="-", autolinks=False)
    markdown = converter.convert_soup(soup)
    return _BLANK_LINES.sub("\n\n", markdown.replace("\r\n", "\n")).strip() + "\n"


def _allowed_link(href: object) -> bool:
    if not isinstance(href, str):
        return False
    try:
        return urlsplit(href.strip()).scheme.lower() in _LINK_SCHEMES
    except ValueError:
        return False


# --- titles and names --------------------------------------------------


def _title(markdown: str, core_title: str | None, fallback: str) -> str:
    """First H1, else the .docx core title, else the filename without extension."""
    first_h1 = next((text for level, text in headings(markdown) if level == 1 and text), None)
    for candidate in (first_h1 and _plain(first_h1), core_title, fallback):
        cleaned = _clean_title(candidate or "")
        if cleaned:
            return cleaned
    return UNTITLED


def _plain(markdown: str) -> str:
    """Inline Markdown as text: links keep their text, emphasis and code marks go."""
    text = _MARKDOWN_LINK.sub(r"\1", markdown).replace("`", "")
    text = _MARKDOWN_EMPHASIS.sub(r"\2", text)
    return _MARKDOWN_ESCAPE.sub(r"\1", text)


def _clean_title(text: str) -> str:
    """One line, no control or format characters (bidi overrides), bounded length."""
    visible = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in {"Cc", "Cf", "Cs", "Co", "Cn"}
    )
    return " ".join(visible.split())[:MAX_TITLE_CHARS].strip()


def _safe_name(filename: str) -> str:
    """The last path component of an untrusted filename, printable and bounded."""
    base = re.split(r"[\\/]", filename)[-1]
    return _clean_title(base) or UNTITLED


def _stem(name: str) -> str:
    return PurePosixPath(name).stem
