"""The upload converter (ragmt.ingest.convert): detection, limits, conversion, titles.

.md and .html inputs are fixture files in tests/fixtures/ingest; .docx inputs
are built in memory by tests/docx_helpers.py.
"""

from pathlib import Path

import pytest

from ragmt.domain import (
    ConvertedDocument,
    DocumentConverter,
    DocumentTooLargeError,
    UnsupportedDocumentError,
)
from ragmt.ingest.chunking import chunk
from ragmt.ingest.convert import UploadConverter, ZipLimits
from tests.docx_helpers import build_docx, paragraph, zip_of

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingest"
MAX_BYTES = 1024 * 1024


@pytest.fixture
def converter() -> UploadConverter:
    return UploadConverter(MAX_BYTES)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_it_satisfies_the_port(converter: UploadConverter) -> None:
    port: DocumentConverter = converter
    assert isinstance(port.convert(b"text", "a.md"), ConvertedDocument)


# --- Markdown --------------------------------------------------------


def test_markdown_is_kept_as_is(converter: UploadConverter) -> None:
    data = fixture("policy.md")
    result = converter.convert(data, "policy.md")
    assert result.markdown == data.decode()
    assert result.title == "Expense policy"


def test_markdown_bom_and_crlf_are_normalized(converter: UploadConverter) -> None:
    data = b"\xef\xbb\xbf" + fixture("policy.md").replace(b"\n", b"\r\n")
    result = converter.convert(data, "policy.md")
    assert result.markdown == fixture("policy.md").decode()


def test_heading_inside_code_is_not_a_title(converter: UploadConverter) -> None:
    result = converter.convert(b"```\n# not a title\n```\n\ntext", "notes.md")
    assert result.title == "notes"


def test_markdown_may_start_with_an_html_element(converter: UploadConverter) -> None:
    data = b'<p align="center">logo</p>\n\n# Readme\n'
    assert converter.convert(data, "README.md").markdown == data.decode()


# --- HTML ------------------------------------------------------------


def test_html_is_converted_to_markdown_with_atx_headings(converter: UploadConverter) -> None:
    result = converter.convert(fixture("page.html"), "page.html")
    assert result.title == "Onboarding guide"
    markdown = result.markdown
    assert "# Onboarding guide" in markdown
    assert "## First week" in markdown
    assert "- Meet your manager" in markdown
    assert "```\nssh dev.example\n```" in markdown
    assert "| Day | Task |" in markdown
    assert "[handbook](https://intranet.example/handbook)" in markdown
    assert "team photo" in markdown  # the image's alt text, not the image


def test_html_active_content_and_remote_references_are_dropped(
    converter: UploadConverter,
) -> None:
    markdown = converter.convert(fixture("page.html"), "page.html").markdown
    for canary in ("CANARY-COMMENT", "CANARY-SCRIPT", "CANARY-NOSCRIPT", "evil.example"):
        assert canary not in markdown
    assert "javascript:" not in markdown
    assert "this link" in markdown  # the text of a dropped link stays
    assert "Head title" not in markdown
    assert "<" not in markdown


def test_html_fragment_with_script(converter: UploadConverter) -> None:
    data = b"<div><script>alert('x')</script><p>Hello <scr<script>ipt>alert(2)</script></p></div>"
    markdown = converter.convert(data, "x.htm").markdown
    # The script elements are gone; what the broken tag leaves is inert text.
    assert "alert('x')" not in markdown
    assert "<script" not in markdown
    assert markdown.startswith("Hello")


def test_html_without_h1_takes_the_filename(converter: UploadConverter) -> None:
    result = converter.convert(b"<p>body</p>", "dir/sub/Quarterly report.html")
    assert result.title == "Quarterly report"


# --- .docx -----------------------------------------------------------


def test_docx_headings_and_text(converter: UploadConverter) -> None:
    data = build_docx(
        [
            paragraph("Expense policy", "Heading1"),
            paragraph("Applies to everyone."),
            paragraph("Travel", "Heading2"),
            paragraph("Book trains."),
        ],
        core_title="Core title",
    )
    result = converter.convert(data, "policy.docx")
    assert result.title == "Expense policy"
    assert result.markdown == (
        "# Expense policy\n\nApplies to everyone.\n\n## Travel\n\nBook trains.\n"
    )
    assert [c.heading for c in chunk(result.markdown, 200, 0)] == [
        "Expense policy",
        "Expense policy > Travel",
    ]


def test_docx_images_are_ignored(converter: UploadConverter) -> None:
    data = build_docx([paragraph("Before the picture.", image=True)])
    markdown = converter.convert(data, "pic.docx").markdown
    assert markdown == "Before the picture.\n"


def test_docx_title_falls_back_to_core_title_then_filename(converter: UploadConverter) -> None:
    body = [paragraph("Only a paragraph.")]
    assert converter.convert(build_docx(body, core_title="Core title"), "a.docx").title == (
        "Core title"
    )
    assert converter.convert(build_docx(body, core_title=" "), "Plan 2026.docx").title == (
        "Plan 2026"
    )
    assert converter.convert(build_docx(body), "Plan 2026.docx").title == "Plan 2026"


def test_titles_are_one_clean_line(converter: UploadConverter) -> None:
    sneaky = "# Pay‮roll \t report​ " + "x" * 500
    title = converter.convert(sneaky.encode(), "a.md").title
    assert title.startswith("Payroll report x")
    assert len(title) == 200
    assert title.isprintable()
    assert converter.convert(b"text", "Bud\x07get\x1b[31m.md").title == "Budget[31m"


# --- type detection ----------------------------------------------------


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        (b"# Markdown", "notes.html"),  # Markdown named .html
        (b"<!DOCTYPE html><html><body>x</body></html>", "page.md"),  # HTML document as .md
        (b"# Markdown", "notes.docx"),  # text named .docx
        (b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n", "report.docx"),  # PDF renamed .docx
        (b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n", "report.md"),  # PDF renamed .md
        (b"%PDF-1.7\n", "report.pdf"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image.md"),
        (b"\xff\xfe# utf-16", "notes.md"),  # not UTF-8
        (b"text\x00with a NUL", "notes.md"),
        (b"plain text", "notes.txt"),  # extension not supported
        (b"plain text", "notes"),
        (zip_of({"a.txt": b"hello"}), "archive.docx"),  # a ZIP that isn't a .docx
        (b"PK\x03\x04garbage", "broken.docx"),
    ],
)
def test_unsupported_or_mismatched_content_is_refused(
    converter: UploadConverter, data: bytes, filename: str
) -> None:
    with pytest.raises(UnsupportedDocumentError):
        converter.convert(data, filename)


def test_docx_named_as_markdown_is_refused(converter: UploadConverter) -> None:
    with pytest.raises(UnsupportedDocumentError, match="does not match"):
        converter.convert(build_docx([paragraph("x")]), "doc.md")


def test_text_starting_with_pk_is_still_text(converter: UploadConverter) -> None:
    assert converter.convert(b"PKI rollout plan", "pki.md").markdown == "PKI rollout plan"


def test_error_messages_use_only_the_base_filename(converter: UploadConverter) -> None:
    with pytest.raises(UnsupportedDocumentError) as caught:
        converter.convert(b"%PDF-1.7", "../../etc/secret‮.docx")
    assert caught.value.filename == "secret.docx"


# --- limits ------------------------------------------------------------


def test_size_limit_is_checked_before_anything_else() -> None:
    converter = UploadConverter(max_bytes=10)
    with pytest.raises(DocumentTooLargeError) as caught:
        converter.convert(b"%PDF-1.7 and more", "x.pdf")
    assert (caught.value.size, caught.value.limit) == (17, 10)
    assert converter.convert(b"0123456789", "x.md").markdown == "0123456789"


def test_zip_bomb_is_refused_by_compression_ratio(converter: UploadConverter) -> None:
    bomb = build_docx([paragraph("x")], extra={"word/bomb.bin": b"\0" * (4 * 1024 * 1024)})
    assert len(bomb) < MAX_BYTES
    with pytest.raises(UnsupportedDocumentError, match="compressed"):
        converter.convert(bomb, "bomb.docx")


def test_zip_total_uncompressed_size_is_capped() -> None:
    limits = ZipLimits(max_uncompressed_bytes=50_000, max_compression_ratio=10**6)
    data = build_docx([paragraph("x")], extra={"word/big.xml": "<a/>" * 20_000})
    with pytest.raises(UnsupportedDocumentError, match="uncompressed"):
        UploadConverter(MAX_BYTES, limits).convert(data, "big.docx")


def test_zip_entry_count_is_capped() -> None:
    extra: dict[str, bytes | str] = {f"word/part{i}.bin": b"x" for i in range(20)}
    data = build_docx([paragraph("x")], extra=extra)
    with pytest.raises(UnsupportedDocumentError, match="entries"):
        UploadConverter(MAX_BYTES, ZipLimits(max_entries=10)).convert(data, "many.docx")


def test_encrypted_entries_are_refused(converter: UploadConverter) -> None:
    data = build_docx([paragraph("x")], encrypted=["word/document.xml"])
    with pytest.raises(UnsupportedDocumentError, match="encrypted"):
        converter.convert(data, "locked.docx")


def test_xml_with_a_doctype_is_refused(converter: UploadConverter) -> None:
    laughs = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>'
        "<cp:coreProperties"
        ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>&lol2;</dc:title>'
        "</cp:coreProperties>"
    )
    data = build_docx([paragraph("x")], core_title="t", extra={"docProps/core.xml": laughs})
    with pytest.raises(UnsupportedDocumentError, match="DOCTYPE"):
        converter.convert(data, "laughs.docx")


def test_utf16_xml_parts_are_refused(converter: UploadConverter) -> None:
    core = '<?xml version="1.0" encoding="UTF-16"?><a/>'.encode("utf-16")
    data = build_docx([paragraph("x")], core_title="t", extra={"docProps/core.xml": core})
    with pytest.raises(UnsupportedDocumentError, match="UTF-8"):
        converter.convert(data, "utf16.docx")


def test_corrupt_document_xml_is_unsupported(converter: UploadConverter) -> None:
    data = build_docx(extra={"word/document.xml": "<w:document><unclosed>"})
    with pytest.raises(UnsupportedDocumentError, match="readable"):
        converter.convert(data, "corrupt.docx")
