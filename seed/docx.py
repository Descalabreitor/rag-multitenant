"""Build the seed's .docx files from small text sources, so the repo holds no binaries.

A source is Markdown-like: `# ` lines become Heading 1, `## ` lines Heading 2,
and every other non-blank line a paragraph. The first Heading 1 is also the core
title.

The output is the same bytes on every machine and every run, because the seed is
deduplicated by the SHA-256 of what it uploads: entries are stored uncompressed
(deflate output can change with the zlib version), in a fixed order, with a
fixed timestamp and creator system.
"""

import io
import zipfile
from xml.sax.saxutils import escape

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_RELS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest timestamp a ZIP entry can hold

_CONTENT_TYPES = (
    _XML + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels"'
    ' ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    '<Override PartName="/docProps/core.xml"'
    ' ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    "</Types>"
)
_PACKAGE_RELS = (
    _XML + f'<Relationships xmlns="{_RELS}">'
    f'<Relationship Id="rId1" Type="{_OFFICE_RELS}/officeDocument" Target="word/document.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/'
    'relationships/metadata/core-properties" Target="docProps/core.xml"/>'
    "</Relationships>"
)
_DOCUMENT_RELS = (
    _XML + f'<Relationships xmlns="{_RELS}">'
    f'<Relationship Id="rId1" Type="{_OFFICE_RELS}/styles" Target="styles.xml"/>'
    "</Relationships>"
)
# mammoth maps paragraphs to headings by style name ("heading 1" -> <h1>).
_STYLES = (
    _XML + f'<w:styles xmlns:w="{_W}">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>'
    "</w:styles>"
)
_HEADINGS = {"# ": "Heading1", "## ": "Heading2"}


def build_docx(source: str) -> bytes:
    """The .docx for `source` (see the module docstring)."""
    paragraphs: list[str] = []
    title = ""
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        style = None
        for prefix, name in _HEADINGS.items():
            if line.startswith(prefix):
                style, line = name, line[len(prefix) :].strip()
        if style == "Heading1" and not title:
            title = line
        paragraphs.append(_paragraph(line, style))

    document = (
        _XML + f'<w:document xmlns:w="{_W}"><w:body>{"".join(paragraphs)}</w:body></w:document>'
    )
    core = (
        _XML + "<cp:coreProperties"
        ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{escape(title)}</dc:title></cp:coreProperties>"
    )
    entries = {
        "[Content_Types].xml": _CONTENT_TYPES,
        "_rels/.rels": _PACKAGE_RELS,
        "docProps/core.xml": core,
        "word/document.xml": document,
        "word/_rels/document.xml.rels": _DOCUMENT_RELS,
        "word/styles.xml": _STYLES,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, content in entries.items():
            info = zipfile.ZipInfo(name, date_time=_EPOCH)
            info.create_system = 0  # otherwise it records the OS the seed runs on
            archive.writestr(info, content)
    return buffer.getvalue()


def _paragraph(text: str, style: str | None) -> str:
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{props}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
