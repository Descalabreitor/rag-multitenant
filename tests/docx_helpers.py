"""Builds minimal .docx files in memory for the converter tests.

Generated rather than committed, so each test shows exactly what the archive
holds and the malicious variants (zip bombs, encrypted entries, DOCTYPEs) are
one argument away.
"""

import io
import zipfile
from collections.abc import Sequence
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
IMAGE_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
STYLES_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
CORE_TYPE = "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"
# A 1x1 transparent PNG.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml"
 ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>
</w:styles>"""

IMAGE_RUN = f"""<w:r><w:drawing>
<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
<wp:docPr id="1" name="Picture 1" descr="a picture"/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:blipFill><a:blip xmlns:r="{R}" r:embed="rIdImage"/></pic:blipFill>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"""


def paragraph(text: str, style: str | None = None, *, image: bool = False) -> str:
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    run = f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    return f"<w:p>{props}{run}{IMAGE_RUN if image else ''}</w:p>"


def document_xml(paragraphs: Sequence[str]) -> str:
    body = "".join(paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>'
    )


def core_xml(title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        "<cp:coreProperties"
        ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{escape(title)}</dc:title></cp:coreProperties>"
    )


def build_docx(
    paragraphs: Sequence[str] = (),
    *,
    core_title: str | None = None,
    extra: dict[str, bytes | str] | None = None,
    encrypted: Sequence[str] = (),
) -> bytes:
    """A .docx with `paragraphs` (from `paragraph`) as its body.

    `extra` adds or replaces entries; `encrypted` names entries whose
    "encrypted" flag is set (the data itself stays plain, which is enough for
    a check that reads the flag). zipfile clears that flag when writing, so it
    is patched into the bytes afterwards.
    """
    package_rels = [f'<Relationship Id="rId1" Type="{DOC_TYPE}" Target="word/document.xml"/>']
    if core_title is not None:
        package_rels.append(
            f'<Relationship Id="rId2" Type="{CORE_TYPE}" Target="docProps/core.xml"/>'
        )
    entries: dict[str, bytes | str] = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": f'<Relationships xmlns="{RELS}">{"".join(package_rels)}</Relationships>',
        "word/document.xml": document_xml(paragraphs),
        "word/_rels/document.xml.rels": (
            f'<Relationships xmlns="{RELS}">'
            f'<Relationship Id="rIdStyles" Type="{STYLES_TYPE}" Target="styles.xml"/>'
            f'<Relationship Id="rIdImage" Type="{IMAGE_TYPE}" Target="media/image1.png"/>'
            "</Relationships>"
        ),
        "word/styles.xml": STYLES,
        "word/media/image1.png": PNG,
    }
    if core_title is not None:
        entries["docProps/core.xml"] = core_xml(core_title)
    entries.update(extra or {})

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return _set_encrypted_flag(buffer.getvalue(), set(encrypted))


def _set_encrypted_flag(data: bytes, names: set[str]) -> bytes:
    """Set bit 0 of the flags of `names`, in the central directory and local headers."""
    patched = bytearray(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.filename in names:
                patched[info.header_offset + 6] |= 0x1
    position = patched.find(b"PK\x01\x02")
    while position >= 0:
        name_length = int.from_bytes(patched[position + 28 : position + 30], "little")
        name = patched[position + 46 : position + 46 + name_length].decode()
        if name in names:
            patched[position + 8] |= 0x1
        position = patched.find(b"PK\x01\x02", position + 46)
    return bytes(patched)


def zip_of(entries: dict[str, bytes]) -> bytes:
    """A plain ZIP archive (not a .docx)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()
