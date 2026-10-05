"""Fakes for the ingest ports, so ingest tests need neither real files nor Ollama.

`FakeConverter` reads UTF-8 Markdown and takes the first `# ` line as the title.
`chunk_paragraphs` makes one chunk per paragraph that isn't a heading, so a test
knows exactly which chunks a document will have. `CountingEmbeddings` is the
hash-based fake embedder plus a call counter.
"""

from collections.abc import Sequence

from ragmt.adapters.llm.fake import FakeEmbeddings
from ragmt.domain import ChunkDraft, ConvertedDocument


class FakeConverter:
    def convert(self, data: bytes, filename: str) -> ConvertedDocument:
        markdown = data.decode("utf-8")
        title = next(
            (line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), filename
        )
        return ConvertedDocument(title=title, markdown=markdown)


def chunk_paragraphs(markdown: str, max_chars: int, overlap_chars: int) -> list[ChunkDraft]:
    """Same signature as the real chunker; sizes are ignored."""
    drafts: list[ChunkDraft] = []
    heading: str | None = None
    for paragraph in (p.strip() for p in markdown.split("\n\n")):
        if paragraph.startswith("#"):
            heading = paragraph.lstrip("#").strip()
        elif paragraph:
            drafts.append(ChunkDraft(len(drafts), heading, paragraph))
    return drafts


class CountingEmbeddings(FakeEmbeddings):
    """The hash-based fake from ragmt.adapters.llm, counting embed_documents calls."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.document_calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return await super().embed_documents(texts)


def markdown_document(title: str, *paragraphs: str) -> bytes:
    return "\n\n".join([f"# {title}", *paragraphs]).encode("utf-8")
