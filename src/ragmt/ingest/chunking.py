"""Markdown → `ChunkDraft`s, split by headings. Pure and deterministic.

How a document is cut:

1. **Sections.** Every ATX heading (`#` to `######`, outside fenced code)
   starts a section. `ChunkDraft.heading` is the path of headings above the
   chunk ("Policies > Travel"), or None before the first heading. The heading
   line itself is the first line of its section's content, so its words are
   embedded with the text they introduce.
2. **Blocks.** A section is a list of blocks: runs of non-blank lines
   (paragraphs, lists, tables) and fenced code blocks (blank lines included).
   A block is never cut while it fits in `max_chars`.
3. **Packing.** Blocks are packed, separated by a blank line, into chunks of at
   most `max_chars`. Each chunk after the first in a section starts with up to
   `overlap_chars` taken from the end of the previous chunk: whole blocks, or the
   tail of a prose paragraph cut at a word boundary. Code and tables are only
   repeated whole, so an overlap never opens half a fence or half a table.
   Overlap never crosses a heading: sections are packed independently.

**Oversized blocks.** A block longer than `max_chars` is split before packing:

- Fenced code is split at line boundaries, and each piece is a complete fenced
  block: it repeats the opening fence line (with its info string) and gets a
  closing fence.
- A table is split between rows, and every piece repeats the header and
  delimiter rows. If some row doesn't fit together with them, the table is
  split between lines instead and only the first piece has the header.
- Prose is split at the last line break that fits, else at the last whitespace.

A piece can still exceed `max_chars` in one case only: it holds a single unit
that can't be cut without breaking it, i.e. one code line (plus its fences),
one table row (plus the header), or one word. Words are never cut.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from ragmt.domain.ingest import ChunkDraft

HEADING_SEPARATOR = " > "

_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+$")
_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_TABLE_DELIMITER = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
_BLOCK_SEPARATOR = "\n\n"

BlockKind = Literal["text", "code", "table"]


@dataclass(frozen=True, slots=True)
class _Block:
    kind: BlockKind
    text: str


@dataclass(slots=True)
class _Section:
    heading: str | None
    lines: list[str]


def chunk(markdown: str, max_chars: int, overlap_chars: int) -> list[ChunkDraft]:
    """Cut `markdown` into chunks of at most `max_chars` (see the module docstring)."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be at least 0 and smaller than max_chars")

    drafts: list[ChunkDraft] = []
    for section in _sections(_normalize(markdown)):
        blocks = [piece for block in _blocks(section.lines) for piece in _fit(block, max_chars)]
        for content in _pack(blocks, max_chars, overlap_chars):
            drafts.append(ChunkDraft(len(drafts), section.heading, content))
    return drafts


def headings(markdown: str) -> Iterator[tuple[int, str]]:
    """(level, text) of each ATX heading outside fenced code, in order."""
    fence: str | None = None
    for line in _normalize(markdown).split("\n"):
        fence, in_code = _track_fence(fence, line)
        if in_code:
            continue
        parsed = _parse_heading(line)
        if parsed is not None:
            yield parsed


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_heading(line: str) -> tuple[int, str] | None:
    match = _HEADING.match(line)
    if match is None:
        return None
    text = _CLOSING_HASHES.sub("", match.group(2) or "").strip()
    return len(match.group(1)), text


def _track_fence(fence: str | None, line: str) -> tuple[str | None, bool]:
    """Update the open fence for `line`. Returns (fence after it, line is code)."""
    match = _FENCE.match(line)
    if fence is None:
        if match is not None and not (match.group(2)[0] == "`" and "`" in match.group(3)):
            return match.group(2), True
        return None, False
    if match is not None and match.group(3).strip() == "":
        marker = match.group(2)
        if marker[0] == fence[0] and len(marker) >= len(fence):
            return None, True
    return fence, True


def _sections(markdown: str) -> Iterator[_Section]:
    path: list[tuple[int, str]] = []
    current = _Section(None, [])
    fence: str | None = None
    for line in markdown.split("\n"):
        fence, in_code = _track_fence(fence, line)
        parsed = None if in_code else _parse_heading(line)
        if parsed is None:
            current.lines.append(line)
            continue
        if any(item.strip() for item in current.lines):
            yield current
        level, text = parsed
        path = [(lvl, txt) for lvl, txt in path if lvl < level] + [(level, text)]
        title = HEADING_SEPARATOR.join(txt for _, txt in path if txt)
        current = _Section(title or None, [line])
    if any(item.strip() for item in current.lines):
        yield current


def _blocks(lines: list[str]) -> Iterator[_Block]:
    run: list[str] = []
    fence: str | None = None
    for line in lines:
        was_open = fence is not None
        fence, in_code = _track_fence(fence, line)
        if in_code:
            if not was_open:  # an opening fence ends the paragraph before it
                yield from _flush_text(run)
                run = []
            run.append(line)
            if fence is None:  # closing fence
                yield _Block("code", "\n".join(run))
                run = []
        elif line.strip():
            run.append(line)
        else:
            yield from _flush_text(run)
            run = []
    if fence is not None:  # unclosed fence: code runs to the end of the section
        yield _Block("code", "\n".join(run))
    else:
        yield from _flush_text(run)


def _flush_text(run: list[str]) -> Iterator[_Block]:
    if not run:
        return
    is_table = len(run) >= 2 and "|" in run[0] and _TABLE_DELIMITER.match(run[1]) is not None
    yield _Block("table" if is_table else "text", "\n".join(run).strip())


# --- oversized blocks ------------------------------------------------


def _fit(block: _Block, max_chars: int) -> list[_Block]:
    if len(block.text) <= max_chars:
        return [block]
    if block.kind == "code":
        return [_Block("code", text) for text in _split_code(block.text, max_chars)]
    if block.kind == "table":
        return [_Block("table", text) for text in _split_table(block.text, max_chars)]
    return [_Block("text", text) for text in _split_text(block.text, max_chars)]


def _split_code(text: str, max_chars: int) -> list[str]:
    opening, *body = text.split("\n")
    match = _FENCE.match(opening)
    if match is None or not body:  # not reachable for blocks built by _blocks
        return _split_text(text, max_chars)
    marker = match.group(2)
    closing = match.group(1) + marker
    if _track_fence(marker, body[-1])[0] is None:  # reuse the original closing fence
        closing = body.pop()
    overhead = len(opening) + len(closing) + 2
    groups = _pack_lines(body, max_chars - overhead) or [[]]
    return ["\n".join([opening, *group, closing]) for group in groups]


def _split_table(text: str, max_chars: int) -> list[str]:
    lines = text.split("\n")
    header = lines[:2]
    rows = lines[2:]
    header_len = len(header[0]) + len(header[1]) + 2
    first = _pack_lines(rows, max_chars - header_len)
    if first and all(len("\n".join([*header, *group])) <= max_chars for group in first):
        return ["\n".join([*header, *group]) for group in first]
    # The header doesn't fit with the rows: send it once, in the first piece.
    return ["\n".join(group) for group in _pack_lines(lines, max_chars)]


def _pack_lines(lines: list[str], budget: int) -> list[list[str]]:
    """Group consecutive lines so each group, joined by newlines, fits `budget`.

    A line longer than the budget gets a group of its own.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        added = len(line) + (1 if current else 0)
        if current and size + added > budget:
            groups.append(current)
            current, size = [], 0
            added = len(line)
        current.append(line)
        size += added
    if current:
        groups.append(current)
    return groups


def _split_text(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        window = rest[: max_chars + 1]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = max((i for i, ch in enumerate(window) if ch.isspace()), default=-1)
        if cut <= 0:  # one word longer than max_chars: keep it whole
            match = re.search(r"\s", rest)
            cut = match.start() if match else len(rest)
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


# --- packing ---------------------------------------------------------


def _pack(blocks: list[_Block], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[_Block] = []
    for block in blocks:
        if current and _joined_len([*current, block]) > max_chars:
            chunks.append(_BLOCK_SEPARATOR.join(b.text for b in current))
            # The overlap must leave room for the block that follows it.
            budget = min(overlap_chars, max_chars - len(block.text) - len(_BLOCK_SEPARATOR))
            current = _overlap(current, budget)
        current.append(block)
    if current:
        chunks.append(_BLOCK_SEPARATOR.join(b.text for b in current))
    return chunks


def _joined_len(blocks: list[_Block]) -> int:
    return sum(len(b.text) for b in blocks) + len(_BLOCK_SEPARATOR) * (len(blocks) - 1)


def _overlap(previous: list[_Block], budget: int) -> list[_Block]:
    """The end of the previous chunk, at most `budget` characters, as blocks."""
    taken: list[_Block] = []
    used = 0
    for block in reversed(previous):
        cost = len(block.text) + (len(_BLOCK_SEPARATOR) if taken else 0)
        if used + cost <= budget:
            taken.insert(0, block)
            used += cost
            continue
        if block.kind == "text":
            tail = _tail(block.text, budget - used - (len(_BLOCK_SEPARATOR) if taken else 0))
            if tail:
                taken.insert(0, _Block("text", tail))
        break
    return taken


def _tail(text: str, budget: int) -> str:
    """The longest suffix of `text` of at most `budget` chars that starts at a word."""
    if budget <= 0:
        return ""
    start = len(text) - budget
    if start <= 0:
        return text
    if text[start - 1].isspace():
        return text[start:].strip()
    match = re.search(r"\s", text[start:])
    return text[start + match.start() :].strip() if match else ""
