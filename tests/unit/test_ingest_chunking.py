"""The heading-based chunker (ragmt.ingest.chunking): examples, then properties."""

import re
import string
from collections.abc import Sequence
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ragmt.domain import ChunkDraft
from ragmt.ingest.chunking import chunk, headings

# --- examples ----------------------------------------------------------


@pytest.mark.parametrize("markdown", ["", " ", "\n\n  \n\t\n"])
def test_empty_input_gives_no_chunks(markdown: str) -> None:
    assert chunk(markdown, 100, 10) == []


@pytest.mark.parametrize(("max_chars", "overlap_chars"), [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_bad_limits_are_refused(max_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk("text", max_chars, overlap_chars)


def test_heading_paths() -> None:
    markdown = (
        "Preamble.\n\n# Policies\n\nAll staff.\n\n### Deep\n\nSkipped a level.\n\n"
        "## Travel ##\n\nTrains.\n\n# Other\n\nEnd.\n"
    )
    assert [(c.heading, c.content) for c in chunk(markdown, 1000, 0)] == [
        (None, "Preamble."),
        ("Policies", "# Policies\n\nAll staff."),
        ("Policies > Deep", "### Deep\n\nSkipped a level."),
        ("Policies > Travel", "## Travel ##\n\nTrains."),
        ("Other", "# Other\n\nEnd."),
    ]


def test_hashes_that_are_not_headings() -> None:
    markdown = "#hashtag\n\n```\n# comment in code\n```\n\n    # indented code\n"
    assert list(headings(markdown)) == []
    assert [c.heading for c in chunk(markdown, 1000, 0)] == [None]


def test_paragraphs_are_packed_with_overlap_inside_a_section() -> None:
    paragraphs = [f"Paragraph {i} talks about travel." for i in range(6)]
    drafts = chunk("\n\n".join(paragraphs), 80, 40)
    assert [c.ordinal for c in drafts] == list(range(len(drafts)))
    assert len(drafts) > 1
    for previous, current in pairwise(drafts):
        assert len(current.content) <= 80
        first_paragraph = current.content.split("\n\n")[0]
        assert previous.content.endswith(first_paragraph)  # the overlap


def test_overlap_of_a_long_paragraph_starts_at_a_word() -> None:
    words = " ".join(f"w{i}" for i in range(40))
    drafts = chunk(f"{words}\n\nnext paragraph", len(words) + 5, 20)
    assert drafts[0].content == words
    overlap = drafts[1].content.split("\n\n")[0]
    assert 0 < len(overlap) <= 20
    assert words.endswith(" " + overlap)


def test_overlap_does_not_cross_a_heading() -> None:
    drafts = chunk("# A\n\nalpha text\n\n# B\n\nbeta text", 1000, 500)
    assert [c.content for c in drafts] == ["# A\n\nalpha text", "# B\n\nbeta text"]


def test_code_and_tables_are_not_split_while_they_fit() -> None:
    code = "```python\ndef f():\n\n    return 1\n```"
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    drafts = chunk(f"intro\n\n{code}\n\n{table}", len(code) + 2, 0)
    assert [c.content for c in drafts] == ["intro", code, table]


def test_oversized_code_is_split_at_lines_and_refenced() -> None:
    lines = [f"print({i})" for i in range(40)]
    code = "~~~~python title\n" + "\n".join(lines) + "\n~~~~~"
    drafts = chunk(code, 100, 30)
    assert len(drafts) > 1
    for draft in drafts:
        assert len(draft.content) <= 100
        assert draft.content.startswith("~~~~python title\n")
        assert draft.content.endswith("\n~~~~~")
    body = [line for d in drafts for line in d.content.split("\n")[1:-1]]
    assert body == lines  # every line once, in order: whole code pieces aren't overlapped here


def test_unclosed_fence_runs_to_the_end_and_is_closed_when_split() -> None:
    markdown = "```\n" + "\n".join(f"line {i}" for i in range(30)) + "\n# not a heading"
    drafts = chunk(markdown, 60, 0)
    assert {d.heading for d in drafts} == {None}
    assert all(d.content.startswith("```\n") and d.content.endswith("\n```") for d in drafts)


def test_oversized_table_repeats_its_header() -> None:
    header = "| name | value |\n|------|-------|"
    rows = [f"| row{i} | {i} |" for i in range(30)]
    drafts = chunk(header + "\n" + "\n".join(rows), 120, 0)
    assert len(drafts) > 1
    assert all(d.content.startswith(header + "\n") for d in drafts)
    assert all(len(d.content) <= 120 for d in drafts)
    assert [r for d in drafts for r in d.content.split("\n")[2:]] == rows


def test_documented_oversized_pieces() -> None:
    long_word = "x" * 50
    assert [c.content for c in chunk(f"short {long_word} tail", 20, 5)] == [
        "short",
        long_word,
        "tail",
    ]

    long_line = "y = " + "1 + " * 20 + "1"
    drafts = chunk(f"```\na = 1\n{long_line}\nb = 2\n```", 30, 0)
    assert [d.content for d in drafts] == [
        "```\na = 1\n```",
        f"```\n{long_line}\n```",
        "```\nb = 2\n```",
    ]


def test_crlf_is_normalized() -> None:
    assert chunk("# A\r\n\r\ntext\r\n", 100, 0) == [ChunkDraft(0, "A", "# A\n\ntext")]


# --- properties ---------------------------------------------------------

WORD = st.text(string.ascii_lowercase, min_size=1, max_size=10)


def tag(section: int, word: str) -> str:
    return f"s{section}x{word}"


@st.composite
def section_body(draw: st.DrawFn, section: int) -> list[str]:
    def words(min_size: int, max_size: int) -> str:
        items = draw(st.lists(WORD, min_size=min_size, max_size=max_size))
        return " ".join(tag(section, w) for w in items)

    blocks: list[str] = []
    for kind in draw(st.lists(st.sampled_from(["text", "code", "table"]), max_size=5)):
        if kind == "text":
            lines = [words(1, 8) for _ in range(draw(st.integers(1, 4)))]
            blocks.append("\n".join(lines))
        elif kind == "code":
            fence = draw(st.sampled_from(["```", "~~~", "````"]))
            info = draw(st.sampled_from(["", "python", "sql"]))
            lines = [words(0, 3) for _ in range(draw(st.integers(0, 15)))]
            blocks.append("\n".join([fence + info, *lines, fence]))
        else:
            rows = [f"| {words(1, 1)} | {words(1, 1)} |" for _ in range(draw(st.integers(1, 15)))]
            blocks.append("\n".join([f"| {words(1, 1)} | {words(1, 1)} |", "|---|---|", *rows]))
    return blocks


@st.composite
def documents(draw: st.DrawFn) -> tuple[str, list[str]]:
    """Markdown whose words are tagged with their section (0 = before any heading).

    Returns the Markdown and each section's heading text ("" for section 0).
    """
    parts: list[str] = draw(section_body(0))
    titles = [""]
    for section in range(1, draw(st.integers(0, 6)) + 1):
        level = draw(st.integers(1, 3))
        count = draw(st.integers(1, 6))
        title = " ".join(
            tag(section, w) for w in draw(st.lists(WORD, min_size=count, max_size=count))
        )
        titles.append(title)
        parts.append("#" * level + " " + title)
        parts.extend(draw(section_body(section)))
    return "\n\n".join(parts), titles


@st.composite
def limits(draw: st.DrawFn) -> tuple[int, int]:
    max_chars = draw(st.integers(80, 600))
    return max_chars, draw(st.integers(0, max_chars - 1))


def is_subsequence(needles: Sequence[str], haystack: Sequence[str]) -> bool:
    remaining = iter(haystack)
    return all(any(token == needle for token in remaining) for needle in needles)


def check_common_properties(markdown: str, max_chars: int, overlap: int) -> list[ChunkDraft]:
    drafts = chunk(markdown, max_chars, overlap)
    # Ordinals are 0, 1, 2, ...
    assert [d.ordinal for d in drafts] == list(range(len(drafts)))
    # Every word of the input is in some chunk, in order (overlap only repeats words).
    output = [token for d in drafts for token in d.content.split()]
    assert is_subsequence(markdown.split(), output)
    # Deterministic.
    assert chunk(markdown, max_chars, overlap) == drafts
    return drafts


@settings(deadline=None)
@given(documents(), limits())
def test_structured_documents(document: tuple[str, list[str]], limit: tuple[int, int]) -> None:
    markdown, titles = document
    max_chars, overlap = limit
    drafts = check_common_properties(markdown, max_chars, overlap)

    tagged = re.compile(r"s(\d+)x")
    for draft in drafts:
        # No line or word here is longer than the budget, so no chunk may be either.
        assert len(draft.content) <= max_chars
        # Overlap never crosses a heading: a chunk's words all come from one section,
        # the section its heading names.
        sections = {int(m.group(1)) for m in map(tagged.match, draft.content.split()) if m}
        assert len(sections) <= 1, draft
        if not sections:  # e.g. an empty code block
            continue
        (section,) = sections
        if section == 0:
            assert draft.heading is None
        else:
            assert draft.heading is not None
            assert draft.heading.endswith(titles[section])


@settings(deadline=None)
@given(st.text(), limits())
def test_arbitrary_text(markdown: str, limit: tuple[int, int]) -> None:
    check_common_properties(markdown, *limit)


MARKDOWNISH = st.lists(
    st.sampled_from(
        ["#", "##", "```", "~~~", "|", "|---|", "-", "\n", "\n\n", " ", "a", "bb", "x" * 30]
    ),
    max_size=80,
).map("".join)


@settings(deadline=None)
@given(MARKDOWNISH, st.integers(1, 60), st.data())
def test_markdown_like_noise(markdown: str, max_chars: int, data: st.DataObject) -> None:
    overlap = data.draw(st.integers(0, max_chars - 1))
    check_common_properties(markdown, max_chars, overlap)
