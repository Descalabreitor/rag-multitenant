"""Scores of the quality baseline, from per-case results. Pure functions, no I/O.

Documents are compared by title (titles are unique in the seed corpus).

- **Citation precision**: of the documents an answer cites, the share the case
  expects. Undefined for an answer that cites nothing.
- **Citation recall**: of the documents the case expects, the share the answer
  cites. Defined for answerable cases only.
- **Retrieval hit**: the same recall, over the documents of the retrieved chunks
  (from the audit row). It separates "not retrieved" from "retrieved, not cited".
- **Abstention**: an unanswerable case is right only if the answer says it
  doesn't know; an answerable case that abstains is a false abstention.
- **Faithfulness** (judge): supported claims / claims, against the retrieved
  chunks, for answers that don't abstain.
- **Correctness** (judge): the answer gives the reference answer's facts.

Means are over cases (macro), each case weighing the same.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from eval.quality.questions import Kind


@dataclass
class CaseResult:
    case: str
    question: str
    user: str
    kind: Kind
    expected: list[str]
    answer: str
    model: str | None
    latency_s: float
    cited: list[str]
    retrieved: list[str]
    chunk_ids: list[str] = field(default_factory=list)
    abstained: bool | None = None
    correct: bool | None = None
    supported_claims: int | None = None
    claims: int | None = None
    unsupported: list[str] = field(default_factory=list)
    judge_error: str | None = None

    @property
    def answerable(self) -> bool:
        return self.kind is Kind.ANSWERABLE

    @property
    def faithfulness(self) -> float | None:
        if not self.claims or self.supported_claims is None:
            return None
        return self.supported_claims / self.claims


def precision(found: Iterable[str], expected: Iterable[str]) -> float | None:
    found_set = set(found)
    if not found_set:
        return None
    return len(found_set & set(expected)) / len(found_set)


def recall(found: Iterable[str], expected: Iterable[str]) -> float | None:
    expected_set = set(expected)
    if not expected_set:
        return None
    return len(set(found) & expected_set) / len(expected_set)


def mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def nearest_rank(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _count(results: Sequence[CaseResult], test: Callable[[CaseResult], bool | None]) -> str:
    return f"{sum(1 for r in results if test(r))}/{len(results)}"


def answerable_table(results: Sequence[CaseResult]) -> str:
    rows = [r for r in results if r.answerable]
    citing = [r for r in rows if r.cited]
    answered = [r for r in rows if r.abstained is False]
    faithful = [r for r in answered if r.faithfulness is not None]
    lines = [
        f"Answerable cases: {len(rows)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Expected document retrieved (doc recall in top-k) "
        f"| {_pct(mean(recall(r.retrieved, r.expected) for r in rows))} |",
        f"| Citation precision (mean over the {len(citing)} answers that cite) "
        f"| {_pct(mean(precision(r.cited, r.expected) for r in citing))} |",
        f"| Citation recall (mean) | {_pct(mean(recall(r.cited, r.expected) for r in rows))} |",
        f"| Answers citing nothing | {len(rows) - len(citing)}/{len(rows)} |",
        f'| False "I don\'t know" | {_count(rows, lambda r: r.abstained)} |',
        f"| Correct vs reference (judge) | {_count(rows, lambda r: r.correct)} |",
        f"| Faithfulness, mean supported-claim ratio (judge, {len(faithful)} answers) "
        f"| {_pct(mean(r.faithfulness for r in faithful))} |",
        f"| Fully faithful answers (judge) "
        f"| {sum(1 for r in faithful if r.faithfulness == 1.0)}/{len(faithful)} |",
    ]
    return "\n".join(lines)


def unanswerable_table(results: Sequence[CaseResult]) -> str:
    lines = [
        '| Why unanswerable | Cases | Said "I don\'t know" | Cited a document '
        "| Answered: faithfulness (judge) |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in (Kind.DENIED_BY_ACL, Kind.OTHER_TENANT, Kind.OUT_OF_CORPUS):
        rows = [r for r in results if r.kind is kind]
        if not rows:
            continue
        answered = [r for r in rows if r.abstained is False]
        lines.append(
            f"| {kind} | {len(rows)} | {_count(rows, lambda r: r.abstained)} "
            f"| {_count(rows, lambda r: bool(r.cited))} "
            f"| {_pct(mean(r.faithfulness for r in answered))} ({len(answered)}) |"
        )
    rows = [r for r in results if not r.answerable]
    lines.append(
        f"| **all** | {len(rows)} | {_count(rows, lambda r: r.abstained)} "
        f"| {_count(rows, lambda r: bool(r.cited))} | |"
    )
    return "\n".join(lines)


def latency_line(results: Sequence[CaseResult]) -> str:
    times = [r.latency_s for r in results]
    if not times:
        return ""
    return (
        f"`POST /ask` latency over {len(times)} calls: median "
        f"{nearest_rank(times, 0.5):.1f} s, p95 {nearest_rank(times, 0.95):.1f} s, "
        f"max {max(times):.1f} s."
    )


def _flag(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def case_table(results: Sequence[CaseResult]) -> str:
    lines = [
        "| Case | Kind | Retrieved expected | Cited | P | R | IDK | Correct | Faithful |",
        "|---|---|---:|---|---:|---:|---|---|---:|",
    ]
    for r in results:
        cited = ", ".join(r.cited) or "-"
        retrieved = _pct(recall(r.retrieved, r.expected))
        faithful = "-" if r.faithfulness is None else f"{r.supported_claims}/{r.claims}"
        lines.append(
            f"| {r.case} | {r.kind} | {retrieved} | {cited} "
            f"| {_pct(precision(r.cited, r.expected))} | {_pct(recall(r.cited, r.expected))} "
            f"| {_flag(r.abstained)} | {_flag(r.correct)} | {faithful} |"
        )
    return "\n".join(lines)


def summary(results: Sequence[CaseResult]) -> str:
    errors = [r.case for r in results if r.judge_error]
    parts = [
        "## Answerable",
        answerable_table(results),
        "## Unanswerable",
        unanswerable_table(results),
        latency_line(results),
    ]
    if errors:
        parts.append(f"Judge errors (excluded from judge scores): {', '.join(errors)}")
    parts += ["## Per case", case_table(results)]
    return "\n\n".join(parts) + "\n"
