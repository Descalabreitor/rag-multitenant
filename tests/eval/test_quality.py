"""The answer-quality baseline (eval.quality), offline: the question set, the
scores and the judge's parsing. No database, no models."""

from collections import Counter

import pytest

from eval.quality.judge import (
    JudgeConfig,
    first_json_object,
    parse_faithfulness,
    parse_verdict,
    plain_abstention,
    strip_markers,
)
from eval.quality.questions import DOCUMENTS, QUESTIONS, USERS, Kind, cases
from eval.quality.scoring import CaseResult, precision, recall, summary
from ragmt.domain import NO_CONTEXT_ANSWER
from seed.corpus import TENANTS


def test_questions_name_seed_documents_and_users() -> None:
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)
    assert 20 <= len(QUESTIONS) <= 30
    for question in QUESTIONS:
        assert set(question.sources) <= DOCUMENTS.keys()
        assert set(question.askers) <= USERS.keys()
        assert len(set(question.askers)) == len(question.askers)
        assert bool(question.sources) == bool(question.reference)


def test_every_document_and_user_is_covered() -> None:
    all_cases = cases()
    answered = {title for case in all_cases for title in case.expected}
    assert answered == DOCUMENTS.keys()
    assert {case.user.username for case in all_cases} == USERS.keys()


def test_every_kind_of_unanswerable_case_is_there() -> None:
    kinds = Counter(case.kind for case in cases())
    assert set(kinds) == set(Kind)
    assert kinds[Kind.ANSWERABLE] >= 15


def test_readability_follows_the_seed_acls() -> None:
    by_user = {
        name: {title for title, doc in DOCUMENTS.items() if user.can_read(doc)}
        for name, user in USERS.items()
    }
    assert by_user == {
        "alice": {"Employee handbook", "Q3 budget", "Performance review: Alice"},
        "bob": {"Employee handbook", "Routing service runbook"},
        "erin": {"Employee handbook"},
        "carol": {"Lab safety policy", "Compound UB-7 trial results"},
        "dave": {"Lab safety policy", "Annual budget"},
    }
    admins = {name: user.admin.username for name, user in USERS.items()}
    assert admins == {
        "alice": "bob",
        "bob": "bob",
        "erin": "bob",
        "carol": "carol",
        "dave": "carol",
    }
    assert {t.uploader for t in TENANTS} == {USERS["bob"].sub, USERS["carol"].sub}


def test_kinds_of_specific_cases() -> None:
    kinds = {case.id: case.kind for case in cases()}
    assert kinds["fleet-budget/alice"] is Kind.ANSWERABLE
    assert kinds["fleet-budget/bob"] is Kind.DENIED_BY_ACL  # the admin isn't in finance
    assert kinds["fleet-budget/dave"] is Kind.OTHER_TENANT  # finance, but Umbra's
    assert kinds["capital/dave"] is Kind.OUT_OF_CORPUS


def test_cases_can_be_limited_to_some_users() -> None:
    assert {c.user.username for c in cases(frozenset({"erin"}))} == {"erin"}


def test_precision_and_recall() -> None:
    assert precision([], ["a"]) is None
    assert precision(["a", "b"], ["a"]) == 0.5
    assert recall(["a"], []) is None
    assert recall(["b"], ["a"]) == 0.0
    assert recall(["a", "b"], ["a"]) == 1.0


def _result(case: str, kind: Kind, **kwargs: object) -> CaseResult:
    fields: dict[str, object] = {
        "question": "q",
        "user": "u",
        "expected": [],
        "answer": "",
        "model": None,
        "latency_s": 1.0,
        "cited": [],
        "retrieved": [],
    }
    fields.update(kwargs)
    return CaseResult(case=case, kind=kind, **fields)  # type: ignore[arg-type]


def test_summary_counts_abstentions_and_citations() -> None:
    results = [
        _result(
            "a/x",
            Kind.ANSWERABLE,
            expected=["A"],
            cited=["A", "B"],
            retrieved=["A", "B"],
            abstained=False,
            correct=True,
            supported_claims=1,
            claims=2,
        ),
        _result("b/x", Kind.ANSWERABLE, expected=["A"], abstained=True, correct=False),
        _result("c/x", Kind.DENIED_BY_ACL, abstained=True, correct=True),
        _result("d/x", Kind.OUT_OF_CORPUS, cited=["C"], abstained=False, correct=False),
    ]
    text = summary(results)
    assert "| Citation precision (mean over the 1 answers that cite) | 0.50 |" in text
    assert "| Citation recall (mean) | 0.50 |" in text
    assert "| Expected document retrieved (doc recall in top-k) | 0.50 |" in text
    assert '| False "I don\'t know" | 1/2 |' in text
    assert "| denied by ACL | 1 | 1/1 | 0/1 |" in text
    assert "| out of corpus | 1 | 0/1 | 1/1 |" in text
    assert "| a/x | answerable | 1.00 | A, B | 0.50 | 1.00 | no | yes | 1/2 |" in text


@pytest.mark.parametrize(
    ("answer", "plain"),
    [
        ("I don't know.", True),
        ("I do not know [doc:00000000-0000-0000-0000-000000000000#1].", True),
        (NO_CONTEXT_ANSWER, True),
        ("I don't know. But core hours are 10:00 to 15:00.", False),
        ("Core hours are 10:00 to 15:00.", False),
    ],
)
def test_plain_abstention(answer: str, plain: bool) -> None:
    assert plain_abstention(answer) is plain


def test_markers_are_stripped_for_the_judge() -> None:
    assert strip_markers("Friday [doc:abc#1] [doc:abc#2].") == "Friday."


def test_judge_replies_are_parsed_leniently() -> None:
    reply = 'Sure:\n```json\n{"abstains": false, "correct": true}\n```'
    verdict = parse_verdict(first_json_object(reply))
    assert (verdict.abstains, verdict.correct) == (False, True)
    faith = parse_faithfulness(
        {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}
    )
    assert (faith.supported, faith.claims, faith.unsupported, faith.score) == (1, 2, ("b",), 0.5)
    assert parse_faithfulness({"claims": []}).score is None
    with pytest.raises(ValueError):
        first_json_object("no json here")
    with pytest.raises(ValueError):
        parse_verdict({"abstains": "no", "correct": True})
    with pytest.raises(ValueError):
        parse_faithfulness({"claims": [{"claim": "a"}]})


def test_judge_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("QUALITY_JUDGE_PROVIDER", "QUALITY_JUDGE_MODEL", "QUALITY_JUDGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    assert JudgeConfig.from_env() == JudgeConfig("ollama", "llama3.1:8b", "http://ollama:11434")

    monkeypatch.setenv("QUALITY_JUDGE_PROVIDER", "openai_compat")
    with pytest.raises(ValueError, match="QUALITY_JUDGE_MODEL"):
        JudgeConfig.from_env()
    monkeypatch.setenv("QUALITY_JUDGE_MODEL", "judge")
    monkeypatch.setenv("QUALITY_JUDGE_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("QUALITY_JUDGE_API_KEY", "k")
    config = JudgeConfig.from_env()
    assert (config.provider, config.model, config.describe()) == (
        "openai_compat",
        "judge",
        "openai_compat:judge",
    )
    assert config.api_key is not None and config.api_key.get_secret_value() == "k"

    monkeypatch.setenv("QUALITY_JUDGE_PROVIDER", "nope")
    with pytest.raises(ValueError, match="unknown"):
        JudgeConfig.from_env()
