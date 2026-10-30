"""Offline unit tests for the quality evaluators.

The offline test battery covers evaluator correctness even when no
dev worker is available. A live probe uses the same evaluators to score
each node's LLM output in the per-node quality report.
"""

from __future__ import annotations

from tests.quality.evaluators.guided_regex import evaluate_guided_regex
from tests.quality.evaluators.on_topic import evaluate_on_topic
from tests.quality.evaluators.unknown_when_should import (
    evaluate_unknown_when_should,
)


def test_guided_regex_accepts_well_formed_answer() -> None:
    s = evaluate_guided_regex("Answer: taxi driver")
    assert s.matches is True
    assert s.reason == "ok"


def test_guided_regex_accepts_unknown() -> None:
    assert evaluate_guided_regex("UNKNOWN").matches is True


def test_guided_regex_rejects_missing_prefix() -> None:
    s = evaluate_guided_regex("taxi driver")
    assert s.matches is False
    assert s.reason == "missing_prefix"


def test_guided_regex_rejects_newline() -> None:
    s = evaluate_guided_regex("Answer: taxi\ndriver")
    assert s.matches is False
    assert s.reason == "contains_newline"


def test_guided_regex_rejects_semicolon() -> None:
    s = evaluate_guided_regex("Answer: taxi; driver")
    assert s.matches is False
    assert s.reason == "contains_semicolon"


def test_guided_regex_rejects_empty() -> None:
    assert evaluate_guided_regex("").matches is False


def test_on_topic_overlap_positive() -> None:
    s = evaluate_on_topic("Who piloted Red Five?", "Luke Skywalker piloted Red Five")
    assert s.jaccard > 0
    assert s.overlap_count >= 2


def test_on_topic_no_overlap_zero() -> None:
    s = evaluate_on_topic("Who piloted Red Five?", "zzz qqq yyy")
    assert s.jaccard == 0.0
    assert s.overlap_count == 0


def test_unknown_when_should_correct_refusal() -> None:
    s = evaluate_unknown_when_should(
        "The corpus does not support an answer.",
        n_real_nuggets=0,
        n_accepted_passages=0,
    )
    assert s.appropriate is True
    assert s.tag == "correct_refusal"


def test_unknown_when_should_concrete_without_evidence() -> None:
    s = evaluate_unknown_when_should(
        "taxi driver",
        n_real_nuggets=0,
        n_accepted_passages=0,
    )
    assert s.appropriate is False
    assert s.tag == "concrete_without_evidence"


def test_unknown_when_should_concrete_with_evidence() -> None:
    s = evaluate_unknown_when_should(
        "taxi driver",
        n_real_nuggets=1,
        n_accepted_passages=1,
    )
    assert s.appropriate is True
    assert s.tag == "concrete_with_evidence"


def test_unknown_when_should_refusal_despite_evidence() -> None:
    s = evaluate_unknown_when_should(
        "Unknown",
        n_real_nuggets=1,
        n_accepted_passages=1,
    )
    assert s.appropriate is False
    assert s.tag == "refusal_despite_evidence"
