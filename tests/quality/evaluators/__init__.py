"""Per-dimension quality evaluators.

Each evaluator is a pure function ``(question, answer, context) -> Score``
that returns a per-qid score and a short human-readable explanation.
"""

from __future__ import annotations

from tests.quality.evaluators.guided_regex import evaluate_guided_regex
from tests.quality.evaluators.on_topic import evaluate_on_topic
from tests.quality.evaluators.unknown_when_should import evaluate_unknown_when_should

__all__ = [
    "evaluate_guided_regex",
    "evaluate_on_topic",
    "evaluate_unknown_when_should",
]
