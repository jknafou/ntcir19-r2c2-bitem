"""On-topic evaluator — Jaccard overlap of question entities vs answer words.

Approximates "does the answer engage with the question's subject" by
measuring the Jaccard similarity of lowercased content tokens (stop-
word filtered) between the question and the answer. It is a
plausibility signal for the node-quality table, not a gold-answer match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["OnTopicScore", "evaluate_on_topic"]

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "with",
        "by",
        "from",
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
        "did",
        "does",
        "do",
        "be",
        "been",
        "being",
        "are",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _STOP_WORDS and len(t) > 1
    }


@dataclass(frozen=True)
class OnTopicScore:
    """Jaccard similarity score + overlapping token count."""

    jaccard: float
    overlap_count: int
    overlap_tokens: tuple[str, ...]


def evaluate_on_topic(question: str, answer: str) -> OnTopicScore:
    """Compute a lowercase-token Jaccard between question and answer.

    Args:
        question: The question text.
        answer: The emitted answer string.

    Returns:
        ``OnTopicScore`` with jaccard in [0, 1], overlap_count, and a
        tuple of the overlapping tokens (for debugging).
    """
    q_toks = _tokens(question)
    a_toks = _tokens(answer)
    if not q_toks or not a_toks:
        return OnTopicScore(0.0, 0, ())
    overlap = q_toks & a_toks
    union = q_toks | a_toks
    jacc = len(overlap) / max(1, len(union))
    return OnTopicScore(jacc, len(overlap), tuple(sorted(overlap)))
