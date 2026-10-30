"""UNKNOWN-appropriateness evaluator.

Checks whether the pipeline emits UNKNOWN / canonical-refusal when the
corpus surface is empty (no nuggets, no accepted passages). Emitting a
concrete guess with zero evidence is worse than a refusal under
calibration scoring, so this evaluator flags such cases.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["UnknownAppropriateScore", "evaluate_unknown_when_should"]


@dataclass(frozen=True)
class UnknownAppropriateScore:
    """Per-qid UNKNOWN-appropriateness tag."""

    appropriate: bool
    tag: str


def evaluate_unknown_when_should(
    answer: str,
    *,
    n_real_nuggets: int,
    n_accepted_passages: int,
) -> UnknownAppropriateScore:
    """Assess whether the emitted answer is UNKNOWN-appropriate.

    Args:
        answer: The emitted answer string.
        n_real_nuggets: Count of verified_supporting nuggets in the DB
            at emission time.
        n_accepted_passages: Count of accepted-status passages.

    Returns:
        ``UnknownAppropriateScore``:
          - appropriate=True and tag="correct_refusal" when the corpus
            is empty and the answer is a refusal.
          - appropriate=True and tag="concrete_with_evidence" when
            evidence is present and the answer is concrete.
          - appropriate=False and tag="concrete_without_evidence" when
            evidence is absent and the answer is a concrete guess.
          - appropriate=False and tag="refusal_despite_evidence" when
            evidence is present and the answer is a refusal.
    """
    is_refusal = _looks_like_refusal_text(answer)
    has_evidence = n_real_nuggets > 0 or n_accepted_passages > 0
    if is_refusal and not has_evidence:
        return UnknownAppropriateScore(True, "correct_refusal")
    if not is_refusal and has_evidence:
        return UnknownAppropriateScore(True, "concrete_with_evidence")
    if not is_refusal and not has_evidence:
        return UnknownAppropriateScore(False, "concrete_without_evidence")
    return UnknownAppropriateScore(False, "refusal_despite_evidence")


def _looks_like_refusal_text(answer: str) -> bool:
    if not answer:
        return True
    lo = answer.lower()
    return (
        "corpus does not support" in lo
        or "corpus does not contain" in lo
        or "corpus does not mention" in lo
        or "insufficient evidence" in lo
        or "cannot determine" in lo
        or lo.strip() == "unknown"
    )
