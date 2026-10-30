"""Per-answer format validator, run inside the pipeline at tier emission.

Every answer-emission site in
``src/r2c2_agent/orchestrator/pipeline_handlers.py`` calls it: tier-1
``handle_submit_answer``, the tier-2+ ``force_deliver`` ladder, the
``answer_search_support`` auto-commit and the final-fallback
submit-history picker. The call comes before the answer goes into the
envelope or is committed to the DB. On failure the caller substitutes
``CANONICAL_REFUSAL_TEXT``; no code path raises.

The rules derive from the R2C2 submission format.
``r2c2_agent.agent.submission_validator.validate_ac_file`` checks the same
rules again, line by line, when the AC file is written; this module is
the earlier gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from r2c2_agent.agent.pipeline._r2c2_rules import (
    ANSWER_MAX_LEN,
    BANNED_WHITESPACE_RE,
    CANONICAL_REFUSAL_TEXT,
    SENTINEL_FRAGMENTS,
)

__all__ = [
    "ANSWER_MAX_LEN",
    "AnswerValidationResult",
    "CANONICAL_REFUSAL_TEXT",
    "validate_r2c2_answer",
]


@dataclass(frozen=True)
class AnswerValidationResult:
    """Result of ``validate_r2c2_answer``.

    Attributes:
        ok: True when the (possibly cleaned) answer passes every R2C2
            rule and is safe to emit verbatim into the AC file. False
            means the caller MUST substitute ``CANONICAL_REFUSAL_TEXT``
            before emitting — no answer mode should raise.
        reason: Empty string on success. Short machine-readable tag on
            failure (``empty`` / ``contains_semicolon`` /
            ``sentinel_leak`` / ``too_long``).
        cleaned: The sanitized answer. Whitespace is normalized and
            surrounding spaces stripped. On failure, this still carries
            the best-effort cleaned form for logging — it is NOT safe to
            emit when ``ok=False``.
    """

    ok: bool
    reason: str
    cleaned: str


def validate_r2c2_answer(
    answer: str,
    qtype: str | None = None,  # noqa: ARG001 — reserved for future per-qtype rules
) -> AnswerValidationResult:
    """Validate a single answer string against R2C2 AC-file rules.

    Rules applied, in order:
      1. Non-str / None / empty → fail ``empty``.
      2. Exact match against ``CANONICAL_REFUSAL_TEXT`` → pass (the
         canonical refusal contains a ``;`` inside prose and has to
         bypass the ``;`` rule at step 6).
      3. Strip surrounding whitespace.
      4. Collapse internal ``\\n``/``\\r``/``\\t`` runs to a single space;
         collapse resulting multi-space runs; re-strip. Empty result →
         fail ``empty``.
      5. Any sentinel fragment present (``<tool_call``, ``<|``, …) →
         fail ``sentinel_leak``.
      6. Any ``;`` remaining → fail ``contains_semicolon``. The ``;`` is
         the AC-file field separator (``answer;confidence``); an answer
         containing it would corrupt the emitted line.
      7. Length > ``ANSWER_MAX_LEN`` → truncate and fail ``too_long``.
         The caller decides whether to emit the truncated form or fall
         back to refusal.

    ``qtype`` is accepted so the signature matches the shape-aware
    ``_answer_shape_check`` in ``pipeline_handlers``. It is not used
    here: shape validation lives there.

    Args:
        answer: Raw answer candidate to validate.
        qtype: Question type (unused; reserved for per-qtype rules such
            as a max-word count for "simple" vs sentence-shape
            aggregations).

    Returns:
        AnswerValidationResult. On ``ok=False`` the caller substitutes
        ``CANONICAL_REFUSAL_TEXT`` before emitting.
    """
    if not isinstance(answer, str) or not answer:
        return AnswerValidationResult(ok=False, reason="empty", cleaned="")

    # The canonical refusal contains a ``;`` inside its prose, so an exact
    # match passes before the ``;`` rule is reached.
    if answer == CANONICAL_REFUSAL_TEXT:
        return AnswerValidationResult(ok=True, reason="", cleaned=answer)

    cleaned = answer.strip()

    # Each run of ``\n``/``\r``/``\t`` becomes one space, which can leave
    # double spaces behind; those are collapsed so the cleaned form can be
    # emitted as it is.
    cleaned = BANNED_WHITESPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if not cleaned:
        return AnswerValidationResult(ok=False, reason="empty", cleaned="")

    # Sentinels are matched as case-insensitive substrings.
    lower = cleaned.lower()
    for sentinel in SENTINEL_FRAGMENTS:
        if sentinel in lower:
            return AnswerValidationResult(
                ok=False, reason="sentinel_leak", cleaned=cleaned
            )

    # ``;`` is the AC-file field separator.
    if ";" in cleaned:
        return AnswerValidationResult(
            ok=False, reason="contains_semicolon", cleaned=cleaned
        )

    # ``cleaned`` carries the truncated form, so the caller can log it or
    # emit it at a lower confidence in place of a refusal.
    if len(cleaned) > ANSWER_MAX_LEN:
        truncated = cleaned[:ANSWER_MAX_LEN].rstrip()
        return AnswerValidationResult(ok=False, reason="too_long", cleaned=truncated)

    return AnswerValidationResult(ok=True, reason="", cleaned=cleaned)
