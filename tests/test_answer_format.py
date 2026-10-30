"""Tests for the per-answer R2C2 format validator.

Covers ``r2c2_agent.agent.pipeline.answer_format.validate_r2c2_answer``, which
the pipeline calls at every site that emits an answer, not as a
post-processing pass over the output file.

Rules under test:
  * empty / None / non-str → reason="empty"
  * exact CANONICAL_REFUSAL_TEXT → pass
  * internal \\n / \\r / \\t → cleaned to space, still ok
  * contains ``;`` → reason="contains_semicolon"
  * sentinel fragments (``<tool_call``, ``<|``, …) → reason="sentinel_leak"
  * length > ANSWER_MAX_LEN → reason="too_long" with cleaned truncated
  * unicode, plain words, sentence-shape → pass
"""

from __future__ import annotations

import pytest

from r2c2_agent.agent.pipeline.answer_format import (
    ANSWER_MAX_LEN,
    CANONICAL_REFUSAL_TEXT,
    AnswerValidationResult,
    validate_r2c2_answer,
)

# Pass cases


class TestPassCases:
    def test_plain_single_word(self) -> None:
        r = validate_r2c2_answer("Spielberg")
        assert r.ok is True
        assert r.reason == ""
        assert r.cleaned == "Spielberg"

    def test_plain_sentence(self) -> None:
        r = validate_r2c2_answer("Anthony Mackie played Captain America")
        assert r.ok is True
        assert r.cleaned == "Anthony Mackie played Captain America"

    def test_multi_word_noun_phrase(self) -> None:
        r = validate_r2c2_answer("the Nobel Prize in Literature")
        assert r.ok is True

    def test_canonical_refusal_exact_match_passes(self) -> None:
        r = validate_r2c2_answer(CANONICAL_REFUSAL_TEXT)
        assert r.ok is True
        assert r.reason == ""
        assert r.cleaned == CANONICAL_REFUSAL_TEXT

    def test_leading_trailing_whitespace_stripped(self) -> None:
        r = validate_r2c2_answer("   Steven Spielberg   ")
        assert r.ok is True
        assert r.cleaned == "Steven Spielberg"

    def test_unicode_accents(self) -> None:
        r = validate_r2c2_answer("Pedro Almodóvar")
        assert r.ok is True
        assert r.cleaned == "Pedro Almodóvar"

    def test_unicode_cjk(self) -> None:
        r = validate_r2c2_answer("宮崎駿")
        assert r.ok is True

    def test_internal_newline_cleaned_to_space(self) -> None:
        r = validate_r2c2_answer("Anthony\nMackie")
        assert r.ok is True
        assert r.cleaned == "Anthony Mackie"

    def test_internal_carriage_return_cleaned(self) -> None:
        r = validate_r2c2_answer("Anthony\rMackie")
        assert r.ok is True
        assert r.cleaned == "Anthony Mackie"

    def test_internal_tab_cleaned(self) -> None:
        r = validate_r2c2_answer("Anthony\tMackie")
        assert r.ok is True
        assert r.cleaned == "Anthony Mackie"

    def test_mixed_whitespace_collapsed(self) -> None:
        r = validate_r2c2_answer("Anthony \n\t Mackie")
        assert r.ok is True
        assert r.cleaned == "Anthony Mackie"

    def test_length_exactly_at_cap_passes(self) -> None:
        s = "A" * ANSWER_MAX_LEN
        r = validate_r2c2_answer(s)
        assert r.ok is True
        assert len(r.cleaned) == ANSWER_MAX_LEN


# Fail cases


class TestFailCases:
    def test_empty_string_fails(self) -> None:
        r = validate_r2c2_answer("")
        assert r.ok is False
        assert r.reason == "empty"
        assert r.cleaned == ""

    def test_whitespace_only_fails(self) -> None:
        r = validate_r2c2_answer("   \t\n  ")
        assert r.ok is False
        assert r.reason == "empty"

    def test_none_fails(self) -> None:
        # type: ignore[arg-type] — deliberately passing wrong type
        r = validate_r2c2_answer(None)  # type: ignore[arg-type]
        assert r.ok is False
        assert r.reason == "empty"

    def test_non_string_fails(self) -> None:
        # type: ignore[arg-type] — deliberately passing wrong type
        r = validate_r2c2_answer(42)  # type: ignore[arg-type]
        assert r.ok is False
        assert r.reason == "empty"

    def test_semicolon_in_answer_fails(self) -> None:
        r = validate_r2c2_answer("Spielberg; director")
        assert r.ok is False
        assert r.reason == "contains_semicolon"

    def test_multiple_semicolons_fail(self) -> None:
        r = validate_r2c2_answer("a;b;c")
        assert r.ok is False
        assert r.reason == "contains_semicolon"

    def test_tool_call_sentinel_fails(self) -> None:
        r = validate_r2c2_answer("Steven<tool_call>{...}</tool_call>")
        assert r.ok is False
        assert r.reason == "sentinel_leak"

    def test_chat_template_sentinel_fails(self) -> None:
        r = validate_r2c2_answer("Steven <|im_end|>")
        assert r.ok is False
        assert r.reason == "sentinel_leak"

    def test_json_code_fence_sentinel_fails(self) -> None:
        r = validate_r2c2_answer("```json\n{'answer': 'Steven'}\n```")
        assert r.ok is False
        # Semicolon-less JSON code fences hit the sentinel gate first.
        assert r.reason == "sentinel_leak"

    def test_too_long_truncates_and_fails(self) -> None:
        s = "A" * (ANSWER_MAX_LEN + 50)
        r = validate_r2c2_answer(s)
        assert r.ok is False
        assert r.reason == "too_long"
        assert len(r.cleaned) == ANSWER_MAX_LEN

    def test_refusal_with_extra_text_fails_semicolon(self) -> None:
        # Anything that is not an exact match of CANONICAL_REFUSAL_TEXT
        # falls back to the normal rules, so a pseudo-refusal with
        # ``;`` in it is rejected. The caller substitutes the canonical
        # version on ``ok=False``.
        r = validate_r2c2_answer(
            "Sorry; the corpus does not support an answer to this question."
        )
        assert r.ok is False
        assert r.reason == "contains_semicolon"


# Return shape / API contract


class TestCanonicalRefusalAlias:
    """Pin that ``pipeline_handlers.CANONICAL_REFUSAL_ANSWER`` is the
    same string as ``r2c2_agent.agent.pipeline.answer_format.CANONICAL_REFUSAL_TEXT``
    and that the validator accepts it, so the two constants cannot
    drift apart.
    """

    def test_canonical_refusal_answer_matches_text(self) -> None:
        import sys
        from pathlib import Path

        orch = Path(__file__).resolve().parent.parent / "scripts" / "orchestrator"
        if str(orch) not in sys.path:
            sys.path.insert(0, str(orch))

        from r2c2_agent.orchestrator.pipeline_handlers import CANONICAL_REFUSAL_ANSWER  # noqa: PLC0415

        assert CANONICAL_REFUSAL_ANSWER == CANONICAL_REFUSAL_TEXT
        # Both must pass the validator.
        assert validate_r2c2_answer(CANONICAL_REFUSAL_ANSWER).ok is True
        assert validate_r2c2_answer(CANONICAL_REFUSAL_TEXT).ok is True


class TestAPIContract:
    def test_result_is_frozen_dataclass(self) -> None:
        r = validate_r2c2_answer("Spielberg")
        with pytest.raises(Exception):
            r.ok = False  # type: ignore[misc] — frozen

    def test_qtype_accepted_but_ignored(self) -> None:
        r_none = validate_r2c2_answer("Spielberg")
        r_simple = validate_r2c2_answer("Spielberg", qtype="simple")
        r_multihop = validate_r2c2_answer("Spielberg", qtype="multi_hop")
        assert r_none.ok == r_simple.ok == r_multihop.ok is True
        assert r_none.cleaned == r_simple.cleaned == r_multihop.cleaned

    def test_result_is_instance(self) -> None:
        r = validate_r2c2_answer("Spielberg")
        assert isinstance(r, AnswerValidationResult)
