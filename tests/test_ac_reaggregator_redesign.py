"""Unit tests for the trace-based answer helpers of the AC re-aggregator.

Covers these helpers of ``r2c2_agent.orchestrator.ac_reaggregator``:

- ``_extract_submit_attempts_from_trace``
- ``_classify_attempt`` (via ``_extract_submit_attempts_from_trace``)
- ``_sanitize_ac_answer``
- ``_pick_per_iter_answer``
- ``_tier_a_star_candidates`` (via ``select_v2_nuggets`` integration)

The traces reproduce the per-attempt schema of recorded runs for
qids 0017, 0022, 0025, 0047 and 0058, which include refusals and
answers rejected by the confidence gate.
"""

from __future__ import annotations

import pytest
from r2c2_agent.orchestrator.ac_reaggregator import (
    ATTEMPT_OK,
    ATTEMPT_REFUSAL_TEXT,
    ATTEMPT_REJ_FORMAT,
    ATTEMPT_REJ_HARDGATE,
    FrozenPR,
    _extract_submit_attempts_from_trace,
    _is_canonical_refusal_text,
    _pick_per_iter_answer,
    _sanitize_ac_answer,
    _tier_a_star_candidates,
)


def _make_trace(
    submit_attempts: list[dict],
) -> dict:
    """Build a minimal trace record with a list of submit_answer attempts.

    Each entry of ``submit_attempts`` is a dict with keys::

        answer (str), rationale (str), citation_pids (list[str]),
        committed (bool), confidence (float|None),
        reason (str|None), error (str|None), n_eff (int|None)

    The function emits the corresponding interleaved
    ``assistant`` / ``tool`` messages.
    """
    msgs: list[dict] = []
    for i, att in enumerate(submit_attempts):
        tcid = f"tc-{i}"
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tcid,
                        "function": {
                            "name": "submit_answer",
                            "arguments": {
                                "answer": att["answer"],
                                "rationale": att.get("rationale", ""),
                                "citation_pids": att.get("citation_pids", []),
                            },
                        },
                    }
                ],
            }
        )
        result: dict = {"committed": att["committed"]}
        if att.get("confidence") is not None:
            result["confidence"] = att["confidence"]
        if att.get("reason") is not None:
            result["reason"] = att["reason"]
        if att.get("error") is not None:
            result["error"] = att["error"]
        if att.get("n_eff") is not None:
            result["n_effective_supporting"] = att["n_eff"]
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": tcid,
                "content": result,
            }
        )
    return {"messages": msgs}


# _is_canonical_refusal_text


class TestRefusalTextDetection:
    def test_canonical_refusal_matches(self):
        text = (
            "The corpus does not support an answer to this question; "
            "insufficient evidence to respond."
        )
        assert _is_canonical_refusal_text(text)

    def test_non_refusal_does_not_match(self):
        assert not _is_canonical_refusal_text("Avatar")
        assert not _is_canonical_refusal_text("3")
        assert not _is_canonical_refusal_text(
            "Imperial death troopers killed both characters"
        )

    def test_empty_string(self):
        assert not _is_canonical_refusal_text("")
        assert not _is_canonical_refusal_text(None)  # type: ignore[arg-type]


# _sanitize_ac_answer


class TestSanitizeAcAnswer:
    def test_replaces_semicolon_with_comma(self):
        # The R2C2 line uses ';' as a delimiter; the answer must not contain raw ';'.
        assert _sanitize_ac_answer("Katharine Hepburn: 4; Audrey Hepburn: 1") == (
            "Katharine Hepburn: 4, Audrey Hepburn: 1"
        )

    def test_strips_double_quotes(self):
        assert _sanitize_ac_answer('"Avatar"') == "Avatar"

    def test_strips_single_quotes(self):
        assert _sanitize_ac_answer("'Avatar'") == "Avatar"

    def test_strips_quotes_with_semicolon_inside(self):
        # qid 0058 case: `"I'm one with the Force; the Force is with me"`
        # → semicolons get replaced first, then surrounding quotes get stripped.
        out = _sanitize_ac_answer('"I am one with the Force; the Force is with me"')
        assert out == "I am one with the Force, the Force is with me"

    def test_collapses_whitespace(self):
        assert _sanitize_ac_answer("Avatar    is\t\there") == "Avatar is here"

    def test_empty_string_passthrough(self):
        assert _sanitize_ac_answer("") == ""
        assert _sanitize_ac_answer(None) == ""  # type: ignore[arg-type]

    def test_does_not_strip_internal_quotes(self):
        assert _sanitize_ac_answer('she said "hello" today') == (
            'she said "hello" today'
        )


# _extract_submit_attempts_from_trace


class TestExtractSubmitAttempts:
    def test_qid_0017_iter0_five_hardgate_rejections(self):
        # Five low-confidence rejections in a row, as recorded for qid 0017.
        trace = _make_trace(
            [
                {
                    "answer": "Avatar",
                    "citation_pids": ["a", "b"],
                    "committed": False,
                    "confidence": 0.195,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 2,
                },
                {
                    "answer": "Avatar",
                    "citation_pids": ["a", "b", "c"],
                    "committed": False,
                    "confidence": 0.249,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 3,
                },
                {
                    "answer": "Avatar",
                    "citation_pids": ["a", "b", "c"],
                    "committed": False,
                    "confidence": 0.249,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 3,
                },
                {
                    "answer": "Avatar",
                    "citation_pids": ["a", "b", "c", "d"],
                    "committed": False,
                    "confidence": 0.303,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 4,
                },
                {
                    "answer": "Avatar",
                    "citation_pids": ["a", "b", "c", "d"],
                    "committed": False,
                    "confidence": 0.303,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 4,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        assert len(attempts) == 5
        assert all(a.classification == ATTEMPT_REJ_HARDGATE for a in attempts)
        assert [a.confidence for a in attempts] == pytest.approx(
            [0.195, 0.249, 0.249, 0.303, 0.303]
        )
        assert [a.n_eff for a in attempts] == [2, 3, 3, 4, 4]

    def test_qid_0022_iter0_committed_with_semicolon_in_answer(self):
        # The model emitted the real answer; force_deliver overrode it.
        trace = _make_trace(
            [
                {
                    "answer": (
                        "Katharine Hepburn: 4 Academy Award for Best Actress wins; "
                        "Audrey Hepburn: 1 Academy Award for Best Actress"
                    ),
                    "citation_pids": ["Q103618__00001", "Q42786__00040"],
                    "committed": True,
                    "confidence": 0.05,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        assert len(attempts) == 1
        # The answer is read from the call arguments, because the tool result
        # may carry a substituted refusal. The argument is not a refusal, so
        # the attempt is classified OK.
        assert attempts[0].classification == ATTEMPT_OK
        assert attempts[0].confidence == 0.05
        assert attempts[0].citation_pids == ["Q103618__00001", "Q42786__00040"]

    def test_qid_0025_iter0_invalid_rationale_then_refusal_commit(self):
        trace = _make_trace(
            [
                {
                    "answer": "The corpus does not support an answer to this question.",
                    "citation_pids": ["a", "b", "c"],
                    "committed": False,
                    "error": "invalid_rationale",
                },
                {
                    "answer": "The corpus does not support an answer to this question.",
                    "citation_pids": ["a", "b", "c"],
                    "committed": True,
                    "confidence": 0.05,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        assert len(attempts) == 2
        # Both are refusal-text class regardless of committed/error state.
        assert all(a.classification == ATTEMPT_REFUSAL_TEXT for a in attempts)

    def test_invalid_rationale_non_refusal_is_rej_format(self):
        trace = _make_trace(
            [
                {
                    "answer": "Imperial death troopers",
                    "citation_pids": ["a"],
                    "committed": False,
                    "error": "invalid_rationale",
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        assert len(attempts) == 1
        assert attempts[0].classification == ATTEMPT_REJ_FORMAT

    def test_no_submit_calls_returns_empty(self):
        # Trace has tool calls but none are submit_answer.
        trace = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x",
                            "function": {
                                "name": "search_passages",
                                "arguments": {"query": "foo"},
                            },
                        }
                    ],
                }
            ]
        }
        assert _extract_submit_attempts_from_trace(trace) == []

    def test_tool_call_id_pairing(self):
        # Two submit_answer calls with non-monotonic message ordering — the
        # walker must pair each by tool_call_id, not by physical position.
        trace = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc-A",
                            "function": {
                                "name": "submit_answer",
                                "arguments": {
                                    "answer": "first",
                                    "citation_pids": [],
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc-A",
                    "content": {"committed": True, "confidence": 0.4},
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc-B",
                            "function": {
                                "name": "submit_answer",
                                "arguments": {
                                    "answer": "second",
                                    "citation_pids": [],
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc-B",
                    "content": {"committed": True, "confidence": 0.7},
                },
            ]
        }
        attempts = _extract_submit_attempts_from_trace(trace)
        assert [a.answer_text for a in attempts] == ["first", "second"]
        assert [a.confidence for a in attempts] == [0.4, 0.7]

    def test_arguments_as_json_string(self):
        # Some traces store function.arguments as a JSON-encoded string.
        trace = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc-A",
                            "function": {
                                "name": "submit_answer",
                                "arguments": (
                                    '{"answer": "Avatar", "citation_pids": ["x"]}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc-A",
                    "content": '{"committed": false, "confidence": 0.3, '
                    '"reason": "low_confidence_or_insufficient_evidence"}',
                },
            ]
        }
        attempts = _extract_submit_attempts_from_trace(trace)
        assert len(attempts) == 1
        assert attempts[0].answer_text == "Avatar"
        assert attempts[0].citation_pids == ["x"]
        assert attempts[0].confidence == 0.3
        assert attempts[0].classification == ATTEMPT_REJ_HARDGATE


# _pick_per_iter_answer


class TestPickPerIterAnswer:
    def test_qid_0017_picks_max_conf_hardgate(self):
        trace = _make_trace(
            [
                {
                    "answer": "Avatar",
                    "citation_pids": ["a"],
                    "committed": False,
                    "confidence": 0.195,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 2,
                },
                {
                    "answer": "Avatar",
                    "citation_pids": ["a", "b", "c", "d"],
                    "committed": False,
                    "confidence": 0.303,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 4,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        # Frozen AC for 0017 is "Avatar;41" — confidence policy uses
        # max(tool*100, frozen).
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer="Avatar",
            frozen_pr_conf=41,
            pr_rank_map={},
            qid="0017",
        )
        assert result is not None
        ans, conf, cits = result
        assert ans == "Avatar"
        # tool_conf*100 = 30, frozen = 41 → emit 41 (floor preserved)
        assert conf == 41
        assert cits == ["a", "b", "c", "d"]

    def test_qid_0022_strips_semicolon_and_uses_frozen_floor(self):
        trace = _make_trace(
            [
                {
                    "answer": (
                        "Katharine Hepburn: 4 Academy Award for Best Actress wins; "
                        "Audrey Hepburn: 1 Academy Award for Best Actress"
                    ),
                    "citation_pids": ["Q103618__00001", "Q42786__00040"],
                    "committed": True,
                    "confidence": 0.05,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        # Frozen AC for 0022 is the canonical refusal at conf=25.
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer=(
                "The corpus does not support an answer to this question; "
                "insufficient evidence to respond."
            ),
            frozen_pr_conf=25,
            pr_rank_map={"Q103618__00001": 3, "Q42786__00040": 2},
            qid="0022",
        )
        assert result is not None
        ans, conf, _cits = result
        assert ";" not in ans
        assert ans.startswith("Katharine Hepburn: 4")
        # conf = max(0.05*100=5, 25) = 25
        assert conf == 25

    def test_qid_0058_strips_quotes_and_semicolon(self):
        trace = _make_trace(
            [
                {
                    "answer": '"I\'m one with the Force; the Force is with me"',
                    "citation_pids": ["539115__00001", "539115__00037"],
                    "committed": True,
                    "confidence": 0.05,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer=(
                "The corpus does not support an answer to this question; "
                "insufficient evidence to respond."
            ),
            frozen_pr_conf=25,
            pr_rank_map={"539115__00001": 2, "539115__00037": 3},
            qid="0058",
        )
        assert result is not None
        ans, conf, _ = result
        assert ";" not in ans
        assert not ans.startswith('"')
        assert "the Force" in ans
        assert conf == 25

    def test_qid_0025_returns_none_when_only_refusal_attempts(self):
        # Both attempts emit canonical-refusal text — no non-refusal answer
        # exists in the trace, so the picker returns None and the caller
        # falls back to the canonical-refusal pipeline.
        trace = _make_trace(
            [
                {
                    "answer": "The corpus does not support an answer to this question.",
                    "citation_pids": [],
                    "committed": False,
                    "error": "invalid_rationale",
                },
                {
                    "answer": "The corpus does not support an answer to this question.",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.05,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer=(
                "The corpus does not support an answer to this question; "
                "insufficient evidence to respond."
            ),
            frozen_pr_conf=25,
            pr_rank_map={},
            qid="0025",
        )
        assert result is None

    def test_empty_attempts_returns_none(self):
        result = _pick_per_iter_answer(
            [],
            frozen_pr_answer="anything",
            frozen_pr_conf=42,
            pr_rank_map={},
            qid="9999",
        )
        assert result is None

    def test_ok_class_outranks_rej_hardgate(self):
        # Class priority is OK > REJ_HARDGATE, whatever the confidence.
        trace = _make_trace(
            [
                {
                    "answer": "3",
                    "citation_pids": [],
                    "committed": False,
                    "confidence": 0.4,
                    "reason": "low_confidence_or_insufficient_evidence",
                },
                {
                    "answer": "3",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.15,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer="3",
            frozen_pr_conf=33,
            pr_rank_map={},
            qid="0047",
        )
        assert result is not None
        # OK class wins; tool*100 = 15 < frozen 33 → emit 33 (floor).
        ans, conf, _ = result
        assert ans == "3"
        assert conf == 33

    def test_refusal_floor_preserved_when_tool_conf_higher(self):
        trace = _make_trace(
            [
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.85,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer="Avatar",
            frozen_pr_conf=20,
            pr_rank_map={},
            qid="0099",
        )
        assert result is not None
        _, conf, _ = result
        # tool_conf*100 = 85 > frozen 20 → emit 85.
        assert conf == 85

    def test_pr_roster_alignment_breaks_tie_within_class(self):
        # Two REJ_HARDGATE attempts with identical conf/n_eff; the one
        # whose citations land in the PR roster wins.
        trace = _make_trace(
            [
                {
                    "answer": "Avatar",
                    "citation_pids": ["unknown1", "unknown2"],
                    "committed": False,
                    "confidence": 0.3,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 2,
                },
                {
                    "answer": "Avatar",
                    "citation_pids": ["pid_in_roster"],
                    "committed": False,
                    "confidence": 0.3,
                    "reason": "low_confidence_or_insufficient_evidence",
                    "n_eff": 2,
                },
            ]
        )
        attempts = _extract_submit_attempts_from_trace(trace)
        result = _pick_per_iter_answer(
            attempts,
            frozen_pr_answer="Avatar",
            frozen_pr_conf=0,
            pr_rank_map={"pid_in_roster": 1},
            qid="0017",
        )
        assert result is not None
        _, _, cits = result
        # Picker selects the second attempt (1 PR-roster hit > 0).
        assert cits == ["pid_in_roster"]


# _tier_a_star_candidates


class TestTierAStarCandidates:
    def test_synthesises_from_pr_text_for_each_resolvable_pid(self):
        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={
                "0022": [
                    (
                        2,
                        "Q42786",
                        (
                            "Hepburn received numerous awards and honours during her "
                            "career. From five nominations, she won the 1953 Academy "
                            "Award for Best Actress for her performance in Roman "
                            "Holiday."
                        ),
                    ),
                    (
                        3,
                        "Q103618",
                        (
                            "The Academy Award for Best Actress is an award presented "
                            "annually by the Academy of Motion Picture Arts and "
                            "Sciences. Katharine Hepburn holds the record for most "
                            "wins."
                        ),
                    ),
                ],
            },
        )
        rank_map = {"Q42786__00040": 2, "Q103618__00001": 3}
        cands = _tier_a_star_candidates(
            qid="0022",
            seed_pids=["Q103618__00001", "Q42786__00040"],
            rank_map=rank_map,
            pr=pr,
        )
        assert len(cands) == 2
        ranks = {c.pr_rank for c in cands}
        assert ranks == {2, 3}
        for c in cands:
            assert c.text  # non-empty
            assert c.passage_id in rank_map
            assert c.nli_score > 0  # gets the NLI bonus

    def test_drops_pids_without_pr_rank(self):
        # Passage text must be ≥ RESCUE_NUGGET_MIN_LEN (20) chars to be kept.
        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={
                "0022": [
                    (
                        1,
                        "Qfoo",
                        (
                            "Some passage text that is long enough to survive "
                            "the rescue-nugget length filter."
                        ),
                    )
                ]
            },
        )
        cands = _tier_a_star_candidates(
            qid="0022",
            seed_pids=["unknown_pid", "Qfoo__00001"],
            rank_map={"Qfoo__00001": 1},
            pr=pr,
        )
        assert len(cands) == 1
        assert cands[0].pr_rank == 1

    def test_empty_seed_pids_returns_empty(self):
        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={"0022": [(1, "Q", "text long enough to pass rescue filter")]},
        )
        assert _tier_a_star_candidates("0022", [], {"a": 1}, pr) == []

    def test_dedup_on_rank(self):
        # Two seed pids that resolve to the same rank produce one candidate.
        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={
                "0022": [
                    (
                        1,
                        "Q",
                        (
                            "Some passage text that is long enough to survive "
                            "the rescue-nugget length filter. More follows."
                        ),
                    )
                ]
            },
        )
        cands = _tier_a_star_candidates(
            qid="0022",
            seed_pids=["a", "b"],
            rank_map={"a": 1, "b": 1},
            pr=pr,
        )
        assert len(cands) == 1
