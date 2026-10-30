"""Unit and integration tests for the AC re-aggregation strategies.

Covers the shared helpers, the strategies (trace-redesign, best-rollout,
plurality, llm-judge, max-conf, conf-sum-cluster), and the byte-identity
regression pin for AC-1 (frozen SHA ``895fb48e…1310``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from r2c2_agent.orchestrator.ac_reaggregator import (
    ALLOWED_STRATEGIES,
    ATTEMPT_OK,
    ATTEMPT_REFUSAL_TEXT,
    BEST_ROLLOUT_NON_REFUSAL_BONUS,
    CANONICAL_REFUSAL_TEXT,
    PLURALITY_CLASS_WEIGHT_FORMAT,
    PLURALITY_CLASS_WEIGHT_HARDGATE,
    PLURALITY_CLASS_WEIGHT_OK,
    REFUSAL_FLOOR_CONF,
    STRATEGY_BEST_ROLLOUT,
    STRATEGY_LLM_JUDGE,
    STRATEGY_PLURALITY,
    STRATEGY_TRACE_REDESIGN,
    FrozenAC,
    FrozenPR,
    _apply_per_iter_to_chosen_iter,
    _class_weight,
    _cluster_attempts,
    _count_verified_supporting_nuggets,
    _dedupe_nuggets_by_pid_text,
    _normalise_answer,
    _NuggetCandidate,
    _pick_best_iter,
    _pool_attempts_across_iters,
    _Rollout,
    _score_attempt,
    _score_rollout,
    _strategy_best_rollout_qid,
    _strategy_llm_judge_qid,
    _strategy_plurality_qid,
    build_v2_ac_text,
    load_frozen_ac,
    load_frozen_pr,
    load_rollouts_for_run,
    run_validation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = REPO_ROOT / "submission"
# Real agent traces are not shipped in the repo (data/ is gitignored). Point
# R2C2_TRACES_DIR at a directory of trace JSONLs to exercise the integration
# tests; otherwise the repo-relative default does not exist on a fresh clone and
# the trace-dependent tests skip (see _have_real_traces / @pytest.mark.skipif).
TRACES_DIR = Path(
    os.environ.get(
        "R2C2_TRACES_DIR",
        str(REPO_ROOT / "data" / "runs" / "run2_ensemble_adaptive_postfix"),
    )
)
TEST_QUESTIONS = REPO_ROOT / "data" / "r2c2" / "test_questions.jsonl"
FROZEN_AC1_SHA = "895fb48e26d94f3316713088b7e279d541507fa88d86012db8dc34b339e61310"


def _have_real_traces() -> bool:
    return TRACES_DIR.exists() and any(
        (TRACES_DIR / "iter_0").glob("test_traces.jsonl")
    )


def _make_rollout(
    qid: str,
    *,
    iter_dir: int = 0,
    ensemble_idx: int = 0,
    final_answer: str = "Avatar",
    submit_attempts: list[dict] | None = None,
    nuggets: list[dict] | None = None,
    passages: list[dict] | None = None,
) -> _Rollout:
    """Build a _Rollout with a synthetic trace and pre-extracted nuggets."""
    from tests.conftest import _build_submit_attempt_messages

    raw_trace = {
        "question_id": qid,
        "iterations": 1,
        "ensemble_idx": ensemble_idx,
        "termination": "test",
        "final_answer": final_answer,
        "messages": _build_submit_attempt_messages(submit_attempts or []),
    }
    return _Rollout(
        qid=qid,
        ensemble_idx=ensemble_idx,
        iter_dir=iter_dir,
        termination="test",
        final_answer=final_answer,
        nuggets=list(nuggets or []),
        passages=list(passages or []),
        raw_trace=raw_trace,
    )


class TestNormaliseAnswer:
    def test_empty_returns_empty_sentinel(self):
        assert _normalise_answer("") == "__empty__"

    def test_whitespace_only_returns_empty_sentinel(self):
        assert _normalise_answer("   ") == "__empty__"

    def test_canonical_refusal_returns_refusal_sentinel(self):
        assert _normalise_answer(CANONICAL_REFUSAL_TEXT) == "__refusal__"

    def test_paraphrased_refusal_returns_refusal_sentinel(self):
        text = (
            "The corpus does not support an answer to this question; "
            "insufficient evidence to respond."
        )
        assert _normalise_answer(text) == "__refusal__"

    def test_lowercases_and_strips_punctuation(self):
        out = _normalise_answer("Avatar, the movie!")
        assert out == "avatar the movie"

    def test_no_truncation_on_long_answer(self):
        # Long enumerated answers must round-trip without truncation.
        long_text = (
            "Katharine Hepburn: 4 Academy Awards for Best Actress; "
            "Audrey Hepburn: 1 Academy Award for Best Actress"
        ) * 3
        out = _normalise_answer(long_text)
        assert len(out) > 120, "must not truncate at 120 chars"

    def test_unicode_punctuation_stripped(self):
        # U+2014 em-dash
        out = _normalise_answer("Avatar—the movie")
        assert "—" not in out
        # collapses to single space
        assert "  " not in out


class TestClassWeight:
    def _att(self, classification: str):
        from r2c2_agent.orchestrator.ac_reaggregator import _SubmitAttempt

        return _SubmitAttempt(
            answer_text="x",
            rationale="",
            citation_pids=[],
            committed=False,
            reason=None,
            error=None,
            confidence=0.5,
            n_eff=1,
            ts_idx=0,
            classification=classification,
        )

    def test_ok_returns_1_0(self):
        assert _class_weight(self._att(ATTEMPT_OK)) == PLURALITY_CLASS_WEIGHT_OK

    def test_hardgate_returns_0_4(self):
        from r2c2_agent.orchestrator.ac_reaggregator import ATTEMPT_REJ_HARDGATE

        assert (
            _class_weight(self._att(ATTEMPT_REJ_HARDGATE))
            == PLURALITY_CLASS_WEIGHT_HARDGATE
        )

    def test_format_returns_0_1(self):
        from r2c2_agent.orchestrator.ac_reaggregator import ATTEMPT_REJ_FORMAT

        assert (
            _class_weight(self._att(ATTEMPT_REJ_FORMAT))
            == PLURALITY_CLASS_WEIGHT_FORMAT
        )

    def test_refusal_returns_0(self):
        assert _class_weight(self._att(ATTEMPT_REFUSAL_TEXT)) == 0.0

    def test_unknown_returns_0(self):
        assert _class_weight(self._att("WEIRD")) == 0.0


class TestPoolAttemptsAcrossIters:
    def test_pools_across_iters(self):
        r0 = _make_rollout(
            "q1",
            iter_dir=0,
            submit_attempts=[
                {
                    "answer": "A",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                    "n_eff": 1,
                },
            ],
        )
        r1 = _make_rollout(
            "q1",
            iter_dir=1,
            submit_attempts=[
                {
                    "answer": "B",
                    "citation_pids": [],
                    "committed": False,
                    "confidence": 0.3,
                    "n_eff": 2,
                    "reason": "low_confidence_or_insufficient_evidence",
                },
            ],
        )
        pairs = _pool_attempts_across_iters([r0, r1])
        assert len(pairs) == 2
        assert pairs[0][1].answer_text == "A"
        assert pairs[1][1].answer_text == "B"

    def test_empty_rollouts_returns_empty(self):
        assert _pool_attempts_across_iters([]) == []

    def test_rollout_with_no_attempts_contributes_nothing(self):
        r0 = _make_rollout("q1", submit_attempts=[])
        assert _pool_attempts_across_iters([r0]) == []


class TestApplyPerIterToChosenIter:
    def test_returns_frozen_when_no_attempts(self):
        r = _make_rollout("q1", submit_attempts=[])
        out = _apply_per_iter_to_chosen_iter(
            r,
            "frozen-ans",
            50,
            {"pidA": 1},
            "q1",
        )
        assert out == ("frozen-ans", 50, None)

    def test_returns_frozen_when_only_refusal(self):
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": CANONICAL_REFUSAL_TEXT,
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        out = _apply_per_iter_to_chosen_iter(r, "frozen-ans", 50, {"pidA": 1}, "q1")
        assert out == ("frozen-ans", 50, None)

    def test_returns_picked_when_ok_present(self):
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA"],
                    "committed": True,
                    "confidence": 0.8,
                    "n_eff": 3,
                },
            ],
        )
        ans, conf, seeds = _apply_per_iter_to_chosen_iter(
            r,
            "frozen-ans",
            50,
            {"pidA": 1},
            "q1",
        )
        assert ans == "Avatar"
        # conf = max(0.8*100, 50) = 80
        assert conf == 80
        assert seeds == ["pidA"]

    def test_confidence_floor_max_with_frozen(self):
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.10,
                    "n_eff": 1,
                },
            ],
        )
        _ans, conf, _seeds = _apply_per_iter_to_chosen_iter(
            r,
            "frozen-ans",
            95,
            {},
            "q1",
        )
        # tool_conf=10 vs frozen=95 → 95 wins
        assert conf == 95


class TestCountVerifiedSupportingNuggets:
    def test_returns_zero_for_empty(self):
        r = _make_rollout("q1")
        assert _count_verified_supporting_nuggets(r) == 0

    def test_returns_len_nuggets(self):
        r = _make_rollout("q1", nuggets=[{"text": "a"}, {"text": "b"}, {"text": "c"}])
        assert _count_verified_supporting_nuggets(r) == 3


class TestDedupeNuggetsByPidText:
    def _cand(self, pid: str, text: str, rank: int = 1) -> _NuggetCandidate:
        return _NuggetCandidate(
            passage_id=pid,
            text=text,
            nli_score=0.0,
            earliest_ens=0,
            co_occurrence=1,
            pr_rank=rank,
        )

    def test_dedup_drops_repeat(self):
        cands = [
            self._cand("p1", "foo"),
            self._cand("p1", "foo"),
            self._cand("p2", "bar"),
        ]
        out = _dedupe_nuggets_by_pid_text(cands)
        assert len(out) == 2

    def test_dedup_keeps_first(self):
        c1 = self._cand("p1", "foo", rank=1)
        c2 = self._cand("p1", "foo", rank=2)
        out = _dedupe_nuggets_by_pid_text([c1, c2])
        assert out[0].pr_rank == 1


class TestScoreRollout:
    def test_non_refusal_bonus_only(self):
        r = _make_rollout("q1", final_answer="Avatar", submit_attempts=[])
        # 10 (non-refusal) + 0 (no OK) + 0 (no nuggets)
        assert _score_rollout(r) == BEST_ROLLOUT_NON_REFUSAL_BONUS

    def test_ok_attempt_contributes_conf_times_neff(self):
        r = _make_rollout(
            "q1",
            final_answer="Avatar",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                    "n_eff": 4,
                },
            ],
        )
        # 10 + 0.5*4 + 0 = 12
        assert _score_rollout(r) == pytest.approx(12.0)

    def test_ok_attempt_with_none_neff_contributes_zero_from_neff(self):
        r = _make_rollout(
            "q1",
            final_answer="Avatar",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                    "n_eff": None,
                },
            ],
        )
        # 10 + 0 (n_eff None → 0) + 0 = 10
        assert _score_rollout(r) == BEST_ROLLOUT_NON_REFUSAL_BONUS

    def test_refusal_final_answer_no_bonus(self):
        r = _make_rollout(
            "q1",
            final_answer=CANONICAL_REFUSAL_TEXT,
            submit_attempts=[],
        )
        assert _score_rollout(r) == 0.0

    def test_rej_hardgate_zero_contribution(self):
        r = _make_rollout(
            "q1",
            final_answer="Avatar",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": False,
                    "confidence": 0.5,
                    "n_eff": 4,
                    "reason": "low_confidence_or_insufficient_evidence",
                },
            ],
        )
        # 10 + 0 (REJ_HG ignored) + 0 = 10
        assert _score_rollout(r) == BEST_ROLLOUT_NON_REFUSAL_BONUS

    def test_nugget_count_one_per_nugget(self):
        r = _make_rollout(
            "q1",
            final_answer="Avatar",
            submit_attempts=[],
            nuggets=[{"text": "a"}, {"text": "b"}],
        )
        # 10 + 0 + 2 = 12
        assert _score_rollout(r) == 12.0


class TestPickBestIter:
    def test_picks_highest_score(self):
        r0 = _make_rollout("q1", iter_dir=0, final_answer="Avatar")
        r1 = _make_rollout(
            "q1",
            iter_dir=1,
            final_answer="Avatar",
            nuggets=[{"text": "n1"}, {"text": "n2"}],
        )
        out = _pick_best_iter([r0, r1])
        assert out is r1

    def test_tie_breaks_by_lowest_iter(self):
        r0 = _make_rollout("q1", iter_dir=0, final_answer="A")
        r2 = _make_rollout("q1", iter_dir=2, final_answer="A")
        out = _pick_best_iter([r0, r2])
        assert out is r0

    def test_returns_none_when_all_zero(self):
        r = _make_rollout("q1", final_answer=CANONICAL_REFUSAL_TEXT)
        assert _pick_best_iter([r]) is None

    def test_returns_none_when_empty(self):
        assert _pick_best_iter([]) is None

    def test_injectable_score_fn(self):
        r0 = _make_rollout("q1", iter_dir=0, final_answer="A")
        r1 = _make_rollout("q1", iter_dir=1, final_answer="A")
        # Reverse-favouring score_fn — picks the later iter.
        out = _pick_best_iter([r0, r1], score_fn=lambda r: float(r.iter_dir + 1))
        assert out is r1


class TestStrategyBestRolloutQid:
    def _setup(self, qid: str = "q1"):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                qid: [(1, "doc1", "Some long enough passage text for rescue use.")]
            },
        )
        ac = FrozenAC(
            run_name="BITEM-PG-2",
            answers_by_qid={qid: ("frozen-ans", 50)},
        )
        return pr, ac

    def test_falls_back_to_frozen_when_no_rollouts(self):
        pr, ac = self._setup()
        ans, conf, _cands, warns = _strategy_best_rollout_qid(
            "q1",
            pr,
            ac,
            [],
            {"pidA": 1},
            max_nuggets=10,
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "no_rollouts" in warns

    def test_falls_back_when_all_zero_score(self):
        pr, ac = self._setup()
        r = _make_rollout(
            "q1",
            final_answer=CANONICAL_REFUSAL_TEXT,
            submit_attempts=[],
        )
        ans, conf, _cands, warns = _strategy_best_rollout_qid(
            "q1",
            pr,
            ac,
            [r],
            {"pidA": 1},
            max_nuggets=10,
        )
        assert ans == "frozen-ans"
        assert "all_zero_score" in warns

    def test_chosen_iter_drives_output(self):
        pr, ac = self._setup()
        r = _make_rollout(
            "q1",
            final_answer="Avatar",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA"],
                    "committed": True,
                    "confidence": 0.9,
                    "n_eff": 2,
                },
            ],
        )
        ans, conf, _cands, warns = _strategy_best_rollout_qid(
            "q1",
            pr,
            ac,
            [r],
            {"pidA": 1},
            max_nuggets=10,
        )
        assert ans == "Avatar"
        assert conf == 90  # max(90, 50)
        assert any(w.startswith("best_rollout_iter_") for w in warns)


class TestScoreAttempt:
    def _att(self, classification, conf, n_eff, answer="x"):
        from r2c2_agent.orchestrator.ac_reaggregator import _SubmitAttempt

        return _SubmitAttempt(
            answer_text=answer,
            rationale="",
            citation_pids=[],
            committed=False,
            reason=None,
            error=None,
            confidence=conf,
            n_eff=n_eff,
            ts_idx=0,
            classification=classification,
        )

    def test_ok_with_n_eff(self):
        # 1.0 * 4 * 0.5 = 2.0
        assert _score_attempt(self._att(ATTEMPT_OK, 0.5, 4)) == pytest.approx(2.0)

    def test_ok_with_none_n_eff_uses_floor_1(self):
        # n_eff=None → 1 for OK only → 1.0*1*0.5 = 0.5
        assert _score_attempt(self._att(ATTEMPT_OK, 0.5, None)) == pytest.approx(0.5)

    def test_rej_hardgate_with_none_n_eff_returns_zero(self):
        from r2c2_agent.orchestrator.ac_reaggregator import ATTEMPT_REJ_HARDGATE

        # n_eff=None for non-OK → 0
        out = _score_attempt(self._att(ATTEMPT_REJ_HARDGATE, 0.5, None))
        assert out == 0.0

    def test_format_attempt_uses_0_1_weight(self):
        from r2c2_agent.orchestrator.ac_reaggregator import ATTEMPT_REJ_FORMAT

        # 0.1 * 2 * 0.5 = 0.1
        out = _score_attempt(self._att(ATTEMPT_REJ_FORMAT, 0.5, 2))
        assert out == pytest.approx(0.1)


class TestClusterAttempts:
    def _build_pair(self, answer: str, classification: str = ATTEMPT_OK):
        from r2c2_agent.orchestrator.ac_reaggregator import _SubmitAttempt

        r = _make_rollout("q1")
        a = _SubmitAttempt(
            answer_text=answer,
            rationale="",
            citation_pids=[],
            committed=True,
            reason=None,
            error=None,
            confidence=0.5,
            n_eff=1,
            ts_idx=0,
            classification=classification,
        )
        return (r, a)

    def test_groups_by_normalised_answer(self):
        pairs = [
            self._build_pair("Avatar"),
            self._build_pair("avatar."),  # normalises to same key
            self._build_pair("Tarkin"),
        ]
        out = _cluster_attempts(pairs)
        assert len(out) == 2

    def test_drops_refusal_cluster(self):
        pairs = [
            self._build_pair(
                CANONICAL_REFUSAL_TEXT, classification=ATTEMPT_REFUSAL_TEXT
            ),
            self._build_pair("Avatar"),
        ]
        out = _cluster_attempts(pairs)
        assert "__refusal__" not in out
        assert len(out) == 1


class TestStrategyPluralityQid:
    def _setup(self, qid: str = "q1"):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                qid: [(1, "doc1", "Some long enough passage text for rescue.")]
            },
        )
        return pr

    def test_no_attempts_falls_back(self):
        pr = self._setup()
        # Frozen answer is a refusal → fallback preserves it.
        ans, conf, _cands, warns = _strategy_plurality_qid(
            "q1",
            pr,
            [],
            {"pidA": 1},
            CANONICAL_REFUSAL_TEXT,
            54,
        )
        assert ans == CANONICAL_REFUSAL_TEXT
        assert conf == 54
        assert "no_attempts" in warns

    def test_no_attempts_non_refusal_frozen_emits_canonical_refusal(self):
        pr = self._setup()
        # Frozen is non-refusal → fallback emits canonical refusal at floor.
        ans, conf, _cands, warns = _strategy_plurality_qid(
            "q1",
            pr,
            [],
            {"pidA": 1},
            "frozen-ans",
            50,
        )
        assert ans == CANONICAL_REFUSAL_TEXT
        assert conf == REFUSAL_FLOOR_CONF
        assert "no_attempts" in warns

    def test_all_refusal_falls_back_to_refusal(self):
        pr = self._setup()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": CANONICAL_REFUSAL_TEXT,
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, conf, _cands, warns = _strategy_plurality_qid(
            "q1",
            pr,
            [r],
            {"pidA": 1},
            CANONICAL_REFUSAL_TEXT,
            54,
        )
        # Frozen was a refusal → preserve
        assert ans == CANONICAL_REFUSAL_TEXT
        assert conf == 54
        assert "all_refusal" in warns

    def test_unanimous_winner(self):
        pr = self._setup()
        r0 = _make_rollout(
            "q1",
            iter_dir=0,
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA"],
                    "committed": True,
                    "confidence": 0.8,
                    "n_eff": 2,
                },
            ],
        )
        r1 = _make_rollout(
            "q1",
            iter_dir=1,
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA"],
                    "committed": True,
                    "confidence": 0.9,
                    "n_eff": 3,
                },
            ],
        )
        ans, _conf, _cands, warns = _strategy_plurality_qid(
            "q1",
            pr,
            [r0, r1],
            {"pidA": 1},
            "frozen",
            50,
        )
        assert ans == "Avatar"
        assert any(w.startswith("cluster_winner:avatar") for w in warns)

    def test_below_threshold_refusal(self):
        pr = self._setup()
        # Score < 0.5 — REJ_FORMAT with low conf.
        r0 = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": False,
                    "confidence": 0.1,
                    "n_eff": 1,
                    "error": "invalid_rationale",
                },
            ],
        )
        # class_weight FORMAT=0.1 × n_eff=1 × conf=0.1 = 0.01 — well below 0.5.
        ans, _conf, _cands, warns = _strategy_plurality_qid(
            "q1",
            pr,
            [r0],
            {},
            "frozen",
            50,
        )
        # Frozen is non-refusal → fallback emits canonical refusal.
        assert any("below_threshold" in w for w in warns)


# Unit level: the judge decisions are built by hand.
class TestStrategyLlmJudgeQid:
    def _setup(self, qid: str = "q1"):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                qid: [(1, "doc1", "Some long enough passage text for rescue.")]
            },
        )
        return pr

    def test_no_decision_falls_back_to_frozen(self):
        pr = self._setup()
        ans, conf, _cands, warns = _strategy_llm_judge_qid(
            "q1",
            pr,
            [],
            {"pidA": 1},
            "frozen-ans",
            50,
            None,
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "judge_fallback_to_frozen" in warns

    def test_parse_failed_falls_back(self):
        from r2c2_agent.orchestrator.llm_judge import JudgeDecision

        pr = self._setup()
        dec = JudgeDecision(
            qid="q1",
            model="x",
            raw_choice="",
            raw_rationale="",
            raw_evidence_pids=[],
            final_choice="default",
            final_answer="frozen-ans",
            final_confidence=50,
            final_citation_pids=[],
            fallback_reason="parse_failed",
            parse_failed=True,
            ts="",
        )
        ans, conf, _cands, warns = _strategy_llm_judge_qid(
            "q1",
            pr,
            [],
            {"pidA": 1},
            "frozen-ans",
            50,
            dec,
        )
        assert ans == "frozen-ans"
        assert "judge_fallback_to_frozen" in warns

    def test_default_choice_uses_decision_answer(self):
        from r2c2_agent.orchestrator.llm_judge import JudgeDecision

        pr = self._setup()
        dec = JudgeDecision(
            qid="q1",
            model="x",
            raw_choice="default",
            raw_rationale="r",
            raw_evidence_pids=["pidA"],
            final_choice="default",
            final_answer="frozen-ans",
            final_confidence=50,
            final_citation_pids=["pidA"],
            fallback_reason=None,
            parse_failed=False,
            ts="",
        )
        ans, conf, _cands, warns = _strategy_llm_judge_qid(
            "q1",
            pr,
            [],
            {"pidA": 1},
            "frozen-ans",
            50,
            dec,
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "judge_choice_default" in warns

    def test_candidate_choice_emits_candidate(self):
        from r2c2_agent.orchestrator.llm_judge import JudgeDecision

        pr = self._setup()
        dec = JudgeDecision(
            qid="q1",
            model="x",
            raw_choice="candidate_2",
            raw_rationale="r",
            raw_evidence_pids=["pidA"],
            final_choice="candidate_2",
            final_answer="8",
            final_confidence=41,
            final_citation_pids=["pidA"],
            fallback_reason=None,
            parse_failed=False,
            ts="",
        )
        ans, conf, _cands, _warns = _strategy_llm_judge_qid(
            "q1",
            pr,
            [],
            {"pidA": 1},
            "frozen",
            30,
            dec,
        )
        assert ans == "8"
        assert conf == 41


class TestProcessRunDispatcher:
    def test_unknown_strategy_raises(self):
        pr = FrozenPR(
            run_name="BITEM-PG-2", rows_by_qid={"q1": [(1, "d", "text long enough")]}
        )
        ac = FrozenAC(run_name="BITEM-PG-2", answers_by_qid={"q1": ("ans", 50)})
        with pytest.raises(ValueError):
            build_v2_ac_text(pr, ac, {"q1": []}, strategy="bogus-strategy")

    def test_pg1_under_best_rollout_falls_back_to_trace_redesign(self):
        # On PG-1 (apply_trace_redesign=True), best-rollout is overridden to
        # trace-redesign. Verify by checking that the output is identical.
        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={"q1": [(1, "d", "text long enough for rescue")]},
        )
        ac = FrozenAC(run_name="BITEM-PG-1", answers_by_qid={"q1": ("frozen", 50)})
        rollouts = {"q1": []}
        text_a, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=STRATEGY_TRACE_REDESIGN,
        )
        text_b, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=STRATEGY_BEST_ROLLOUT,
        )
        assert text_a == text_b


class TestLlmJudgeGuardrail:
    """Hard guardrail: ``build_v2_ac_text(strategy='llm-judge')`` on PG-2
    must SystemExit unless every PR qid has a real, non-parse_failed
    judge decision. Otherwise a degenerate AC-4 that duplicates frozen
    PG-2 verbatim could be produced without any error.

    PG-1 (``apply_trace_redesign=True``) is exempt — it always reduces
    to trace-redesign on K=1.
    """

    def _setup_pr_ac(self, qids: list[str]) -> tuple[FrozenPR, FrozenAC]:
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                q: [(1, "doc1", "Some long enough passage text for rescue.")]
                for q in qids
            },
        )
        ac = FrozenAC(
            run_name="BITEM-PG-2",
            answers_by_qid={q: ("frozen-ans", 50) for q in qids},
        )
        return pr, ac

    def _make_decision(
        self,
        qid: str,
        *,
        parse_failed: bool = False,
        fallback_reason: str | None = None,
    ):
        from r2c2_agent.orchestrator.llm_judge import JudgeDecision

        return JudgeDecision(
            qid=qid,
            model="test-model",
            raw_choice="default",
            raw_rationale="r",
            raw_evidence_pids=[],
            final_choice="default",
            final_answer="frozen-ans",
            final_confidence=50,
            final_citation_pids=[],
            fallback_reason=fallback_reason,
            parse_failed=parse_failed,
            ts="",
        )

    def test_empty_decisions_hard_fails_on_pg2(self):
        pr, ac = self._setup_pr_ac(["q1", "q2", "q3"])
        with pytest.raises(SystemExit, match="--strategy llm-judge requires"):
            build_v2_ac_text(
                pr,
                ac,
                {q: [] for q in pr.qids()},
                apply_trace_redesign=False,
                strategy=STRATEGY_LLM_JUDGE,
                judge_decisions={},
            )

    def test_none_decisions_hard_fails_on_pg2(self):
        pr, ac = self._setup_pr_ac(["q1", "q2"])
        with pytest.raises(SystemExit, match="--strategy llm-judge requires"):
            build_v2_ac_text(
                pr,
                ac,
                {q: [] for q in pr.qids()},
                apply_trace_redesign=False,
                strategy=STRATEGY_LLM_JUDGE,
                judge_decisions=None,
            )

    def test_missing_qid_hard_fails_on_pg2(self):
        pr, ac = self._setup_pr_ac(["q1", "q2", "q3"])
        decisions = {
            "q1": self._make_decision("q1"),
            "q2": self._make_decision("q2"),
            # q3 missing
        }
        with pytest.raises(SystemExit, match="LLM-judge strategy requires"):
            build_v2_ac_text(
                pr,
                ac,
                {q: [] for q in pr.qids()},
                apply_trace_redesign=False,
                strategy=STRATEGY_LLM_JUDGE,
                judge_decisions=decisions,
            )

    def test_parse_failed_qid_hard_fails_on_pg2(self):
        pr, ac = self._setup_pr_ac(["q1", "q2"])
        decisions = {
            "q1": self._make_decision("q1"),
            "q2": self._make_decision("q2", parse_failed=True),
        }
        with pytest.raises(SystemExit, match="LLM-judge strategy requires"):
            build_v2_ac_text(
                pr,
                ac,
                {q: [] for q in pr.qids()},
                apply_trace_redesign=False,
                strategy=STRATEGY_LLM_JUDGE,
                judge_decisions=decisions,
            )

    def test_endpoint_pending_fallback_hard_fails_on_pg2(self):
        # Deferred / placeholder decisions (e.g. fallback_reason='endpoint_pending'
        # or 'deferred') indicate no real LLM call happened. These must be
        # rejected even though parse_failed=False.
        pr, ac = self._setup_pr_ac(["q1", "q2"])
        decisions = {
            "q1": self._make_decision("q1"),
            "q2": self._make_decision("q2", fallback_reason="endpoint_pending"),
        }
        with pytest.raises(SystemExit, match="LLM-judge strategy requires"):
            build_v2_ac_text(
                pr,
                ac,
                {q: [] for q in pr.qids()},
                apply_trace_redesign=False,
                strategy=STRATEGY_LLM_JUDGE,
                judge_decisions=decisions,
            )

    def test_legitimate_fallback_reason_passes_on_pg2(self):
        # ``invalid_choice:foo`` and ``candidate_2_unavailable`` are real LLM
        # decisions that the finaliser re-mapped to default — these are still
        # real calls and must not trip the guardrail.
        pr, ac = self._setup_pr_ac(["q1", "q2"])
        decisions = {
            "q1": self._make_decision("q1", fallback_reason="invalid_choice:foo"),
            "q2": self._make_decision("q2", fallback_reason="candidate_2_unavailable"),
        }
        text, _, _ = build_v2_ac_text(
            pr,
            ac,
            {q: [] for q in pr.qids()},
            apply_trace_redesign=False,
            strategy=STRATEGY_LLM_JUDGE,
            judge_decisions=decisions,
        )
        assert "<q1>" in text and "<q2>" in text

    def test_complete_decisions_passes_on_pg2(self):
        pr, ac = self._setup_pr_ac(["q1", "q2"])
        decisions = {
            "q1": self._make_decision("q1"),
            "q2": self._make_decision("q2"),
        }
        # Every qid has a real decision, so this does not raise.
        text, _, _ = build_v2_ac_text(
            pr,
            ac,
            {q: [] for q in pr.qids()},
            apply_trace_redesign=False,
            strategy=STRATEGY_LLM_JUDGE,
            judge_decisions=decisions,
        )
        assert "<q1>" in text and "<q2>" in text

    def test_allow_incomplete_judge_escape_hatch(self):
        # Development-only override.
        pr, ac = self._setup_pr_ac(["q1", "q2"])
        # No decisions — would normally hard-fail.
        text, _, _ = build_v2_ac_text(
            pr,
            ac,
            {q: [] for q in pr.qids()},
            apply_trace_redesign=False,
            strategy=STRATEGY_LLM_JUDGE,
            judge_decisions={},
            allow_incomplete_judge=True,
        )
        assert "<q1>" in text and "<q2>" in text

    def test_pg1_exempt_from_guardrail(self):
        # PG-1 always reduces to trace-redesign — guardrail doesn't apply
        # even with empty decisions.
        pr, ac = self._setup_pr_ac(["q1"])
        text, _, _ = build_v2_ac_text(
            pr,
            ac,
            {q: [] for q in pr.qids()},
            apply_trace_redesign=True,
            strategy=STRATEGY_LLM_JUDGE,
            judge_decisions={},
        )
        assert "<q1>" in text


@pytest.mark.skipif(
    not _have_real_traces() or not TEST_QUESTIONS.exists(),
    reason="real iter_*/test_traces.jsonl or test_questions.jsonl not available",
)
class TestByteIdentityAC1:
    """Lock-in: the default --strategy trace-redesign MUST produce the
    shipped AC-1 SHA byte-for-byte. Repeated under all 4 strategies for
    PG-1 (which falls back to trace-redesign on K=1)."""

    def _build_pg1(self, strategy: str, tmp_path: Path) -> Path:
        pr_path = SUBMISSION_DIR / "BITEM-PG-1"
        ac_path = SUBMISSION_DIR / "BITEM-AC-1"
        pr = load_frozen_pr(pr_path, "BITEM-PG-1")
        ac = load_frozen_ac(ac_path, "BITEM-PG-1")
        rollouts = load_rollouts_for_run(TRACES_DIR, first_iter_only=True)
        text, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=strategy,
            judge_decisions=None,
        )
        out_path = tmp_path / "BITEM-AC-1.v2"
        out_path.write_text(text)
        return out_path

    def _sha(self, p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    def test_pg1_trace_redesign_matches_shipped(self, tmp_path):
        out = self._build_pg1(STRATEGY_TRACE_REDESIGN, tmp_path)
        assert self._sha(out) == FROZEN_AC1_SHA

    def test_pg1_best_rollout_falls_back_to_trace_redesign(self, tmp_path):
        out = self._build_pg1(STRATEGY_BEST_ROLLOUT, tmp_path)
        assert self._sha(out) == FROZEN_AC1_SHA

    def test_pg1_plurality_falls_back_to_trace_redesign(self, tmp_path):
        out = self._build_pg1(STRATEGY_PLURALITY, tmp_path)
        assert self._sha(out) == FROZEN_AC1_SHA

    def test_pg1_llm_judge_falls_back_to_trace_redesign(self, tmp_path):
        out = self._build_pg1(STRATEGY_LLM_JUDGE, tmp_path)
        assert self._sha(out) == FROZEN_AC1_SHA


@pytest.mark.skipif(
    not _have_real_traces() or not TEST_QUESTIONS.exists(),
    reason="real iter_*/test_traces.jsonl or test_questions.jsonl not available",
)
class TestIntegrationFiveQid:
    """End-to-end smoke test on real trace data — all 4 strategies must
    survive the 8 spec checks + 0 subset violations on PG-2."""

    def _run_strategy(self, strategy: str, tmp_path: Path):
        pr_path = SUBMISSION_DIR / "BITEM-PG-2"
        ac_path = SUBMISSION_DIR / "BITEM-AC-2"
        pr = load_frozen_pr(pr_path, "BITEM-PG-2")
        ac = load_frozen_ac(ac_path, "BITEM-PG-2")
        rollouts = load_rollouts_for_run(TRACES_DIR, first_iter_only=False)
        text, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=False,
            strategy=strategy,
        )
        out_path = tmp_path / "BITEM-AC-X"
        out_path.write_text(text)
        source_qids = set()
        with open(TEST_QUESTIONS) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid:
                    source_qids.add(qid)
        outcome = run_validation(pr_path, out_path, pr, "BITEM-PG-2", source_qids)
        return text, outcome

    def test_best_rollout_passes_all_spec_and_subset(self, tmp_path):
        _text, outcome = self._run_strategy(STRATEGY_BEST_ROLLOUT, tmp_path)
        assert outcome.spec_failed == 0
        assert outcome.subset_violations == []

    def test_plurality_passes_all_spec_and_subset(self, tmp_path):
        _text, outcome = self._run_strategy(STRATEGY_PLURALITY, tmp_path)
        assert outcome.spec_failed == 0
        assert outcome.subset_violations == []

    def test_plurality_qid_0020_pins_to_8(self, tmp_path):
        text, _ = self._run_strategy(STRATEGY_PLURALITY, tmp_path)
        idx = text.find("<0020>\n")
        assert idx >= 0
        # First line after the open tag is the Answer;Confidence line.
        block = text[idx:].split("\n", 3)
        ans_line = block[1]
        assert ans_line.startswith("8;"), f"qid 0020 regressed: {ans_line!r}"

    def test_best_rollout_qid_0020_known_regression(self, tmp_path):
        # Known trade-off of best-rollout: AC-2 answers qid 0020 with "18: ...".
        text, _ = self._run_strategy(STRATEGY_BEST_ROLLOUT, tmp_path)
        idx = text.find("<0020>\n")
        assert idx >= 0
        block = text[idx:].split("\n", 3)
        ans_line = block[1]
        assert ans_line.startswith("18"), (
            f"AC-2 0020 should be '18'-prefix: {ans_line!r}"
        )


def test_allowed_strategies_contains_all_six():
    from r2c2_agent.orchestrator.ac_reaggregator import (
        STRATEGY_CONF_SUM_CLUSTER,
        STRATEGY_MAX_CONF,
    )

    assert ALLOWED_STRATEGIES == frozenset(
        {
            STRATEGY_TRACE_REDESIGN,
            STRATEGY_BEST_ROLLOUT,
            STRATEGY_PLURALITY,
            STRATEGY_LLM_JUDGE,
            STRATEGY_MAX_CONF,
            STRATEGY_CONF_SUM_CLUSTER,
        }
    )


# max-conf: order-statistic variance reduction.
class TestStrategyMaxConfQid:
    """Synthetic-fixture coverage for the AC-2 ``max-conf`` strategy."""

    def _setup_pr(self, qid: str = "q1") -> FrozenPR:
        return FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                qid: [(1, "doc1", "Some long enough passage text for rescue use.")]
            },
        )

    def test_single_non_refusal_attempt_emits_round_tool_conf(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            iter_dir=0,
            final_answer="Avatar",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA"],
                    "committed": True,
                    "confidence": 0.73,
                    "n_eff": 2,
                },
            ],
        )
        ans, conf, _cands, warns = _strategy_run1_max_conf_qid(
            "q1",
            [r],
            {"pidA": 1},
            pr,
            "frozen-ans",
            50,
            max_nuggets=10,
        )
        assert ans == "Avatar"
        # conf = round(0.73 * 100) = 73, with no frozen floor.
        assert conf == 73
        assert any(w.startswith("max_conf_winner_conf") for w in warns)

    def test_no_frozen_floor_low_conf_emits_low_conf(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Foo",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.10,
                    "n_eff": 1,
                },
            ],
        )
        _ans, conf, _cands, _warns = _strategy_run1_max_conf_qid(
            "q1", [r], {}, pr, "frozen-ans", 95, max_nuggets=10
        )
        # No frozen floor: emit round(0.10 * 100) = 10.
        assert conf == 10

    def test_argmax_picks_highest_tool_conf_across_iters(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r0 = _make_rollout(
            "q1",
            iter_dir=0,
            submit_attempts=[
                {
                    "answer": "low-conf",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.25,
                    "n_eff": 1,
                },
            ],
        )
        r1 = _make_rollout(
            "q1",
            iter_dir=1,
            submit_attempts=[
                {
                    "answer": "high-conf",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.91,
                    "n_eff": 1,
                },
            ],
        )
        ans, conf, _cands, _warns = _strategy_run1_max_conf_qid(
            "q1", [r0, r1], {}, pr, "frozen", 50
        )
        assert ans == "high-conf"
        assert conf == 91

    def test_all_refusal_falls_back_to_frozen(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": CANONICAL_REFUSAL_TEXT,
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, conf, _cands, warns = _strategy_run1_max_conf_qid(
            "q1", [r], {}, pr, "frozen-ans", 50
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "max_conf_no_attempts" in warns

    def test_no_rollouts_falls_back_to_frozen(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        ans, conf, _cands, warns = _strategy_run1_max_conf_qid(
            "q1", [], {}, pr, "frozen-ans", 50
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "max_conf_no_attempts" in warns

    def test_semicolon_in_answer_replaced_with_comma(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "alpha; beta; gamma",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, _conf, _cands, _warns = _strategy_run1_max_conf_qid(
            "q1", [r], {}, pr, "frozen", 50
        )
        # `;` must be replaced with `,` so the AC line delimiter is unambiguous.
        assert ";" not in ans
        assert ans == "alpha, beta, gamma"

    def test_citations_restricted_to_pr_roster(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA", "pidNotInRoster"],
                    "committed": True,
                    "confidence": 0.8,
                },
            ],
        )
        # Only pidA is in the rank_map, so only pidA can appear in the seed
        # citation pids passed downstream. The full pipeline enforces this
        # with its subset-invariant check; this test checks the strategy
        # function alone.
        ans, _conf, _cands, _warns = _strategy_run1_max_conf_qid(
            "q1", [r], {"pidA": 1}, pr, "frozen", 50
        )
        # Every candidate pid must be in rank_map or be a synthetic pid
        # prefixed `__pr_rank_`.
        assert ans == "Avatar"
        for cand in _cands:
            assert cand.passage_id in {"pidA"} or cand.passage_id.startswith(
                "__pr_rank_"
            )

    def test_tie_break_on_message_order_when_equal_conf(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run1_max_conf_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "first",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
                {
                    "answer": "second",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, _conf, _cands, _warns = _strategy_run1_max_conf_qid(
            "q1", [r], {}, pr, "frozen", 50
        )
        # Equal-confidence attempts → earliest (ts_idx=0) wins.
        assert ans == "first"


# conf-sum-cluster: agreement between rollouts pooled as evidence.
class TestStrategyConfSumClusterQid:
    """Synthetic-fixture coverage for the AC-3 ``conf-sum-cluster`` strategy."""

    def _setup_pr(self, qid: str = "q1") -> FrozenPR:
        return FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                qid: [(1, "doc1", "Some long enough passage text for rescue use.")]
            },
        )

    def test_single_cluster_capped_at_100(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        # 4 rollouts all converging on "Avatar" with conf=0.5 each.
        # Σ = 4 × 50 = 200, capped to 100.
        rollouts = [
            _make_rollout(
                "q1",
                iter_dir=i,
                submit_attempts=[
                    {
                        "answer": "Avatar",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.5,
                    },
                ],
            )
            for i in range(4)
        ]
        ans, conf, _cands, warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", rollouts, {}, pr, "frozen", 50
        )
        assert ans == "Avatar"
        # Cap at 100.
        assert conf == 100
        assert any(w.startswith("conf_sum_cluster:") for w in warns)
        assert any("n_members:4" == w for w in warns)

    def test_single_cluster_below_cap_emits_round_sum(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        rollouts = [
            _make_rollout(
                "q1",
                iter_dir=0,
                submit_attempts=[
                    {
                        "answer": "Avatar",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.35,
                    },
                ],
            ),
            _make_rollout(
                "q1",
                iter_dir=1,
                submit_attempts=[
                    {
                        "answer": "Avatar",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.42,
                    },
                ],
            ),
        ]
        ans, conf, _cands, _warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", rollouts, {}, pr, "frozen", 50
        )
        assert ans == "Avatar"
        # Σ = round(35 + 42) = 77.  Below 100 cap.
        assert conf == 77

    def test_multi_cluster_highest_score_wins(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        # Cluster A: two rollouts at conf=0.6 each → score = 120 → capped 100.
        # Cluster B: one rollout at conf=0.95 → score = 95.
        rollouts = [
            _make_rollout(
                "q1",
                iter_dir=0,
                submit_attempts=[
                    {
                        "answer": "alpha",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.6,
                    },
                ],
            ),
            _make_rollout(
                "q1",
                iter_dir=1,
                submit_attempts=[
                    {
                        "answer": "alpha",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.6,
                    },
                ],
            ),
            _make_rollout(
                "q1",
                iter_dir=2,
                submit_attempts=[
                    {
                        "answer": "beta",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.95,
                    },
                ],
            ),
        ]
        ans, conf, _cands, warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", rollouts, {}, pr, "frozen", 50
        )
        # alpha-cluster sum = 120, beta-cluster sum = 95 → alpha wins.
        assert ans == "alpha"
        assert conf == 100
        assert any("n_members:2" == w for w in warns)

    def test_answer_from_max_tool_conf_member_of_winning_cluster(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        # Two answers normalise to the same key but the surface form is
        # the higher-conf one's verbatim text.
        rollouts = [
            _make_rollout(
                "q1",
                iter_dir=0,
                submit_attempts=[
                    {
                        "answer": "avatar the movie",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.3,
                    },
                ],
            ),
            _make_rollout(
                "q1",
                iter_dir=1,
                submit_attempts=[
                    {
                        "answer": "Avatar, the movie!",
                        "citation_pids": [],
                        "committed": True,
                        "confidence": 0.9,
                    },
                ],
            ),
        ]
        ans, _conf, _cands, _warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", rollouts, {}, pr, "frozen", 50
        )
        # Higher tool_conf within the cluster supplies the surface form.
        assert ans == "Avatar, the movie!"

    def test_all_refusal_falls_back_to_frozen(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": CANONICAL_REFUSAL_TEXT,
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, conf, _cands, warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", [r], {}, pr, "frozen-ans", 50
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "conf_sum_no_attempts" in warns

    def test_no_rollouts_falls_back_to_frozen(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        ans, conf, _cands, warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", [], {}, pr, "frozen-ans", 50
        )
        assert ans == "frozen-ans"
        assert conf == 50
        assert "conf_sum_no_attempts" in warns

    def test_semicolon_in_answer_replaced_with_comma(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "x; y; z",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, _conf, _cands, _warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", [r], {}, pr, "frozen", 50
        )
        assert ";" not in ans
        assert ans == "x, y, z"

    def test_citations_restricted_to_pr_roster(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        # Citations include one in-roster and one out-of-roster pid; the
        # downstream nugget pipeline must only emit candidates whose
        # passage_id is in rank_map (or a synthetic `__pr_rank_` pid).
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA", "pidOutOfRoster"],
                    "committed": True,
                    "confidence": 0.5,
                },
            ],
        )
        ans, _conf, cands, _warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", [r], {"pidA": 1}, pr, "frozen", 50
        )
        assert ans == "Avatar"
        for cand in cands:
            assert cand.passage_id in {"pidA"} or cand.passage_id.startswith(
                "__pr_rank_"
            )

    def test_no_frozen_floor_on_low_score_cluster(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            _strategy_run3_conf_sum_cluster_qid,
        )

        pr = self._setup_pr()
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": [],
                    "committed": True,
                    "confidence": 0.05,
                },
            ],
        )
        _ans, conf, _cands, _warns = _strategy_run3_conf_sum_cluster_qid(
            "q1", [r], {}, pr, "frozen", 95
        )
        # Σ = 5, no frozen-floor → conf == 5.
        assert conf == 5


class TestPg1FallbackForNewStrategies:
    """PG-1 (K=1, ``apply_trace_redesign=True``) must emit the same text
    under any strategy: ``max-conf`` and ``conf-sum-cluster`` reduce to
    trace-redesign as ``best-rollout``, ``plurality`` and ``llm-judge`` do.
    Together with TestByteIdentityAC1 (which uses real traces), this pins
    the AC-1 frozen SHA against accidental regressions.
    """

    def test_pg1_max_conf_falls_back_to_trace_redesign(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            STRATEGY_MAX_CONF,
        )

        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={"q1": [(1, "d", "text long enough for rescue.")]},
        )
        ac = FrozenAC(run_name="BITEM-PG-1", answers_by_qid={"q1": ("frozen", 50)})
        rollouts = {"q1": []}
        text_a, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=STRATEGY_TRACE_REDESIGN,
        )
        text_b, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=STRATEGY_MAX_CONF,
        )
        assert text_a == text_b

    def test_pg1_conf_sum_cluster_falls_back_to_trace_redesign(self):
        from r2c2_agent.orchestrator.ac_reaggregator import (
            STRATEGY_CONF_SUM_CLUSTER,
        )

        pr = FrozenPR(
            run_name="BITEM-PG-1",
            rows_by_qid={"q1": [(1, "d", "text long enough for rescue.")]},
        )
        ac = FrozenAC(run_name="BITEM-PG-1", answers_by_qid={"q1": ("frozen", 50)})
        rollouts = {"q1": []}
        text_a, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=STRATEGY_TRACE_REDESIGN,
        )
        text_b, _, _ = build_v2_ac_text(
            pr,
            ac,
            rollouts,
            apply_trace_redesign=True,
            strategy=STRATEGY_CONF_SUM_CLUSTER,
        )
        assert text_a == text_b
