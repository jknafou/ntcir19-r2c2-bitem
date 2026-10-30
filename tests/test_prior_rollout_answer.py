"""Unit tests for ``load_prior_rollout_answer``.

Scope (the function is dependency-free, like ``load_touched_set``):

  * picks the highest-confidence committed answer across iters
  * filters out refusals (termination_reason not in committed set)
  * filters out answers below ``min_confidence``
  * tolerates missing iter files (non-strict)
  * normalizes legacy float confidences (0..1) to int (0..100)
  * tie-break: same confidence → earliest iter (more independent)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from r2c2_agent.agent.ensemble.prior_state import (
    MissingPriorIterError,
    load_prior_rollout_answer,
)


def _write_trace(
    path: Path,
    qid: str,
    rollout_idx: int,
    ensemble_idx: int,
    answer: str,
    confidence: int | float,
    termination: str,
    *,
    conf_field: str = "raw_confidence",
) -> None:
    rec = {
        "question_id": qid,
        "rollout_idx": rollout_idx,
        "ensemble_idx": ensemble_idx,
        "answer": answer,
        conf_field: confidence,
        "termination_reason": termination,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(json.dumps(rec) + "\n")


# Happy path: find best candidate


class TestBasic:
    def test_picks_highest_confidence(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        p1 = tmp_path / "iter_1" / "test_traces.jsonl"
        p2 = tmp_path / "iter_2" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "low", 72, "answer_submitted")
        _write_trace(p1, "qa", 0, 1, "WINNER", 95, "answer_submitted")
        _write_trace(p2, "qa", 0, 2, "mid", 80, "answer_submitted")
        result = load_prior_rollout_answer([p0, p1, p2], "qa", 0)
        assert result.best_answer == "WINNER"
        assert result.best_confidence == 95
        assert result.source_iter == 1
        assert len(result.all_prior) == 3

    def test_none_below_threshold(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "unsure", 40, "answer_submitted")
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer is None
        assert result.best_confidence == 0
        assert result.source_iter is None

    def test_respects_custom_min_confidence(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "mid_conf", 55, "answer_submitted")
        # Default threshold 70: filtered out.
        result_default = load_prior_rollout_answer([p0], "qa", 0)
        assert result_default.best_answer is None
        # Lower threshold 50: accepted.
        result_low = load_prior_rollout_answer([p0], "qa", 0, min_confidence=50)
        assert result_low.best_answer == "mid_conf"


class TestTerminationFilter:
    def test_refusal_excluded(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        # High conf but termination is a refusal, so it must not win.
        _write_trace(p0, "qa", 0, 0, "refusal prose", 95, "low_confidence_refusal")
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer is None

    def test_force_deliver_accepted(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "force_guess", 75, "force_deliver")
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer == "force_guess"
        assert result.best_confidence == 75

    def test_unknown_termination_excluded(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "weird", 90, "unknown_custom_tag")
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer is None

    def test_empty_answer_excluded(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "", 95, "answer_submitted")
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer is None


class TestConfidenceNormalization:
    def test_float_0_to_1_normalized(self, tmp_path: Path) -> None:
        """Legacy traces may carry ``confidence: 0.95`` (float in 0..1)."""
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(
            p0,
            "qa",
            0,
            0,
            "legacy",
            0.95,
            "answer_submitted",
            conf_field="confidence",
        )
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer == "legacy"
        assert result.best_confidence == 95

    def test_int_already_normalized(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "modern", 85, "answer_submitted")
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_confidence == 85


# Tie-break + missing paths


class TestEdgeCases:
    def test_tie_break_earliest_iter(self, tmp_path: Path) -> None:
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        p1 = tmp_path / "iter_1" / "test_traces.jsonl"
        _write_trace(p0, "qa", 0, 0, "early", 85, "answer_submitted")
        _write_trace(p1, "qa", 0, 1, "late", 85, "answer_submitted")
        result = load_prior_rollout_answer([p0, p1], "qa", 0)
        assert result.best_answer == "early"
        assert result.source_iter == 0

    def test_missing_path_non_strict(self, tmp_path: Path) -> None:
        p_missing = tmp_path / "iter_0" / "gone.jsonl"
        p1 = tmp_path / "iter_1" / "test_traces.jsonl"
        _write_trace(p1, "qa", 0, 1, "survivor", 85, "answer_submitted")
        result = load_prior_rollout_answer([p_missing, p1], "qa", 0)
        assert result.best_answer == "survivor"
        assert str(p_missing) in result.missing_paths

    def test_missing_path_strict_raises(self, tmp_path: Path) -> None:
        p_missing = tmp_path / "iter_0" / "gone.jsonl"
        with pytest.raises(MissingPriorIterError):
            load_prior_rollout_answer([p_missing], "qa", 0, strict=True)

    def test_rollout_filter(self, tmp_path: Path) -> None:
        """Records for a different rollout_idx must not leak."""
        p0 = tmp_path / "iter_0" / "test_traces.jsonl"
        p0.parent.mkdir(parents=True, exist_ok=True)
        with p0.open("w") as fh:
            fh.write(
                json.dumps(
                    {
                        "question_id": "qa",
                        "rollout_idx": 1,  # different rollout
                        "ensemble_idx": 0,
                        "answer": "wrong_rollout",
                        "raw_confidence": 99,
                        "termination_reason": "answer_submitted",
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "question_id": "qa",
                        "rollout_idx": 0,  # target
                        "ensemble_idx": 0,
                        "answer": "right_rollout",
                        "raw_confidence": 80,
                        "termination_reason": "answer_submitted",
                    }
                )
                + "\n"
            )
        result = load_prior_rollout_answer([p0], "qa", 0)
        assert result.best_answer == "right_rollout"
        assert result.best_confidence == 80
