"""Unit tests for :mod:`r2c2_agent.agent.ensemble.prior_state`.

The loader is used by the per-iter ensemble dispatch (node-independent
ensemble feature) to reconstruct ``touched_pids`` from prior iter
records on disk. It must handle missing files, malformed JSON, cross-
qid contamination, and rollout-index isolation correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from r2c2_agent.agent.ensemble.prior_state import (
    MissingPriorIterError,
    PriorIterResult,
    load_touched_set,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_empty_prior_paths_returns_empty_result() -> None:
    result = load_touched_set([], qid="0001", rollout_idx=0)
    assert isinstance(result, PriorIterResult)
    assert result.touched == frozenset()
    assert result.iters_found == ()
    assert result.missing_paths == ()


def test_single_file_with_matching_record(tmp_path: Path) -> None:
    p = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_jsonl(
        p,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A", "pid_B", "pid_C"],
            },
        ],
    )
    result = load_touched_set([p], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A", "pid_B", "pid_C"})
    assert result.iters_found == (0,)
    assert result.missing_paths == ()


def test_two_files_dedup_union(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    p1 = tmp_path / "iter_1" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A", "pid_B"],
            },
        ],
    )
    _write_jsonl(
        p1,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 1,
                "ensemble_touched_delta": ["pid_B", "pid_C"],
            },
        ],
    )
    result = load_touched_set([p0, p1], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A", "pid_B", "pid_C"})
    assert result.iters_found == (0, 1)
    assert result.missing_paths == ()


def test_missing_file_lenient_populates_missing_paths(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    p1 = tmp_path / "iter_1" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    # p1 never created
    result = load_touched_set([p0, p1], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A"})
    assert result.iters_found == (0,)
    assert result.missing_paths == (str(p1),)


def test_missing_file_strict_raises(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    p1 = tmp_path / "iter_1" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    with pytest.raises(MissingPriorIterError) as excinfo:
        load_touched_set([p0, p1], qid="0001", rollout_idx=0, strict=True)
    assert "0001" in str(excinfo.value)
    assert "rollout=0" in str(excinfo.value)


def test_wrong_qid_is_skipped(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0002",  # different qid
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_X"],
            },
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A", "pid_B"],
            },
        ],
    )
    result = load_touched_set([p0], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A", "pid_B"})


def test_wrong_rollout_is_skipped(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 1,  # different rollout
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_Z"],
            },
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    result = load_touched_set([p0], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A"})


def test_missing_rollout_field_defaults_to_zero(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                # no rollout_idx field
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    result = load_touched_set([p0], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A"})


def test_malformed_jsonl_lines_skipped(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    p0.parent.mkdir(parents=True)
    p0.write_text(
        "\n"
        "not a json line\n"
        '{"question_id": "0001", "rollout_idx": 0, '
        '"ensemble_idx": 0, "ensemble_touched_delta": ["pid_A"]}\n'
        "another garbage\n",
        encoding="utf-8",
    )
    result = load_touched_set([p0], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A"})
    assert result.iters_found == (0,)


def test_absent_ensemble_idx_falls_back_to_positional(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    p1 = tmp_path / "iter_1" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                # no ensemble_idx
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    _write_jsonl(
        p1,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                # no ensemble_idx
                "ensemble_touched_delta": ["pid_B"],
            },
        ],
    )
    result = load_touched_set([p0, p1], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A", "pid_B"})
    assert result.iters_found == (0, 1)


def test_empty_touched_delta_contributes_nothing(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": [],
            },
        ],
    )
    result = load_touched_set([p0], qid="0001", rollout_idx=0)
    assert result.touched == frozenset()
    assert result.iters_found == (0,)
    assert result.missing_paths == ()


def test_string_paths_accepted(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    result = load_touched_set([str(p0)], qid="0001", rollout_idx=0)
    assert result.touched == frozenset({"pid_A"})


def test_missing_paths_counted_before_strict_raise(tmp_path: Path) -> None:
    p0 = tmp_path / "iter_0" / "test_traces.jsonl"
    p1 = tmp_path / "iter_1" / "test_traces.jsonl"
    _write_jsonl(
        p0,
        [
            {
                "question_id": "0001",
                "rollout_idx": 0,
                "ensemble_idx": 0,
                "ensemble_touched_delta": ["pid_A"],
            },
        ],
    )
    # In strict mode the missing-paths detection must survive the first
    # match so that the raised exception lists exactly the defective paths.
    with pytest.raises(MissingPriorIterError) as excinfo:
        load_touched_set([p0, p1], qid="0001", rollout_idx=0, strict=True)
    assert str(p1) in str(excinfo.value)
    assert str(p0) not in str(excinfo.value)
