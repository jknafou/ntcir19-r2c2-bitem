"""Tests for ``r2c2_agent.orchestrator.orchestrator._wait_iter_ready``.

The helper polls a BeeGFS path until a (qid, rollout) record is visible
and the file size has stabilised across a tunable window. These tests
exercise the core correctness guarantees without booting the
orchestrator or the workers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def orch_module() -> Iterator:
    """Import orchestrator.py via its public package path."""
    from r2c2_agent.orchestrator import orchestrator as mod

    yield mod


def _write_record(path: Path, qid: str, rollout_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "question_id": qid,
        "rollout_idx": rollout_idx,
        "ensemble_idx": 0,
        "ensemble_touched_delta": [],
        "ensemble_excluded_pids": [],
        "termination": "answer_submitted",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def test_missing_file_times_out(orch_module, tmp_path: Path) -> None:
    target = tmp_path / "iter_0" / "test_traces.jsonl"
    result = asyncio.run(
        orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=1.0,
            stability_window_s=0.05,
        )
    )
    assert result is False


def test_existing_record_is_detected(orch_module, tmp_path: Path) -> None:
    target = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_record(target, "0001", 0)
    result = asyncio.run(
        orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=2.0,
            stability_window_s=0.1,
        )
    )
    assert result is True


def test_wrong_qid_returns_false(orch_module, tmp_path: Path) -> None:
    target = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_record(target, "0002", 0)
    result = asyncio.run(
        orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=1.0,
            stability_window_s=0.05,
        )
    )
    assert result is False


def test_wrong_rollout_returns_false(orch_module, tmp_path: Path) -> None:
    target = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_record(target, "0001", 1)
    result = asyncio.run(
        orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=1.0,
            stability_window_s=0.05,
        )
    )
    assert result is False


def test_record_written_during_poll(orch_module, tmp_path: Path) -> None:
    """Simulate a BeeGFS flush mid-poll — file appears after the call starts."""
    target = tmp_path / "iter_0" / "test_traces.jsonl"

    async def scenario() -> bool:
        async def delayed_write() -> None:
            await asyncio.sleep(0.3)
            _write_record(target, "0001", 0)

        write_task = asyncio.create_task(delayed_write())
        ok = await orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=3.0,
            stability_window_s=0.1,
        )
        await write_task
        return ok

    assert asyncio.run(scenario()) is True


def test_growing_file_waits_for_stability(orch_module, tmp_path: Path) -> None:
    """File keeps growing after the target record — wait must see stable size."""
    target = tmp_path / "iter_0" / "test_traces.jsonl"
    _write_record(target, "0001", 0)

    async def scenario() -> bool:
        async def keep_appending() -> None:
            for _ in range(3):
                await asyncio.sleep(0.05)
                _write_record(target, "0002", 0)

        growth_task = asyncio.create_task(keep_appending())
        ok = await orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=2.0,
            stability_window_s=0.3,
        )
        await growth_task
        return ok

    assert asyncio.run(scenario()) is True


def test_malformed_jsonl_lines_tolerated(orch_module, tmp_path: Path) -> None:
    target = tmp_path / "iter_0" / "test_traces.jsonl"
    target.parent.mkdir(parents=True)
    good_line = json.dumps(
        {
            "question_id": "0001",
            "rollout_idx": 0,
            "ensemble_idx": 0,
            "ensemble_touched_delta": [],
        }
    )
    target.write_text(
        "\nnot json\n" + good_line + "\n",
        encoding="utf-8",
    )
    result = asyncio.run(
        orch_module._wait_iter_ready(
            target,
            qid="0001",
            rollout_idx=0,
            timeout_s=1.0,
            stability_window_s=0.05,
        )
    )
    assert result is True
