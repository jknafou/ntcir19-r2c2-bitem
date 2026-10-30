# Shared test fixtures.

from __future__ import annotations

from typing import Any

import pytest


def _build_submit_attempt_messages(submit_attempts: list[dict[str, Any]]) -> list[dict]:
    """Render submit_answer attempts into interleaved assistant/tool messages.

    Each entry of ``submit_attempts`` is a dict with keys::

        answer (str), rationale (str), citation_pids (list[str]),
        committed (bool), confidence (float|None),
        reason (str|None), error (str|None), n_eff (int|None)

    Returns a flat list of messages ready to drop into the ``messages``
    field of a trace record (the schema produced by the live orchestrator
    and consumed by ``ac_reaggregator._extract_submit_attempts_from_trace``).
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
    return msgs


@pytest.fixture
def make_trace():
    """Factory fixture: build a minimal trace record from submit_answer attempts.

    Usage::

        def test_something(make_trace):
            rec = make_trace([
                {"answer": "Avatar", "citation_pids": ["a"], "committed": True,
                 "confidence": 0.85, "n_eff": 3},
            ])

    Returns a single-record dict shaped like one row of
    ``iter_<n>/test_traces.jsonl``.
    """

    def _factory(submit_attempts: list[dict[str, Any]]) -> dict:
        return {"messages": _build_submit_attempt_messages(submit_attempts)}

    return _factory


@pytest.fixture(autouse=True)
def _scoring_flag_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``R2C2_DECOUPLED_SCORING`` OFF for every test by default.

    The decoupled scoring
    is gated behind this flag. Legacy numeric fixtures throughout the
    suite assume the legacy additive formula (flag OFF). This autouse
    fixture guarantees they run under the legacy path regardless of the
    outer shell env.

    Tests that want the decoupled path (``tests/test_scoring_decouple.py``)
    override this via their own ``_flag_on`` fixture, which uses the
    same ``monkeypatch`` instance and runs after this one in declaration
    order — ``monkeypatch.setenv`` therefore wins.
    """
    monkeypatch.delenv("R2C2_DECOUPLED_SCORING", raising=False)
