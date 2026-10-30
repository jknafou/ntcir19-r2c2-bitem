"""Unit tests for the LLM-as-judge module.

Covers the deterministic surface of ``r2c2_agent.orchestrator.llm_judge``:
evidence-pack assembly, prompt rendering, tool-arg validation,
retry-until-valid loop (with a stub HTTP layer), and JSONL load/dedupe.
End-to-end LLM behaviour needs a live vLLM endpoint and is not covered
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from r2c2_agent.orchestrator.ac_reaggregator import (
    ATTEMPT_OK,
    CANONICAL_REFUSAL_TEXT,
    FrozenPR,
    _Rollout,
)
from r2c2_agent.orchestrator.llm_judge import (
    TOOL_SPEC,
    JudgeDecision,
    JudgeEvidencePack,
    _AttemptRow,
    _NuggetRow,
    _render_judge_prompt,
    _render_passage_block,
    _validate_tool_args,
    build_evidence_pack,
    call_judge_with_retries,
    load_judge_decisions,
)


def _make_rollout(
    qid: str,
    iter_dir: int = 0,
    *,
    submit_attempts: list[dict] | None = None,
    final_answer: str = "Avatar",
    nuggets: list[dict] | None = None,
) -> _Rollout:
    from tests.conftest import _build_submit_attempt_messages

    msgs = _build_submit_attempt_messages(submit_attempts or [])
    return _Rollout(
        qid=qid,
        ensemble_idx=0,
        iter_dir=iter_dir,
        termination="test",
        final_answer=final_answer,
        nuggets=nuggets or [],
        passages=[],
        raw_trace={
            "question_id": qid,
            "messages": msgs,
            "iterations": 1,
            "ensemble_idx": 0,
            "termination": "test",
            "final_answer": final_answer,
        },
    )


def _make_pack(qid: str = "q1") -> JudgeEvidencePack:
    return JudgeEvidencePack(
        qid=qid,
        question_text="What movie?",
        pr_roster=(("pidA", 1, "Avatar is a movie."),),
        attempts=(
            _AttemptRow(
                iter_dir=0,
                attempt_idx=1,
                classification=ATTEMPT_OK,
                confidence=0.8,
                n_eff=3,
                answer_text="Avatar",
                citation_pids=("pidA",),
            ),
        ),
        nuggets=(
            _NuggetRow(nli_score=0.91, primary_pid="pidA", text="Avatar is a movie."),
        ),
    )


class TestRenderPassageBlock:
    def test_basic_render(self):
        out = _render_passage_block("pidA", "Avatar is a movie.")
        assert out.startswith("[P:pidA]")
        assert "Avatar is a movie." in out

    def test_truncates_long_text(self):
        long_text = "x " * 400
        out = _render_passage_block("p", long_text, max_chars=20)
        # 20-char cap on snippet plus "[P:p] " prefix.
        assert len(out) <= 6 + 20 + 2


class TestRenderJudgePrompt:
    def test_includes_question_text(self):
        pack = _make_pack()
        _system, user = _render_judge_prompt(pack)
        assert "What movie?" in user

    def test_includes_pr_roster_with_rank(self):
        pack = _make_pack()
        _system, user = _render_judge_prompt(pack)
        assert "PR-2 roster" in user
        assert "[P:pidA rank 1]" in user

    def test_includes_attempt_block(self):
        pack = _make_pack()
        _system, user = _render_judge_prompt(pack)
        assert "iter 0, attempt 1" in user
        assert "class=OK" in user
        assert "Avatar" in user

    def test_includes_nugget_block(self):
        pack = _make_pack()
        _system, user = _render_judge_prompt(pack)
        assert "Verified-supporting nuggets" in user
        assert "NLI=0.91" in user

    def test_system_prompt_mentions_tool_call(self):
        pack = _make_pack()
        system, _user = _render_judge_prompt(pack)
        # The system prompt forces a single tool call.
        assert "submit_answer" in system
        assert "tool" in system.lower()


class TestToolSpec:
    def test_tool_spec_shape(self):
        assert TOOL_SPEC["type"] == "function"
        fn = TOOL_SPEC["function"]
        assert fn["name"] == "submit_answer"
        props = fn["parameters"]["properties"]
        assert set(props.keys()) == {"answer", "confidence", "evidence_pids"}
        assert fn["parameters"]["required"] == ["answer", "confidence", "evidence_pids"]
        assert props["confidence"]["minimum"] == 0
        assert props["confidence"]["maximum"] == 100
        assert props["evidence_pids"]["minItems"] == 1
        assert props["evidence_pids"]["maxItems"] == 10


class TestValidateToolArgs:
    def test_valid_call(self):
        errs, pids = _validate_tool_args(
            {"answer": "Bor Gullet", "confidence": 80, "evidence_pids": ["pidA"]},
            {"pidA": 1, "pidB": 2},
        )
        assert errs == []
        assert pids == ["pidA"]

    def test_empty_answer(self):
        errs, _ = _validate_tool_args(
            {"answer": "", "confidence": 80, "evidence_pids": ["pidA"]},
            {"pidA": 1},
        )
        assert any("answer" in e for e in errs)

    def test_missing_answer(self):
        errs, _ = _validate_tool_args(
            {"confidence": 80, "evidence_pids": ["pidA"]},
            {"pidA": 1},
        )
        assert any("answer" in e for e in errs)

    def test_confidence_out_of_range(self):
        errs, _ = _validate_tool_args(
            {"answer": "x", "confidence": 150, "evidence_pids": ["pidA"]},
            {"pidA": 1},
        )
        assert any("confidence" in e for e in errs)

    def test_confidence_non_integer(self):
        errs, _ = _validate_tool_args(
            {"answer": "x", "confidence": "high", "evidence_pids": ["pidA"]},
            {"pidA": 1},
        )
        assert any("confidence" in e for e in errs)

    def test_confidence_float_integer_value_ok(self):
        # vLLM tool parsers sometimes emit 80.0; accept that.
        errs, _ = _validate_tool_args(
            {"answer": "x", "confidence": 80.0, "evidence_pids": ["pidA"]},
            {"pidA": 1},
        )
        assert errs == []

    def test_empty_evidence_pids(self):
        errs, _ = _validate_tool_args(
            {"answer": "x", "confidence": 50, "evidence_pids": []},
            {"pidA": 1},
        )
        assert any("evidence_pids" in e for e in errs)

    def test_pids_not_in_roster(self):
        errs, pids = _validate_tool_args(
            {"answer": "x", "confidence": 50, "evidence_pids": ["pidX", "pidY"]},
            {"pidA": 1},
        )
        assert any("roster" in e for e in errs)
        assert pids == []

    def test_partial_pids_in_roster(self):
        # ``pidB`` valid, ``pidX`` invalid → keep only the valid one,
        # no errors.
        errs, pids = _validate_tool_args(
            {"answer": "x", "confidence": 50, "evidence_pids": ["pidB", "pidX"]},
            {"pidA": 1, "pidB": 2},
        )
        assert errs == []
        assert pids == ["pidB"]


def _mock_response(
    *,
    tool_calls: list[dict] | None = None,
    content: str = "",
) -> dict:
    msg: dict = {"content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


class TestCallJudgeWithRetries:
    def test_valid_tool_call_first_try(self):
        rank_map = {"pidA": 1}
        ok_args = json.dumps(
            {"answer": "Avatar", "confidence": 80, "evidence_pids": ["pidA"]}
        )
        resp = _mock_response(
            tool_calls=[
                {
                    "id": "c0",
                    "function": {"name": "submit_answer", "arguments": ok_args},
                }
            ],
        )
        with mock.patch(
            "r2c2_agent.orchestrator.llm_judge._post_json",
            return_value=resp,
        ):
            out = call_judge_with_retries(
                "http://x/v1",
                "m",
                "sys",
                "user",
                rank_map,
                max_retries=3,
            )
        assert out is not None
        assert out["answer"] == "Avatar"
        assert out["confidence"] == 80
        assert out["evidence_pids"] == ["pidA"]
        assert out["n_retries"] == 0

    def test_malformed_tool_call_then_retry_succeeds(self):
        rank_map = {"pidA": 1}
        bad_args = "this is not json {{{"
        ok_args = json.dumps(
            {"answer": "Avatar", "confidence": 80, "evidence_pids": ["pidA"]}
        )
        responses = [
            _mock_response(
                tool_calls=[
                    {
                        "id": "c0",
                        "function": {"name": "submit_answer", "arguments": bad_args},
                    }
                ]
            ),
            _mock_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {"name": "submit_answer", "arguments": ok_args},
                    }
                ]
            ),
        ]
        with mock.patch(
            "r2c2_agent.orchestrator.llm_judge._post_json",
            side_effect=responses,
        ):
            out = call_judge_with_retries(
                "http://x/v1",
                "m",
                "sys",
                "user",
                rank_map,
                max_retries=3,
            )
        assert out is not None
        assert out["n_retries"] == 1

    def test_invalid_pid_then_retry_succeeds(self):
        rank_map = {"pidA": 1}
        bad_args = json.dumps(
            {"answer": "Avatar", "confidence": 80, "evidence_pids": ["pidX"]}
        )
        ok_args = json.dumps(
            {"answer": "Avatar", "confidence": 80, "evidence_pids": ["pidA"]}
        )
        responses = [
            _mock_response(
                tool_calls=[
                    {
                        "id": "c0",
                        "function": {"name": "submit_answer", "arguments": bad_args},
                    }
                ]
            ),
            _mock_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {"name": "submit_answer", "arguments": ok_args},
                    }
                ]
            ),
        ]
        with mock.patch(
            "r2c2_agent.orchestrator.llm_judge._post_json",
            side_effect=responses,
        ):
            out = call_judge_with_retries(
                "http://x/v1",
                "m",
                "sys",
                "user",
                rank_map,
                max_retries=3,
            )
        assert out is not None
        assert out["evidence_pids"] == ["pidA"]
        assert out["n_retries"] == 1

    def test_no_tool_call_then_retry_succeeds(self):
        rank_map = {"pidA": 1}
        ok_args = json.dumps(
            {"answer": "Avatar", "confidence": 80, "evidence_pids": ["pidA"]}
        )
        responses = [
            _mock_response(content="Some prose, no tool call.", tool_calls=None),
            _mock_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {"name": "submit_answer", "arguments": ok_args},
                    }
                ]
            ),
        ]
        with mock.patch(
            "r2c2_agent.orchestrator.llm_judge._post_json",
            side_effect=responses,
        ):
            out = call_judge_with_retries(
                "http://x/v1",
                "m",
                "sys",
                "user",
                rank_map,
                max_retries=3,
            )
        assert out is not None
        assert out["n_retries"] == 1

    def test_exhausted_retries_returns_none(self):
        rank_map = {"pidA": 1}
        bad_args = "still not json"
        responses = [
            _mock_response(
                tool_calls=[
                    {
                        "id": "c0",
                        "function": {"name": "submit_answer", "arguments": bad_args},
                    }
                ]
            )
            for _ in range(4)
        ]
        with mock.patch(
            "r2c2_agent.orchestrator.llm_judge._post_json",
            side_effect=responses,
        ):
            out = call_judge_with_retries(
                "http://x/v1",
                "m",
                "sys",
                "user",
                rank_map,
                max_retries=3,
            )
        assert out is None

    def test_captures_reasoning_as_rationale(self):
        rank_map = {"pidA": 1}
        ok_args = json.dumps(
            {"answer": "Avatar", "confidence": 80, "evidence_pids": ["pidA"]}
        )
        resp = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "I think Avatar because evidence...",
                        "tool_calls": [
                            {
                                "id": "c0",
                                "function": {
                                    "name": "submit_answer",
                                    "arguments": ok_args,
                                },
                            }
                        ],
                    }
                }
            ]
        }
        with mock.patch(
            "r2c2_agent.orchestrator.llm_judge._post_json",
            return_value=resp,
        ):
            out = call_judge_with_retries(
                "http://x/v1",
                "m",
                "sys",
                "user",
                rank_map,
                max_retries=3,
            )
        assert out is not None
        assert "Avatar because" in out["rationale"]


class TestBuildEvidencePack:
    def test_basic_extraction(self):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={
                "q1": [(1, "doc1", "Avatar is a movie.")],
            },
        )
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA"],
                    "committed": True,
                    "confidence": 0.9,
                    "n_eff": 3,
                },
            ],
            nuggets=[
                {
                    "passage_id": "pidA",
                    "supporting_pids": ["pidA"],
                    "pid_nli_scores": {"pidA": 0.88},
                    "nli_score": 0.88,
                    "text": "Avatar is a James Cameron movie.",
                },
            ],
        )
        pack = build_evidence_pack(
            "q1",
            "Q?",
            [r],
            {"pidA": 1},
            pr,
        )
        assert pack.qid == "q1"
        assert pack.question_text == "Q?"
        assert pack.pr_roster == (("pidA", 1, "Avatar is a movie."),)
        assert len(pack.attempts) == 1
        assert pack.attempts[0].answer_text == "Avatar"
        assert pack.attempts[0].classification == ATTEMPT_OK
        assert pack.attempts[0].citation_pids == ("pidA",)
        assert len(pack.nuggets) == 1
        assert pack.nuggets[0].primary_pid == "pidA"
        assert pack.nuggets[0].nli_score == 0.88

    def test_refusal_attempt_still_emitted(self):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={"q1": [(1, "doc1", "Long passage text.")]},
        )
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
        pack = build_evidence_pack(
            "q1",
            "Q?",
            [r],
            {},
            pr,
        )
        # Even refusal-text attempts are surfaced so the judge sees them.
        assert len(pack.attempts) >= 1

    def test_drops_attempts_with_empty_answer(self):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={"q1": [(1, "doc1", "Text")]},
        )
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "",
                    "citation_pids": [],
                    "committed": False,
                },
            ],
        )
        pack = build_evidence_pack("q1", "Q?", [r], {}, pr)
        assert pack.attempts == ()

    def test_filters_out_of_roster_pids_from_attempts(self):
        pr = FrozenPR(
            run_name="BITEM-PG-2",
            rows_by_qid={"q1": [(1, "doc1", "Text")]},
        )
        r = _make_rollout(
            "q1",
            submit_attempts=[
                {
                    "answer": "Avatar",
                    "citation_pids": ["pidA", "pidX"],
                    "committed": True,
                    "confidence": 0.9,
                },
            ],
        )
        pack = build_evidence_pack("q1", "Q?", [r], {"pidA": 1}, pr)
        assert pack.attempts[0].citation_pids == ("pidA",)


class TestLoadJudgeDecisions:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_judge_decisions(tmp_path / "no-such.jsonl") == {}

    def test_loads_records(self, tmp_path: Path):
        path = tmp_path / "judge.jsonl"
        d1 = JudgeDecision(
            qid="q1",
            model="m",
            raw_choice="tool_call",
            raw_rationale="r",
            raw_evidence_pids=["pidA"],
            final_choice="tool_call",
            final_answer="a",
            final_confidence=50,
            final_citation_pids=["pidA"],
            fallback_reason=None,
            parse_failed=False,
            ts="",
            n_retries=0,
        )
        d2 = JudgeDecision(
            qid="q2",
            model="m",
            raw_choice="tool_call",
            raw_rationale="r2",
            raw_evidence_pids=["p"],
            final_choice="tool_call",
            final_answer="b",
            final_confidence=80,
            final_citation_pids=["p"],
            fallback_reason=None,
            parse_failed=False,
            ts="",
            n_retries=1,
        )
        with open(path, "w") as f:
            f.write(d1.to_json() + "\n")
            f.write(d2.to_json() + "\n")
        out = load_judge_decisions(path)
        assert set(out.keys()) == {"q1", "q2"}
        assert out["q2"].final_answer == "b"
        assert out["q2"].n_retries == 1

    def test_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "judge.jsonl"
        d1 = JudgeDecision(
            qid="q1",
            model="m",
            raw_choice="tool_call",
            raw_rationale="r",
            raw_evidence_pids=[],
            final_choice="tool_call",
            final_answer="a",
            final_confidence=50,
            final_citation_pids=[],
            fallback_reason=None,
            parse_failed=False,
            ts="",
        )
        with open(path, "w") as f:
            f.write("not-json\n")
            f.write(d1.to_json() + "\n")
        out = load_judge_decisions(path)
        assert set(out.keys()) == {"q1"}
