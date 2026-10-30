"""Integration tests for ``answer_question_pipeline`` — the split-mode driver.

Covers:
- Happy path (record_evidence → submit_answer).
- Refusal path (submit_answer(refusal=True) on the first WRITE).
- Unknown-pid handling (record_evidence against a pid never seeded).
- MCP timeout surfaces a wrapped error to the LLM.
- No-tool-call on WRITE triggers implicit refusal.
- Iteration cap forces a refusal via handle_submit_answer.
- Budget envelope + session_status present on every tool response.
- Feature-flag dispatch at the worker.py call sites.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from r2c2_agent.agent.config import LoopConfig  # noqa: E402
from r2c2_agent.agent.interfaces.types import (  # noqa: E402
    LLMResponse,
    Question,
    ToolCall,
)
from r2c2_agent.agent.loop import answer_question_pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _bypass_fpp_exhaustion_gate(monkeypatch: pytest.MonkeyPatch):
    """Decouple these driver-plumbing tests from the F++ exhaustion gate.

    The gate's own semantics are covered exhaustively in
    ``test_exhaustion_check_gates.py`` and
    ``test_refusal_rejection_then_force_deliver.py``. Here we're testing
    that the driver correctly propagates a scripted LLM's refusal /
    happy-path decisions into the trace, so we patch the gate to a
    permissive stub — the same way we'd mock an NLI server.
    """
    from r2c2_agent.orchestrator import pipeline_handlers

    monkeypatch.setattr(
        pipeline_handlers,
        "_exhaustion_check",
        lambda _db: (True, []),
    )


# Canned LLM and MCP clients for the single-turn driver


class _ScriptedLLMClient:
    """Replays a fixed sequence of LLMResponse objects across chat() calls.

    The driver makes one LLM call per turn, with thinking on and tools
    attached, so a test scripts exactly one response per turn. Running
    out of responses raises, because the test scripted too few.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "enable_thinking": enable_thinking,
            }
        )
        if self._idx >= len(self._responses):
            raise RuntimeError(
                f"_ScriptedLLMClient exhausted (idx={self._idx}, "
                f"scripted={len(self._responses)})"
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class _StubMCPClient:
    """MCP stub. Returns canned dicts (not JSON strings) per tool name.

    ``fail_tools`` is a set of tool names that should raise
    ``asyncio.TimeoutError`` on ``call_tool``.
    """

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        fail_tools: set[str] | None = None,
    ) -> None:
        self._responses = responses or {}
        self._fail_tools = fail_tools or set()
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        self.calls.append({"name": name, "arguments": arguments})
        if name in self._fail_tools:
            raise TimeoutError("simulated mcp timeout")
        return self._responses.get(name, {"passages": []})


# Helpers


def _write_tool(
    name: str,
    args: dict[str, Any],
    call_id: str = "call_1",
    content: str = "",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        reasoning=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
    )


def _write_text(text: str = "Done.") -> LLMResponse:
    return LLMResponse(content=text, reasoning=None, tool_calls=[])


def _passage_hit(pid: str, text: str, title: str = "t", score: float = 0.9) -> dict:
    return {
        "passage_id": pid,
        "text": text,
        "title": title,
        "score": score,
        "doc_id": "doc_1",
    }


def _tools_catalogue() -> list[dict[str, Any]]:
    """Minimal OpenAI-format tool list — only names matter for the driver."""
    names = [
        "search_passages",
        "search_entities",
        "get_passage",
        "extract_from_passage",
        "record_evidence",
        "submit_answer",
    ]
    return [
        {
            "type": "function",
            "function": {"name": n, "description": n, "parameters": {"type": "object"}},
        }
        for n in names
    ]


def _default_config() -> LoopConfig:
    return LoopConfig(
        max_iterations=10,
        context_budget=999_999,
        token_budget=999_999,
        tool_timeout=2.0,
        concurrency=1,
    )


def _patch_nli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``_batch_nli`` with a synchronous stub returning entails=1.0.

    Ensures test-time record_evidence never tries to reach a real NLI server.
    """
    from r2c2_agent.orchestrator import pipeline_handlers as handlers

    async def _fake_batch_nli(
        pairs: list[tuple[str, str]],
        *,
        nli_url: str | None = None,
        http_post: Any | None = None,
    ) -> list[dict[str, Any]]:
        return [{"label": "entails", "score": 0.95} for _ in pairs]

    monkeypatch.setattr(handlers, "_batch_nli", _fake_batch_nli, raising=True)


# Tests


class TestHappyPath:
    def test_record_then_submit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_nli_env(monkeypatch)
        passage_text = (
            "Steven Spielberg directed Jaws in 1975. "
            "The film was a major commercial success."
        )
        passage_text_2 = (
            "Jaws, directed by Steven Spielberg, was released in June 1975 "
            "by Universal Pictures."
        )
        # n_eff counts distinct supporting pids: two nuggets on the same
        # pid count once, and the answer gate rejects the submission.
        # The scenario therefore surfaces two passages and records
        # evidence on each.
        # Turn 1: search_passages
        # Turn 2: record_evidence on p_0001
        # Turn 3: record_evidence on p_0002
        # Turn 4: submit_answer
        responses = [
            _write_tool(
                "search_passages",
                {"queries": ["Jaws director"], "max_results": 5},
                call_id="c1",
            ),
            _write_tool(
                "record_evidence",
                {
                    "pid": "p_0001",
                    "stance": "supporting",
                    "rationale": "Passage names Spielberg as director.",
                    "nuggets": [
                        {
                            "claim": "Spielberg directed Jaws.",
                            "polarity": "affirming",
                            "quoted_span": "Steven Spielberg directed Jaws",
                        },
                    ],
                },
                call_id="c2",
            ),
            _write_tool(
                "record_evidence",
                {
                    "pid": "p_0002",
                    "stance": "supporting",
                    "rationale": "Second passage confirms Spielberg.",
                    "nuggets": [
                        {
                            "claim": "Spielberg directed Jaws.",
                            "polarity": "affirming",
                            "quoted_span": "directed by Steven Spielberg",
                        },
                    ],
                },
                call_id="c3",
            ),
            _write_tool(
                "submit_answer",
                {
                    "answer": "Steven Spielberg",
                    "refusal": False,
                    "rationale": "Two passages directly support this.",
                    "citation_pids": ["p_0001", "p_0002"],
                },
                call_id="c4",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(
            responses={
                "search_passages": {
                    "passages": [
                        _passage_hit("p_0001", passage_text),
                        _passage_hit("p_0002", passage_text_2),
                    ],
                }
            }
        )
        trace = asyncio.run(
            answer_question_pipeline(
                question=Question(
                    qid="q1", text="Who directed Jaws?", question_type="simple"
                ),
                llm_client=llm,
                mcp_client=mcp,
                tools=_tools_catalogue(),
                config=_default_config(),
            )
        )

        assert trace.draft_answer == "Steven Spielberg"
        assert trace.termination == "answer_submitted"
        assert len(trace.nuggets) >= 1
        assert trace.nuggets[0].passage_id in {"p_0001", "p_0002"}
        # Single-turn driver: 4 turns = 4 LLM calls.
        assert len(llm.calls) == 4
        # Every call has tools attached and thinking=True.
        assert all(c["tools"] is not None for c in llm.calls)
        assert all(c["enable_thinking"] is True for c in llm.calls)

    def test_budget_envelope_on_every_tool_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        # Just search then refuse — tool message will carry budget + session.
        responses = [
            _write_tool("search_passages", {"queries": ["x"]}, call_id="c1"),
            _write_tool(
                "submit_answer",
                {"refusal": True, "rationale": "no useful hits.", "answer": ""},
                call_id="c2",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(
            responses={
                "search_passages": {"passages": [_passage_hit("p_0001", "Some text")]}
            }
        )
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # Both tool responses must carry budget + session_status envelopes.
        tool_msgs = [m for m in trace.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        for tmsg in tool_msgs:
            body = json.loads(tmsg["content"])
            assert "budget" in body
            assert "session_status" in body
            assert set(body["budget"]) >= {
                "calls_used",
                "calls_remaining",
                "nuggets_recorded",
                "recorded_pids_count",
            }


class TestRefusalPath:
    def test_immediate_refusal_on_first_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        responses = [
            _write_tool(
                "submit_answer",
                {
                    "refusal": True,
                    "rationale": "no evidence available for this question.",
                    "answer": "",
                },
                call_id="c1",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient()
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # A refusal on the first write routes through force_deliver. With
        # no retrieved passages the envelope falls to tier 5, the
        # canonical refusal at the 0.05 confidence floor.
        assert trace.termination == "force_deliver"
        assert trace.raw_confidence == 5  # 0-100 scale, 0.05 floor
        # Refusal nugget present but excluded from the trace's AC nuggets list.
        assert all(n.passage_id != "__refusal__" for n in trace.nuggets)
        # Draft answer is the canonical refusal string (tier-5).
        assert "insufficient" in trace.draft_answer.lower()


class TestUnknownPid:
    def test_record_evidence_on_unseen_pid_surfaces_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        # LLM tries to record evidence on a pid that was not seeded.
        # Handler returns error; driver surfaces it to the LLM; then refuse.
        responses = [
            _write_tool(
                "record_evidence",
                {
                    "pid": "p_unknown",
                    "stance": "supporting",
                    "rationale": "trying hard.",
                    "nuggets": [
                        {
                            "claim": "Something.",
                            "polarity": "affirming",
                            "quoted_span": "foo",
                        }
                    ],
                },
                call_id="c1",
            ),
            _write_tool(
                "submit_answer",
                {
                    "refusal": True,
                    "rationale": "no retrieved passage available.",
                    "answer": "",
                },
                call_id="c2",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient()
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # Inspect the tool result for record_evidence — should contain
        # "unknown_pid" in the body.
        tool_msgs = [m for m in trace.messages if m.get("role") == "tool"]
        body_rec = json.loads(tool_msgs[0]["content"])
        assert body_rec.get("error") == "unknown_pid"
        assert body_rec.get("committed") is False
        # Refusals route through force_deliver, so the loop terminates there.
        assert trace.termination == "force_deliver"


class TestMCPTimeout:
    def test_search_passages_timeout_surfaces_wrapped_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        responses = [
            _write_tool("search_passages", {"queries": ["x"]}, call_id="c1"),
            _write_tool(
                "submit_answer",
                {
                    "refusal": True,
                    "rationale": "retrieval unavailable.",
                    "answer": "",
                },
                call_id="c2",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(fail_tools={"search_passages"})
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        tool_msgs = [m for m in trace.messages if m.get("role") == "tool"]
        body = json.loads(tool_msgs[0]["content"])
        assert body.get("error") == "mcp_timeout"
        # Budget envelope still present despite MCP failure.
        assert "budget" in body


class TestNoToolCallOnWrite:
    def test_empty_write_triggers_implicit_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        responses = [
            _write_text("I don't know."),  # no tool calls at all
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient()
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # no_tool_call routes through force_deliver. The assistant's
        # written text is taken as preferred_answer (tier 1). Refusal
        # sentinel still fires because the DB has no real passages, so
        # confidence is clamped (raw ≤ 15, ×100 → 0.1 max).
        assert trace.termination == "force_deliver"
        assert trace.raw_confidence <= 15


class TestIterationCap:
    def test_hits_max_iterations_and_forces_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)

        # Every turn the model calls search_passages; never submits. Cap at 2.
        def _turn(i: int) -> list[LLMResponse]:
            return [
                _write_tool(
                    "search_passages",
                    {"queries": [f"q{i}"]},
                    call_id=f"c{i}",
                ),
            ]

        responses = _turn(1) + _turn(2)
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(responses={"search_passages": {"passages": []}})
        cfg = _default_config()
        cfg.max_iterations = 2
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                cfg,
            )
        )
        # max_iterations routes through force_deliver, which on
        # the refusal-sentinel path clamps confidence to 0.1 (×100 → 10).
        assert trace.termination == "force_deliver"
        assert trace.raw_confidence <= 15
        assert "insufficient" in trace.draft_answer.lower()


class TestMessageShape:
    def test_single_turn_assistant_and_tool_messages_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        responses = [
            _write_tool("search_passages", {"queries": ["x"]}, call_id="c1"),
            _write_tool(
                "submit_answer",
                {"refusal": True, "rationale": "no hits.", "answer": ""},
                call_id="c2",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(responses={"search_passages": {"passages": []}})
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # Single-turn driver: system, user, 2 assistant (one per turn), 2 tool = 6.
        roles = [m["role"] for m in trace.messages]
        assert roles.count("assistant") == 2
        assert roles.count("tool") == 2
        assert roles[0] == "system"
        assert roles[1] == "user"
        # Every assistant message carries tool_calls (tools are always attached).
        asst_msgs = [m for m in trace.messages if m.get("role") == "assistant"]
        assert all("tool_calls" in m for m in asst_msgs)
        assert len(asst_msgs) == 2


class TestDBState:
    def test_trace_carries_pipeline_db_with_final_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        passage_text = "Spielberg directed Jaws in 1975."
        responses = [
            _write_tool("search_passages", {"queries": ["jaws"]}, call_id="c1"),
            _write_tool(
                "record_evidence",
                {
                    "pid": "p_0001",
                    "stance": "supporting",
                    "rationale": "names spielberg.",
                    "nuggets": [
                        {
                            "claim": "Spielberg directed Jaws.",
                            "polarity": "affirming",
                            "quoted_span": "Spielberg directed Jaws",
                        },
                        {
                            "claim": "Jaws was released in 1975.",
                            "polarity": "affirming",
                            "quoted_span": "directed Jaws in 1975",
                        },
                    ],
                },
                call_id="c2",
            ),
            _write_tool(
                "submit_answer",
                {
                    "answer": "Spielberg",
                    "refusal": False,
                    "rationale": "p_0001 supports.",
                    "citation_pids": ["p_0001"],
                },
                call_id="c3",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(
            responses={
                "search_passages": {"passages": [_passage_hit("p_0001", passage_text)]}
            }
        )
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="Who directed Jaws?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        db = getattr(trace, "m11_db", None)
        assert db is not None
        assert db.terminated is True
        # Passage was seeded from MCP response, then stance applied.
        assert "p_0001" in db.passages
        assert db.passages["p_0001"].status == "accepted"
        # At least one verified nugget. The answer gate needs n_eff>=2 to
        # commit at the default confidence, so this test seeds 2 supporters.
        verified = [n for n in db.nuggets.values() if n.status == "verified_supporting"]
        assert len(verified) >= 1


class TestHandlerExceptionIsolation:
    def test_handler_exception_does_not_crash_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        # First WRITE emits submit_answer with invalid args (empty rationale).
        # Handler should return a well-formed error instead of raising.
        # Then the driver's safety-net refusal fires when the loop exits
        # without db.terminated.
        responses = [
            _write_tool(
                "submit_answer",
                {"refusal": False, "answer": "x", "rationale": ""},  # invalid
                call_id="c1",
            ),
            _write_tool(
                "submit_answer",
                {"refusal": True, "rationale": "invalid args.", "answer": ""},
                call_id="c2",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient()
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # The failed submit should have surfaced an error body to the model.
        tool_msgs = [m for m in trace.messages if m.get("role") == "tool"]
        first_body = json.loads(tool_msgs[0]["content"])
        assert first_body.get("committed") is False
        assert "error" in first_body
        # The second submit, a refusal, routes through force_deliver.
        assert trace.termination == "force_deliver"


class TestSystemAndUserMessage:
    def test_system_prompt_and_user_with_qtype_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        responses = [
            _write_tool(
                "submit_answer",
                {"refusal": True, "rationale": "no.", "answer": ""},
                call_id="c1",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient()
        trace = asyncio.run(
            answer_question_pipeline(
                Question(
                    qid="q1",
                    text="How many Star Wars films are there?",
                    question_type="aggregation",
                ),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # User message carries the qtype tag.
        assert trace.messages[1]["role"] == "user"
        assert "[type: aggregation]" in trace.messages[1]["content"]
        # System message is non-empty and contains XML sections.
        assert trace.messages[0]["role"] == "system"
        assert "<role>" in trace.messages[0]["content"]


class TestToolCallsJsonSerializable:
    def test_write_message_tool_calls_have_json_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        responses = [
            _write_tool(
                "search_passages",
                {"queries": ["x"], "max_results": 3},
                call_id="c1",
            ),
            _write_tool(
                "submit_answer",
                {"refusal": True, "rationale": "n.", "answer": ""},
                call_id="c2",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(responses={"search_passages": {"passages": []}})
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="foo?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        # WRITE assistant messages: arguments must be a JSON string.
        for m in trace.messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc_dict in m["tool_calls"]:
                    assert isinstance(tc_dict["function"]["arguments"], str)
                    json.loads(tc_dict["function"]["arguments"])


# Feature-flag dispatch at the worker.py call sites


def _import_worker() -> Any:
    """Dynamically import worker.py via its filesystem path."""
    import r2c2_agent.agent.interfaces  # noqa: F401  — prime namespace

    worker_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "r2c2_agent"
        / "orchestrator"
        / "worker.py"
    )
    spec = importlib.util.spec_from_file_location("_worker_probe", worker_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFeatureFlagSource:
    """Check that the feature-flag dispatch is written at both call sites.

    Reads the worker.py source, because importing and calling the full
    aiohttp handler needs a running MCP server and vLLM, which unit tests
    do not have.
    """

    def test_source_contains_pipeline_dispatch_twice(self) -> None:
        worker_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "r2c2_agent"
            / "orchestrator"
            / "worker.py"
        )
        src = worker_path.read_text()
        # Each call site must check AGENTIC_PIPELINE + call answer_question_pipeline.
        assert src.count("answer_question_pipeline") >= 2, (
            "expected answer_question_pipeline imported/called at both sites in worker.py"
        )
        assert src.count('os.environ.get("AGENTIC_PIPELINE"') >= 2

    def test_env_var_default_is_unified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver dispatch defaults to the unified pipeline when env unset.

        A launcher that does not export ``AGENTIC_PIPELINE`` must not
        route every qid through the legacy driver, which crashes on the
        unified tool schemas. Run the legacy pipeline by explicitly
        setting ``AGENTIC_PIPELINE=legacy``.
        """
        worker_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "r2c2_agent"
            / "orchestrator"
            / "worker.py"
        )
        src = worker_path.read_text()
        # The default must be "unified".
        assert 'os.environ.get("AGENTIC_PIPELINE", "unified")' in src
        # Neither earlier default may be present.
        assert 'os.environ.get("AGENTIC_PIPELINE", "legacy")' not in src
        assert 'os.environ.get("AGENTIC_PIPELINE", "m11_unified")' not in src


class TestConfidenceMapping:
    def test_high_conf_answer_maps_to_int_0_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_nli_env(monkeypatch)
        passage_text = "Spielberg directed Jaws."
        passage_text_2 = "Jaws was directed by Steven Spielberg in 1975."
        # Record evidence on two distinct pids so that n_eff=2 gives
        # supportability='sufficient' (base 0.85).
        responses = [
            _write_tool("search_passages", {"queries": ["jaws"]}, call_id="c1"),
            _write_tool(
                "record_evidence",
                {
                    "pid": "p_0001",
                    "stance": "supporting",
                    "rationale": "names spielberg.",
                    "nuggets": [
                        {
                            "claim": "Spielberg directed Jaws.",
                            "polarity": "affirming",
                            "quoted_span": "Spielberg directed Jaws",
                        }
                    ],
                },
                call_id="c2",
            ),
            _write_tool(
                "record_evidence",
                {
                    "pid": "p_0002",
                    "stance": "supporting",
                    "rationale": "second passage confirms.",
                    "nuggets": [
                        {
                            "claim": "Spielberg was the director.",
                            "polarity": "affirming",
                            "quoted_span": "directed by Steven Spielberg",
                        }
                    ],
                },
                call_id="c3",
            ),
            _write_tool(
                "submit_answer",
                {
                    "answer": "Spielberg",
                    "refusal": False,
                    "rationale": "strong support across two passages.",
                },
                call_id="c4",
            ),
        ]
        llm = _ScriptedLLMClient(responses)
        mcp = _StubMCPClient(
            responses={
                "search_passages": {
                    "passages": [
                        _passage_hit("p_0001", passage_text),
                        _passage_hit("p_0002", passage_text_2),
                    ]
                }
            }
        )
        trace = asyncio.run(
            answer_question_pipeline(
                Question(qid="q1", text="Who directed Jaws?", question_type="simple"),
                llm,
                mcp,
                _tools_catalogue(),
                _default_config(),
            )
        )
        assert 0 <= trace.raw_confidence <= 100
        assert trace.termination == "answer_submitted"
