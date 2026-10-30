"""Tests for the foundation: types, config, exceptions, prompts."""

from __future__ import annotations

import json

import pytest

from r2c2_agent.agent.config import NER_TOOLS, LoopConfig, Mode, RunConfig, ServiceEndpoints
from r2c2_agent.agent.exceptions import (
    AgentBiTemError,
    BudgetExceededError,
    LLMError,
    MCPError,
    ParseError,
    ToolDispatchError,
    ValidationError,
)
from r2c2_agent.agent.interfaces.types import (
    ACResult,
    CitedPassage,
    LLMResponse,
    Nugget,
    PREntry,
    Question,
    QuestionTrace,
    ToolCall,
    ToolResult,
    VerificationResult,
)
from r2c2_agent.agent.prompt import (
    CONFIDENCE_PATTERN,
    NUGGET_PATTERN,
    REFORMULATION_TEMPLATES,
    SYNTHESIZE_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    VERIFIER_SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
    build_verification_user_message,
)

# Dataclass construction and serialization


class TestQuestion:
    def test_construct(self) -> None:
        q = Question(qid="Q001", text="Who directed Jaws?")
        assert q.qid == "Q001"
        assert q.question_type == "default"

    def test_with_type(self) -> None:
        q = Question(qid="Q002", text="Compare X and Y", question_type="comparison")
        assert q.question_type == "comparison"


class TestToolCall:
    def test_construct(self) -> None:
        tc = ToolCall(id="call_1", name="search_passages", arguments={"query": "Jaws"})
        assert tc.name == "search_passages"
        assert tc.arguments["query"] == "Jaws"


class TestToolResult:
    def test_construct(self) -> None:
        tr = ToolResult(tool_call_id="call_1", content={"passages": []})
        assert not tr.is_error

    def test_error(self) -> None:
        tr = ToolResult(
            tool_call_id="call_1", content={"error": "timeout"}, is_error=True
        )
        assert tr.is_error


class TestLLMResponse:
    def test_text_only(self) -> None:
        r = LLMResponse(content="The answer is 42")
        assert r.content == "The answer is 42"
        assert r.reasoning is None
        assert r.tool_calls == []

    def test_with_reasoning_and_tools(self) -> None:
        tc = ToolCall(id="c1", name="search_passages", arguments={"query": "test"})
        r = LLMResponse(content="", reasoning="Let me search...", tool_calls=[tc])
        assert r.reasoning == "Let me search..."
        assert len(r.tool_calls) == 1


class TestNugget:
    def test_construct(self) -> None:
        n = Nugget(passage_id="wiki_Q123_p2", text="Spielberg directed Jaws")
        assert n.passage_id == "wiki_Q123_p2"
        assert n.num == 0  # default


class TestVerificationResult:
    def test_agree(self) -> None:
        v = VerificationResult(
            verdict="AGREE",
            confidence_adjustment=5,
            supporting_passages=["p1", "p2"],
            reasoning="Well supported",
        )
        assert v.verdict == "AGREE"
        assert v.thinking is None

    def test_with_thinking(self) -> None:
        v = VerificationResult(
            verdict="DISAGREE",
            confidence_adjustment=-15,
            reasoning="Contradicted",
            alternative_answer="Different answer",
            thinking="I noticed that...",
        )
        assert v.thinking == "I noticed that..."


class TestQuestionTrace:
    def test_round_trip(self) -> None:
        """Serialize to dict and back — all fields preserved."""
        trace = QuestionTrace(
            question_id="Q042",
            question_text="Who is the actor?",
            question_type="multi_hop",
            mode="b",
            model="Qwen3.5-122B",
            draft_answer="Anthony Mackie",
            raw_confidence=88,
            final_confidence=85,
            termination="natural",
            iterations=4,
            total_tokens=12500,
            wall_time_seconds=8.3,
        )
        trace.nuggets = [
            Nugget(passage_id="p1", text="Mackie in Manchurian Candidate", num=1),
            Nugget(passage_id="p2", text="Mackie played Captain America", num=2),
        ]
        trace.verification = VerificationResult(
            verdict="AGREE",
            confidence_adjustment=3,
            supporting_passages=["p1"],
            reasoning="Confirmed",
            raw_response="VERDICT: AGREE...",
            thinking="The evidence is clear",
        )

        d = trace.to_dict()
        restored = QuestionTrace.from_dict(d)

        assert restored.question_id == "Q042"
        assert restored.draft_answer == "Anthony Mackie"
        assert restored.raw_confidence == 88
        assert restored.final_confidence == 85
        assert len(restored.nuggets) == 2
        assert restored.nuggets[0].passage_id == "p1"
        assert restored.verification is not None
        assert restored.verification.verdict == "AGREE"
        assert restored.verification.thinking == "The evidence is clear"

    def test_json_serialization(self) -> None:
        """to_dict output is JSON-serializable."""
        trace = QuestionTrace(question_id="Q001", draft_answer="test")
        d = trace.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["question_id"] == "Q001"

    def test_append_turn(self) -> None:
        trace = QuestionTrace()
        msg = {"role": "assistant", "content": "Hello"}
        trace.append_turn(msg)
        assert len(trace.messages) == 1

    def test_append_tool_result(self) -> None:
        trace = QuestionTrace()
        tc = ToolCall(id="c1", name="search_passages", arguments={"query": "test"})
        trace.append_tool_result(tc, {"passages": [{"id": "p1"}]})
        assert len(trace.messages) == 1
        assert trace.messages[0]["role"] == "tool"
        assert trace.messages[0]["tool_call_id"] == "c1"

    def test_add_nuggets(self) -> None:
        trace = QuestionTrace()
        trace.add_nuggets([Nugget(passage_id="p1", text="fact 1")])
        trace.add_nuggets([Nugget(passage_id="p2", text="fact 2")])
        assert len(trace.nuggets) == 2


class TestPREntry:
    def test_construct(self) -> None:
        e = PREntry(question_id="Q001", rank=1, doc_id="Q123", passage_text="Some text")
        assert e.rank == 1


class TestACResult:
    def test_construct(self) -> None:
        r = ACResult(answer="Anthony Mackie", confidence=85)
        assert r.confidence == 85
        assert r.nuggets == []
        assert r.passage_rank_map == {}


class TestCitedPassage:
    def test_construct(self) -> None:
        cp = CitedPassage(passage_id="p1", citation_count=3, first_appearance=2)
        assert cp.citation_count == 3


# Config


class TestConfig:
    def test_loop_config_defaults(self) -> None:
        lc = LoopConfig()
        assert lc.max_iterations == 10
        assert lc.context_budget == 34816
        # tool_timeout is a safety net and is not tuned for speed. Bulk
        # HPC workflows do not cascade per-call timeouts; the per-qid
        # wall-clock bound at the orchestrator covers that.
        assert lc.tool_timeout == 900.0
        assert lc.concurrency == 1

    def test_mode_enum(self) -> None:
        assert Mode.HUMAN_GOLD.value == "a"
        assert Mode.STUDENT_AGENT.value == "b"
        assert Mode.HUMAN_JUDGE.value == "c"
        assert Mode.AUTO_JUDGE.value == "d"

    def test_ner_tools_set(self) -> None:
        assert "search_by_entity" in NER_TOOLS
        assert "search_entities" in NER_TOOLS
        assert "search_passages" not in NER_TOOLS

    def test_service_endpoints(self) -> None:
        ep = ServiceEndpoints(
            main_model="http://localhost:8000/v1",
            verifier="http://localhost:8001/v1",
            mcp_standard="http://localhost:9000",
            mcp_ner="http://localhost:9001",
        )
        assert ep.main_model == "http://localhost:8000/v1"
        assert ep.verifier is not None

    def test_service_endpoints_no_verifier(self) -> None:
        ep = ServiceEndpoints(main_model="http://localhost:8000/v1")
        assert ep.verifier is None

    def test_run_config_defaults(self) -> None:
        rc = RunConfig()
        assert rc.run_name == "BITEM-PG-1"
        assert rc.mode == Mode.STUDENT_AGENT
        assert rc.dummy is False


# Exceptions


class TestExceptions:
    def test_hierarchy(self) -> None:
        """All custom exceptions inherit from AgentBiTemError."""
        for exc_class in [
            LLMError,
            MCPError,
            ToolDispatchError,
            ParseError,
            BudgetExceededError,
            ValidationError,
        ]:
            assert issubclass(exc_class, AgentBiTemError)
            assert issubclass(exc_class, Exception)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(AgentBiTemError):
            raise LLMError("connection refused")


# Prompts and patterns


class TestNuggetPattern:
    def test_basic_match(self) -> None:
        text = "NUGGET[wiki_Q123_p2]: Spielberg directed Jaws in 1975"
        matches = NUGGET_PATTERN.findall(text)
        assert len(matches) == 1
        assert matches[0][0] == "wiki_Q123_p2"
        assert matches[0][1] == "Spielberg directed Jaws in 1975"

    def test_multiple_matches(self) -> None:
        text = "NUGGET[p1]: First fact\nSome other text\nNUGGET[p2]: Second fact\n"
        matches = NUGGET_PATTERN.findall(text)
        assert len(matches) == 2
        assert matches[0] == ("p1", "First fact")
        assert matches[1] == ("p2", "Second fact")

    def test_no_match(self) -> None:
        text = "This text has no nuggets"
        matches = NUGGET_PATTERN.findall(text)
        assert len(matches) == 0

    def test_passage_id_with_underscore(self) -> None:
        text = "NUGGET[wook_17_p5]: Star Wars fact"
        matches = NUGGET_PATTERN.findall(text)
        assert matches[0][0] == "wook_17_p5"


class TestConfidencePattern:
    def test_basic(self) -> None:
        text = "Confidence: 85"
        match = CONFIDENCE_PATTERN.search(text)
        assert match is not None
        assert match.group(1) == "85"

    def test_with_percent(self) -> None:
        text = "Confidence: 75%"
        match = CONFIDENCE_PATTERN.search(text)
        assert match is not None
        assert match.group(1) == "75"

    def test_lowercase(self) -> None:
        text = "confidence: 60"
        match = CONFIDENCE_PATTERN.search(text)
        assert match is not None


class TestBuildSystemPrompt:
    def test_no_tools(self) -> None:
        prompt = build_system_prompt([])
        assert "(No tools available)" in prompt
        # The unified agentic loop uses an XML-sectioned prompt and records
        # evidence through record_evidence. The NUGGET[pid] free-text format
        # is kept only as a regex fallback.
        assert "<corpus_card>" in prompt
        assert "record_evidence" in prompt

    def test_corpus_enum_matches_mcp_wire_schema(self) -> None:
        """The prompt must advertise the corpus enum values MCP accepts.

        The server accepts only ``all`` / ``wikipedia`` /
        ``wookieepedia``. A prompt that mentions R2C2 or SiBiLS as a
        corpus costs the model a turn on ``Invalid corpus: R2C2``.
        """
        prompt = build_system_prompt([])
        # Positive: each wire-level literal appears at least once.
        assert "`wikipedia`" in prompt
        assert "`wookieepedia`" in prompt
        assert "`all`" in prompt
        # Negative: the LLM echoes any corpus name it reads into
        # search_passages and hits the server-side enum guard. The server
        # tool's description may mention "R2C2 corpus" in prose, but the
        # <tools> and <corpus_card> sections must not imply a ``R2C2`` enum.
        assert "corpus ∈ {r2c2" not in prompt
        assert "corpus ∈ {R2C2" not in prompt
        # format_answer must not appear: the unified loop does not bind it.
        assert "format_answer" not in prompt

    def test_with_tools(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_passages",
                    "description": "Search for passages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        prompt = build_system_prompt(tools)
        assert "search_passages" in prompt
        assert "query, top_k" in prompt


class TestBuildUserMessage:
    def test_no_qtype(self) -> None:
        msg = build_user_message("Who directed Jaws?")
        assert msg == "Who directed Jaws?"

    def test_none_qtype(self) -> None:
        msg = build_user_message("Who directed Jaws?", None)
        assert "[type:" not in msg

    def test_with_qtype(self) -> None:
        msg = build_user_message("Who directed Jaws?", "simple")
        assert msg.startswith("Who directed Jaws?")
        assert msg.endswith("[type: simple]")

    def test_strips_whitespace(self) -> None:
        msg = build_user_message("  Question?  \n", "aggregation")
        assert msg.startswith("Question?")
        assert msg.endswith("[type: aggregation]")


class TestBuildVerificationMessage:
    def test_basic(self) -> None:
        msg = build_verification_user_message(
            question="What about the movie?",
            draft_answer="Anthony Mackie",
            passages=[
                {
                    "id": "p1",
                    "text": "Some passage text",
                    "doc_id": "Q123",
                    "title": "Film",
                },
            ],
        )
        assert "What about the movie?" in msg
        assert "Anthony Mackie" in msg
        assert "Some passage text" in msg
        assert "[Passage p1]" in msg


class TestReformulationTemplates:
    def test_all_types_present(self) -> None:
        expected = {
            "simple",
            "simple_w_condition",
            "comparison",
            "aggregation",
            "multi_hop",
            "post_processing",
            "default",
        }
        assert set(REFORMULATION_TEMPLATES.keys()) == expected

    def test_default_template_has_placeholder(self) -> None:
        template = REFORMULATION_TEMPLATES["default"]
        result = template.format(question="Who directed Jaws?")
        assert "Who directed Jaws?" in result

    def test_all_templates_have_question_placeholder(self) -> None:
        for name, template in REFORMULATION_TEMPLATES.items():
            result = template.format(question="test question")
            assert "test question" in result, f"Template '{name}' missing {{question}}"


class TestPromptConstants:
    def test_system_prompt_has_placeholders(self) -> None:
        assert "{tool_descriptions}" in SYSTEM_PROMPT_TEMPLATE

    def test_synthesize_prompt_nonempty(self) -> None:
        assert len(SYNTHESIZE_PROMPT) > 50

    def test_verifier_prompt_has_format(self) -> None:
        assert "VERDICT:" in VERIFIER_SYSTEM_PROMPT
        assert "CONFIDENCE_ADJUSTMENT:" in VERIFIER_SYSTEM_PROMPT
        assert "SOURCES_SUPPORTING:" in VERIFIER_SYSTEM_PROMPT
        assert "ALTERNATIVE:" in VERIFIER_SYSTEM_PROMPT
