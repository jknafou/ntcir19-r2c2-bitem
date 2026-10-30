"""Tests for the post-processing pipeline (postprocess.py).

Tests passage extraction, ranking, PR/AC formatting, and validation.
"""

from __future__ import annotations

import json

from r2c2_agent.agent.interfaces.types import (
    ACResult,
    CitedPassage,
    Nugget,
    PREntry,
    QuestionTrace,
)
from r2c2_agent.agent.postprocess import (
    extract_cited_passages,
    format_ac_file,
    format_ac_result,
    format_pr_file,
    rank_passages,
    validate_submission,
)

# Helpers to build test traces


def _make_tool_result(
    tool_call_id: str,
    passages: list[dict],
) -> dict:
    """Build a tool result message with passage data."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps({"passages": passages}),
    }


def _make_single_passage_result(
    tool_call_id: str,
    passage_id: str,
    text: str = "passage text",
    doc_id: str = "DOC-1",
) -> dict:
    """Build a single-passage tool result."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(
            {
                "passage_id": passage_id,
                "text": text,
                "doc_id": doc_id,
            }
        ),
    }


def _make_assistant_msg(
    content: str = "",
    reasoning: str | None = None,
) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning"] = reasoning
    return msg


def _trace_with_messages(messages: list[dict], **kwargs) -> QuestionTrace:
    """Build a QuestionTrace with given messages."""
    trace = QuestionTrace(
        question_id=kwargs.get("question_id", "Q001"),
        question_text=kwargs.get("question_text", "test question"),
        messages=messages,
        draft_answer=kwargs.get("draft_answer", "test answer"),
        raw_confidence=kwargs.get("raw_confidence", 80),
        final_confidence=kwargs.get("final_confidence", 75),
    )
    if "nuggets" in kwargs:
        trace.nuggets = kwargs["nuggets"]
    return trace


class TestExtractCitedPassages:
    """Tests for extract_cited_passages."""

    def test_empty_trace(self):
        trace = _trace_with_messages([])
        result = extract_cited_passages(trace)
        assert result == []

    def test_nuggets_in_reasoning(self):
        """NUGGET tags in reasoning increment citation count."""
        messages = [
            _make_assistant_msg(
                reasoning="NUGGET[P-001]: fact one\nNUGGET[P-001]: fact two\nNUGGET[P-002]: other fact"
            ),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        by_id = {c.passage_id: c for c in cited}
        assert "P-001" in by_id
        assert by_id["P-001"].citation_count == 2
        assert "P-002" in by_id
        assert by_id["P-002"].citation_count == 1

    def test_passages_from_tool_results(self):
        """Tool results with passages are counted."""
        messages = [
            _make_tool_result(
                "tc1",
                [
                    {"passage_id": "P-001", "text": "hello", "doc_id": "D1"},
                    {"passage_id": "P-002", "text": "world", "doc_id": "D2"},
                ],
            ),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        by_id = {c.passage_id: c for c in cited}
        assert len(by_id) == 2
        assert by_id["P-001"].doc_id == "D1"
        assert by_id["P-001"].text == "hello"

    def test_single_passage_tool_result(self):
        """Single-passage format (get_passage) is extracted."""
        messages = [
            _make_single_passage_result("tc1", "P-100", "some text", "DOC-5"),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        assert len(cited) == 1
        assert cited[0].passage_id == "P-100"
        assert cited[0].doc_id == "DOC-5"

    def test_combined_sources(self):
        """Nuggets + tool results accumulate correctly."""
        messages = [
            _make_tool_result(
                "tc1",
                [
                    {"passage_id": "P-001", "text": "t1", "doc_id": "D1"},
                ],
            ),
            _make_assistant_msg(reasoning="NUGGET[P-001]: important fact"),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        by_id = {c.passage_id: c for c in cited}
        # 1 from tool result + 1 from nugget tag + 1 from text mention in reasoning
        assert by_id["P-001"].citation_count == 3

    def test_first_appearance_tracking(self):
        """First appearance is recorded as message index."""
        messages = [
            _make_assistant_msg(content="filler"),  # idx 0
            _make_tool_result(
                "tc1",
                [  # idx 1
                    {"passage_id": "P-001", "text": "t1", "doc_id": "D1"},
                ],
            ),
            _make_assistant_msg(  # idx 2
                reasoning="NUGGET[P-002]: late fact"
            ),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        by_id = {c.passage_id: c for c in cited}
        assert by_id["P-001"].first_appearance == 1
        assert by_id["P-002"].first_appearance == 2

    def test_nuggets_in_content(self):
        """NUGGET tags in assistant content (not just reasoning) are counted."""
        messages = [
            _make_assistant_msg(
                content="NUGGET[P-003]: inline fact",
            ),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        by_id = {c.passage_id: c for c in cited}
        assert "P-003" in by_id
        assert by_id["P-003"].citation_count == 1

    def test_malformed_tool_result_skipped(self):
        """Non-JSON tool results are safely skipped."""
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "not json"},
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)
        assert cited == []

    def test_text_mentions_only_count_for_known_passages(self):
        """Free-text passage ID mentions only count if passage seen in tool results."""
        messages = [
            _make_tool_result(
                "tc1",
                [
                    {"passage_id": "P-001", "text": "t1", "doc_id": "D1"},
                ],
            ),
            _make_assistant_msg(
                reasoning="I looked at P-001 again and it confirms the fact"
            ),
        ]
        trace = _trace_with_messages(messages)
        cited = extract_cited_passages(trace)

        by_id = {c.passage_id: c for c in cited}
        # 1 from tool + 1 from text mention
        assert by_id["P-001"].citation_count == 2


class TestRankPassages:
    """Tests for rank_passages."""

    def test_sort_by_frequency(self):
        cited = [
            CitedPassage(passage_id="A", citation_count=1, first_appearance=0),
            CitedPassage(passage_id="B", citation_count=3, first_appearance=1),
            CitedPassage(passage_id="C", citation_count=2, first_appearance=2),
        ]
        ranked = rank_passages(cited)
        assert [p.passage_id for p in ranked] == ["B", "C", "A"]

    def test_tie_broken_by_first_appearance(self):
        cited = [
            CitedPassage(passage_id="A", citation_count=2, first_appearance=5),
            CitedPassage(passage_id="B", citation_count=2, first_appearance=1),
        ]
        ranked = rank_passages(cited)
        assert [p.passage_id for p in ranked] == ["B", "A"]

    def test_max_passages_limit(self):
        cited = [
            CitedPassage(passage_id=f"P{i}", citation_count=100 - i, first_appearance=i)
            for i in range(30)
        ]
        ranked = rank_passages(cited, max_passages=20)
        assert len(ranked) == 20

    def test_empty_input(self):
        assert rank_passages([]) == []


class TestFormatPREntries:
    """Tests for PR entry formatting."""

    def test_basic_format(self):
        from r2c2_agent.agent.postprocess import format_pr_entries

        passages = [
            CitedPassage(
                passage_id="P1", citation_count=2, doc_id="D1", text="hello world"
            ),
            CitedPassage(
                passage_id="P2", citation_count=1, doc_id="D2", text="goodbye"
            ),
        ]
        entries = format_pr_entries("Q001", passages)

        assert len(entries) == 2
        assert entries[0].rank == 1
        assert entries[0].doc_id == "D1"
        assert entries[1].rank == 2

    def test_whitespace_sanitization(self):
        from r2c2_agent.agent.postprocess import format_pr_entries

        passages = [
            CitedPassage(
                passage_id="P1",
                citation_count=1,
                doc_id="D1",
                text="line one\nline two\ttabbed  double-space",
            ),
        ]
        entries = format_pr_entries("Q001", passages)
        assert "\n" not in entries[0].passage_text
        assert "\t" not in entries[0].passage_text
        assert "  " not in entries[0].passage_text


class TestFormatACResult:
    """Tests for AC result formatting."""

    def test_basic_ac_result(self):
        trace = _trace_with_messages(
            [],
            draft_answer="42",
            final_confidence=85,
            nuggets=[
                Nugget(passage_id="P1", text="fact one"),
                Nugget(passage_id="P2", text="fact two"),
            ],
        )
        passage_rank_map = {"P1": 1, "P2": 3}

        result = format_ac_result(trace, "BITEM-PG-1", passage_rank_map)
        assert result.answer == "42"
        assert result.confidence == 85
        assert len(result.nuggets) == 2
        assert result.nuggets[0].num == 1
        assert result.nuggets[0].pr_run_name == "BITEM-PG-1"
        assert result.nuggets[0].passage_rank == 1

    def test_nuggets_without_pr_passage_dropped(self):
        trace = _trace_with_messages(
            [],
            draft_answer="answer",
            final_confidence=70,
            nuggets=[
                Nugget(passage_id="P1", text="fact one"),
                Nugget(passage_id="P-MISSING", text="orphan fact"),
            ],
        )
        passage_rank_map = {"P1": 1}

        result = format_ac_result(trace, "BITEM-PG-1", passage_rank_map)
        assert len(result.nuggets) == 1
        assert result.nuggets[0].passage_id == "P1"

    def test_max_nuggets_enforced(self):
        nuggets = [Nugget(passage_id="P1", text=f"fact {i}") for i in range(15)]
        trace = _trace_with_messages(
            [],
            draft_answer="answer",
            final_confidence=70,
            nuggets=nuggets,
        )
        passage_rank_map = {"P1": 1}

        result = format_ac_result(trace, "RUN", passage_rank_map, max_nuggets=10)
        assert len(result.nuggets) == 10

    def test_duplicate_nuggets_deduplicated(self):
        nuggets = [
            Nugget(passage_id="P1", text="same fact"),
            Nugget(passage_id="P1", text="Same Fact"),  # case-insensitive dup
            Nugget(passage_id="P1", text="different fact"),
        ]
        trace = _trace_with_messages(
            [],
            draft_answer="answer",
            final_confidence=70,
            nuggets=nuggets,
        )
        result = format_ac_result(trace, "RUN", {"P1": 1})
        assert len(result.nuggets) == 2


class TestFileFormatting:
    """Tests for PR/AC file formatting."""

    def test_pr_file_format(self):
        entries = {
            "Q002": [PREntry("Q002", 1, "D1", "text two")],
            "Q001": [
                PREntry("Q001", 1, "D1", "text one"),
                PREntry("Q001", 2, "D2", "text two"),
            ],
        }
        content = format_pr_file(entries)
        lines = content.strip().split("\n")
        # Q001 before Q002 (sorted)
        assert lines[0] == "Q001;1;D1;text one"
        assert lines[1] == "Q001;2;D2;text two"
        assert lines[2] == "Q002;1;D1;text two"

    def test_ac_file_format(self):
        results = {
            "Q001": ACResult(
                answer="42",
                confidence=85,
                nuggets=[
                    Nugget(
                        passage_id="P1",
                        text="fact",
                        num=1,
                        pr_run_name="RUN",
                        passage_rank=1,
                    ),
                ],
            ),
        }
        content = format_ac_file(results)
        assert "<Q001>" in content
        assert "42;85" in content
        assert "1;RUN;1;fact" in content
        assert "</Q001>" in content

    def test_pr_file_empty(self):
        assert format_pr_file({}) == ""

    def test_ac_file_empty(self):
        assert format_ac_file({}) == ""

    def test_ac_file_multiple_questions_sorted(self):
        results = {
            "Q003": ACResult(answer="c", confidence=60, nuggets=[]),
            "Q001": ACResult(answer="a", confidence=80, nuggets=[]),
        }
        content = format_ac_file(results)
        q1_pos = content.index("<Q001>")
        q3_pos = content.index("<Q003>")
        assert q1_pos < q3_pos

    def test_pr_semicolons_in_passage_text(self):
        """Semicolons in passage text are safe (parser takes first 3 only)."""
        entries = {
            "Q001": [PREntry("Q001", 1, "D1", "text;with;semicolons;here")],
        }
        content = format_pr_file(entries)
        line = content.strip()
        parts = line.split(";", 3)
        assert parts[0] == "Q001"
        assert parts[1] == "1"
        assert parts[2] == "D1"
        assert parts[3] == "text;with;semicolons;here"


class TestValidateSubmission:
    """Tests for submission validation."""

    def test_valid_submission(self):
        pr = {"Q001": [PREntry("Q001", 1, "D1", "text")]}
        ac = {
            "Q001": ACResult(
                answer="42",
                confidence=85,
                nuggets=[
                    Nugget(
                        passage_id="P1",
                        text="fact",
                        num=1,
                        pr_run_name="RUN",
                        passage_rank=1,
                    ),
                ],
            ),
        }
        errors = validate_submission(pr, ac, "RUN")
        assert errors == []

    def test_too_many_pr_passages(self):
        pr = {"Q001": [PREntry("Q001", i, "D1", f"text{i}") for i in range(1, 25)]}
        ac = {"Q001": ACResult(answer="x", confidence=50, nuggets=[])}
        errors = validate_submission(pr, ac, "RUN")
        assert any("24 PR passages" in e for e in errors)

    def test_empty_answer(self):
        pr = {"Q001": [PREntry("Q001", 1, "D1", "text")]}
        ac = {"Q001": ACResult(answer="  ", confidence=50, nuggets=[])}
        errors = validate_submission(pr, ac, "RUN")
        assert any("empty answer" in e for e in errors)

    def test_confidence_out_of_range(self):
        pr = {"Q001": [PREntry("Q001", 1, "D1", "text")]}
        ac = {"Q001": ACResult(answer="x", confidence=150, nuggets=[])}
        errors = validate_submission(pr, ac, "RUN")
        assert any("out of range" in e for e in errors)

    def test_too_many_nuggets(self):
        pr = {"Q001": [PREntry("Q001", 1, "D1", "text")]}
        nuggets = [
            Nugget(
                passage_id="P1", text=f"f{i}", num=i, pr_run_name="RUN", passage_rank=1
            )
            for i in range(1, 13)
        ]
        ac = {"Q001": ACResult(answer="x", confidence=50, nuggets=nuggets)}
        errors = validate_submission(pr, ac, "RUN")
        assert any("12 nuggets" in e for e in errors)

    def test_wrong_run_name(self):
        pr = {"Q001": [PREntry("Q001", 1, "D1", "text")]}
        ac = {
            "Q001": ACResult(
                answer="x",
                confidence=50,
                nuggets=[
                    Nugget(
                        passage_id="P1",
                        text="f",
                        num=1,
                        pr_run_name="WRONG",
                        passage_rank=1,
                    ),
                ],
            ),
        }
        errors = validate_submission(pr, ac, "RUN")
        assert any("WRONG" in e for e in errors)

    def test_invalid_passage_rank_reference(self):
        pr = {"Q001": [PREntry("Q001", 1, "D1", "text")]}
        ac = {
            "Q001": ACResult(
                answer="x",
                confidence=50,
                nuggets=[
                    Nugget(
                        passage_id="P1",
                        text="f",
                        num=1,
                        pr_run_name="RUN",
                        passage_rank=99,
                    ),
                ],
            ),
        }
        errors = validate_submission(pr, ac, "RUN")
        assert any("rank 99" in e for e in errors)
