"""Shared dataclasses for the Agent-BiTeM pipeline.

All modules depend on these types. They form the contract between
the LLM client, MCP client, agentic loop, verifier, post-processing,
and orchestrator layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ACResult",
    "CitedPassage",
    "LLMResponse",
    "Nugget",
    "PREntry",
    "Question",
    "QuestionTrace",
    "ToolCall",
    "ToolResult",
    "VerificationResult",
]


@dataclass
class Question:
    """A single R2C2 question from topics.xml or dummy JSON."""

    qid: str
    text: str
    question_type: str = "default"


@dataclass
class ToolCall:
    """A tool call extracted from an LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of dispatching a tool call to an MCP server."""

    tool_call_id: str
    content: dict[str, Any]
    is_error: bool = False


@dataclass
class LLMResponse:
    """Parsed response from an LLM call.

    The ``reasoning`` field holds thinking tokens (mapped from the
    OpenAI SDK's ``reasoning_content`` attribute produced by vLLM's
    ``--reasoning-parser qwen3``).
    """

    content: str
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass
class Nugget:
    """An atomic factual claim extracted from a passage.

    In the final AC file, nuggets reference a passage via
    ``(pr_run_name, passage_rank)`` — the PassageKey.
    """

    passage_id: str
    text: str
    num: int = 0
    pr_run_name: str = ""
    passage_rank: int = 0


@dataclass
class VerificationResult:
    """Output of the 27B verification agent."""

    verdict: str  # "AGREE" | "DISAGREE" | "PARTIAL"
    confidence_adjustment: int  # -30 to +10
    supporting_passages: list[str] = field(default_factory=list)
    contradicting_passages: list[str] = field(default_factory=list)
    reasoning: str = ""
    alternative_answer: str | None = None
    raw_response: str = ""
    thinking: str | None = None


@dataclass
class QuestionTrace:
    """Full trace of answering a single question.

    One trace = one line in ``conversations.jsonl``.
    """

    question_id: str = ""
    question_text: str = ""
    question_type: str = "default"
    mode: str = "b"
    model: str = ""
    timestamp: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    draft_answer: str = ""
    raw_confidence: int = 0
    final_confidence: int = 0
    nuggets: list[Nugget] = field(default_factory=list)
    verification: VerificationResult | None = None
    termination: str = ""  # "natural" | "max_iterations" | "human"
    iterations: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0

    # Helper methods used during loop execution.

    def append_turn(self, message: dict[str, Any]) -> None:
        """Append an assistant or tool message to the trace."""
        self.messages.append(message)

    def append_tool_result(
        self, tool_call: ToolCall, result: dict[str, Any]
    ) -> None:
        """Record a tool call and its result in the trace."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    def add_nuggets(self, nuggets: list[Nugget]) -> None:
        """Add nuggets extracted from thinking tokens."""
        self.nuggets.extend(nuggets)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for conversations.jsonl."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "mode": self.mode,
            "model": self.model,
            "timestamp": self.timestamp,
            "messages": self.messages,
            "draft_answer": self.draft_answer,
            "raw_confidence": self.raw_confidence,
            "final_confidence": self.final_confidence,
            "nuggets": [
                {
                    "passage_id": n.passage_id,
                    "text": n.text,
                    "num": n.num,
                    "pr_run_name": n.pr_run_name,
                    "passage_rank": n.passage_rank,
                }
                for n in self.nuggets
            ],
            "verification": (
                {
                    "verdict": self.verification.verdict,
                    "confidence_adjustment": self.verification.confidence_adjustment,
                    "supporting_passages": self.verification.supporting_passages,
                    "contradicting_passages": self.verification.contradicting_passages,
                    "reasoning": self.verification.reasoning,
                    "alternative_answer": self.verification.alternative_answer,
                    "raw_response": self.verification.raw_response,
                    "thinking": self.verification.thinking,
                }
                if self.verification
                else None
            ),
            "termination": self.termination,
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
            "wall_time_seconds": self.wall_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionTrace:
        """Deserialize from a conversations.jsonl record."""
        verification_data = data.get("verification")
        verification = (
            VerificationResult(**verification_data)
            if verification_data
            else None
        )

        nuggets_data = data.get("nuggets", [])
        nuggets = [Nugget(**n) for n in nuggets_data]

        return cls(
            question_id=data.get("question_id", ""),
            question_text=data.get("question_text", ""),
            question_type=data.get("question_type", "default"),
            mode=data.get("mode", "b"),
            model=data.get("model", ""),
            timestamp=data.get("timestamp", ""),
            messages=data.get("messages", []),
            draft_answer=data.get("draft_answer", ""),
            raw_confidence=data.get("raw_confidence", 0),
            final_confidence=data.get("final_confidence", 0),
            nuggets=nuggets,
            verification=verification,
            termination=data.get("termination", ""),
            iterations=data.get("iterations", 0),
            total_tokens=data.get("total_tokens", 0),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
        )


@dataclass
class PREntry:
    """One passage in the PR submission file."""

    question_id: str
    rank: int
    doc_id: str
    passage_text: str


@dataclass
class ACResult:
    """Per-question result for the AC submission file."""

    answer: str
    confidence: int  # 0-100
    nuggets: list[Nugget] = field(default_factory=list)
    passage_rank_map: dict[str, int] = field(default_factory=dict)


@dataclass
class CitedPassage:
    """A passage cited in a trace, with citation frequency metadata."""

    passage_id: str
    citation_count: int = 0
    first_appearance: int = 0  # message index of first mention
    doc_id: str = ""
    text: str = ""
