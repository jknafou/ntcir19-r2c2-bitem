"""Dummy LLM client with scripted responses for testing.

Returns deterministic responses that exercise the full agentic loop:
call 1 → tool call (search_passages), call 2 → text with NUGGET tags +
confidence.  One script per question type (6 types).
"""

from __future__ import annotations

import logging
from typing import Any

from r2c2_agent.agent.interfaces.types import LLMResponse, ToolCall

__all__ = ["DummyLLMClient"]

log = logging.getLogger(__name__)


# Scripted response sequences, one per question type.


def _search_tool_call(query: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="search_passages",
        arguments={"query": query, "top_k": 5},
    )


def _text_response(
    answer: str,
    confidence: int,
    nuggets: list[tuple[str, str]],
    reasoning: str = "",
) -> LLMResponse:
    nugget_lines = "\n".join(f"NUGGET[{pid}]: {text}" for pid, text in nuggets)
    thinking = (
        f"{reasoning}\n{nugget_lines}\nConfidence: {confidence}"
        if reasoning
        else f"{nugget_lines}\nConfidence: {confidence}"
    )
    content = f"Answer: {answer}\nConfidence: {confidence}"
    return LLMResponse(content=content, reasoning=thinking, tool_calls=[])


def _verifier_response(
    verdict: str = "AGREE",
    adjustment: int = 3,
) -> LLMResponse:
    content = (
        f"VERDICT: {verdict}\n"
        f"CONFIDENCE_ADJUSTMENT: {adjustment}\n"
        f"SOURCES_SUPPORTING: dummy_p1, dummy_p2\n"
        f"SOURCES_CONTRADICTING: NONE\n"
        f"REASONING: The evidence supports the proposed answer.\n"
        f"ALTERNATIVE: N/A"
    )
    return LLMResponse(
        content=content,
        reasoning="Let me check the passages carefully...",
        tool_calls=[],
    )


# Pre-built sequences: list of LLMResponse per question type
# Each sequence is consumed in order by the dummy client.
RESPONSE_SEQUENCES: dict[str, list[LLMResponse]] = {
    "simple": [
        LLMResponse(
            content="",
            reasoning="Simple factual question. Let me search for the answer.",
            tool_calls=[_search_tool_call("Jaws director")],
        ),
        _text_response(
            answer="Steven Spielberg",
            confidence=92,
            nuggets=[("dummy_p1", "Spielberg directed Jaws in 1975")],
            reasoning="Found the answer in the first passage. High confidence.",
        ),
    ],
    "simple_w_condition": [
        LLMResponse(
            content="",
            reasoning="Need to find entity with a condition. Search first.",
            tool_calls=[_search_tool_call("Nolan films after 2010")],
        ),
        _text_response(
            answer="Interstellar (2014)",
            confidence=85,
            nuggets=[
                ("dummy_p2", "Nolan directed Interstellar in 2014"),
                ("dummy_p3", "Interstellar was a science fiction film"),
            ],
            reasoning="Found the film matching the condition.",
        ),
    ],
    "comparison": [
        LLMResponse(
            content="",
            reasoning="Comparison question. Need data on both entities.",
            tool_calls=[_search_tool_call("Titanic runtime length")],
        ),
        LLMResponse(
            content="",
            reasoning="Got Titanic data. Now search for the other film.",
            tool_calls=[_search_tool_call("Avatar runtime length")],
        ),
        _text_response(
            answer="Titanic is longer at 195 minutes vs Avatar at 162 minutes",
            confidence=80,
            nuggets=[
                ("dummy_p4", "Titanic has a runtime of 195 minutes"),
                ("dummy_p5", "Avatar has a runtime of 162 minutes"),
            ],
            reasoning="Both runtimes found. Titanic is longer.",
        ),
    ],
    "aggregation": [
        LLMResponse(
            content="",
            reasoning="Aggregation: need to count. Broad search first.",
            tool_calls=[_search_tool_call("Meryl Streep Oscar wins")],
        ),
        _text_response(
            answer="Meryl Streep has won three Academy Awards",
            confidence=75,
            nuggets=[
                ("dummy_p6", "Streep won Oscar for Kramer vs. Kramer"),
                ("dummy_p7", "Streep won Oscar for Sophie's Choice"),
                ("dummy_p6", "Streep won Oscar for The Iron Lady"),
            ],
            reasoning="Found 3 wins. May have missed some, moderate confidence.",
        ),
    ],
    "multi_hop": [
        LLMResponse(
            content="",
            reasoning="Multi-hop: decompose into sub-questions. First find cast.",
            tool_calls=[_search_tool_call("The Manchurian Candidate cast 2004")],
        ),
        LLMResponse(
            content="",
            reasoning="Found Mackie in the cast. Now check Captain America.",
            tool_calls=[_search_tool_call("Captain America actor MCU")],
        ),
        _text_response(
            answer="Anthony Mackie",
            confidence=88,
            nuggets=[
                (
                    "dummy_p8",
                    "Anthony Mackie appeared in The Manchurian Candidate 2004",
                ),
                ("dummy_p9", "Anthony Mackie portrayed Captain America in the MCU"),
            ],
            reasoning="Two independent sources confirm. Chain complete.",
        ),
    ],
    "post_processing": [
        LLMResponse(
            content="",
            reasoning="Post-processing: need dates to compute duration.",
            tool_calls=[
                _search_tool_call("Thurgood Marshall Supreme Court tenure dates")
            ],
        ),
        _text_response(
            answer="Thurgood Marshall served 8,765 days as Supreme Court Justice",
            confidence=78,
            nuggets=[
                ("dummy_p10", "Marshall appointed to Supreme Court October 2, 1967"),
                ("dummy_p10", "Marshall retired from Supreme Court October 1, 1991"),
            ],
            reasoning="Dates found. Computed: 1967-10-02 to 1991-10-01 = 8765 days.",
        ),
    ],
}

# Default sequence for unknown question types
RESPONSE_SEQUENCES["default"] = RESPONSE_SEQUENCES["simple"]

# Verifier response (used when the dummy client serves the 27B role)
VERIFIER_SEQUENCE: list[LLMResponse] = [_verifier_response()]


class DummyLLMClient:
    """LLM client that returns scripted responses for testing.

    Each ``chat()`` call pops the next response from a per-question
    sequence.  Call ``set_question_type()`` before starting a new
    question to select the appropriate script.

    For verifier calls (no tools), returns a canned AGREE response.

    Args:
        default_type: Question type to use if ``set_question_type``
            is not called.
    """

    def __init__(self, default_type: str = "simple") -> None:
        self._current_type = default_type
        self._call_index = 0
        self._is_verifier = False

    def set_question_type(self, question_type: str) -> None:
        """Select the response script for the next question."""
        self._current_type = question_type
        self._call_index = 0

    def set_verifier_mode(self, is_verifier: bool = True) -> None:
        """Switch to verifier mode (returns AGREE verdict)."""
        self._is_verifier = is_verifier
        self._call_index = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
        user: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        guided_regex: str | None = None,
    ) -> LLMResponse:
        """Return the next scripted response.

        Args:
            messages: Ignored (responses are pre-scripted).
            tools: If empty/None and not in verifier mode, returns a
                text-only response (forced synthesis path).
            enable_thinking: Ignored.
            user: Ignored. Accepted for compatibility with the driver's
                per-request qid/iter/phase hint.
            max_tokens: Ignored. Accepted for signature parity with
                OpenAILLMClient so the finalizer ladder can pass through
                ``FINALIZER_MAX_TOKENS``.
            temperature: Ignored. Accepted for signature parity.
            guided_regex: Ignored. Accepted for signature parity so
                Tier-2+ emission sites can pass their guided_regex
                pattern without a conditional on client type.

        Returns:
            Next LLMResponse in the sequence.
        """
        _ = user  # compatibility-only, DummyLLMClient ignores the vLLM hint
        # The finalizer kwargs are unused: the responses are scripted
        # whatever these hints say.
        _ = max_tokens
        _ = temperature
        _ = guided_regex
        if self._is_verifier:
            return VERIFIER_SEQUENCE[0]

        # If no tools provided, force text response (synthesis path)
        if not tools:
            seq = RESPONSE_SEQUENCES.get(
                self._current_type,
                RESPONSE_SEQUENCES["default"],
            )
            # Return the last response (the text answer)
            return seq[-1]

        seq = RESPONSE_SEQUENCES.get(
            self._current_type,
            RESPONSE_SEQUENCES["default"],
        )

        if self._call_index >= len(seq):
            # Exhausted script — return last (text) response again
            log.warning(
                "DummyLLMClient: exhausted script for type=%s, repeating last",
                self._current_type,
            )
            return seq[-1]

        response = seq[self._call_index]
        self._call_index += 1
        return response
