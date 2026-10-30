"""Core agentic loop for the R2C2 pipeline.

Call the LLM, extract tool calls, dispatch them to MCP, append the
results, and repeat until natural termination or max iterations.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from typing import Any

from r2c2_agent.agent.config import LoopConfig
from r2c2_agent.agent.interfaces.llm import LLMClientProtocol
from r2c2_agent.agent.interfaces.mcp import MCPClientProtocol
from r2c2_agent.agent.interfaces.types import (
    LLMResponse,
    Nugget,
    Question,
    QuestionTrace,
    ToolCall,
)
from r2c2_agent.agent.loop_hooks import DraftContext
from r2c2_agent.agent.prompt import (
    ANSWER_PATTERN,
    CONFIDENCE_PATTERN,
    NUGGET_PATTERN,
    SYNTHESIZE_PROMPT,
    build_system_prompt,
    build_user_message,
)

__all__ = [
    "answer_question",
    "answer_question_pipeline",
    "compact_messages",
    "count_tokens",
    "extract_confidence",
    "extract_nuggets_from_response",
    "extract_nuggets_from_thinking",
]

log = logging.getLogger(__name__)


# Fraction of ``config.token_budget`` above which the driver compacts the
# conversation before the next LLM call.
PREFLIGHT_COMPACT_FRAC = 0.75
# Fraction of ``config.token_budget`` above which the driver stops the
# loop before the next LLM call. The ``count_tokens`` estimator
# (char/3.5) can under-count tool-JSON-heavy conversations by up to
# ~25% because short keys + digits tokenize at ~1 token/2-2.5 chars.
# 0.75 * 124518 estimated-tokens = 93388 estimated ≈ 116735 real tokens
# worst-case, which stays under the 131072 vLLM cap.
PREFLIGHT_ABORT_FRAC = 0.75
# Absolute floor on ``trace.raw_confidence`` (0-100 scale). Mirrors the
# 0.05 floor in ``pipeline_handlers.CONFIDENCE_FLOOR``.
CONFIDENCE_FLOOR_INT = 5


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count using char/3.5 approximation.

    Args:
        messages: Conversation messages.

    Returns:
        Estimated token count.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        reasoning = msg.get("reasoning", "")
        if isinstance(reasoning, str):
            total_chars += len(reasoning)
    return int(total_chars / 3.5)


def compact_messages(
    messages: list[dict[str, Any]],
    keep_recent: int = 3,
) -> list[dict[str, Any]]:
    """Reduce message list to fit context budget.

    Three-phase eviction:
      1. Old tool results → truncated summary
      2. Old thinking blocks → extract NUGGETs, drop rest
      3. Drop early turns (keep system + user query)

    Always preserves: system prompt (index 0), user query (index 1),
    and the last ``keep_recent`` round-trips.

    Args:
        messages: Full conversation history.
        keep_recent: Number of recent messages to preserve fully.

    Returns:
        Compacted message list.
    """
    if len(messages) <= 2 + keep_recent:
        return messages

    # Boundary: messages to compact vs preserve
    preserve_start = max(2, len(messages) - keep_recent)
    head = messages[:2]  # system + user query
    middle = messages[2:preserve_start]
    tail = messages[preserve_start:]

    compacted_middle: list[dict[str, Any]] = []
    for msg in middle:
        role = msg.get("role", "")

        # Phase 1: truncate tool results
        if role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                compacted_middle.append(
                    {
                        **msg,
                        "content": "[truncated tool result]",
                    }
                )
            else:
                compacted_middle.append(msg)
            continue

        # Phase 2: compact assistant thinking — preserve NUGGETs
        if role == "assistant":
            reasoning = msg.get("reasoning", "")
            if isinstance(reasoning, str) and reasoning:
                nugget_matches = NUGGET_PATTERN.findall(reasoning)
                if nugget_matches:
                    preserved = "\n".join(
                        f"NUGGET[{pid}]: {text}" for pid, text in nugget_matches
                    )
                    compacted_middle.append(
                        {
                            **msg,
                            "reasoning": (
                                f"[compacted thinking — nuggets preserved]\n{preserved}"
                            ),
                        }
                    )
                else:
                    compacted_middle.append(
                        {
                            **msg,
                            "reasoning": "[compacted — no nuggets]",
                        }
                    )
            else:
                compacted_middle.append(msg)
            continue

        compacted_middle.append(msg)

    return head + compacted_middle + tail


def extract_nuggets_from_thinking(thinking: str) -> list[Nugget]:
    """Extract NUGGET[passage_id]: text from thinking tokens.

    Args:
        thinking: Raw thinking/reasoning text from the LLM.

    Returns:
        List of Nugget objects.
    """
    matches = NUGGET_PATTERN.findall(thinking)
    return [Nugget(passage_id=pid.strip(), text=text.strip()) for pid, text in matches]


def extract_nuggets_from_response(response: LLMResponse) -> list[Nugget]:
    """Extract NUGGET tags from both reasoning and content fields.

    Qwen3.5 may place NUGGETs in either the reasoning (thinking) tokens
    or the visible content, depending on model configuration and whether
    reasoning mode is active.  This function searches both fields to
    ensure nuggets are always captured.

    Args:
        response: Parsed LLM response.

    Returns:
        Deduplicated list of Nugget objects.
    """
    nuggets: list[Nugget] = []
    seen: set[tuple[str, str]] = set()

    for text in [response.reasoning or "", response.content]:
        if not text:
            continue
        for pid, claim in NUGGET_PATTERN.findall(text):
            key = (pid.strip(), claim.strip())
            if key not in seen:
                seen.add(key)
                nuggets.append(Nugget(passage_id=key[0], text=key[1]))

    return nuggets


def extract_answer_and_confidence_from_response(
    response: LLMResponse,
) -> tuple[str, int]:
    """Extract answer and confidence from an LLM response.

    Algorithm (see ``src/r2c2_agent/shared/answer_extraction.py``):
    1. Strip markdown from the entire content
    2. Find the last ``Answer:`` occurrence
    3. Extract text between that ``Answer:`` and ``Confidence:``
    4. Parse confidence as int
    5. If answer is empty, fall back to full stripped content

    Args:
        response: Parsed LLM response.

    Returns:
        Tuple of (answer_string, confidence_score).
    """
    try:
        from r2c2_agent.shared.answer_extraction import extract_answer_and_confidence

        return extract_answer_and_confidence(response.content)
    except ImportError:
        # Fallback if shared module not on path — use basic extraction
        match = ANSWER_PATTERN.search(response.content)
        answer = match.group(1).strip() if match else response.content.strip()
        conf_match = CONFIDENCE_PATTERN.search(response.content)
        confidence = int(conf_match.group(1)) if conf_match else 50
        return answer, max(0, min(100, confidence))


# Thin wrappers kept for external callers.
def extract_answer(response: LLMResponse) -> str:
    """Extract answer from LLM response."""
    answer, _ = extract_answer_and_confidence_from_response(response)
    return answer


def extract_confidence(response: LLMResponse) -> int:
    """Extract confidence from LLM response.

    Scans ``response.content`` first (canonical location for the
    ``Confidence: N`` token). If content has no match, falls back to
    ``response.reasoning`` — Qwen3.5 may emit the confidence line inside
    the thinking trace when reasoning mode is active (see
    :func:`extract_nuggets_from_response` for the same content+reasoning
    scan precedent). Returns the default ``50`` only when neither field
    contains a ``Confidence:`` match.

    Args:
        response: Parsed LLM response.

    Returns:
        Confidence integer, clamped to ``[0, 100]``.
    """
    _, confidence = extract_answer_and_confidence_from_response(response)
    # ``extract_answer_and_confidence`` returns 50 both when no
    # Confidence: token is found *and* when an explicit "Confidence: 50"
    # is present. Disambiguate by re-checking content for any match.
    if CONFIDENCE_PATTERN.search(response.content or ""):
        return confidence
    reasoning = response.reasoning or ""
    conf_match = CONFIDENCE_PATTERN.search(reasoning)
    if conf_match:
        try:
            return max(0, min(100, int(conf_match.group(1))))
        except (TypeError, ValueError):
            return confidence
    return confidence


_REFORMULATION_PROMPT = (
    "Your previous response did not include a properly formatted answer. "
    "Based on your analysis above, please provide your final answer.\n\n"
    "Rules:\n"
    "- Answer MUST be a single line (no line breaks)\n"
    "- For list questions, use comma-separated values on one line\n"
    "- For numeric questions, just provide the number\n"
    "- Include both Answer and Confidence on separate lines\n\n"
    "Format:\n"
    "Answer: <your answer on a single line>\n"
    "Confidence: <integer 0-100>"
)


def _needs_reformulation(answer: str, raw_content: str) -> bool:
    """Check if the extracted answer needs LLM reformulation.

    Triggers on:
    - Empty answer (extraction failed entirely)
    - Multi-line answer (contains newlines — bad format for R2C2)
    """
    if not answer.strip():
        return True
    if "\n" in answer:
        return True
    return False


async def _reformulate_answer(
    llm_client: LLMClientProtocol,
    messages: list[dict[str, Any]],
    trace: QuestionTrace,
    config: LoopConfig,
) -> tuple[str, int] | None:
    """Ask the LLM to restate its answer in the correct format.

    Injects a user message asking for format correction, then extracts
    the answer from the model's response. Both turns are appended to
    messages and trace.

    Returns (answer, confidence) or None if reformulation also fails.
    """
    log.info("Answer needs reformulation — requesting format correction")

    # Inject reformulation prompt into full history
    reformulation_msg = {"role": "user", "content": _REFORMULATION_PROMPT}
    messages.append(reformulation_msg)
    trace.append_turn(reformulation_msg)

    # But only send the last assistant message + prompt to the LLM
    # (no need to resend the entire conversation for a format fix)
    last_assistant = None
    for msg in reversed(messages[:-1]):  # skip the reformulation msg itself
        if msg.get("role") == "assistant":
            last_assistant = msg
            break
    short_messages = []
    if last_assistant:
        content = last_assistant.get("content", "") or ""
        reasoning = last_assistant.get("reasoning", "") or ""
        if reasoning:
            content = f"{reasoning}\n\n{content}"
        short_messages.append(
            {
                "role": "user",
                "content": (
                    f"Here is your previous response:\n\n{content}\n\n"
                    f"{_REFORMULATION_PROMPT}"
                ),
            }
        )
    else:
        short_messages.append(reformulation_msg)

    try:
        reformulated = await llm_client.chat(
            messages=short_messages,
            enable_thinking=True,
        )
    except Exception:
        log.warning("Reformulation LLM call failed", exc_info=True)
        return None

    reform_msg = _build_assistant_message(reformulated)
    messages.append(reform_msg)
    trace.append_turn(reform_msg)

    new_answer, new_confidence = extract_answer_and_confidence_from_response(
        reformulated
    )

    if not new_answer.strip():
        log.warning("Reformulation still produced empty answer")
        return None

    log.info("Reformulation successful: %s (conf=%d)", new_answer[:50], new_confidence)
    return new_answer, new_confidence


def _handle_format_answer(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle the local format_answer tool call.

    Returns the formatted answer as a tool result dict. Falls back
    gracefully if the shared module is not importable.
    """
    try:
        from r2c2_agent.shared.answer_normalization import format_answer_tool_handler

        formatted = format_answer_tool_handler(
            question_type=arguments.get("question_type", "simple"),
            draft_answer=arguments.get("draft_answer", ""),
            answer_format=arguments.get("answer_format", "single_entity"),
            items=arguments.get("items"),
        )
        return {"formatted_answer": formatted}
    except ImportError:
        # Shared module not on path — return draft unchanged
        return {"formatted_answer": arguments.get("draft_answer", "")}


def _sanitize_tool_call(
    tc: ToolCall,
    known_tools: set[str],
) -> ToolCall | None:
    """Validate a tool call. Returns None if invalid.

    Args:
        tc: Tool call from LLM response.
        known_tools: Set of valid tool names.

    Returns:
        The tool call if valid, or None if it should be skipped.
    """
    if tc.name not in known_tools:
        log.warning("Unknown tool: %s (known: %s)", tc.name, known_tools)
        return None
    if "_parse_error" in tc.arguments:
        log.warning("Malformed tool call arguments for %s", tc.name)
        # Still return it — the loop will create an error tool result
        return tc
    return tc


def _build_assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Build an assistant message dict from an LLMResponse.

    Preserves reasoning tokens and tool calls in the message format
    expected by OpenAI-compatible APIs.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": response.content}

    if response.reasoning:
        msg["reasoning"] = response.reasoning

    if response.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in response.tool_calls
        ]

    return msg


async def answer_question(
    question: Question,
    llm_client: LLMClientProtocol,
    mcp_client: MCPClientProtocol,
    tools: list[dict[str, Any]],
    config: LoopConfig,
) -> QuestionTrace:
    """Run the main agentic loop for a single question.

    Call the LLM, dispatch tool calls, and repeat until natural
    termination or max iterations.

    Args:
        question: The question to answer.
        llm_client: LLM client (real or dummy).
        mcp_client: MCP client (real or dummy).
        tools: Tool definitions in OpenAI format.
        config: Loop configuration.

    Returns:
        Complete QuestionTrace with answer, confidence, nuggets.
    """
    start_time = time.monotonic()

    # Render the minimal template so the LLM is not told about the
    # unified-pipeline tools (record_evidence/submit_answer/…) that
    # this dispatcher cannot route.
    system_prompt = build_system_prompt(tools, pipeline="legacy")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.text},
    ]

    trace = QuestionTrace(
        question_id=question.qid,
        question_text=question.text,
        question_type=question.question_type,
        mode="b",
    )

    known_tools = (
        {t.get("function", t).get("name", "") for t in tools} if tools else set()
    )

    iteration = 0

    # Counter and audit trail for the optional on_draft callback. Both are
    # set on the trace below so downstream code can consult them without
    # a change to the QuestionTrace schema.
    self_correction_injections = 0
    verifier_verdicts: list[dict[str, Any]] = []
    # Hard cap on injected reflections per trace.
    cp2_max_injections = 5

    while True:
        token_count = count_tokens(messages)
        if token_count > config.context_budget:
            log.info(
                "Context budget exceeded (%d > %d), compacting",
                token_count,
                config.context_budget,
            )
            messages = compact_messages(messages)
            token_count = count_tokens(messages)
            if token_count > config.context_budget:
                log.warning("Still over budget after compaction")

        # When the context is nearly full, ask the model for a final answer
        # from the evidence it has instead of killing the trace. 90% of the
        # budget leaves room for the forced-answer response.
        if token_count > int(config.token_budget * 0.9):
            if trace.termination == "token_budget_forced_answer":
                # Already asked once, model didn't stop — hard kill
                log.warning("Forced answer failed, killing trace")
                trace.draft_answer = ""
                trace.raw_confidence = 0
                trace.termination = "token_budget_exceeded"
                break
            log.warning(
                "Token budget exceeded (%d > %d) — forcing final answer",
                token_count,
                config.token_budget,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "IMPORTANT: You are running out of context. Stop searching "
                        "and provide your best answer NOW based on what you have "
                        "found so far.\n\n"
                        "Answer: <your answer on a single line>\n"
                        "Confidence: <integer 0-100>"
                    ),
                }
            )
            trace.termination = "token_budget_forced_answer"

        # After a forced-answer prompt, disable tools so the model has to
        # produce a text answer instead of continuing to make tool calls.
        forced = trace.termination == "token_budget_forced_answer"
        response = await llm_client.chat(
            messages=messages,
            tools=None if forced else (tools if tools else None),
            enable_thinking=True,
        )

        assistant_msg = _build_assistant_message(response)
        messages.append(assistant_msg)
        trace.append_turn(assistant_msg)

        nuggets = extract_nuggets_from_response(response)
        if nuggets:
            trace.add_nuggets(nuggets)

        # Forced-answer safety: if the model was asked for a final answer and
        # still made tool calls (not expected with tools=None), end the trace.
        if forced and response.tool_calls:
            log.warning("Model made tool calls after forced-answer prompt — hard kill")
            draft_answer_text, _ = extract_answer_and_confidence_from_response(response)
            trace.draft_answer = draft_answer_text or ""
            trace.raw_confidence = 0
            trace.termination = "token_budget_exceeded"
            break

        # Natural termination: no tool calls.
        if not response.tool_calls:
            draft_answer_text, draft_confidence = (
                extract_answer_and_confidence_from_response(response)
            )

            # If the answer is empty or contains newlines, ask the model
            # to restate in the correct format. One attempt only.
            if _needs_reformulation(draft_answer_text, response.content):
                reformulated = await _reformulate_answer(
                    llm_client, messages, trace, config
                )
                if reformulated is not None:
                    draft_answer_text, draft_confidence = reformulated

            trace.draft_answer = draft_answer_text
            trace.raw_confidence = draft_confidence

            # If the caller supplied an on_draft callback, give it a
            # chance to inject a reflection turn and keep the loop going,
            # up to cp2_max_injections times per trace.
            if config.on_draft is not None:
                recent_tool_msgs = [
                    str(m.get("content", ""))
                    for m in messages
                    if m.get("role") == "tool"
                ][-3:]
                ctx = DraftContext(
                    question=question.text,
                    # expected_answer is plumbed via the caller's closure;
                    # the loop itself has no access to gold.
                    expected_answer="",
                    draft_answer=draft_answer_text,
                    retrieved_passages=recent_tool_msgs,
                    injections_so_far=self_correction_injections,
                )
                try:
                    decision = await config.on_draft(ctx)
                except Exception as hook_exc:  # fail-open
                    log.warning("on_draft callback failed: %s", hook_exc)
                    decision = None

                if decision is not None:
                    if decision.verdict_info:
                        verifier_verdicts.append(decision.verdict_info)
                        vi = decision.verdict_info
                        log.info(
                            "CP2 verifier call: verdict=%s conf=%.2f "
                            "inject=%s abort=%s injections=%d/%d",
                            vi.get("verdict", "?"),
                            float(vi.get("confidence", 0.0)),
                            decision.inject,
                            decision.abort,
                            self_correction_injections + (1 if decision.inject else 0),
                            cp2_max_injections,
                        )
                    if decision.abort:
                        trace.termination = "cp2_abort"
                        break
                    if (
                        decision.inject
                        and decision.reflection_text
                        and self_correction_injections < cp2_max_injections
                    ):
                        messages.append(
                            {
                                "role": "user",
                                "content": decision.reflection_text,
                                "_injected_reflection": True,
                            }
                        )
                        self_correction_injections += 1
                        # No break: the agent takes another turn.
                        continue
                    elif (
                        decision.inject
                        and self_correction_injections >= cp2_max_injections
                    ):
                        # Cap exceeded — abort instead of accepting a
                        # natural miss. Saves tokens and prevents the
                        # student from learning confident wrong answers.
                        log.info(
                            "CP2 injection cap exceeded (%d >= %d), aborting",
                            self_correction_injections,
                            cp2_max_injections,
                        )
                        trace.termination = "injection_cap_exceeded"
                        break

            trace.termination = "natural"
            break

        iteration += 1
        if iteration >= config.max_iterations:
            log.info(
                "Max iterations (%d) reached, forcing synthesis", config.max_iterations
            )
            messages.append({"role": "user", "content": SYNTHESIZE_PROMPT})

            final = await llm_client.chat(
                messages=messages,
                tools=None,  # No tools → forces text
                enable_thinking=True,
            )

            final_msg = _build_assistant_message(final)
            messages.append(final_msg)
            trace.append_turn(final_msg)

            final_nuggets = extract_nuggets_from_response(final)
            if final_nuggets:
                trace.add_nuggets(final_nuggets)

            final_answer, final_confidence = (
                extract_answer_and_confidence_from_response(final)
            )

            # Same format reformulation as on natural termination.
            if _needs_reformulation(final_answer, final.content):
                reformulated = await _reformulate_answer(
                    llm_client, messages, trace, config
                )
                if reformulated is not None:
                    final_answer, final_confidence = reformulated

            trace.draft_answer = final_answer
            trace.raw_confidence = final_confidence
            trace.termination = "max_iterations"
            break

        # Dispatch tool calls.
        for tc in response.tool_calls:
            valid_tc = _sanitize_tool_call(tc, known_tools)

            if valid_tc is None:
                # Unknown tool — create error result
                trace.append_tool_result(tc, {"error": f"Unknown tool: {tc.name}"})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {tc.name}"}),
                    }
                )
                continue

            if "_parse_error" in valid_tc.arguments:
                # Malformed arguments — create error result
                error_result = {
                    "error": (
                        f"Malformed arguments: {valid_tc.arguments['_parse_error']}"
                    ),
                    "tool": valid_tc.name,
                }
                trace.append_tool_result(valid_tc, error_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": valid_tc.id,
                        "content": json.dumps(error_result),
                    }
                )
                continue

            # Local tool handlers (no MCP round-trip)
            if valid_tc.name == "format_answer":
                result = _handle_format_answer(valid_tc.arguments)
            else:
                result = await mcp_client.call_tool(
                    name=valid_tc.name,
                    arguments=valid_tc.arguments,
                    timeout=config.tool_timeout,
                )

            trace.append_tool_result(valid_tc, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": valid_tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    trace.iterations = iteration
    trace.wall_time_seconds = time.monotonic() - start_time
    trace.messages = messages

    # Reject traces with 0 tool calls — the model answered from
    # parametric knowledge without searching, so the answer is not
    # grounded in the corpus.
    tool_call_count = sum(1 for m in messages if m.get("role") == "tool")
    if tool_call_count == 0 and trace.termination == "natural":
        log.info("No tool calls in trace — marking as no_tool_call")
        trace.termination = "no_tool_call"

    # Set the injection count and verdicts directly on the trace object.
    # QuestionTrace has no __slots__, so this is safe; downstream code
    # (``src/r2c2_agent/orchestrator/worker.py``) reads the attrs with
    # ``getattr(trace, ..., default)``. This leaves the QuestionTrace
    # interface unchanged.
    trace.self_correction_injections = self_correction_injections  # type: ignore[attr-defined]
    trace.verifier_verdicts = verifier_verdicts  # type: ignore[attr-defined]

    return trace


# Unified agentic loop.
#
# Separate entry point from ``answer_question``, behind the
# ``AGENTIC_PIPELINE=unified`` feature flag in ``worker.py``. Each turn is
# one LLM call with thinking on and the tool catalogue attached.
#
# Tool dispatch goes through pipeline_handlers for record_evidence / submit_answer
# (synchronous, run in a thread pool so their internal asyncio.run() is safe)
# and directly to the MCP client for the 4 retrieval tools. Every response is
# decorated by ``wrap_tool_response`` (budget envelope + reflect nudges).
# Retrieved passages from MCP are seeded into ``db.passages`` before wrapping
# so the ``already_recorded`` flag + subsequent record_evidence lookups work.


def _build_assistant_message_from_dict(
    response: LLMResponse,
    *,
    tool_calls_as_openai: bool = True,
) -> dict[str, Any]:
    """Build an assistant-role message dict from an LLMResponse.

    Duplicates the logic of ``_build_assistant_message`` but with an explicit
    knob for whether to serialize tool_calls into OpenAI format (the unified
    driver always wants the OpenAI format because the raw ``ToolCall``
    dataclasses aren't JSON-friendly and the message gets fed back to the
    LLM on the next turn).
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": response.content or "",
    }
    if response.reasoning:
        msg["reasoning"] = response.reasoning
    if response.tool_calls and tool_calls_as_openai:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in response.tool_calls
        ]
    return msg


def _normalize_search_queries(arguments: dict[str, Any]) -> list[str]:
    """Extract search_passages queries into a flat list of strings.

    The tool schema permits ``queries: list[str]`` or ``query: str``; normalize
    both so wrap_tool_response logs a consistent per-query history for
    the no_answer_keyword_search nudge.
    """
    q = arguments.get("queries")
    if q is None:
        q = arguments.get("query")
    if q is None:
        return []
    if isinstance(q, list):
        return [str(x) for x in q if x]
    return [str(q)]


def _seed_passages_from_mcp(
    db: Any,
    raw: Any,
    *,
    tool_call_id: str,
    iteration: int,
) -> None:
    """Upsert passages from an MCP tool response into ``db.passages``.

    MCP tools return JSON strings or dicts with ``{"passages": [...]}`` or a
    single-passage envelope. Parse it, then insert each passage via the
    ``nugget_reconciler.Passage`` model if not already present. Existing pid
    rows are left untouched — their stance/status is authoritative.
    """
    try:
        # Lazy import to avoid forcing the orchestrator deps onto callers
        # who only use the legacy loop.
        from r2c2_agent.orchestrator.nugget_reconciler import Passage, PassageAlias
    except ImportError:  # pragma: no cover — only when orchestrator path missing
        log.debug("nugget_reconciler not on sys.path; skipping passage seeding")
        return

    data: dict[str, Any] | None = None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(parsed, dict):
            data = parsed
    elif isinstance(raw, dict):
        data = raw

    if data is None:
        return

    candidates: list[dict[str, Any]] = []
    passages = data.get("passages")
    if isinstance(passages, list):
        for p in passages:
            if isinstance(p, dict):
                candidates.append(p)
    # Also handle a single-passage envelope (``get_passage`` with one pid).
    if not passages and "passage_id" in data and isinstance(data, dict):
        candidates.append(data)

    for i, p in enumerate(candidates):
        pid = p.get("passage_id") or p.get("pid")
        if not isinstance(pid, str) or not pid:
            continue
        if pid in db.passages:
            # Log alias so downstream replay can see multi-hit retrieval.
            existing = db.passages[pid]
            try:
                existing.aliases.append(
                    PassageAlias(
                        tool_call_id=tool_call_id,
                        msg_idx=i,
                        score=p.get("score"),
                    )
                )
            except Exception:  # pragma: no cover — defensive, never fatal
                pass
            continue
        text = p.get("text") or p.get("passage_text") or p.get("passage") or ""
        try:
            db.passages[pid] = Passage(
                passage_id=pid,
                text=text,
                title=p.get("title") or None,
                score=p.get("score"),
                source_tool_call_id=tool_call_id,
                source_msg_idx=i,
                first_seen_iter=iteration,
                status="unclassified",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("pipeline: failed to seed pid=%s: %s", pid, exc)


async def _dispatch_pipeline_tool(
    db: Any,
    tc: ToolCall,
    mcp_client: MCPClientProtocol,
    config: LoopConfig,
    *,
    iteration: int,
) -> dict[str, Any]:
    """Dispatch one tool call and return the wrapped response dict.

    Dispatch rules:
    - ``record_evidence`` / ``submit_answer`` → orchestrator-side handler
      (synchronous). Run in a thread-pool executor so their internal ``asyncio.run``
      does not collide with the driver's running event loop.
    - ``search_passages`` / ``search_entities`` / ``get_passage`` /
      ``extract_from_passage`` → MCP round-trip with a tool-timeout.
    - Unknown tool → wrapped ``unknown_tool`` error.

    Side-effects on the DB:
    - For ``search_passages`` we pre-log the query into ``db.search_queries``
      so the no_answer_keyword_search reflect nudge can fire on the
      following turn. Done before wrap_tool_response (which increments the
      call counter but does not see the request args).
    - For retrieval tools we seed new passages into ``db.passages`` before
      wrapping so the next ``record_evidence`` call can reference them.
    """
    # Lazy import of the pipeline handlers keeps the legacy loop cheap when
    # AGENTIC_PIPELINE=legacy and the orchestrator path is unneeded.
    from r2c2_agent.orchestrator.pipeline_handlers import (
        handle_batch_record_stance,
        handle_flag_nugget_bogus,
        handle_list_nuggets,
        handle_record_evidence,
        handle_submit_answer,
        wrap_tool_response,
    )

    from r2c2_agent.agent.obs import emit_event

    tool_name = tc.name
    arguments = tc.arguments if isinstance(tc.arguments, dict) else {}
    qid = getattr(db, "question_id", "?")

    # Pre-log search_passages queries for the no_answer_keyword_search nudge.
    if tool_name == "search_passages":
        try:
            db.search_queries.extend(_normalize_search_queries(arguments))
        except AttributeError:
            # A DB without search_queries should not reach here; defensive.
            pass

    raw: Any
    if tool_name in ("record_evidence", "submit_answer"):
        # Synchronous handlers: offload to the thread pool so their internal
        # asyncio.run() (NLI batch) has a fresh event loop.
        running = asyncio.get_running_loop()
        if tool_name == "record_evidence":
            fn = functools.partial(handle_record_evidence, db, arguments)
        else:
            fn = functools.partial(handle_submit_answer, db, arguments)
        # run_in_executor does not auto-copy contextvars; without ctx.run
        # the finalizer client registry is invisible to handle_submit_answer.
        import contextvars as _cv_loop

        ctx = _cv_loop.copy_context()
        try:
            raw = await running.run_in_executor(None, ctx.run, fn)
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline %s handler raised", tool_name)
            raw = {
                "committed": False,
                "error": "handler_exception",
                "message": str(exc),
            }
    elif tool_name in ("list_nuggets", "flag_nugget_bogus", "batch_record_stance"):
        # read/curate handlers: pure Python, no NLI, no MCP. Run inline
        # — offloading adds latency with no concurrency benefit.
        try:
            if tool_name == "list_nuggets":
                raw = handle_list_nuggets(db, arguments)
            elif tool_name == "flag_nugget_bogus":
                raw = handle_flag_nugget_bogus(db, arguments)
            else:
                raw = handle_batch_record_stance(db, arguments)
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline %s handler raised", tool_name)
            raw = {
                "committed": False,
                "error": "handler_exception",
                "message": str(exc),
            }
    elif tool_name in (
        "search_passages",
        "search_entities",
        "get_passage",
        "extract_from_passage",
    ):
        # Time the MCP round-trip. There is no outer asyncio.wait_for:
        # per-call timeouts in a bulk HPC workflow compound load rather
        # than bounding it. The ``config.tool_timeout`` value (900 s
        # default) is still passed through as a safety net in case a
        # wedged call must be unblocked; actual per-qid wall-clock is
        # bounded at the orchestrator level (~30 min). ``TimeoutError``
        # here only surfaces from the socket layer if the server drops;
        # it is handled like a generic transport exception so the trace
        # stays complete.
        t_mcp = time.monotonic()
        try:
            raw = await mcp_client.call_tool(
                name=tool_name,
                arguments=arguments,
                timeout=config.tool_timeout,
            )
            _mcp_dt = time.monotonic() - t_mcp
            _n_pass = (
                len(raw.get("passages", []))
                if isinstance(raw, dict) and isinstance(raw.get("passages"), list)
                else 0
            )
            log.info(
                "pipeline mcp qid=%s tool=%s dt=%.1fs passages=%d",
                qid,
                tool_name,
                _mcp_dt,
                _n_pass,
            )
            emit_event(
                "pipeline.mcp.call",
                qid=qid,
                iter=iteration,
                tool=tool_name,
                dt_s=round(_mcp_dt, 3),
                passages=_n_pass,
            )
        except TimeoutError:
            _mcp_dt = time.monotonic() - t_mcp
            log.warning(
                "pipeline mcp TIMEOUT qid=%s tool=%s after %.1fs",
                qid,
                tool_name,
                _mcp_dt,
            )
            emit_event(
                "pipeline.mcp.timeout",
                qid=qid,
                iter=iteration,
                tool=tool_name,
                dt_s=round(_mcp_dt, 3),
            )
            raw = {"error": "mcp_timeout", "tool": tool_name}
        except Exception as exc:  # noqa: BLE001
            _mcp_dt = time.monotonic() - t_mcp
            log.warning(
                "pipeline mcp call failed qid=%s tool=%s dt=%.1fs: %s",
                qid,
                tool_name,
                _mcp_dt,
                exc,
            )
            emit_event(
                "pipeline.mcp.error",
                qid=qid,
                iter=iteration,
                tool=tool_name,
                dt_s=round(_mcp_dt, 3),
                error=str(exc)[:200],
            )
            raw = {"error": "mcp_exception", "tool": tool_name, "message": str(exc)}

        # Seed newly retrieved passages into the DB before wrapping, so
        # already_recorded flags + subsequent record_evidence lookups work.
        if tool_name in ("search_passages", "get_passage"):
            _seed_passages_from_mcp(
                db,
                raw,
                tool_call_id=tc.id,
                iteration=iteration,
            )
    else:
        raw = {
            "error": "unknown_tool",
            "tool": tool_name,
            "message": f"tool {tool_name!r} is not part of the m11 catalogue",
        }

    wrapped = wrap_tool_response(db, tool_name, raw)
    return wrapped


def _finalize_trace_from_db(trace: QuestionTrace, db: Any) -> None:
    """Populate ``trace`` from the final PipelineState.

    Sets ``draft_answer``, ``raw_confidence`` and ``nuggets`` from the DB.
    ``trace.termination`` is only set from ``db.answer.exit_reason`` when
    the driver has not already chosen a more-specific value (e.g.
    ``no_tool_call``, ``max_iterations``, ``token_budget_exceeded``,
    ``error``). Skips refusal nuggets from the AC output — the adapter
    remaps them at serialization time.
    """
    # ``seed_db_empty`` always creates ``db.answer``. Guard against None
    # anyway, so the finalizer cannot crash before the invariant-rescue
    # block that follows it runs.
    answer_state = db.answer
    if answer_state is None:
        trace.draft_answer = ""
        trace.raw_confidence = 0
    else:
        trace.draft_answer = answer_state.draft_answer or ""
        # raw_confidence is stored as int on QuestionTrace;
        # compute_confidence returns a float in [0, 1]. Map to 0-100.
        conf_float = float(answer_state.confidence or 0.0)
        trace.raw_confidence = int(round(max(0.0, min(1.0, conf_float)) * 100))

    # Build nugget list for the trace — exclude refusal (virtual) and
    # bogus (retracted via flag_nugget_bogus or cascade) nuggets, so that
    # retracted nuggets do not reach the submission.
    nuggets_out: list[Nugget] = []
    for n in db.nuggets.values():
        if n.status in ("refusal", "bogus"):
            continue
        nuggets_out.append(
            Nugget(
                passage_id=n.passage_id,
                text=n.text,
            )
        )
    if nuggets_out:
        trace.nuggets.extend(nuggets_out)

    # Driver-set termination values (forced refusal paths) take precedence —
    # they carry more information than the generic "refusal" that
    # exit_reason=low_confidence_refusal would yield.
    if trace.termination:
        return

    # Map exit_reason → trace.termination string.
    exit_reason = (answer_state.exit_reason or "") if answer_state else ""
    if exit_reason == "low_confidence_refusal":
        trace.termination = "refusal"
    elif exit_reason == "high_confidence":
        trace.termination = "answer_submitted"
    elif exit_reason == "max_iter":
        trace.termination = "max_iterations"
    elif exit_reason == "error":
        trace.termination = "error"
    elif exit_reason == "force_deliver":
        # Model-initiated refusals exit via force_deliver, not
        # low_confidence_refusal. Surface it explicitly so the
        # termination string matches the exit path.
        trace.termination = "force_deliver"
    else:
        trace.termination = "natural"


async def answer_question_pipeline(
    question: Question,
    llm_client: LLMClientProtocol,
    mcp_client: MCPClientProtocol,
    tools: list[dict[str, Any]],
    config: LoopConfig,
) -> QuestionTrace:
    """Run the unified agentic loop for a single question.

    Each turn is one LLM call with ``enable_thinking=True`` and the tools
    supplied. The response carries tool calls, which are dispatched, or
    final text with no tool call, which is taken as a candidate answer
    and ends the loop through a fallback path.

    Terminates when ``db.terminated`` flips (via ``submit_answer``), or
    on the fallback paths below.

    Args:
        question: The question to answer.
        llm_client: LLM client (real or dummy).
        mcp_client: MCP client for the 4 retrieval tools.
        tools: OpenAI-format tool catalogue (6 tools: search_passages,
            search_entities, get_passage, extract_from_passage,
            record_evidence, submit_answer).
        config: Loop configuration.

    Returns:
        Complete ``QuestionTrace`` with answer, confidence, nuggets, and a
        termination string mapped from the server-side exit_reason.
    """
    # Lazy imports — keeps the legacy loop cheap.
    from r2c2_agent.orchestrator.pipeline_handlers import (
        clear_finalizer_clients,
        force_deliver,
        parse_regex_nugget_fallback,
        seed_db_empty,
        set_finalizer_clients,
        set_finalizer_session_messages,
        set_finalizer_tools,
    )

    from r2c2_agent.agent.loop_phase2 import decide_terminal_route, run_phase2_loop
    from r2c2_agent.agent.obs import emit_event

    start_time = time.monotonic()

    async def _terminal_exit(reason: str) -> None:
        """Route a Phase 1 terminal branch through Phase 2 when
        the DB is adequate; otherwise fall through to force_deliver.

        ``force_deliver`` is the backstop in both branches — the outer
        Phase 2 loop calls it itself on timeout / stuck / exhausted, so
        on return ``db.terminated`` is always True.
        """
        route = decide_terminal_route(db, reason)
        if route == "phase2":
            try:
                await run_phase2_loop(
                    db,
                    question,
                    llm_client=llm_client,
                    mcp_client=mcp_client,
                    tools=tools,
                    trace=trace,
                    parent_reason=reason,
                )
            except Exception:  # noqa: BLE001
                log.exception("pipeline: run_phase2_loop raised on reason=%s", reason)
                try:
                    force_deliver(db, f"phase2_failed:{reason}")
                except Exception:  # noqa: BLE001
                    log.exception("pipeline: fallback force_deliver raised")
        else:
            try:
                force_deliver(db, reason)
            except Exception:  # noqa: BLE001
                log.exception("pipeline: force_deliver raised on reason=%s", reason)

    db = seed_db_empty(
        question.qid,
        question.text,
        question.question_type,
    )

    # Copy the prior-rollout answer (if the per-iter worker supplied one
    # via LoopConfig) onto the PipelineState. force_deliver's final
    # fallback ladder uses these fields when every earlier path fails.
    if getattr(config, "prior_rollout_answer", None):
        db.prior_rollout_answer = config.prior_rollout_answer
        db.prior_rollout_confidence = int(
            getattr(config, "prior_rollout_confidence", 0) or 0
        )
        db.prior_rollout_source_iter = getattr(
            config, "prior_rollout_source_iter", None
        )

    # Wire clients for the finalizer educated-guess ladder. Idempotent:
    # if ``force_deliver`` runs, the educated-guess ladder has clients; otherwise no-op.
    # Cleared before return so the next qid does not inherit a stale ref.
    set_finalizer_clients(llm_client, mcp_client)
    # Stash the OpenAI-format tool catalogue for the recursive submission
    # cycle (``_reenter_agentic_loop``). Without this, the re-entry call
    # falls back to ``tools=None`` and cannot legally pass
    # ``tool_choice="required"`` to vLLM.
    set_finalizer_tools(tools)

    system_prompt = build_system_prompt(tools)
    user_message = build_user_message(
        question.text,
        question.question_type if question.question_type != "default" else None,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    trace = QuestionTrace(
        question_id=question.qid,
        question_text=question.text,
        question_type=question.question_type,
        mode="b",
    )

    iteration = 0
    qid = question.qid
    # Per-iteration wall-clock, indexed by iteration number.
    turn_timings: list[float] = []
    turn_start = time.monotonic()

    emit_event(
        "pipeline.qid.start",
        qid=qid,
        question_type=question.question_type,
        calls_total=db.calls_total,
    )

    # Token-budget watermark trigger for Phase 2. Imported here to avoid
    # a circular import cost when loop_phase2 is not exercised.
    from r2c2_agent.agent.loop_phase2 import (
        PHASE2_TOKEN_BUDGET_WATERMARK,
        at_token_budget_watermark,
    )

    while not db.terminated:
        # Token-budget watermark trigger. Fires when the agent has
        # spent >= PHASE2_TOKEN_BUDGET_WATERMARK of its call budget
        # without committing, and routes through Phase 2 (or straight
        # to force_deliver when the DB is not adequate).
        if at_token_budget_watermark(db, watermark=PHASE2_TOKEN_BUDGET_WATERMARK):
            log.info(
                "pipeline qid=%s token-budget watermark reached "
                "(calls_used=%d/%d >= %.2f)",
                qid,
                db.calls_used,
                db.calls_total,
                PHASE2_TOKEN_BUDGET_WATERMARK,
            )
            emit_event(
                "pipeline.token_budget_watermark",
                qid=qid,
                calls_used=db.calls_used,
                calls_total=db.calls_total,
                watermark=PHASE2_TOKEN_BUDGET_WATERMARK,
            )
            await _terminal_exit(
                f"token_budget_watermark:{db.calls_used}/{db.calls_total}"
            )
            # Only stamp "force_deliver" when force_deliver actually
            # fired. _terminal_exit may route through Phase-2, whose
            # successful submit_answer commits with
            # exit_reason="high_confidence" and leaves
            # force_deliver_envelope={}. An unconditional stamp here
            # would overwrite termination to "force_deliver" while the
            # envelope is still empty. Fall through to
            # _finalize_trace_from_db for the Phase-2 commit so
            # exit_reason="high_confidence" maps to
            # termination="answer_submitted".
            if not db.answer or db.answer.exit_reason == "force_deliver":
                trace.termination = "force_deliver"
            break

        # Safety: iteration cap.
        if iteration >= config.max_iterations:
            log.info(
                "pipeline FORCED_REFUSAL qid=%s reason=max_iterations cap=%d",
                qid,
                config.max_iterations,
            )
            emit_event(
                "pipeline.forced_refusal",
                qid=qid,
                iter=iteration,
                reason="max_iterations",
                cap=config.max_iterations,
            )
            # force_deliver, unlike a bare refusal, guarantees a
            # non-empty answer + ≥1 nugget + a citation pid.
            # Phase 2 is tried first if the DB is adequate.
            await _terminal_exit(
                f"max_iterations:iter={iteration}/{config.max_iterations}"
            )
            # See the token-budget-watermark guard above for the
            # rationale. Phase-2 may have committed via submit_answer
            # (exit_reason="high_confidence") — don't relabel that as
            # force_deliver (which would leave an empty envelope).
            if not db.answer or db.answer.exit_reason == "force_deliver":
                trace.termination = "force_deliver"
            break

        # Safety: context budget. Tokens are measured before the LLM
        # call so that no request is sent that vLLM would reject for
        # context overflow. The thresholds are the module constants
        # PREFLIGHT_COMPACT_FRAC and PREFLIGHT_ABORT_FRAC.
        token_count = count_tokens(messages)
        compact_threshold = max(
            config.context_budget,
            int(config.token_budget * PREFLIGHT_COMPACT_FRAC),
        )
        if token_count > compact_threshold:
            log.info(
                "pipeline qid=%s context budget exceeded (%d > %d), compacting",
                qid,
                token_count,
                compact_threshold,
            )
            emit_event(
                "pipeline.compact",
                qid=qid,
                iter=iteration,
                tokens=token_count,
                budget=compact_threshold,
            )
            messages = compact_messages(messages)
            token_count = count_tokens(messages)

        if token_count > int(config.token_budget * PREFLIGHT_ABORT_FRAC):
            log.warning(
                "pipeline FORCED_REFUSAL qid=%s reason=preflight_token_budget "
                "(%d > %.2f * %d)",
                qid,
                token_count,
                PREFLIGHT_ABORT_FRAC,
                config.token_budget,
            )
            emit_event(
                "pipeline.forced_refusal",
                qid=qid,
                iter=iteration,
                reason="preflight_token_budget",
                tokens=token_count,
                cap=config.token_budget,
                frac=PREFLIGHT_ABORT_FRAC,
            )
            await _terminal_exit(
                f"preflight_token_budget:{token_count}/{config.token_budget}"
            )
            # See the token-budget-watermark guard above — stamp only
            # when force_deliver actually fired. Phase-2 commit via
            # submit_answer leaves exit_reason="high_confidence" and
            # an empty envelope, so an unconditional stamp would produce
            # orphan (termination=force_deliver, envelope={}) records.
            if not db.answer or db.answer.exit_reason == "force_deliver":
                trace.termination = "force_deliver"
            break

        # Per-turn status line.
        _verified_count = sum(
            1 for n in db.nuggets.values() if n.status == "verified_supporting"
        )
        log.info(
            "pipeline turn qid=%s iter=%d budget=%d/%d "
            "nuggets=%d/%d verified=%d tokens=%d",
            qid,
            iteration,
            db.calls_used,
            db.calls_total,
            db.nuggets_recorded,
            db.max_nuggets_global,
            _verified_count,
            token_count,
        )
        emit_event(
            "pipeline.turn.banner",
            qid=qid,
            iter=iteration,
            calls_used=db.calls_used,
            calls_total=db.calls_total,
            nuggets_recorded=db.nuggets_recorded,
            verified=_verified_count,
            tokens=token_count,
        )

        # Single turn: thinking on, tools attached. Same shape as
        # answer_question: interleaved reasoning and structured
        # tool_calls in one LLM call. Splitting the turn into a reasoning
        # call with tools=None and a separate tool call puts XML-shaped
        # tool calls from the first into the context of the second, which
        # derails the tool-call parser.
        t_write = time.monotonic()
        _write_user = f"qid={qid};iter={iteration}"
        try:
            try:
                write_response = await llm_client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    enable_thinking=True,
                    user=_write_user,
                )
            except TypeError:
                write_response = await llm_client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    enable_thinking=True,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline WRITE call failed qid=%s: %s", qid, exc)
            emit_event(
                "pipeline.write.error",
                qid=qid,
                iter=iteration,
                error=str(exc)[:200],
            )
            # LLM call exceptions (vLLM context overflow, 5xx, network
            # drop) do not go to handle_submit_answer with refusal=True:
            # the exhaustion gate can reject it, leaving the DB
            # un-terminated and the trace with an empty answer.
            # _terminal_exit routes through Phase-2 (if the DB is
            # adequate) or force_deliver directly, either way
            # guaranteeing a non-empty answer commit.
            await _terminal_exit(f"write_error:{type(exc).__name__}:{str(exc)[:100]}")
            # See the token-budget-watermark guard above — stamp only
            # when force_deliver actually fired. Phase-2 submit_answer
            # exits with exit_reason="high_confidence" + empty envelope;
            # an unconditional stamp here would produce orphan records.
            if not db.answer or db.answer.exit_reason == "force_deliver":
                trace.termination = "force_deliver"
            break
        _write_dt = time.monotonic() - t_write

        write_msg = _build_assistant_message_from_dict(write_response)
        messages.append(write_msg)
        trace.append_turn(write_msg)

        # Snapshot the conversation for the finalizer fallback path.
        # If force_deliver fires this turn (tool dispatch below, iter
        # cap, exhaustion), the finalizer replays this list + a nudge.
        set_finalizer_session_messages(messages)

        # Natural fallback: the response carries no tool call.
        if not write_response.tool_calls:
            log.info(
                "pipeline WRITE qid=%s iter=%d dt=%.1fs tool=NONE — implicit refusal",
                qid,
                iteration,
                _write_dt,
            )
            emit_event(
                "pipeline.no_tool_call",
                qid=qid,
                iter=iteration,
                dt_s=round(_write_dt, 3),
                content_preview=(write_response.content or "")[:240],
            )
            # Scrape assistant content as a Tier-1 candidate answer.
            # Salvages cases where the model produced valid prose but
            # skipped the submit_answer tool call.
            # On an adequate DB, route through Phase 2 first.
            # Park the scraped prose in db.answer.draft_answer before
            # entering Phase 2 so every subsequent force_deliver call
            # (from inside Phase 2 or from our fallback path) picks it up
            # via Tier 2 automatically, independent of call path.
            _reason_nt = f"no_tool_call:iter={iteration}"
            _preferred = (write_response.content or "").strip()[:2000] or None
            if (
                _preferred
                and db.answer is not None
                and not (db.answer.draft_answer or "").strip()
            ):
                db.answer.draft_answer = _preferred
            if decide_terminal_route(db, _reason_nt) == "phase2":
                try:
                    await run_phase2_loop(
                        db,
                        question,
                        llm_client=llm_client,
                        mcp_client=mcp_client,
                        tools=tools,
                        trace=trace,
                        parent_reason=_reason_nt,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("pipeline: run_phase2_loop(no_tool_call) raised")
                    try:
                        force_deliver(
                            db,
                            f"phase2_failed:{_reason_nt}",
                            preferred_answer=_preferred,
                        )
                    except Exception:  # noqa: BLE001
                        log.exception("pipeline: fallback force_deliver raised")
            else:
                try:
                    force_deliver(
                        db,
                        _reason_nt,
                        preferred_answer=_preferred,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("pipeline: force_deliver(no_tool_call) raised")
            # Only stamp force_deliver when the DB didn't already commit
            # a successful answer through Phase-2 (or any nested path).
            # Without this guard, Phase-2's successful ``submit_answer``
            # sets ``exit_reason="high_confidence"`` + ``db.terminated=True``
            # but the unconditional stamp here overwrites the trace
            # termination to ``"force_deliver"``. The downstream
            # ``_finalize_trace_from_db`` short-circuits on an already-set
            # termination, so ``exit_reason`` is ignored — producing a
            # mismatched (termination="force_deliver",
            # exit_reason="high_confidence") trace.
            if not db.terminated:
                trace.termination = "force_deliver"
            break

        # Dispatch tool calls (usually one, but the model may emit several).
        for tc in write_response.tool_calls:
            _args_preview = json.dumps(tc.arguments, ensure_ascii=False)[:180]
            log.info(
                "pipeline WRITE qid=%s iter=%d dt=%.1fs tool=%s args=%s",
                qid,
                iteration,
                _write_dt,
                tc.name,
                _args_preview,
            )
            emit_event(
                "pipeline.write",
                qid=qid,
                iter=iteration,
                dt_s=round(_write_dt, 3),
                tool=tc.name,
                args_preview=_args_preview,
                arg_types={
                    k: type(v).__name__ for k, v in (tc.arguments or {}).items()
                },
            )
            wrapped = await _dispatch_pipeline_tool(
                db, tc, mcp_client, config, iteration=iteration
            )
            # Per-nugget outcome.
            if tc.name == "record_evidence":
                _per = wrapped.get("per_nugget") or []
                _reflect = wrapped.get("reflect") or ""
                for _entry in _per:
                    if not isinstance(_entry, dict):
                        continue
                    _i = _entry.get("index", "?")
                    _ok = bool(_entry.get("committed"))
                    _score = float(_entry.get("nli_score", 0.0) or 0.0)
                    _reason = _entry.get("reason", "trust_band" if _ok else "unknown")
                    _label = _entry.get("nli_label", "?")
                    log.info(
                        "pipeline record_evidence qid=%s pid=%s n#%s %s nli=%.2f "
                        "label=%s reason=%s",
                        qid,
                        tc.arguments.get("pid"),
                        _i,
                        "COMMIT" if _ok else "REJECT",
                        _score,
                        _label,
                        _reason,
                    )
                    emit_event(
                        "pipeline.record_evidence.outcome",
                        qid=qid,
                        iter=iteration,
                        pid=tc.arguments.get("pid"),
                        n_idx=_i,
                        committed=_ok,
                        nli_score=round(_score, 3),
                        nli_label=_label,
                        reason=_reason,
                    )
                if _reflect:
                    log.info(
                        "pipeline reflect qid=%s source=record_evidence text=%r",
                        qid,
                        _reflect[:160],
                    )
                    emit_event(
                        "pipeline.reflect",
                        qid=qid,
                        iter=iteration,
                        source="record_evidence",
                        text=_reflect[:240],
                    )
            # submit_answer request and outcome.
            elif tc.name == "submit_answer":
                _refusal_req = bool(tc.arguments.get("refusal"))
                _committed = wrapped.get("committed")
                _exit = wrapped.get("exit_reason") or wrapped.get("reason") or "?"
                _conf = float(wrapped.get("confidence") or 0.0)
                _cits_field = wrapped.get("citation_pids") or []
                if isinstance(_cits_field, str):
                    _cits = 0
                else:
                    try:
                        _cits = len(_cits_field)
                    except TypeError:
                        _cits = 0
                _n_eff = wrapped.get("n_effective_supporting", "?")
                log.info(
                    "pipeline submit_answer qid=%s refusal_requested=%s "
                    "committed=%s exit=%s conf=%.2f n_eff=%s citations=%d",
                    qid,
                    _refusal_req,
                    _committed,
                    _exit,
                    _conf,
                    _n_eff,
                    _cits,
                )
                emit_event(
                    "pipeline.submit_answer",
                    qid=qid,
                    iter=iteration,
                    refusal_requested=_refusal_req,
                    committed=_committed,
                    exit_reason=_exit,
                    confidence=round(_conf, 3),
                    n_effective_supporting=_n_eff,
                    citation_count=_cits,
                )
                if wrapped.get("reflect"):
                    log.info(
                        "pipeline reflect qid=%s source=submit_answer text=%r",
                        qid,
                        str(wrapped["reflect"])[:160],
                    )
                    emit_event(
                        "pipeline.reflect",
                        qid=qid,
                        iter=iteration,
                        source="submit_answer",
                        text=str(wrapped["reflect"])[:240],
                    )
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(wrapped, ensure_ascii=False),
            }
            messages.append(tool_msg)
            trace.append_tool_result(tc, wrapped)

            # Terminate immediately when submit_answer commits — the DB flag
            # drives the while loop, but bail out mid-batch to avoid
            # dispatching stale tool calls that would mutate the DB post-exit.
            if db.terminated:
                break

        turn_timings.append(round(time.monotonic() - turn_start, 3))
        turn_start = time.monotonic()
        iteration += 1

    # Finalize the trace. The post-loop safety net must not produce an
    # empty answer or a zero-confidence refusal, so it calls force_deliver
    # directly, which bypasses the exhaustion gate by design. It fires
    # when the while-loop exited for reasons other than commit, or when a
    # committed DB has an empty draft_answer (can happen if an earlier
    # terminal branch wrote through a handler that the gate rejected).
    needs_force_deliver = (not db.terminated) or not (
        (db.answer.draft_answer or "").strip() if db.answer else ""
    )
    if needs_force_deliver:
        try:
            force_deliver(
                db,
                f"post_loop_safety_net:termination={trace.termination or 'none'}",
            )
        except Exception:  # noqa: BLE001
            log.exception("pipeline: post-loop force_deliver raised (non-fatal)")

    # Scrape free-text NUGGET[pid]: claim markers the LLM may have typed
    # in prose instead of calling record_evidence. The fallback itself
    # skips on the refusal path.
    try:
        fallback_stats = parse_regex_nugget_fallback(db, messages)
        if fallback_stats.get("scraped", 0) > 0:
            log.info(
                "pipeline regex_fallback qid=%s scraped=%d stats=%s",
                qid,
                fallback_stats["scraped"],
                fallback_stats,
            )
            emit_event(
                "pipeline.regex_fallback",
                qid=qid,
                scraped=fallback_stats.get("scraped", 0),
                deduped=fallback_stats.get("deduped", 0),
                unknown_pid=fallback_stats.get("unknown_pid", 0),
            )
    except Exception:  # noqa: BLE001
        log.exception("pipeline regex_fallback failed (non-fatal)")
        fallback_stats = {"scraped": 0, "error": "exception"}

    _finalize_trace_from_db(trace, db)

    # Last-resort invariant: no trace leaves with an empty draft_answer or
    # a raw_confidence below the floor. This should be unreachable after
    # the force_deliver block above; it covers the case where
    # force_deliver itself raised.
    if not (trace.draft_answer or "").strip():
        from r2c2_agent.orchestrator.pipeline_handlers import CANONICAL_REFUSAL_ANSWER

        log.error(
            "pipeline INVARIANT_RESCUE qid=%s empty draft_answer at finalize — "
            "emitting canonical fallback",
            qid,
        )
        emit_event("pipeline.invariant_rescue", qid=qid, field="draft_answer")
        trace.draft_answer = CANONICAL_REFUSAL_ANSWER
    if trace.raw_confidence < CONFIDENCE_FLOOR_INT:
        log.warning(
            "pipeline INVARIANT_RESCUE qid=%s raw_confidence=%d below floor=%d",
            qid,
            trace.raw_confidence,
            CONFIDENCE_FLOOR_INT,
        )
        emit_event(
            "pipeline.invariant_rescue",
            qid=qid,
            field="raw_confidence",
            old=trace.raw_confidence,
            new=CONFIDENCE_FLOOR_INT,
        )
        trace.raw_confidence = CONFIDENCE_FLOOR_INT

    trace.iterations = iteration
    trace.wall_time_seconds = time.monotonic() - start_time
    trace.messages = messages

    # Stash the PipelineState on the trace so downstream code
    # (adapter, serializer) can read the full event log / budget state.
    trace.m11_db = db  # type: ignore[attr-defined]
    # Telemetry on how often the LLM bypassed record_evidence, which
    # tells whether the regex fallback is still needed.
    trace.m11_regex_fallback_stats = fallback_stats  # type: ignore[attr-defined]
    trace.m11_turn_timings = turn_timings  # type: ignore[attr-defined]

    # Termination summary.
    _verified_final = sum(
        1 for n in db.nuggets.values() if n.status == "verified_supporting"
    )
    log.info(
        "pipeline FINAL qid=%s termination=%s iter=%d wall=%.1fs nuggets=%d "
        "verified=%d conf=%d answer=%s",
        qid,
        trace.termination,
        trace.iterations,
        trace.wall_time_seconds,
        len(trace.nuggets),
        _verified_final,
        trace.raw_confidence,
        (trace.draft_answer or "")[:80],
    )
    emit_event(
        "pipeline.final",
        level="quiet",
        qid=qid,
        termination=trace.termination,
        iter=trace.iterations,
        wall_s=round(trace.wall_time_seconds, 2),
        nuggets=len(trace.nuggets),
        verified=_verified_final,
        raw_confidence=trace.raw_confidence,
        answer_preview=(trace.draft_answer or "")[:120],
    )

    # Reset the finalizer client registry so the next qid cannot observe
    # a stale client reference (orchestrator may process multiple qids
    # sequentially in one process).
    clear_finalizer_clients()

    return trace
