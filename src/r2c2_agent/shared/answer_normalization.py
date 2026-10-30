"""Answer format normalization for R2C2 submissions.

Shared module consumed by ``src/r2c2_agent/orchestrator/worker.py`` (inline
normalization on the inference path).

Two modes:
- With-gold mode (``expected != None``): uses the gold answer to produce
  the ideal format. Used only for offline data-prep / audit; not on the
  inference path.
- Inference mode (``expected is None``): only universal cleanup
  (markdown / outer quotes / trailing period). Structural reshaping is
  delegated to the model via the ``format_answer`` tool (see
  ``FORMAT_ANSWER_TOOL_SPEC`` below).
"""

from __future__ import annotations

import re

__all__ = [
    "FORMAT_ANSWER_TOOL_SPEC",
    "format_answer_tool_handler",
    "normalize_answer",
    "normalize_answer_in_trace",
]

# Core normalization

_MARKDOWN_RE = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_LEADING_STARS_RE = re.compile(r"^\*+\s*")
_QUOTES_RE = re.compile(r'^["\']|["\']$')
# Used by ``format_answer_tool_handler`` to drop a parenthetical once the
# model has chosen the "single_entity" format. The model decides the
# format; the regex only cleans up.
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_ANSWER_PATTERN = re.compile(r"([Aa]nswer:\s*)(.+?)(\n|$)")


def _strip_markdown(s: str) -> str:
    """Remove markdown bold/italic markers."""
    s = _MARKDOWN_RE.sub(r"\1", s)
    s = _LEADING_STARS_RE.sub("", s)
    s = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", s)
    return s.strip()


def normalize_answer(
    draft: str,
    question_type: str,
    expected: str | None = None,
) -> str:
    """Normalize a draft answer for submission.

    Args:
        draft: Raw draft answer from the model's Answer: line.
        question_type: One of simple, simple_w_condition, aggregation,
            comparison, multi_hop, post_processing.
        expected: Gold answer (data-prep only; ``None`` at inference).

    Returns:
        Normalized answer string.
    """
    # Universal cleanup (both modes)
    s = _strip_markdown(draft)
    s = _QUOTES_RE.sub("", s).strip()
    # Strip trailing period if answer is a short phrase
    if s.endswith(".") and s.count(".") == 1 and len(s.split()) <= 6:
        s = s[:-1].strip()

    # With-gold mode: expected_answer available
    if expected is not None:
        if question_type == "aggregation":
            # Always use gold format "N: item1, item2, ..."
            return expected

        if question_type == "comparison":
            el = expected.lower().strip()
            if el in ("yes", "no"):
                if s.lower().startswith("yes"):
                    return "Yes"
                if s.lower().startswith("no"):
                    return "No"
                return expected.capitalize()
            # Entity:value format -- use gold
            return expected

        if question_type == "post_processing":
            if "," in s and "," not in expected:
                return expected
            if expected.lower() in s.lower() and len(s) > len(expected) * 1.3:
                return expected
            return s if s else expected

        # simple, simple_w_condition, multi_hop
        if expected.lower() in s.lower() and len(s) > len(expected) * 1.3:
            return expected
        return s

    # Inference mode: no expected_answer
    # Only the universal cleanup above applies. The model calls
    # ``format_answer`` (see FORMAT_ANSWER_TOOL_SPEC below) when it needs
    # a structured shape. Guessing the shape here with regexes
    # (comma-splitting "Palpatine, Emperor of the Galactic Empire" into
    # "Palpatine", extracting a leading number for post_processing,
    # reformatting aggregation as "N: items") is wrong often enough to
    # lower accuracy overall.
    return s


# Trace-level normalization (modifies trace in-place)


def normalize_answer_in_trace(
    trace: dict,
    expected: str | None = None,
) -> dict:
    """Normalize the Answer: line in the final assistant message of a trace.

    Modifies the trace in-place and returns it.

    Args:
        trace: Full trace dict with 'messages', 'question_type', etc.
        expected: Gold answer. If None, uses trace['expected_answer'] if present.

    Returns:
        The modified trace dict (same object, mutated).
    """
    qtype = trace.get("question_type", "simple")
    if expected is None:
        expected = trace.get("expected_answer")

    messages = trace.get("messages", [])
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            match = _ANSWER_PATTERN.search(content)
            if match:
                prefix = match.group(1)  # "Answer: "
                old_answer = match.group(2)
                suffix = match.group(3)  # "\n" or ""
                new_answer = normalize_answer(old_answer, qtype, expected=expected)
                new_content = (
                    content[: match.start()]
                    + prefix
                    + new_answer
                    + suffix
                    + content[match.end() :]
                )
                messages[i] = {**msg, "content": new_content}
                break

    return trace


# format_answer tool (inference-time, called by the model)
# The question type is not detected here: the orchestrator's
# ``question_classifier.classify_question_type`` is authoritative.

FORMAT_ANSWER_TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": "format_answer",
        "description": (
            "Format your draft answer according to the question type before "
            "giving your final Answer: line. Call this AFTER you have determined "
            "your answer but BEFORE writing Answer:. The tool returns the "
            "properly formatted answer string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question_type": {
                    "type": "string",
                    "enum": [
                        "simple",
                        "simple_w_condition",
                        "aggregation",
                        "comparison",
                        "multi_hop",
                        "post_processing",
                    ],
                    "description": "The type of question being answered",
                },
                "draft_answer": {
                    "type": "string",
                    "description": "Your draft answer text",
                },
                "answer_format": {
                    "type": "string",
                    "enum": [
                        "single_entity",
                        "count_and_list",
                        "entity_value",
                        "yes_no",
                        "single_value",
                    ],
                    "description": (
                        "The format to apply: "
                        "single_entity = one name/date/phrase; "
                        "count_and_list = 'N: item1, item2, ...'; "
                        "entity_value = 'Entity: value'; "
                        "yes_no = Yes or No; "
                        "single_value = one computed number/date"
                    ),
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For count_and_list: the individual items. "
                        "For entity_value: [entity, value]."
                    ),
                },
            },
            "required": ["question_type", "draft_answer", "answer_format"],
        },
    },
}


def format_answer_tool_handler(
    question_type: str,
    draft_answer: str,
    answer_format: str,
    items: list[str] | None = None,
) -> str:
    """Handler for the format_answer tool call.

    Called by the app layer when the model invokes the format_answer tool.
    Returns the formatted answer string that the model should use.

    Args:
        question_type: Model-detected question type.
        draft_answer: Model's draft answer text.
        answer_format: One of single_entity, count_and_list, entity_value,
            yes_no, single_value.
        items: For count_and_list: individual items.
            For entity_value: [entity, value].

    Returns:
        Formatted answer string.
    """
    clean = _strip_markdown(draft_answer)

    if answer_format == "single_entity":
        # Take first entity if comma-separated
        if items:
            return items[0]
        if "," in clean:
            return clean.split(",")[0].strip()
        return _PAREN_RE.sub("", clean).strip()

    if answer_format == "count_and_list":
        if items:
            return f"{len(items)}: {', '.join(items)}"
        # Fallback: parse comma list from draft
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", clean)
        parts = [p.strip() for p in parts if p.strip()]
        return f"{len(parts)}: {', '.join(parts)}"

    if answer_format == "entity_value":
        if items and len(items) >= 2:
            return f"{items[0]}: {items[1]}"
        return clean

    if answer_format == "yes_no":
        if clean.lower().startswith("yes"):
            return "Yes"
        if clean.lower().startswith("no"):
            return "No"
        return clean

    if answer_format == "single_value":
        nums = re.findall(r"-?\$?\d+(?:[.,]\d+)?", clean)
        if nums:
            return nums[0]
        return clean

    return clean
