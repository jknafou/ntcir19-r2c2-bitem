"""Question-type classifier for the agentic pipeline.

At worker entry the R2C2 test set gives only (question_id, question_text);
there is no ``question_type`` label. This module asks the LLM to classify the
question into one of the 6 canonical types before the agentic loop starts.

Downstream consumers use the result for per-type metrics only.

The prompt follows the wording of ``_CLASSIFY_PROMPT`` in
``src/r2c2_agent/orchestrator/nugget_reconciler.py``, adapted to classify from
the question alone: that prompt also takes an answer, which does not exist
at ingest time.

Every failure (LLM unreachable, timeout, malformed JSON, unknown type)
falls through to ``heuristic_classify``, which flags aggregation-like
phrasings and defaults to ``simple``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

__all__ = [
    "VALID_QUESTION_TYPES",
    "classify_question_type",
    "heuristic_classify",
]

log = logging.getLogger(__name__)


VALID_QUESTION_TYPES: frozenset[str] = frozenset(
    {
        "simple",
        "aggregation",
        "comparison",
        "simple_w_condition",
        "multi_hop",
        "post_processing",
    }
)


_PROMPT = """\
Classify this question into exactly one of 6 types.

## Types

1. **simple** — Direct single-fact lookup about one entity and one property.
   Example: "What is Veronica Lang's nationality?"

2. **aggregation** — Asks for a LIST or COUNT of items matching a criterion.
   Example: "What films did Ronald Mann direct?"
   Example: "How many Wes Anderson films are there?"

3. **comparison** — Names TWO entities, asks which has more/less/earlier/later
   of a property.
   Example: "Which was released first, Imagine the Sound or Comic Book Confidential?"

4. **simple_w_condition** — Like simple, but includes a qualifying clause
   (budget, runtime, year, network) that narrows which entity is meant.
   Example: "Who reviewed the 86-minute 2015 film The Lazarus Effect?"

5. **multi_hop** — Requires finding an intermediate entity first. Chains
   two relationships.
   Example: "Which Pope was portrayed by the actor who starred in Guilty Conscience?"

6. **post_processing** — Asks about dates, years, or ratings for specific
   editions of the same work.
   Example: "What are the release dates for the 1997 film Lolita?"

## Priority rules

- Question enumerates OR counts items
  ("how many", "list all", "name all", "count") → **aggregation**
- Question compares two named entities on one property → **comparison**
- Question asks about dates/years/ratings across editions → **post_processing**
- Question has a qualifying clause with a specific number/date
  narrowing the target → **simple_w_condition**
- Question requires finding entity A to look up entity B → **multi_hop**
- None of the above → **simple**

Question: {question}

Output ONLY a JSON object: {{"question_type": "<type>"}}"""


_JSON_RE = re.compile(r"\{[^{}]*\"question_type\"[^{}]*\}")


def heuristic_classify(question: str) -> str:
    """Cheap rule-based fallback.

    Only detects aggregation with high confidence; everything else defaults
    to ``simple``. Mirrors ``nugget_reconciler._infer_question_type``, so
    that callers do not have to import that module.

    Args:
        question: The question text.

    Returns:
        One of ``VALID_QUESTION_TYPES``.
    """
    q = (question or "").lower()
    aggregation_markers = (
        "how many",
        "list all",
        "enumerate",
        "count",
        "name all",
        "which films",
        "which movies",
    )
    if any(kw in q for kw in aggregation_markers):
        return "aggregation"
    return "simple"


async def classify_question_type(
    llm_client: Any,
    question: str,
    *,
    timeout_s: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    """Classify a question into one of the 6 canonical types.

    Args:
        llm_client: An object with an async ``chat(messages, tools=None,
            enable_thinking=False, ...)`` method returning an object with
            a ``content: str`` attribute (matches ``OpenAILLMClient``).
        question: The question text.
        timeout_s: Ignored. The underlying client applies its own
            timeouts.

    Returns:
        ``(question_type, stats)`` where ``stats`` has ``source`` (one of
        ``"llm"`` / ``"heuristic"`` / ``"fallback"``), ``llm_calls`` (0 or 1),
        and ``raw_content`` (the LLM's raw reply, empty on failure).
    """
    del timeout_s  # reserved for future use

    stats: dict[str, Any] = {
        "source": "fallback",
        "llm_calls": 0,
        "raw_content": "",
    }

    # The heuristic result is only the fallback for the LLM-failure
    # paths below; the LLM always classifies. As a short-circuit before
    # the LLM, the heuristic would route any question containing
    # "how many" / "list all" / "which films" to aggregation, which
    # mis-routes multi_hop and post_processing questions that carry
    # those phrases in a sub-clause (e.g. "Which film, of the three
    # Wick sequels, had the highest box-office?").
    pre = heuristic_classify(question)

    messages = [
        {"role": "user", "content": _PROMPT.format(question=question)},
    ]
    try:
        response = await llm_client.chat(
            messages=messages,
            tools=None,
            enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("classifier LLM call failed (%s); falling back", exc)
        return pre, stats

    content = getattr(response, "content", "") or ""
    stats["llm_calls"] = 1
    stats["raw_content"] = content[:400]

    match = _JSON_RE.search(content)
    if not match:
        log.warning("classifier: no JSON found in reply; falling back")
        return pre, stats
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        log.warning("classifier: malformed JSON %r; falling back", match.group(0))
        return pre, stats

    qtype = str(parsed.get("question_type", "")).strip().lower()
    if qtype not in VALID_QUESTION_TYPES:
        log.warning("classifier: invalid type %r; falling back", qtype)
        return pre, stats

    stats["source"] = "llm"
    return qtype, stats
