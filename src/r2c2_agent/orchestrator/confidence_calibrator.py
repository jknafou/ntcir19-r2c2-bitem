"""Confidence calibration via nugget-based reproducibility.

After the reconciler produces validated nuggets, runs 3 LLM calls:
1. Confidence score (parallel): "How confident given only these nuggets?"
2. Re-answer (parallel): "Answer this question using only these nuggets."
3. Match check (sequential): "Are these two answers the same?"

Final confidence = match_score * llm_confidence
  - match_score = 1.0 if re-answer matches original, 0.25 if not
  - Clamped to heuristic bounds (0 nuggets → max 25, etc.)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

__all__ = ["calibrate_confidence"]

log = logging.getLogger(__name__)

_CONFIDENCE_PROMPT = """\
You are a confidence assessor for a QA competition (movie/entertainment domain).

Question: {question}

Supporting evidence (nuggets):
{nuggets}

Based ONLY on the nuggets above (not your own knowledge), how confident \
are you that a correct answer to the question can be derived from this evidence?

Rules:
- Score 90-100: nuggets directly and unambiguously answer the question
- Score 60-89: nuggets strongly suggest an answer but some inference is needed
- Score 30-59: nuggets are relevant but incomplete — the answer requires guessing
- Score 0-29: nuggets don't address the question at all

Output ONLY a JSON object: {{"confidence": <integer 0-100>}}"""

_REANSWER_PROMPT = """\
You are answering a question for a QA competition (movie/entertainment domain).

Question: {question}

You may ONLY use the following evidence (nuggets) to answer. Do NOT use \
any knowledge beyond what is stated in the nuggets.

Evidence:
{nuggets}

If the evidence is insufficient to answer, respond with: INSUFFICIENT EVIDENCE

Otherwise, provide your answer in the same format as the question expects \
(a name, number, title, etc.).

Answer:"""

_MATCH_PROMPT = """\
Do these two answers refer to the same thing? Consider semantic equivalence \
— paraphrasing, abbreviations, name variants, numeric equivalence, and \
list reordering all count as matching.

Answer A: {answer_a}
Answer B: {answer_b}

Output ONLY a JSON object: {{"match": true}} or {{"match": false}}"""


def _fmt_nuggets(nuggets: list[dict]) -> str:
    """Format nuggets for prompt inclusion."""
    if not nuggets:
        return "(no evidence available)"
    lines = []
    for i, n in enumerate(nuggets):
        lines.append(f"- {n['nugget_text']}")
    return "\n".join(lines)


def _parse_json(text: str) -> Any:
    """Extract JSON from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


async def _llm_call(client: AsyncOpenAI, model: str, prompt: str) -> str:
    """Single LLM call with thinking disabled."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""

    calibrated_confidence: int
    llm_confidence: int
    re_answer: str
    match: bool
    match_score: float
    llm_calls: int = 0
    heuristic_cap: int | None = None


async def calibrate_confidence(
    client: AsyncOpenAI,
    model: str,
    question: str,
    original_answer: str,
    nuggets: list[dict],
    is_complete: bool = True,
    iterations_used: int = 0,
) -> CalibrationResult:
    """Run the 3-call confidence calibration pipeline.

    Args:
        client: Async OpenAI-compatible client.
        model: Model name.
        question: The question text.
        original_answer: The model's original answer.
        nuggets: Validated nuggets from the reconciler.
        is_complete: Whether COMPLETE step said "yes" (from reconciler).
        iterations_used: How many reconciler iterations were needed.

    Returns:
        CalibrationResult with the calibrated confidence score.
    """
    # Heuristic bounds
    heuristic_cap = 100
    if not nuggets:
        log.info("Calibration: 0 nuggets, returning 25")
        return CalibrationResult(
            calibrated_confidence=25,
            llm_confidence=0,
            re_answer="",
            match=False,
            match_score=0.25,
            heuristic_cap=25,
        )
    if not is_complete and iterations_used >= 3:
        heuristic_cap = 60
    elif not is_complete:
        heuristic_cap = 75

    nugget_text = _fmt_nuggets(nuggets)

    # Calls 1 and 2 run in parallel
    conf_prompt = _CONFIDENCE_PROMPT.format(
        question=question,
        nuggets=nugget_text,
    )
    reanswer_prompt = _REANSWER_PROMPT.format(
        question=question,
        nuggets=nugget_text,
    )

    conf_raw, reanswer_raw = await asyncio.gather(
        _llm_call(client, model, conf_prompt),
        _llm_call(client, model, reanswer_prompt),
    )

    llm_confidence = 50  # default
    try:
        conf_result = _parse_json(conf_raw)
        if isinstance(conf_result, dict):
            llm_confidence = int(conf_result.get("confidence", 50))
            llm_confidence = min(100, max(0, llm_confidence))
    except Exception:
        log.warning("Failed to parse confidence response: %s", conf_raw[:100])

    re_answer = reanswer_raw.strip()
    if "Answer:" in re_answer:
        re_answer = re_answer.split("Answer:")[-1].strip()

    if "INSUFFICIENT" in re_answer.upper():
        log.info("Calibration: re-answer says INSUFFICIENT EVIDENCE")
        raw_score = int(0.25 * llm_confidence)
        return CalibrationResult(
            calibrated_confidence=min(raw_score, heuristic_cap),
            llm_confidence=llm_confidence,
            re_answer=re_answer,
            match=False,
            match_score=0.25,
            llm_calls=2,
            heuristic_cap=heuristic_cap,
        )

    # Call 3: match check
    match_prompt = _MATCH_PROMPT.format(
        answer_a=original_answer,
        answer_b=re_answer,
    )
    try:
        match_raw = await _llm_call(client, model, match_prompt)
        match_result = _parse_json(match_raw)
        is_match = (
            match_result.get("match", False)
            if isinstance(match_result, dict)
            else False
        )
    except Exception:
        log.warning("Failed to parse match response: %s", match_raw[:100])
        is_match = False

    # Final score
    match_score = 1.0 if is_match else 0.25
    raw_score = int(match_score * llm_confidence)
    calibrated = min(raw_score, heuristic_cap)
    calibrated = max(calibrated, 0)

    log.info(
        "Calibration: llm_conf=%d re_answer=%s match=%s "
        "match_score=%.2f raw=%d cap=%s final=%d",
        llm_confidence,
        re_answer[:40],
        is_match,
        match_score,
        raw_score,
        heuristic_cap,
        calibrated,
    )

    return CalibrationResult(
        calibrated_confidence=calibrated,
        llm_confidence=llm_confidence,
        re_answer=re_answer,
        match=is_match,
        match_score=match_score,
        llm_calls=3,
        heuristic_cap=heuristic_cap,
    )
