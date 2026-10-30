"""LLM-as-judge AC re-aggregation (AC-4 strategy) — tool-call redesign.

Replaces the previous enum-pick design (``candidate_<i> | default |
refusal`` with a "prefer default" rule) which biased the judge into
always picking ``default``. The new judge sees the raw evidence pool
(all submit_answer attempts across iters with confidence + citations,
verified-supporting nuggets with NLI scores, and the frozen PR-2
roster) and must call a single ``submit_answer`` tool with its own
``(answer, confidence, evidence_pids)`` decision. Retry-until-valid
guards against malformed calls and out-of-roster passage IDs.

Per-qid pipeline:

1. Extract ALL ``submit_answer`` attempts across all dispatched iters
   (re-using ``ac_reaggregator._extract_submit_attempts_from_trace``)
   plus all verified-supporting nuggets recovered from those rollouts.
2. Render the PR-2 roster (passage IDs + sanitised text), the per-iter
   submit_answer attempts (class / conf / n_eff / answer / citations),
   and the verified-supporting nuggets into a single user prompt.
3. Issue an OpenAI-compatible chat-completion call against a vLLM
   endpoint launched with ``--tool-call-parser hermes`` +
   ``--enable-auto-tool-choice``, forcing the model to call
   ``submit_answer``.
4. Validate the tool arguments (non-empty answer string, integer
   confidence in ``[0, 100]``, ``evidence_pids`` non-empty and a subset
   of the PR roster). On any failure, append an ERROR message and
   retry up to ``max_retries`` times (3 by default).
5. Persist ``judge_decisions.jsonl`` (one record per qid). Resumable
   via the existing ``load_done_ids`` pattern.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from r2c2_agent.orchestrator.ac_reaggregator import (
    AC_FILE_PG2,
    ATTEMPT_REFUSAL_TEXT,
    CANONICAL_REFUSAL_TEXT,
    PR_FILE_PG2,
    REFUSAL_FLOOR_CONF,
    RUN_NAME_PG2,
    FrozenAC,
    FrozenPR,
    _extract_submit_attempts_from_trace,
    _Rollout,
    _sanitize_ac_answer,
    _synthesize_rescue_text,
    build_pid_index,
    load_frozen_ac,
    load_frozen_pr,
    load_rollouts_for_run,
    map_pr_rows_to_pids,
)

log = logging.getLogger(__name__)

__all__ = [
    "JudgeDecision",
    "JudgeEvidencePack",
    "TOOL_SPEC",
    "build_evidence_pack",
    "call_judge_with_retries",
    "load_judge_decisions",
    "run_judge_pass",
    "main",
]

# Prompt constants.
_JUDGE_SYSTEM_PROMPT = (
    "You are an aggregation judge for an evidence-driven QA task.\n\n"
    "You see one question, a list of passages (the PR roster), the "
    "model's submit_answer attempts across K reasoning iterations "
    "(with confidence scores and citations), and verified-supporting "
    "nuggets the rollouts produced.\n\n"
    "Reason carefully through the evidence and form your own answer:\n"
    "- You may synthesise an answer that combines insights from multiple "
    "attempts, OR pick one attempt's answer verbatim if it is best "
    "supported.\n"
    "- Confidence (0-100) reflects how strongly the corpus supports your "
    "answer. Strong cross-iter agreement + high-NLI nuggets -> high "
    "confidence. Disagreement or weak evidence -> moderate-to-low "
    'confidence. Use 0-30 for "essentially unsupported"; do NOT pick '
    "confidence 0 unless you would refuse to answer.\n"
    "- evidence_pids: the passage IDs (from the PR roster shown above) "
    "that most directly support your answer. Cite at least 1, at most "
    "10.\n\n"
    "You must call the submit_answer tool exactly once. Do not produce "
    "prose outside the tool call."
)

_PASSAGE_TEXT_MAX_CHARS = 300
_NUGGET_TEXT_MAX_CHARS = 200
_MAX_NUGGETS_IN_PROMPT = 20
_MAX_ATTEMPTS_IN_PROMPT = 40
# Qwen3 thinking traces in this judge routinely consume 2-4k tokens before
# the tool call is emitted. When the combined thinking + tool-call output
# is truncated mid-string, vLLM's hermes parser fails with
# ``Unterminated string starting at: line 1 column N``. The budget matches
# the reconciler and worker default, which is large enough to fit Qwen3's
# reasoning channel plus the tool call.
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_MAX_RETRIES = 3

# Tool spec for the judge's submit_answer call. Mirrors the live
# pipeline's tool name so that the model's tool-calling priors transfer
# directly, but we re-validate every field locally.
TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": ("Submit your final answer with confidence and citations."),
        "parameters": {
            "type": "object",
            "required": ["answer", "confidence", "evidence_pids"],
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Your answer text. Avoid raw `;` (will be sanitised)."
                    ),
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "How confident the corpus supports this answer (0-100)."
                    ),
                },
                "evidence_pids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                    "description": ("PR roster passage IDs supporting your answer."),
                },
            },
        },
    },
}


# Datatypes


@dataclass(frozen=True)
class _AttemptRow:
    """One submit_answer attempt rendered into the user prompt."""

    iter_dir: int
    attempt_idx: int  # ordinal within (iter_dir, ensemble_idx)
    classification: str
    confidence: float | None
    n_eff: int | None
    answer_text: str
    citation_pids: tuple[str, ...]


@dataclass(frozen=True)
class _NuggetRow:
    """One verified-supporting nugget rendered into the user prompt."""

    nli_score: float
    primary_pid: str
    text: str


@dataclass(frozen=True)
class JudgeEvidencePack:
    """All evidence shown to the LLM judge for one qid."""

    qid: str
    question_text: str
    pr_roster: tuple[tuple[str, int, str], ...]  # (pid, rank, text)
    attempts: tuple[_AttemptRow, ...]
    nuggets: tuple[_NuggetRow, ...]


@dataclass
class JudgeDecision:
    """Per-qid LLM judge decision (one JSONL record).

    The redesigned schema replaces ``raw_choice``/``final_choice``'s
    closed enum with the open ``answer`` produced by the tool call.
    ``final_choice`` is now always ``"tool_call"`` on success; legacy
    enum strings are retired.
    """

    qid: str
    model: str
    raw_choice: str  # always "tool_call" on success.
    raw_rationale: str  # the model's pre-tool prose (CoT or empty).
    raw_evidence_pids: list[str]  # raw tool-arg list (pre-validation).
    final_choice: str  # "tool_call" | "" (on parse_failed).
    final_answer: str
    final_confidence: int
    final_citation_pids: list[str]
    fallback_reason: str | None
    parse_failed: bool
    ts: str
    n_retries: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# Evidence pack construction


def _render_passage_block(
    pid: str,
    text: str,
    *,
    max_chars: int = _PASSAGE_TEXT_MAX_CHARS,
) -> str:
    """Render one ``[P:<pid> rank N] <passage_text>`` line for the prompt."""
    snippet = _synthesize_rescue_text(text) if text else ""
    if not snippet:
        snippet = text.strip()[:max_chars]
    snippet = snippet[:max_chars].strip()
    return f"[P:{pid}] {snippet}"


def _build_attempt_rows(
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    *,
    max_attempts: int = _MAX_ATTEMPTS_IN_PROMPT,
) -> tuple[_AttemptRow, ...]:
    """Walk every rollout in iter order and emit one row per submit_answer attempt."""
    rows: list[_AttemptRow] = []
    rollouts_sorted = sorted(rollouts, key=lambda r: (r.iter_dir, r.ensemble_idx))
    for r in rollouts_sorted:
        attempts = _extract_submit_attempts_from_trace(r.raw_trace)
        for attempt_idx, a in enumerate(attempts, start=1):
            answer_text = (a.answer_text or "").strip()
            if not answer_text:
                continue
            # Truncate the answer for prompt budget; full text is in trace.
            sanitised = _sanitize_ac_answer(answer_text) or answer_text
            sanitised = sanitised[:600]
            filtered_pids = tuple(p for p in a.citation_pids if p in rank_map)
            rows.append(
                _AttemptRow(
                    iter_dir=r.iter_dir,
                    attempt_idx=attempt_idx,
                    classification=a.classification,
                    confidence=a.confidence,
                    n_eff=a.n_eff,
                    answer_text=sanitised,
                    citation_pids=filtered_pids,
                )
            )
    # Cap at max_attempts, preferring earlier-iter rows (already sorted).
    if len(rows) > max_attempts:
        rows = rows[:max_attempts]
    return tuple(rows)


def _build_nugget_rows(
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    *,
    max_nuggets: int = _MAX_NUGGETS_IN_PROMPT,
) -> tuple[_NuggetRow, ...]:
    """Collect verified-supporting nuggets across rollouts and pick top-K by NLI."""
    rows: list[_NuggetRow] = []
    seen_texts: set[str] = set()
    for r in rollouts:
        for n in r.nuggets:
            if not isinstance(n, dict):
                continue
            text = (n.get("text") or "").strip()
            if not text:
                continue
            pid = n.get("passage_id") or ""
            if pid not in rank_map:
                # Fall back to the first supporting pid that is on the roster.
                supporting = n.get("supporting_pids") or []
                for sp in supporting:
                    if sp in rank_map:
                        pid = sp
                        break
                if pid not in rank_map:
                    continue
            nli = float(n.get("nli_score") or 0.0)
            key = (pid, text[:80])
            key_str = f"{key[0]}::{key[1]}"
            if key_str in seen_texts:
                continue
            seen_texts.add(key_str)
            rows.append(
                _NuggetRow(
                    nli_score=nli,
                    primary_pid=pid,
                    text=text[:_NUGGET_TEXT_MAX_CHARS],
                )
            )
    rows.sort(key=lambda r: -r.nli_score)
    return tuple(rows[:max_nuggets])


def build_evidence_pack(
    qid: str,
    question_text: str,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    pr: FrozenPR,
) -> JudgeEvidencePack:
    """Assemble the per-qid evidence pack (no LLM call yet)."""
    # PR roster: (pid, rank, text), ordered by rank ascending.
    rank_to_pid: dict[int, str] = {}
    for pid, rank in rank_map.items():
        rank_to_pid[rank] = pid
    rows_by_rank: dict[int, str] = {
        rank: passage_text for rank, _doc, passage_text in pr.rows_by_qid.get(qid, [])
    }
    pr_roster: list[tuple[str, int, str]] = []
    for rank in sorted(rank_to_pid.keys()):
        pid = rank_to_pid[rank]
        text = rows_by_rank.get(rank, "")
        pr_roster.append((pid, rank, text))

    attempts = _build_attempt_rows(rollouts, rank_map)
    nuggets = _build_nugget_rows(rollouts, rank_map)
    return JudgeEvidencePack(
        qid=qid,
        question_text=question_text,
        pr_roster=tuple(pr_roster),
        attempts=attempts,
        nuggets=nuggets,
    )


# Prompt rendering


def _render_judge_prompt(pack: JudgeEvidencePack) -> tuple[str, str]:
    """Render (system, user) prompt strings for one qid."""
    lines: list[str] = []
    lines.append(f"Question: {pack.question_text}")
    lines.append("")
    lines.append("Frozen PR-2 roster (passages you may cite):")
    for pid, rank, text in pack.pr_roster:
        lines.append(
            "  "
            + _render_passage_block(pid, text).replace(
                f"[P:{pid}]", f"[P:{pid} rank {rank}]"
            )
        )
    lines.append("")
    if pack.attempts:
        lines.append(
            f"Model attempts across {len({a.iter_dir for a in pack.attempts})} "
            f"iterations (each tried to answer):"
        )
        lines.append("")
        for a in pack.attempts:
            conf_str = f"{a.confidence:.2f}" if a.confidence is not None else "n/a"
            n_eff_str = str(a.n_eff) if a.n_eff is not None else "n/a"
            lines.append(
                f"  iter {a.iter_dir}, attempt {a.attempt_idx}: "
                f"class={a.classification}, conf={conf_str}, n_eff={n_eff_str}"
            )
            lines.append(f"    answer: {a.answer_text!r}")
            cited_str = ", ".join(a.citation_pids) if a.citation_pids else ""
            lines.append(f"    cited: [{cited_str}]")
        lines.append("")
    else:
        lines.append("Model attempts: (no submit_answer attempts in any iter)")
        lines.append("")
    if pack.nuggets:
        lines.append("Verified-supporting nuggets recovered from rollouts:")
        for n in pack.nuggets:
            lines.append(
                f"  - NLI={n.nli_score:.2f} cited P:{n.primary_pid}: {n.text!r}"
            )
        lines.append("")
    else:
        lines.append("Verified-supporting nuggets: (none)")
        lines.append("")
    lines.append(
        "Now decide. Call submit_answer with your answer, your honest "
        "confidence (0-100), and the evidence_pids you stand behind."
    )
    return _JUDGE_SYSTEM_PROMPT, "\n".join(lines)


# HTTP call + retry loop


def _post_json(
    url: str, payload: dict[str, Any], *, timeout: int = 900
) -> dict[str, Any]:
    """Issue one OpenAI-compatible POST, return parsed JSON."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")[:600]
        except Exception:  # pragma: no cover
            pass
        raise RuntimeError(f"HTTPError {e.code} {e.reason}: {body_text}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError from vLLM: {e.reason}") from e
    return json.loads(data)


def _normalise_endpoint(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url + "/chat/completions"


def _validate_tool_args(
    args: dict[str, Any],
    pr_rank_map: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Return ``(errors, valid_pids)`` for one tool_call's arguments."""
    errors: list[str] = []
    answer = args.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer must be a non-empty string")
    conf = args.get("confidence")
    # Accept ints and integer-valued floats (some parsers emit e.g. 80.0).
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        errors.append(f"confidence must be integer in [0, 100]; got {conf!r}")
        conf = None
    elif isinstance(conf, float) and not conf.is_integer():
        errors.append(f"confidence must be integer in [0, 100]; got {conf!r}")
        conf = None
    else:
        conf_int = int(conf)
        if not (0 <= conf_int <= 100):
            errors.append(f"confidence must be integer in [0, 100]; got {conf!r}")
    pids = args.get("evidence_pids") or []
    if not isinstance(pids, list) or not pids:
        errors.append("evidence_pids must be a non-empty list")
        return errors, []
    pids_str = [str(p) for p in pids]
    valid_pids = [p for p in pids_str if p in pr_rank_map]
    if not valid_pids:
        errors.append(
            f"None of {pids_str[:5]} appear in the PR roster shown above. "
            "Use the [P:...] ids exactly as listed."
        )
    return errors, valid_pids


def call_judge_with_retries(
    endpoint: str,
    model: str,
    system: str,
    user: str,
    pr_rank_map: dict[str, int],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    timeout: int = 900,
) -> dict[str, Any] | None:
    """Invoke the LLM judge with retry-until-valid tool calls.

    Returns a dict ``{answer, confidence, evidence_pids, rationale,
    n_retries}`` on success or ``None`` if retries are exhausted.
    """
    url = _normalise_endpoint(endpoint)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        # Determinism is desirable on first attempt but counter-productive on
        # retry: an HTTP 400 from vLLM's hermes parser ("Unterminated string")
        # will recur identically with temperature=0.0 + seed=0. Bump both on
        # each retry to perturb sampling and escape the deterministic failure.
        attempt_temp = (
            temperature if attempt == 0 else min(0.7, temperature + 0.3 * attempt)
        )
        attempt_seed = attempt
        # On retry, drop the forced named-function tool_choice in favour of
        # ``"required"``. vLLM's structured-decoder path for forced named
        # functions interacts poorly with the reasoning parser when the
        # model produces a long thinking trace before the tool call; the
        # less constrained ``"required"`` path runs Qwen3's native tool
        # template (the same one the worker pipeline uses successfully),
        # which the hermes parser handles cleanly.
        tool_choice: str | dict[str, Any]
        if attempt == 0:
            tool_choice = {
                "type": "function",
                "function": {"name": "submit_answer"},
            }
        else:
            tool_choice = "required"
        payload = {
            "model": model,
            "messages": messages,
            "tools": [TOOL_SPEC],
            "tool_choice": tool_choice,
            "temperature": attempt_temp,
            "max_tokens": max_tokens,
            "seed": attempt_seed,
            # Mirror the worker / reconciler invocation: Qwen3 reasoning
            # via ``--reasoning-parser qwen3`` requires the chat template
            # ``enable_thinking`` toggle to surface reasoning in the
            # dedicated channel rather than inline in ``content`` (which
            # would force the hermes tool-call parser to interpret the
            # reasoning prose as part of the tool call).
            "chat_template_kwargs": {"enable_thinking": True},
        }
        try:
            resp = _post_json(url, payload, timeout=timeout)
        except RuntimeError as e:
            last_error = f"transport error: {e}"
            log.warning("judge transport error on attempt %d: %s", attempt, e)
            # Don't append the error to messages — for transport-side parser
            # failures, retrying with perturbed temperature + seed is what
            # actually changes the outcome. Polluting messages with vLLM
            # internal-error strings just confuses the next attempt.
            continue

        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            last_error = "No 'choices[0].message' in response."
            log.warning("judge response malformed on attempt %d", attempt)
            continue
        prose = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        rationale = reasoning if reasoning else prose
        tcs = msg.get("tool_calls") or []
        tc = tcs[0] if tcs else None

        if tc is None:
            last_error = "No tool call returned. You MUST call submit_answer."
            messages.append({"role": "assistant", "content": prose})
            messages.append(
                {
                    "role": "user",
                    "content": f"ERROR: {last_error} Try again.",
                }
            )
            continue

        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        args: dict[str, Any] = {}
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                last_error = f"Tool arguments were not valid JSON: {e}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": prose,
                        "tool_calls": [tc],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": f"ERROR: {last_error} Try again.",
                    }
                )
                continue
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            last_error = "Tool arguments missing."
            messages.append({"role": "assistant", "content": prose, "tool_calls": [tc]})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": f"ERROR: {last_error} Try again.",
                }
            )
            continue

        errors, valid_pids = _validate_tool_args(args, pr_rank_map)
        if errors:
            last_error = "; ".join(errors)
            log.info(
                "judge attempt %d invalid: %s (args=%s)",
                attempt,
                last_error,
                str(args)[:200],
            )
            messages.append({"role": "assistant", "content": prose, "tool_calls": [tc]})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": f"ERROR: {last_error} Try again.",
                }
            )
            continue

        return {
            "answer": _sanitize_ac_answer(str(args["answer"]).strip()) or "",
            "confidence": int(args["confidence"]),
            "evidence_pids": valid_pids[:10],
            "rationale": (rationale or "").strip()[:1500],
            "raw_evidence_pids": [str(p) for p in args.get("evidence_pids") or []],
            "n_retries": attempt,
        }
    log.error("judge exhausted %d retries; last error: %s", max_retries, last_error)
    return None


# Decision finaliser


def _decision_from_tool_result(
    qid: str,
    model: str,
    result: dict[str, Any],
    ts: str,
) -> JudgeDecision:
    """Build a ``JudgeDecision`` from a successful tool-call result."""
    answer = result["answer"]
    conf = max(0, min(100, int(result["confidence"])))
    return JudgeDecision(
        qid=qid,
        model=model,
        raw_choice="tool_call",
        raw_rationale=result.get("rationale", ""),
        raw_evidence_pids=list(result.get("raw_evidence_pids") or []),
        final_choice="tool_call",
        final_answer=answer,
        final_confidence=conf,
        final_citation_pids=list(result["evidence_pids"]),
        fallback_reason=None,
        parse_failed=False,
        ts=ts,
        n_retries=int(result.get("n_retries", 0)),
    )


def _decision_failed(
    qid: str,
    model: str,
    ts: str,
    *,
    reason: str = "retries_exhausted",
) -> JudgeDecision:
    """Build a parse_failed ``JudgeDecision``. Guardrail will reject this."""
    return JudgeDecision(
        qid=qid,
        model=model,
        raw_choice="",
        raw_rationale="",
        raw_evidence_pids=[],
        final_choice="",
        final_answer="",
        final_confidence=0,
        final_citation_pids=[],
        fallback_reason=reason,
        parse_failed=True,
        ts=ts,
        n_retries=0,
    )


# Loaders + driver


def load_judge_decisions(path: Path) -> dict[str, JudgeDecision]:
    """Load decisions from a JSONL produced by ``run_judge_pass``."""
    out: dict[str, JudgeDecision] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed judge_decisions line in %s", path)
                continue
            try:
                dec = JudgeDecision(**rec)
            except TypeError:
                log.warning(
                    "schema mismatch on judge_decisions row for qid=%s", rec.get("qid")
                )
                continue
            out[dec.qid] = dec
    return out


def _load_done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("qid")
            if isinstance(qid, str):
                out.add(qid)
    return out


def _load_question_texts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(rec.get("question_id") or rec.get("qid") or "").strip()
            if not qid:
                continue
            q = rec.get("question") or rec.get("question_text") or ""
            out[qid] = str(q)
    return out


def run_judge_pass(
    traces_dir: Path,
    pr_dir: Path,
    questions_path: Path,
    endpoint: str,
    output_path: Path,
    *,
    model: str = "Qwen/Qwen3.5-122B-A10B-FP8",
    qid_filter: set[str] | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_workers: int = 1,
) -> int:
    """Run the full LLM-judge pass over all (or filtered) qids.

    Idempotent / resumable: appends one JSON record per qid to
    ``output_path``; replays the file to skip already-done qids. ``fsync``
    after each write. When ``max_workers > 1``, qids are dispatched
    concurrently against the vLLM endpoint (the only shared state is the
    output file, which is protected by a ``threading.Lock``).
    """
    pr_path = pr_dir / PR_FILE_PG2
    ac_path = pr_dir / AC_FILE_PG2
    pr = load_frozen_pr(pr_path, RUN_NAME_PG2)
    _ = load_frozen_ac(ac_path, RUN_NAME_PG2)  # parsed for schema check only
    rollouts_by_qid = load_rollouts_for_run(traces_dir, first_iter_only=False)
    pid_index = build_pid_index(rollouts_by_qid)
    rank_maps = map_pr_rows_to_pids(pr, pid_index)
    question_texts = _load_question_texts(questions_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_qids(output_path)

    qids = sorted(pr.qids())
    if qid_filter is not None:
        qids = [q for q in qids if q in qid_filter]
    log.info(
        "llm-judge pass: %d qids total, %d already done, %d to process "
        "(max_workers=%d)",
        len(qids),
        len(done),
        sum(1 for q in qids if q not in done),
        max_workers,
    )

    todo = [q for q in qids if q not in done]
    if not todo:
        log.info("llm-judge pass: nothing to do, all %d qids already done", len(qids))
        return 0

    def _judge_one(qid: str) -> tuple[str, JudgeDecision]:
        rollouts = rollouts_by_qid.get(qid, [])
        rank_map = rank_maps.get(qid, {})
        question_text = question_texts.get(qid, "")
        pack = build_evidence_pack(qid, question_text, rollouts, rank_map, pr)
        system, user = _render_judge_prompt(pack)
        ts = _now_iso()
        result = call_judge_with_retries(
            endpoint,
            model,
            system,
            user,
            rank_map,
            max_retries=max_retries,
        )
        if result is None:
            return qid, _decision_failed(qid, model, ts)
        return qid, _decision_from_tool_result(qid, model, result, ts)

    n_processed = 0
    n_failed = 0
    write_lock = threading.Lock()
    workers = max(1, int(max_workers))
    with open(output_path, "a") as f_out:
        if workers == 1:
            iterator = (_judge_one(qid) for qid in todo)
            for qid, dec in iterator:
                if getattr(dec, "parse_failed", False):
                    n_failed += 1
                else:
                    n_processed += 1
                f_out.write(dec.to_json() + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())
                log.info(
                    "qid=%s answer=%r conf=%d pids=%s retries=%d",
                    qid,
                    dec.final_answer[:60],
                    dec.final_confidence,
                    dec.final_citation_pids[:3],
                    dec.n_retries,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_judge_one, qid): qid for qid in todo}
                for fut in concurrent.futures.as_completed(futures):
                    qid_done, dec = fut.result()
                    if getattr(dec, "parse_failed", False):
                        n_failed += 1
                    else:
                        n_processed += 1
                    with write_lock:
                        f_out.write(dec.to_json() + "\n")
                        f_out.flush()
                        os.fsync(f_out.fileno())
                    log.info(
                        "qid=%s answer=%r conf=%d pids=%s retries=%d",
                        qid_done,
                        dec.final_answer[:60],
                        dec.final_confidence,
                        dec.final_citation_pids[:3],
                        dec.n_retries,
                    )

    log.info("llm-judge pass complete: %d processed, %d failed", n_processed, n_failed)
    return 0 if n_failed == 0 else 1


def _now_iso() -> str:
    """Return UTC timestamp as ISO 8601 (without subsecond)."""
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``r2c2_agent llm-judge``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-dir", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Source questions JSONL (id + question text).",
    )
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        required=True,
        help="OpenAI-compatible base URL, e.g. http://host:11000/v1",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-122B-A10B-FP8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qid-filter",
        type=str,
        default=None,
        help="Comma-separated qids for smoke testing.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=_DEFAULT_MAX_RETRIES,
        help="Max retries on malformed tool calls (default: 3).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    qid_filter: set[str] | None = None
    if args.qid_filter:
        qid_filter = {q.strip() for q in args.qid_filter.split(",") if q.strip()}

    return run_judge_pass(
        traces_dir=args.traces_dir,
        pr_dir=args.pr_dir,
        questions_path=args.questions,
        endpoint=args.vllm_endpoint,
        output_path=args.output,
        model=args.model,
        qid_filter=qid_filter,
        max_retries=args.max_retries,
        max_workers=args.concurrency,
    )


# Module-level re-exports — silence unused-import warnings.
_ = (FrozenAC, REFUSAL_FLOOR_CONF, CANONICAL_REFUSAL_TEXT, ATTEMPT_REFUSAL_TEXT)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
