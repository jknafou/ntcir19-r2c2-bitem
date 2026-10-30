"""Post-answer pipeline: 10-step nugget reconciliation.

After a trace is produced, runs a purpose-built pipeline optimized for R2C2
evaluation metrics (Nugget Precision, PR nDCG@20).

Pipeline:
    Step 0: Answer format verification (regex + optional LLM)
    Step 1: Relevant passage extraction (1 LLM + parallel extract MCP)
    Step 2: Parallel nugget generation (1 LLM per passage)
    Step 3: Parallel bogus verification (1 entailment check per nugget)
    Step 5: Deduplication (cluster-merge with multi-passage support)
    Step 6: Re-verification post-dedup
    Step 7: Completeness check (1 gap-fill retry if critical)
    Step 8: Passage pruning >20 (greedy set cover, deterministic)
    Step 9: Nugget pruning >10 (pre-filter + LLM selection)
    Step 10: Relevance scoring (parallel, 1 LLM per nugget)

Max 20 passages, max 10 nuggets per question.
All calls use the worker's vLLM instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "reconcile_nuggets_and_passages",
    # v2 pipeline: data model and seeding
    "PipelineDB",
    "Passage",
    "Nugget",
    "AnswerState",
    "TransitionEvent",
    "PassageAlias",
    "PassageStatus",
    "NuggetStatus",
    "Stance",
    "NLILabel",
    "ExitReason",
    "Supportability",
    "effective_supporting_ids",
    "effective_supporting_pids",
    "max_support_nli_score",
    "question_parser",
    "seed_db_from_trace",
    "append_event",
    "event_log_path",
    "REFUSAL_PID",
    "REFUSAL_PASSAGE_TEXT",
]

log = logging.getLogger(__name__)

# Constants

MAX_PASSAGES = 20
MAX_NUGGETS = 10
ANSWER_MAX_CHARS = 150
_ENTAILMENT_CHAR_LIMIT = 2000

# Valid question types (from training data)
_VALID_TYPES = {
    "simple",
    "aggregation",
    "comparison",
    "simple_w_condition",
    "multi_hop",
    "post_processing",
}

# Per-passage nugget caps by question type
_NUGGET_CAP_BY_TYPE: dict[str, int] = {
    "simple": 3,
    "simple_w_condition": 3,
    "aggregation": 5,
    "comparison": 4,
    "multi_hop": 4,
    "post_processing": 4,
}
_NUGGET_CAP_DEFAULT = 4
_NUGGET_GLOBAL_CAP = 30


# v2 pipeline: a 6-step agentic loop over typed Pydantic records, offered as
# an alternative to the 10-step JSON-parsing pipeline further down. This part
# holds the data model, the derived helpers, the seeding from a trace and the
# event log.

# Literal types

PassageStatus = Literal["unclassified", "accepted", "rejected", "counter_evidence"]
# "refusal" is the 5th status, synthesized by the controller when no
# supporting evidence exists. It is terminal and never held by real nuggets.
NuggetStatus = Literal[
    "unclassified",
    "verified_supporting",
    "verified_infirming",
    "bogus",
    "refusal",
]
Stance = Literal["supporting", "infirming"]
NLILabel = Literal["entails", "contradicts", "neutral"]
ExitReason = Literal["high_confidence", "low_confidence_refusal", "max_iter", "error"]
Supportability = Literal["sufficient", "insufficient", "contradicted", "unresolvable"]
EventActor = Literal["llm_tool", "seed", "system"]

# Constants for refusal synthesis

REFUSAL_PID = "__refusal__"
REFUSAL_PASSAGE_TEXT = (
    "No passage in the retrieved corpus supports an answer to this question."
)
REFUSAL_ANSWER_TEXT = (
    "No passage in the retrieved corpus was found to support an answer; "
    "insufficient evidence to respond."
)
REFUSAL_REGEX = re.compile(
    r"(?i)\b(cannot|unable|insufficient|no evidence|refuse|nothing (?:was|has) been found)\b"
)

# Budget caps, bound into PipelineDB at seed time.
# The passage cap is soft and set high: parallel search batches can seed more
# than a hundred pids, and a lower cap made validate_finish reject every
# refusal attempt. The hard cap is on the submitted passages
# (MAX_PASSAGES=20) and is enforced when the submission is generated.

V2_MAX_PASSAGES = 200
V2_MAX_NUGGETS_GLOBAL = 40
V2_MAX_NUGGETS_PER_PASSAGE = 8
V2_MAX_TOOL_CALLS_GLOBAL = 300  # S3 alone takes ~120 calls (4 × 30 nuggets)
V2_MAX_ITER = 6

# [strengthens-by-negation] tag parser

_STRENGTHEN_TAG_RE = re.compile(
    r"(?m)^(n_[0-9a-f]{8})\s*:\s*\[?strengthens-by-negation\]?\s*$"
)

# Question-parser constants

_Q_STOPWORDS = {
    "what",
    "who",
    "how",
    "why",
    "when",
    "where",
    "which",
    "the",
    "a",
    "an",
    "is",
    "was",
    "did",
    "does",
    "do",
    "are",
    "were",
    "has",
    "have",
    "had",
}
_WORD_TO_INT: dict[str, int | None] = {"few": 3, "several": 4, "all": None}
_EXPECTED_COUNT_RE = re.compile(
    # Allow an optional article (a/the) between the trigger and the count word,
    # so "name a few X" / "list the 3 Y" parse correctly.
    r"\b(?:how many|list|name|identify|count)\s+(?:(?:a|the)\s+)?(\d+|few|several|all)\b",
    re.IGNORECASE,
)
# For questions like "How many Star Wars films are there?" where "how many"
# has no integer immediately after — fallback to aggregation default.
_HOW_MANY_RE = re.compile(r"\bhow many\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r'"([^"]+)"|([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*)')

# Pydantic models


class TransitionEvent(BaseModel):
    """One mutation to the DB. Append-only; replay-deterministic."""

    ts: float = Field(default_factory=time.time)
    step_id: str
    iteration: int
    actor: EventActor
    tool_call_id: str | None = None
    from_status: str | None = None
    to_status: str
    reason: str = ""
    thinking: str | None = None  # sliced from message.reasoning


class PassageAlias(BaseModel):
    """Secondary occurrence of a pid across multiple MCP tool calls."""

    tool_call_id: str
    msg_idx: int
    score: float | None = None


class Passage(BaseModel):
    passage_id: str
    text: str
    title: str | None = None
    score: float | None = None
    source_tool_call_id: str
    source_msg_idx: int
    first_seen_iter: int = 0
    last_reviewed_iter: int = 0
    aliases: list[PassageAlias] = Field(default_factory=list)
    counter_for_answer: str = ""
    status: PassageStatus = "unclassified"
    status_reason: str = ""
    history: list[TransitionEvent] = Field(default_factory=list)


class Nugget(BaseModel):
    nugget_id: str = Field(default_factory=lambda: f"n_{uuid.uuid4().hex[:8]}")
    passage_id: str
    text: str
    proposed_stance: Stance = "supporting"
    source_step: str
    source_tool_call_id: str
    status: NuggetStatus = "unclassified"
    status_reason: str = ""
    nli_label: NLILabel | None = None
    nli_score: float | None = None
    history: list[TransitionEvent] = Field(default_factory=list)
    # Hybrid-cascade breakdown, populated by ``_commit_nugget`` in
    # ``pipeline_handlers.py``: one of {polarity_flip, entity_swap,
    # quantifier_drift, lexical_overlap_override, llm_judge_tiebreak,
    # minicheck_default}. ``hybrid_signals`` carries the underlying
    # signal values for trace audits and as a training signal. Both are
    # optional so that callers which do not run the cascade can omit them.
    hybrid_rule: str | None = None
    hybrid_signals: dict[str, Any] = Field(default_factory=dict)
    # One nugget per claim; the passages corroborating the claim are
    # listed here. ``passage_id`` (above) remains the primary passage,
    # where the nugget was first recorded, and is used by pid filters and
    # ``_pick_citation_pid``. ``supporting_pids`` includes the primary pid
    # plus every later pid whose quoted span entailed the same text
    # (NLI ≥ VERIFY_ENTAIL_THRESHOLD). No duplicates; every entry has a
    # matching ``pid_nli_scores`` entry.
    supporting_pids: list[str] = Field(default_factory=list)
    pid_nli_scores: dict[str, float] = Field(default_factory=dict)
    # Telemetry-only flag. When true, this nugget is a bookkeeping
    # placeholder synthesised by ``force_deliver`` and is not counted as
    # corpus evidence: ``adapter_candidate_filter`` excludes it from the
    # emitted ``reconciled_nuggets`` list so that nugget counts reflect
    # only nuggets grounded in retrieved passages. It stays in
    # ``db.nuggets`` for replay and audit.
    is_synthetic: bool = False

    @model_validator(mode="after")
    def _seed_supporting_pids(self) -> Nugget:
        """Backward-compat seeding for verified_supporting nuggets.

        Legacy trace rows (pre Option B refactor) have empty
        ``supporting_pids``. When the nugget is verified_supporting and
        anchored to a real ``passage_id``, seed ``supporting_pids`` with
        that pid and ``pid_nli_scores`` with the recorded ``nli_score``
        (falling back to ``VERIFY_ENTAIL_THRESHOLD=0.60`` when the legacy
        row had no score).

        Deduplicate the ``supporting_pids`` list in place so callers
        cannot accidentally double-count a pid.
        """
        # Dedup while preserving order.
        if self.supporting_pids:
            seen: set[str] = set()
            deduped: list[str] = []
            for pid in self.supporting_pids:
                if pid and pid not in seen:
                    seen.add(pid)
                    deduped.append(pid)
            if deduped != self.supporting_pids:
                self.supporting_pids = deduped

        # Legacy seed path: only for verified_supporting rows with a
        # real anchor.
        if (
            self.status == "verified_supporting"
            and self.passage_id
            and not self.supporting_pids
        ):
            self.supporting_pids = [self.passage_id]
            if self.passage_id not in self.pid_nli_scores:
                self.pid_nli_scores[self.passage_id] = (
                    float(self.nli_score) if self.nli_score is not None else 0.60
                )
        return self


class AnswerState(BaseModel):
    draft_answer: str = ""
    confidence: float = 0.0
    reasoning_trace: str = ""
    supportability: Supportability | None = None
    missing: str = ""
    rationale: str = ""
    expected_count: int | None = None
    counting_note: str = ""
    question_entities: list[str] = Field(default_factory=list)
    prev_missing: str = ""
    prev_accepted_pid_count: int = 0
    iteration_count: int = 0
    retry_count: dict[str, int] = Field(default_factory=dict)
    exit_reason: ExitReason | None = None
    history: list[TransitionEvent] = Field(default_factory=list)


class PipelineDB(BaseModel):
    question_id: str
    question: str
    question_type: str = ""
    passages: dict[str, Passage] = Field(default_factory=dict)
    nuggets: dict[str, Nugget] = Field(default_factory=dict)
    answer: AnswerState = Field(default_factory=AnswerState)
    events: list[TransitionEvent] = Field(default_factory=list)

    max_passages: int = V2_MAX_PASSAGES
    max_nuggets_global: int = V2_MAX_NUGGETS_GLOBAL
    max_nuggets_per_passage: int = V2_MAX_NUGGETS_PER_PASSAGE
    max_tool_calls_global: int = V2_MAX_TOOL_CALLS_GLOBAL
    max_iter: int = V2_MAX_ITER


# Derived helpers


def effective_supporting_ids(db: PipelineDB) -> set[str]:
    """Return nugget_ids counted as 'supporting' for gate/validator purposes.

    Union of:
      - verified_supporting nugget_ids (direct)
      - verified_infirming nugget_ids tagged [strengthens-by-negation] in
        answer.rationale (elimination evidence)

    Read before S4→S5 advance, finish validator high_confidence branch,
    refusal trigger, and confidence formula (s_elim term).
    """
    direct = {
        n.nugget_id
        for n in db.nuggets.values()
        if n.status == "verified_supporting" and not getattr(n, "is_synthetic", False)
    }
    tagged = set(_STRENGTHEN_TAG_RE.findall(db.answer.rationale or ""))
    elim = {
        n.nugget_id
        for n in db.nuggets.values()
        if n.status == "verified_infirming"
        and n.nugget_id in tagged
        and not getattr(n, "is_synthetic", False)
    }
    return direct | elim


def effective_supporting_pids(db: PipelineDB) -> set[str]:
    """Return the set of distinct passages supporting verified claims.

    ``n_eff`` counts passages, not nugget rows. One claim corroborated by
    3 passages contributes 3 to ``n_eff`` through a single nugget's
    ``supporting_pids`` list, without needing 3 duplicate nuggets.

    Union of:
      - every pid in ``supporting_pids`` across ``verified_supporting``
        nuggets (direct corroboration);
      - every pid in ``supporting_pids`` of ``verified_infirming``
        nuggets whose ``nugget_id`` is tagged ``[strengthens-by-negation]``
        in ``db.answer.rationale`` (elimination evidence).

    Trace rows without a populated ``supporting_pids`` list are
    backfilled by the ``Nugget`` model validator using the primary
    ``passage_id`` so downstream callers see a consistent view.
    """
    pids: set[str] = set()
    tagged = set(_STRENGTHEN_TAG_RE.findall(db.answer.rationale or ""))
    for n in db.nuggets.values():
        if getattr(n, "is_synthetic", False):
            continue
        if n.status == "verified_supporting":
            fallback = [n.passage_id] if n.passage_id else []
            for pid in n.supporting_pids or fallback:
                if pid:
                    pids.add(pid)
        elif n.status == "verified_infirming" and n.nugget_id in tagged:
            fallback = [n.passage_id] if n.passage_id else []
            for pid in n.supporting_pids or fallback:
                if pid:
                    pids.add(pid)
    return pids


def max_support_nli_score(db: PipelineDB) -> float:
    """Return the highest NLI entailment score across supporting evidence.

    Used by the hard gate's thin-evidence branch: when ``n_eff == 1``,
    the gate can clear a commit if the single supporting passage entails
    the claim with high confidence (NLI score >=
    THIN_EVIDENCE_NLI_THRESHOLD in pipeline_handlers.py).

    Scans ``verified_supporting`` nuggets and returns the max of both
    the legacy ``nli_score`` field and every entry in ``pid_nli_scores``.
    Returns 0.0 if the DB has no verified_supporting nuggets.
    """
    best = 0.0
    for n in db.nuggets.values():
        if n.status != "verified_supporting":
            continue
        if getattr(n, "is_synthetic", False):
            continue
        if n.nli_score is not None and n.nli_score > best:
            best = float(n.nli_score)
        for score in n.pid_nli_scores.values():
            if score is not None and score > best:
                best = float(score)
    return best


def question_parser(question: str, qtype: str) -> tuple[list[str], int | None, str]:
    """Seeding-time deterministic parse. No LLM. No NER library.

    Returns (question_entities, expected_count, counting_note).

    Uses `_ENUM_RULES`, which is defined further down; the forward
    reference is resolved at call time.
    """
    # Entities: capitalized token-runs + quoted strings, minus stopwords
    tokens = _ENTITY_RE.findall(question)
    entities = [t[0] or t[1] for t in tokens if (t[0] or t[1])]
    entities = [e for e in entities if e.lower() not in _Q_STOPWORDS]

    # expected_count: prefer explicit integer, else fuzzy word, else fallback
    expected_count: int | None = None
    m = _EXPECTED_COUNT_RE.search(question)
    if m:
        v = m.group(1).lower()
        if v.isdigit():
            expected_count = int(v)
        else:
            expected_count = _WORD_TO_INT.get(v)

    # Aggregation fallback: "how many X" with no explicit count → default 5
    if qtype == "aggregation" and expected_count is None:
        if _HOW_MANY_RE.search(question):
            expected_count = 5

    # counting_note: _ENUM_RULES is a forward reference, resolved at call
    # time. Safe because question_parser is only called from
    # seed_db_from_trace, which runs after module load completes.
    try:
        counting_note = _ENUM_RULES.get(qtype, "")  # type: ignore[name-defined]
    except NameError:
        counting_note = ""

    return entities, expected_count, counting_note


# Event log writer


def event_log_path(qid: str) -> Path:
    """Per-question event log path in worker-local scratch.

    Events go to $SLURM_TMPDIR/events_<qid>.jsonl and not to the shared
    BeeGFS output directory, so that the orchestrator's load_done_ids()
    glob does not mistake event files for completed records.

    Falls back to /tmp if SLURM_TMPDIR is unset (local dev / tests).
    """
    scratch = os.environ.get("SLURM_TMPDIR") or "/tmp"
    return Path(scratch) / f"events_{qid}.jsonl"


def append_event(db: PipelineDB, event: TransitionEvent) -> None:
    """Append to the in-memory event list and the persistent JSONL log.

    fsync after each write so crash-resume replay sees the event.
    """
    db.events.append(event)
    path = event_log_path(db.question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = event.model_dump_json() + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# Seeder


def _seed_virtual_refusal_passage(db: PipelineDB) -> None:
    """Unconditionally add the virtual __refusal__ passage.

    Required for the refusal-synthesis FK invariant: the synthesized
    refusal nugget must anchor to a real passage row. Kept out of S1/S2/S3
    scope by default (status='rejected'); the adapter uses it for PR rank-1
    on the refusal path.
    """
    if REFUSAL_PID in db.passages:
        return
    ev = TransitionEvent(
        step_id="seed",
        iteration=0,
        actor="seed",
        from_status=None,
        to_status="rejected",
        reason="virtual __refusal__ passage",
    )
    p = Passage(
        passage_id=REFUSAL_PID,
        text=REFUSAL_PASSAGE_TEXT,
        title=REFUSAL_PID,
        score=None,
        source_tool_call_id="seed",
        source_msg_idx=-1,
        first_seen_iter=0,
        status="rejected",
        status_reason="virtual refusal-path anchor",
        history=[ev],
    )
    db.passages[REFUSAL_PID] = p
    append_event(db, ev)


def _extract_draft_answer(messages: list[dict]) -> str:
    """Extract the AC block from the last assistant message, if any.

    Reuses src/r2c2_agent/shared/answer_extraction.py to stay consistent with
    worker.py's own draft extraction.
    """
    try:
        from r2c2_agent.shared.answer_extraction import (  # type: ignore[import]
            extract_answer_and_confidence,
        )
    except Exception:
        extract_answer_and_confidence = None  # type: ignore[assignment]

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        if extract_answer_and_confidence is not None:
            try:
                ans, _conf = extract_answer_and_confidence(content)
            except Exception:
                ans = ""
            if ans:
                return ans
        # Fallback: look for "Answer:" line
        m = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", content)
        if m:
            return m.group(1).strip()
    return ""


def _seed_passages_from_trace(
    db: PipelineDB,
    messages: list[dict],
) -> None:
    """Walk tool-result messages and build Passage rows.

    Reuses `_extract_passages_from_tool_result` from
    `r2c2_agent.agent.postprocess`, which mutates three dicts in place.
    """
    try:
        from r2c2_agent.agent.postprocess import (  # type: ignore[import]
            _extract_passages_from_tool_result,
        )
    except Exception as exc:
        log.warning("v2 seed: postprocess import failed (%s); passages empty", exc)
        return

    freq: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    passage_data: dict[str, dict[str, str]] = {}
    occurrences: dict[str, list[tuple[str, int, float | None]]] = {}

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
        else:
            data = content

        pids_before = set(passage_data.keys())
        try:
            _extract_passages_from_tool_result(
                data, msg_idx, freq, first_seen, passage_data
            )
        except Exception as exc:
            log.debug("v2 seed: tool_result parse failed at msg %d: %s", msg_idx, exc)
            continue

        tool_call_id = msg.get("tool_call_id") or f"msg_{msg_idx}"
        pids_after = set(passage_data.keys())
        for pid in pids_after - pids_before:
            score_raw = passage_data.get(pid, {}).get("score")
            try:
                score = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                score = None
            occurrences.setdefault(pid, []).append((tool_call_id, msg_idx, score))
        for pid in pids_before & pids_after:
            score_raw = passage_data.get(pid, {}).get("score")
            try:
                score = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                score = None
            occs = occurrences.setdefault(pid, [])
            if not any(o[0] == tool_call_id for o in occs):
                occs.append((tool_call_id, msg_idx, score))

    for pid, meta in passage_data.items():
        occs = occurrences.get(pid, [])
        if not occs:
            continue
        first_tool_call_id, first_msg_idx, first_score = occs[0]
        aliases = [
            PassageAlias(tool_call_id=tcid, msg_idx=mi, score=sc)
            for tcid, mi, sc in occs[1:]
        ]
        text = meta.get("text", "") or ""
        title = meta.get("title")
        ev = TransitionEvent(
            step_id="seed",
            iteration=0,
            actor="seed",
            from_status=None,
            to_status="unclassified",
            reason=f"seeded from tool_call {first_tool_call_id}",
        )
        p = Passage(
            passage_id=pid,
            text=text,
            title=title if title else None,
            score=first_score,
            source_tool_call_id=first_tool_call_id,
            source_msg_idx=first_msg_idx,
            first_seen_iter=0,
            aliases=aliases,
            status="unclassified",
            history=[ev],
        )
        db.passages[pid] = p
        append_event(db, ev)


def seed_db_from_trace(
    messages: list[dict],
    question: str,
    qid: str,
    question_type: str = "",
) -> PipelineDB:
    """Build a fresh PipelineDB from an existing agent trace.

    Steps:
      1. Create empty DB.
      2. Walk tool-result messages → Passage rows.
      3. Extract draft_answer from last assistant AC block.
      4. question_parser → entities, expected_count, counting_note.
      5. Nuggets start empty.
      6. Seed the virtual __refusal__ passage unconditionally.
    """
    qtype = question_type
    if not qtype:
        try:
            qtype = _infer_question_type(question)  # type: ignore[name-defined]
        except NameError:
            qtype = ""

    db = PipelineDB(
        question_id=qid,
        question=question,
        question_type=qtype,
    )

    entities, expected_count, counting_note = question_parser(question, qtype)
    db.answer.question_entities = entities
    db.answer.expected_count = expected_count
    db.answer.counting_note = counting_note

    db.answer.draft_answer = _extract_draft_answer(messages)

    _seed_passages_from_trace(db, messages)
    _seed_virtual_refusal_passage(db)

    return db


# v2 pipeline: tool handlers and finish validator.
# The 12 tool handlers the LLM calls during S1-S5 (list_passages, read_passage,
# mark_passage, list_nuggets, read_nugget, propose_nugget, mark_nugget,
# check_entailment, read_answer, set_supportability, update_answer, finish).
#
# Each handler takes a ToolContext (DB + per-turn state) and kwargs and returns
# a dict. Handlers never raise: errors come back as
# {"error": ..., "message": ...} so the LLM can self-correct inside the ReAct
# loop.
#
# `validate_finish(db, exit_reason)` checks the invariants before a finish is
# accepted. An empty-corpus refusal at iteration 0 is valid, so there is no
# universal floor on iteration_count.

from dataclasses import dataclass as _dataclass_decorator  # alias to avoid reimport
from dataclasses import field as _field  # noqa: E402

# ToolContext: per-turn state passed to every handler


@_dataclass_decorator
class ToolContext:
    """Per-step, per-turn state shared across tool handlers.

    - `db`: the PipelineDB being mutated.
    - `step_id`, `iteration`, `tool_call_id`: threaded through TransitionEvents.
    - `nli_cache`: per-turn cache of (pid, nid) → (label, score) so
      `mark_nugget`'s "read-before-mark" contract can verify that
      `check_entailment` ran for that pair in the same turn.
    - `tool_call_count`: incremented by the controller on every handler invoke;
      enforces the per-step cap (default 15) and global cap (default 60).
    - `nli_server_url`: localhost URL for the NLI oracle server
      (`src/r2c2_agent/orchestrator/nli_verifier.py`, Phase 2).
    """

    db: PipelineDB
    step_id: str
    iteration: int
    tool_call_id: str = "tool_call_unknown"
    nli_cache: dict[tuple[str, str], tuple[str, float]] = _field(default_factory=dict)
    # Tracks pids the LLM has already read this session. `handle_read_passage`
    # annotates responses with `previously_read: True` to discourage the LLM
    # from burning tool-call budget re-reading the same passage.
    seen_pids: set[str] = _field(default_factory=set)
    tool_call_count: int = 0
    nli_server_url: str = "http://127.0.0.1:18500"


# Handler constants

# Preview truncation length for list_passages / list_nuggets responses.
_PREVIEW_PASSAGE_CHARS = 200
_PREVIEW_NUGGET_CHARS = 120

# Server-side NLI decision table:
#   MiniCheck p >= 0.5 + supports-draft    → verified_supporting
#   MiniCheck p >= 0.5 + contradicts-draft → verified_infirming
#   MiniCheck p <  0.5                     → bogus
# draft-relation is derived from `proposed_stance` ("supporting" → supports;
# "infirming" → contradicts). The LLM's `reason` field is ignored for the
# automatic derivation but is retained for audit.
_NLI_SUPPORT_THRESHOLD = 0.5


# Error helpers


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Structured error result — never raised, always returned to the LLM."""
    out = {"error": code, "message": message}
    out.update(extra)
    return out


def _real_pids(db: PipelineDB) -> list[str]:
    """Passage ids excluding the virtual __refusal__ sentinel."""
    return [pid for pid in db.passages.keys() if pid != REFUSAL_PID]


def _is_virtual_pid(pid: str) -> bool:
    return pid == REFUSAL_PID


# TransitionEvent shortcut used by the handlers


def _emit_event(
    ctx: ToolContext,
    to_status: str,
    reason: str,
    *,
    from_status: str | None = None,
    thinking: str | None = None,
) -> TransitionEvent:
    ev = TransitionEvent(
        step_id=ctx.step_id,
        iteration=ctx.iteration,
        actor="llm_tool",
        tool_call_id=ctx.tool_call_id,
        from_status=from_status,
        to_status=to_status,
        reason=reason,
        thinking=thinking,
    )
    append_event(ctx.db, ev)
    return ev


# Tool 1: list_passages


def handle_list_passages(
    ctx: ToolContext, *, status: str | None = None
) -> dict[str, Any]:
    """Return [{pid, title, score, status, nugget_count, text_preview}].

    Filters out the virtual __refusal__ passage: it is a sentinel, not
    visible to the LLM at any step.
    """
    valid_statuses = {"unclassified", "accepted", "rejected", "counter_evidence"}
    if status is not None and status not in valid_statuses:
        return _err(
            "invalid_status",
            f"status must be one of {sorted(valid_statuses)}",
            valid_statuses=sorted(valid_statuses),
        )

    # nugget counts per pid (linked_nuggets)
    counts: dict[str, int] = {}
    for n in ctx.db.nuggets.values():
        counts[n.passage_id] = counts.get(n.passage_id, 0) + 1

    rows: list[dict[str, Any]] = []
    for pid, p in ctx.db.passages.items():
        if _is_virtual_pid(pid):
            continue
        if status is not None and p.status != status:
            continue
        rows.append(
            {
                "pid": pid,
                "title": p.title,
                "score": p.score,
                "status": p.status,
                "nugget_count": counts.get(pid, 0),
                "text_preview": (p.text or "")[:_PREVIEW_PASSAGE_CHARS],
            }
        )
    return {"passages": rows, "count": len(rows)}


# Tool 2: read_passage


def handle_read_passage(ctx: ToolContext, *, pid: str) -> dict[str, Any]:
    """Return the full Passage + linked nugget_ids.

    Tracks `ctx.seen_pids` so the response can tag repeat reads with
    `previously_read: True`. The S1 prompt tells the LLM not to re-read;
    the flag makes that contract observable.
    """
    if pid not in ctx.db.passages or _is_virtual_pid(pid):
        return _err(
            "unknown_pid",
            f"pid {pid!r} not found",
            valid_pids=_real_pids(ctx.db)[:20],
        )
    p = ctx.db.passages[pid]
    linked_nuggets = [
        n.nugget_id for n in ctx.db.nuggets.values() if n.passage_id == pid
    ]
    previously_read = pid in ctx.seen_pids
    ctx.seen_pids.add(pid)
    return {
        "pid": pid,
        "title": p.title,
        "text": p.text,
        "score": p.score,
        "status": p.status,
        "status_reason": p.status_reason,
        "counter_for_answer": p.counter_for_answer,
        "first_seen_iter": p.first_seen_iter,
        "last_reviewed_iter": p.last_reviewed_iter,
        "aliases": [a.model_dump() for a in p.aliases],
        "linked_nugget_ids": linked_nuggets,
        "previously_read": previously_read,
    }


# Tool 3: mark_passage (cascades on demotion)


def handle_mark_passage(
    ctx: ToolContext,
    *,
    pid: str,
    status: str,
    reason: str,
    counter_for_answer: str = "",
) -> dict[str, Any]:
    """Set passage status. Demotion to 'rejected' cascades linked verified_*
    nuggets to 'bogus'. 'unclassified' is not allowed: status moves
    monotonically away from unclassified, but can later transition between
    terminals during loopback re-review."""
    if pid not in ctx.db.passages or _is_virtual_pid(pid):
        return _err(
            "unknown_pid",
            f"pid {pid!r} not found",
            valid_pids=_real_pids(ctx.db)[:20],
        )
    if status not in {"accepted", "rejected", "counter_evidence"}:
        return _err(
            "invalid_status",
            "status must be 'accepted', 'rejected', or 'counter_evidence' "
            "(not 'unclassified' — monotonic forward)",
        )
    if not reason or len(reason) < 5 or len(reason) > 200:
        return _err("invalid_reason", "reason must be 5-200 chars")
    if status == "counter_evidence" and not counter_for_answer:
        return _err(
            "missing_counter_for_answer",
            "counter_for_answer is required when status='counter_evidence' "
            "(state the alternative answer ruled out)",
        )

    p = ctx.db.passages[pid]
    from_status = p.status
    if from_status == status:
        return {"ok": True, "pid": pid, "status": status, "no_change": True}

    p.status = status  # type: ignore[assignment]
    p.status_reason = reason
    p.last_reviewed_iter = ctx.iteration
    if status == "counter_evidence":
        p.counter_for_answer = counter_for_answer
    ev = _emit_event(
        ctx,
        to_status=status,
        reason=reason,
        from_status=from_status,
    )
    p.history.append(ev)

    # Cascade on demotion to 'rejected'.
    cascaded: list[str] = []
    if status == "rejected":
        for nid, n in ctx.db.nuggets.items():
            if n.passage_id == pid and n.status in {
                "verified_supporting",
                "verified_infirming",
            }:
                n_from = n.status
                n.status = "bogus"
                n.status_reason = "passage_demoted_to_rejected"
                ev_n = TransitionEvent(
                    step_id=ctx.step_id,
                    iteration=ctx.iteration,
                    actor="system",
                    tool_call_id=ctx.tool_call_id,
                    from_status=n_from,
                    to_status="bogus",
                    reason="cascade: passage demoted to rejected",
                )
                append_event(ctx.db, ev_n)
                n.history.append(ev_n)
                cascaded.append(nid)

    return {
        "ok": True,
        "pid": pid,
        "status": status,
        "from_status": from_status,
        "cascaded_nuggets_to_bogus": cascaded,
    }


# Tool 4: list_nuggets


def handle_list_nuggets(
    ctx: ToolContext,
    *,
    status: str | None = None,
    passage_id: str | None = None,
) -> dict[str, Any]:
    """Return [{nid, pid, text_preview, proposed_stance, status}]."""
    valid_statuses = {
        "unclassified",
        "verified_supporting",
        "verified_infirming",
        "bogus",
        "refusal",
    }
    if status is not None and status not in valid_statuses:
        return _err(
            "invalid_status",
            f"status must be one of {sorted(valid_statuses)}",
            valid_statuses=sorted(valid_statuses),
        )
    if passage_id is not None and passage_id not in ctx.db.passages:
        return _err(
            "unknown_pid",
            f"passage_id {passage_id!r} not found",
            valid_pids=_real_pids(ctx.db)[:20],
        )

    rows: list[dict[str, Any]] = []
    for nid, n in ctx.db.nuggets.items():
        if status is not None and n.status != status:
            continue
        if passage_id is not None and n.passage_id != passage_id:
            continue
        rows.append(
            {
                "nid": nid,
                "pid": n.passage_id,
                "text_preview": (n.text or "")[:_PREVIEW_NUGGET_CHARS],
                "proposed_stance": n.proposed_stance,
                "status": n.status,
            }
        )
    return {"nuggets": rows, "count": len(rows)}


# Tool 5: read_nugget (bundles the linked passage text, which saves a turn in S3)


def handle_read_nugget(ctx: ToolContext, *, nid: str) -> dict[str, Any]:
    if nid not in ctx.db.nuggets:
        # Nid might be valid-looking but absent; return recent nids to help.
        recent = list(ctx.db.nuggets.keys())[-20:]
        return _err("unknown_nid", f"nid {nid!r} not found", recent_nids=recent)
    n = ctx.db.nuggets[nid]
    pid = n.passage_id
    # Virtual __refusal__ passage text can be included for the refusal nugget
    # since it's the anchor; real pids resolve normally.
    passage_text = ""
    passage_title = None
    if pid in ctx.db.passages:
        passage_text = ctx.db.passages[pid].text
        passage_title = ctx.db.passages[pid].title
    return {
        "nid": nid,
        "pid": pid,
        "text": n.text,
        "proposed_stance": n.proposed_stance,
        "status": n.status,
        "status_reason": n.status_reason,
        "nli_label": n.nli_label,
        "nli_score": n.nli_score,
        "passage_title": passage_title,
        "passage_text": passage_text,
    }


# Tool 6: propose_nugget (quota echo + pid validation)


def _per_pid_nugget_count(db: PipelineDB, pid: str) -> int:
    return sum(1 for n in db.nuggets.values() if n.passage_id == pid)


def _real_nugget_count(db: PipelineDB) -> int:
    return sum(1 for n in db.nuggets.values() if n.status != "refusal")


def handle_propose_nugget(
    ctx: ToolContext,
    *,
    pid: str,
    text: str,
    stance: str,
) -> dict[str, Any]:
    if pid not in ctx.db.passages or _is_virtual_pid(pid):
        return _err(
            "unknown_pid",
            f"pid {pid!r} not found",
            valid_pids=_real_pids(ctx.db)[:20],
        )
    p = ctx.db.passages[pid]
    if p.status not in {"accepted", "counter_evidence"}:
        return _err(
            "pid_not_accepted",
            f"pid {pid!r} has status {p.status!r}; nuggets can only be "
            "proposed on 'accepted' or 'counter_evidence' passages",
            pid_status=p.status,
        )
    if not isinstance(text, str) or len(text) < 20 or len(text) > 300:
        return _err(
            "invalid_text",
            "text must be 20-300 chars (one atomic claim, 8-30 words)",
        )
    if stance not in {"supporting", "infirming"}:
        return _err(
            "invalid_stance",
            "stance must be 'supporting' or 'infirming'",
        )

    # Enforce caps server-side
    per_pid = _per_pid_nugget_count(ctx.db, pid)
    if per_pid >= ctx.db.max_nuggets_per_passage:
        return _err(
            "per_pid_cap_exhausted",
            f"pid {pid!r} has {per_pid} nuggets; cap is "
            f"{ctx.db.max_nuggets_per_passage}. Move to a different pid.",
            pid_quota_remaining=0,
        )
    real_total = _real_nugget_count(ctx.db)
    if real_total >= ctx.db.max_nuggets_global:
        return _err(
            "global_cap_exhausted",
            f"global nugget cap ({ctx.db.max_nuggets_global}) reached; "
            "no further proposals accepted. Finalize this step.",
            total_quota_remaining=0,
        )

    # Dedup: reject exact-text duplicates per pid
    normalized = text.strip().lower()
    for existing in ctx.db.nuggets.values():
        if existing.passage_id == pid and existing.text.strip().lower() == normalized:
            return _err(
                "duplicate_text",
                f"a nugget with this text already exists on pid {pid!r}",
                existing_nid=existing.nugget_id,
            )

    n = Nugget(
        passage_id=pid,
        text=text,
        proposed_stance=stance,  # type: ignore[arg-type]
        source_step=ctx.step_id,
        source_tool_call_id=ctx.tool_call_id,
        status="unclassified",
    )
    ev = _emit_event(
        ctx,
        to_status="unclassified",
        reason=f"proposed from pid {pid} (stance={stance})",
    )
    n.history.append(ev)
    ctx.db.nuggets[n.nugget_id] = n

    # Echo remaining quotas
    pid_remaining = ctx.db.max_nuggets_per_passage - (per_pid + 1)
    total_remaining = ctx.db.max_nuggets_global - (real_total + 1)
    return {
        "nid": n.nugget_id,
        "pid": pid,
        "stance": stance,
        "pid_quota_remaining": pid_remaining,
        "total_quota_remaining": total_remaining,
    }


# Tool 7: check_entailment (calls the localhost NLI server; per-turn cache)


def handle_check_entailment(
    ctx: ToolContext,
    *,
    passage_id: str,
    nugget_id: str,
    _http_post: Any | None = None,
) -> dict[str, Any]:
    """POST to NLI server and cache the result in ctx.nli_cache.

    `_http_post` is an optional injected callable for tests:
        _http_post(url, json_body) -> {"label": "entails"|"bogus", "score": float}
    When None (production), uses httpx synchronously.
    """
    if passage_id not in ctx.db.passages:
        return _err(
            "unknown_pid",
            f"passage_id {passage_id!r} not found",
            valid_pids=_real_pids(ctx.db)[:20],
        )
    if nugget_id not in ctx.db.nuggets:
        return _err("unknown_nid", f"nugget_id {nugget_id!r} not found")

    cache_key = (passage_id, nugget_id)
    if cache_key in ctx.nli_cache:
        label, score = ctx.nli_cache[cache_key]
        return {
            "label": label,
            "score": score,
            "cached": True,
            "passage_id": passage_id,
            "nugget_id": nugget_id,
        }

    passage_text = ctx.db.passages[passage_id].text
    claim_text = ctx.db.nuggets[nugget_id].text
    body = {"passage_text": passage_text, "claim_text": claim_text}

    try:
        if _http_post is not None:
            data = _http_post(f"{ctx.nli_server_url}/check_entailment", body)
        else:
            import httpx  # type: ignore[import]

            resp = httpx.post(
                f"{ctx.nli_server_url}/check_entailment",
                json=body,
                timeout=30.0,
            )
            data = resp.json()
    except Exception as exc:
        log.warning("check_entailment HTTP failed: %s", exc)
        # Safe-conservative fallback: bogus
        data = {"label": "bogus", "score": 0.0}

    label = data.get("label") or "bogus"
    score_raw = data.get("score")
    try:
        score = float(score_raw) if score_raw is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    # Clamp + NaN guard
    if not (score == score):
        score = 0.0
    score = max(0.0, min(1.0, score))
    if label not in {"entails", "bogus"}:
        label = "bogus"

    ctx.nli_cache[cache_key] = (label, score)
    return {
        "label": label,
        "score": score,
        "cached": False,
        "passage_id": passage_id,
        "nugget_id": nugget_id,
    }


# Tool 8: mark_nugget (read-before-mark contract + NLI cross-check)


def _expected_status_from_table(
    nli_label: str,
    score: float,
    proposed_stance: str,
) -> str:
    """Decision table applied server-side.

    p >= 0.5 + supports-draft (proposed=supporting)    → verified_supporting
    p >= 0.5 + contradicts-draft (proposed=infirming)  → verified_infirming
    p <  0.5 or label=='bogus'                          → bogus
    """
    if nli_label != "entails" or score < _NLI_SUPPORT_THRESHOLD:
        return "bogus"
    if proposed_stance == "supporting":
        return "verified_supporting"
    if proposed_stance == "infirming":
        return "verified_infirming"
    return "bogus"


def handle_mark_nugget(
    ctx: ToolContext,
    *,
    nid: str,
    status: str,
    reason: str,
    nli_label: str,
    nli_score: float | None = None,  # deprecated — kept for backward-compat
) -> dict[str, Any]:
    """Terminal classification. Server-side contract:

    1. `check_entailment(pid, nid)` must have been called this turn (cache
       lookup). Otherwise reject.
    2. `nli_label` must be consistent with the cached oracle value. The
       `nli_score` parameter is ignored: the server uses the cached score
       directly, which avoids a retry loop where small transcription
       drift failed a tolerance check.
    3. `status` must match the decision table for (nli_label, score,
       proposed_stance). Otherwise reject.

    On success: flip nugget status, record nli_label + cached nli_score,
    append event. The `nli_score` parameter is accepted for backward-compat
    but not validated.
    """
    _ = nli_score  # silence unused-arg linting; deliberately ignored
    if nid not in ctx.db.nuggets:
        return _err("unknown_nid", f"nid {nid!r} not found")
    if status not in {"verified_supporting", "verified_infirming", "bogus"}:
        return _err(
            "invalid_status",
            "status must be 'verified_supporting', 'verified_infirming', or 'bogus'",
        )
    if not reason or len(reason) < 5 or len(reason) > 200:
        return _err("invalid_reason", "reason must be 5-200 chars")
    if nli_label not in {"entails", "contradicts", "neutral"}:
        return _err(
            "invalid_nli_label",
            "nli_label must be 'entails', 'contradicts', or 'neutral'",
        )

    nugget = ctx.db.nuggets[nid]
    pid = nugget.passage_id

    # Contract 0: lifecycle monotonic (first — "already classified" is the
    # cleanest signal to the LLM, independent of oracle/table semantics).
    if nugget.status != "unclassified":
        return _err(
            "already_classified",
            f"nugget {nid!r} is already {nugget.status!r}; no re-classification",
            current_status=nugget.status,
        )

    # Contract 1: read-before-mark
    cache_key = (pid, nid)
    if cache_key not in ctx.nli_cache:
        return _err(
            "check_entailment_required",
            f"call check_entailment(passage_id={pid!r}, nugget_id={nid!r}) "
            "before mark_nugget (this turn)",
        )

    # Contract 2: transcribed label consistent with cache (score ignored).
    cached_label, cached_score = ctx.nli_cache[cache_key]
    # Our NLI server emits 'entails' or 'bogus' (binary). The LLM is allowed
    # to transcribe the underlying 3-label interpretation leniently:
    #   cached 'entails' → LLM transcribes 'entails'
    #   cached 'bogus'   → LLM transcribes 'contradicts' OR 'neutral'
    if cached_label == "entails":
        if nli_label != "entails":
            return _err(
                "nli_label_mismatch",
                f"cached label is 'entails' but you transcribed {nli_label!r}",
                cached_label=cached_label,
                cached_score=cached_score,
            )
    else:  # cached 'bogus'
        if nli_label == "entails":
            return _err(
                "nli_label_mismatch",
                "cached label is 'bogus' but you transcribed 'entails'",
                cached_label=cached_label,
                cached_score=cached_score,
            )

    # Contract 3: status matches decision table
    expected = _expected_status_from_table(
        nli_label=cached_label,
        score=cached_score,
        proposed_stance=nugget.proposed_stance,
    )
    if status != expected:
        return _err(
            "status_mismatch",
            f"based on NLI oracle and proposed_stance, expected status "
            f"{expected!r} but got {status!r}",
            expected_status=expected,
        )

    # Record the cached score, not whatever the LLM transcribed.
    from_status = nugget.status
    nugget.status = status  # type: ignore[assignment]
    nugget.status_reason = reason
    nugget.nli_label = nli_label  # type: ignore[assignment]
    nugget.nli_score = cached_score
    ev = _emit_event(
        ctx,
        to_status=status,
        reason=reason,
        from_status=from_status,
    )
    nugget.history.append(ev)

    return {
        "ok": True,
        "nid": nid,
        "status": status,
        "from_status": from_status,
        "nli_label": nli_label,
        "nli_score": cached_score,
    }


# Tool 9: read_answer


def handle_read_answer(ctx: ToolContext) -> dict[str, Any]:
    a = ctx.db.answer
    return {
        "draft_answer": a.draft_answer,
        "confidence": a.confidence,
        "supportability": a.supportability,
        "missing": a.missing,
        "rationale": a.rationale,
        "expected_count": a.expected_count,
        "counting_note": a.counting_note,
        "question_entities": list(a.question_entities),
        "iteration_count": a.iteration_count,
    }


# Tool 10: set_supportability (S4b write)


def handle_set_supportability(
    ctx: ToolContext,
    *,
    state: str,
    missing: str,
    rationale: str,
) -> dict[str, Any]:
    if state not in {"sufficient", "insufficient", "contradicted", "unresolvable"}:
        return _err(
            "invalid_state",
            "state must be one of {sufficient, insufficient, contradicted, unresolvable}",
        )
    if state in {"insufficient", "contradicted"} and not missing.strip():
        return _err(
            "missing_required",
            "'missing' must be non-empty for state='insufficient' or 'contradicted'",
        )
    if len(missing) > 200:
        return _err("missing_too_long", "missing must be ≤200 chars")
    if not rationale or len(rationale) < 10 or len(rationale) > 500:
        return _err("invalid_rationale", "rationale must be 10-500 chars")

    a = ctx.db.answer
    a.supportability = state  # type: ignore[assignment]
    a.missing = missing
    a.rationale = rationale
    ev = _emit_event(
        ctx,
        to_status=f"supportability={state}",
        reason="S4b WRITE",
    )
    a.history.append(ev)
    return {
        "ok": True,
        "state": state,
        "missing": missing,
        "rationale_len": len(rationale),
    }


# Tool 11: update_answer (S5b write; the controller overrides conf afterwards)


def handle_update_answer(
    ctx: ToolContext,
    *,
    text: str,
    conf: float,
    reason: str,
) -> dict[str, Any]:
    """Write draft_answer. `conf` is recorded for audit but overridden by the
    controller's deterministic formula before the finish call."""
    if not isinstance(text, str) or len(text) < 3 or len(text) > 2000:
        return _err("invalid_text", "text must be 3-2000 chars")
    # Coerce string conf leniently
    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        return _err("invalid_conf", "conf must be a number in [0, 1]")
    conf_f = max(0.0, min(1.0, conf_f))
    if not reason or len(reason) < 10 or len(reason) > 500:
        return _err("invalid_reason", "reason must be 10-500 chars")

    a = ctx.db.answer
    a.draft_answer = text
    a.confidence = conf_f  # the controller overwrites this
    ev = _emit_event(
        ctx,
        to_status="draft_updated",
        reason=reason,
    )
    a.history.append(ev)
    return {
        "ok": True,
        "text_len": len(text),
        "llm_conf": conf_f,
        "note": (
            "conf recorded; orchestrator will override via §5.5 formula "
            "before finish() validation"
        ),
    }


# Tool 12: finish (validated server-side)


def handle_finish(ctx: ToolContext, *, exit_reason: str) -> dict[str, Any]:
    """Terminal tool. Runs `validate_finish(db, exit_reason)` and rejects on
    any invariant failure; the loop continues until the controller forces exit.
    """
    if exit_reason not in {
        "high_confidence",
        "low_confidence_refusal",
        "max_iter",
        "error",
    }:
        return _err("invalid_exit_reason", "unsupported exit_reason")

    err = validate_finish(ctx.db, exit_reason)  # type: ignore[arg-type]
    if err is not None:
        return _err(
            "finish_validator_rejected",
            err,
            exit_reason=exit_reason,
        )

    ctx.db.answer.exit_reason = exit_reason  # type: ignore[assignment]
    ev = _emit_event(
        ctx,
        to_status=f"exit={exit_reason}",
        reason="LLM-issued finish accepted by validator",
    )
    ctx.db.answer.history.append(ev)
    return {"ok": True, "exit_reason": exit_reason}


# Server-side validator for finish()


def validate_finish(db: PipelineDB, exit_reason: str) -> str | None:
    """Check the invariants before accepting any finish() call.

    Returns None on success; returns an error message string on failure.
    Confidence is read from `db.answer.confidence` only; there is no
    LLM-passed conf argument. iteration_count is not required to be ≥1:
    the empty-corpus refusal path at iteration 0 is valid.
    """
    # Universal invariants (apply to all exit_reasons except "error")
    if exit_reason != "error":
        unclassified_p = [
            pid
            for pid, p in db.passages.items()
            if p.status == "unclassified" and pid != REFUSAL_PID
        ]
        if unclassified_p:
            return f"unclassified_passages={len(unclassified_p)}"
        unclassified_n = [
            nid for nid, n in db.nuggets.items() if n.status == "unclassified"
        ]
        if unclassified_n:
            return f"unclassified_nuggets={len(unclassified_n)}"
        for n in db.nuggets.values():
            if n.status.startswith("verified_"):
                if n.passage_id not in db.passages:
                    return f"nugget {n.nugget_id} has unknown pid"
                if db.passages[n.passage_id].status not in {
                    "accepted",
                    "counter_evidence",
                }:
                    return f"nugget {n.nugget_id} anchored to non-accepted pid"
                if n.nli_label is None or n.nli_score is None:
                    return f"nugget {n.nugget_id} missing NLI oracle transcription"
        # No universal iteration_count floor: S1 exits at iteration 0 on an
        # empty corpus.

    eff = effective_supporting_ids(db)
    n_effective = len(eff)

    if exit_reason == "high_confidence":
        if db.answer.confidence < 0.8:
            return f"confidence={db.answer.confidence} < 0.8"
        if n_effective < 1:
            return "no effective-supporting nuggets (direct or elimination)"
        if not db.answer.draft_answer.strip():
            return "empty draft_answer"
        if not db.answer.reasoning_trace.strip():
            return "empty reasoning_trace"
        if db.answer.supportability != "sufficient":
            return f"supportability={db.answer.supportability}"

    elif exit_reason == "low_confidence_refusal":
        if n_effective != 0:
            return f"effective_supporting={n_effective} (must be 0 on refusal)"
        n_refusal = sum(1 for n in db.nuggets.values() if n.status == "refusal")
        if n_refusal != 1:
            return f"refusal_nuggets={n_refusal} (must be exactly 1)"
        n_real = _real_nugget_count(db)
        all_real_passages_classified = all(
            p.status != "unclassified"
            for pid, p in db.passages.items()
            if pid != REFUSAL_PID
        )
        if n_real < 3 and not all_real_passages_classified:
            return (
                f"len(real_nuggets)={n_real} < 3 and not all_real_passages_classified"
            )
        # The draft is not matched against ``REFUSAL_REGEX`` here:
        # ``handle_submit_answer`` (pipeline_handlers.py) always writes the
        # canonical ``CANONICAL_REFUSAL_ANSWER`` before calling
        # validate_finish, and the structural signals are already gated
        # (``exit_reason == "low_confidence_refusal"``,
        # ``n_effective_supporting == 0``, exactly one refusal nugget,
        # ``confidence == 0.0`` on the next line).
        if db.answer.confidence != 0.0:
            return f"confidence={db.answer.confidence} (must be 0.0 on refusal)"

    elif exit_reason == "max_iter":
        # Controller-issued; always valid.
        pass

    elif exit_reason == "error":
        # Fallback path; triggers legacy pipeline.
        pass

    # Global budget checks (universal)
    if len(db.passages) > db.max_passages + 1:  # +1 for virtual __refusal__
        return f"passages={len(db.passages)} exceeds cap {db.max_passages}"
    if _real_nugget_count(db) > db.max_nuggets_global:
        return f"nuggets exceed cap {db.max_nuggets_global}"

    return None


# Tool dispatch entry point


_TOOL_HANDLERS = {
    "list_passages": handle_list_passages,
    "read_passage": handle_read_passage,
    "mark_passage": handle_mark_passage,
    "list_nuggets": handle_list_nuggets,
    "read_nugget": handle_read_nugget,
    "propose_nugget": handle_propose_nugget,
    "mark_nugget": handle_mark_nugget,
    "check_entailment": handle_check_entailment,
    "read_answer": handle_read_answer,
    "set_supportability": handle_set_supportability,
    "update_answer": handle_update_answer,
    "finish": handle_finish,
}


def dispatch_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call to its handler; enforces budget caps.

    Never raises: invalid names and argument errors come back as structured errors.
    """
    if name not in _TOOL_HANDLERS:
        return _err(
            "unknown_tool",
            f"unknown tool {name!r}",
            valid_tools=sorted(_TOOL_HANDLERS.keys()),
        )
    # Global tool-call cap
    if ctx.tool_call_count >= ctx.db.max_tool_calls_global:
        return _err(
            "global_tool_cap_exhausted",
            f"global tool-call cap ({ctx.db.max_tool_calls_global}) reached; "
            "finalize now",
        )
    ctx.tool_call_count += 1
    handler = _TOOL_HANDLERS[name]
    try:
        return handler(ctx, **(args or {}))
    except TypeError as exc:
        # Bad kwargs (missing required, unexpected arg, etc.)
        return _err("invalid_args", str(exc), tool=name)


# Exports: handlers and validator


__all__ += [  # noqa: F821 — __all__ is defined at module top
    "ToolContext",
    "dispatch_tool",
    "validate_finish",
    "handle_list_passages",
    "handle_read_passage",
    "handle_mark_passage",
    "handle_list_nuggets",
    "handle_read_nugget",
    "handle_propose_nugget",
    "handle_mark_nugget",
    "handle_check_entailment",
    "handle_read_answer",
    "handle_set_supportability",
    "handle_update_answer",
    "handle_finish",
]


# v2 pipeline: step controllers, prompts, confidence and the S6 state machine.
# Holds the state machine with its advance rules and refusal synthesis, the
# per-step prompts and the confidence formula, the thinking strategy with its
# split mode, and the S6 escape rules a-g. A ReAct-style inner loop drives the
# LLM tool calls within a single chat session per step.


# Per-step system prompts


_S1_SYSTEM_PROMPT = """\
You triage retrieved passages for a QA pipeline. You work agentically: you call
tools until every passage is classified, then stop. You do NOT write answers.

Buckets (pick exactly one per passage — BE LIBERAL on "accepted";
downstream S3/NLI filters will catch off-topic nuggets):
- accepted: could plausibly bear on answering the question. If the passage
  is topically related to the question's entities, lean accepted.
- counter_evidence: fact about a NEAR-MISS alternative that, by elimination,
  supports the draft. Also applies to numeric drafts where an ordinal/cardinal
  is ruled out. Counter_evidence is NOT rejection — it feeds S4's elimination
  reasoning.
- rejected: off-topic; no fact helps judge the answer either way.

Tools:
- list_passages(status="unclassified") -> [{pid,title,score,status}]
- read_passage(pid) -> full text (only needed the FIRST time for a pid)
- mark_passage(pid, status, reason[, counter_for_answer]) -> commit

Loop: list -> for each UNCLASSIFIED pid, read_passage ONCE then mark_passage.
Stop when list_passages(status="unclassified") returns [].

Constraints:
- You MUST call read_passage(pid) before mark_passage(pid) (first time only).
  Server rejects mark_passage without a prior read this session.
- Do NOT re-read passages you have already seen in this turn's context.
- Rejection target 15%-40%. When unsure, lean accepted — S3 will filter.
- reason must quote a concrete span (<=15 words) from the passage, not generic.
- counter_for_answer required iff status="counter_evidence".
- One tool call at a time. No parallel calls.
"""


_S2_SYSTEM_PROMPT = """\
You extract atomic factual claims (nuggets) from accepted passages. Optimize for
RECALL. Atomic = ONE subject, ONE predicate, ONE value. Compound sentences
become multiple nuggets. "Mackie replaced Farrell in the remake" ->
  n1: "Mackie acted in the remake"
  n2: "Farrell was originally cast in the remake"

Stance:
- supporting: nugget helps a reader accept the draft.
- infirming: nugget challenges the draft OR rules out a near-miss alternative.
  Extract these — S4 will reinterpret elimination-style infirming as supporting.

Scope: only facts present in the passage. No inference, no world knowledge.
Do NOT promote bare table rows / cast lists to role claims.

Tools:
- list_passages(status="accepted"|"counter_evidence")
- read_passage(pid)
- propose_nugget(pid, text, stance) -> {nid, pid_quota_remaining:N}
- list_nuggets(passage_id=pid)

Workflow: for each accepted/counter pid, read_passage then call propose_nugget
until pid_quota_remaining==5 or genuinely no more atomic claims. Per-pid cap 8;
global cap 40. Finish when every accepted/counter pid has >=1 nugget.

Length: 8-30 words per nugget. Self-contained. Present tense.
"""


_S3_SYSTEM_PROMPT = """\
You verify each unclassified nugget against its source passage using an
independent NLI oracle.

Workflow per nugget (only 3 calls — read_nugget already returns passage text):
1. read_nugget(nid) -> returns {text, pid, passage_text}
2. check_entailment(passage_id=pid, nugget_id=nid) -> {label, verdict}
3. mark_nugget(nid, status, reason, nli_label=label)
   (no nli_score — server uses the cached value)

Do NOT call read_passage in S3 — read_nugget already bundles passage_text.
Server rejects mark_nugget if check_entailment was not called for (pid,nid)
this turn. Status is computed server-side from the NLI label; you just pass
the label back verbatim.

Decision guide (nli_label x relation-to-draft -> status):

  nli_label     supports draft        contradicts draft
  ------------  -------------------   ---------------------
  entails       verified_supporting   verified_infirming
  bogus         bogus                 bogus

"supports draft" = if the nugget is true, a reader is more likely to accept
the draft. verified_infirming is a VALID outcome, not a failure.

Calibration: expect 10%-40% bogus. If <5% bogus with >=3 nuggets, re-examine
marginal cases.
"""


_S4A_REASON_PROMPT = """\
You judge whether the verified nuggets support the draft answer. Think step by
step, then emit a structured block.

Verdicts:
- sufficient: verified_supporting nuggets (plus elimination-tagged infirming)
  cover every claim the draft makes at the expected granularity.
- insufficient: some draft claim has no supporting nugget, OR for aggregation
  distinct items < expected_count.
- contradicted: >=1 verified_infirming NEGATES a draft claim AND that claim is
  not rescued by a stronger supporting nugget.
- unresolvable: supporting and infirming conflict on the same claim; more
  retrieval will not help.

ELIMINATION RULE (mandatory when only elim-style evidence exists): a
verified_infirming nugget that refutes a NEAR-MISS alternative (not the draft)
strengthens the draft by elimination. Tag it `[strengthens-by-negation]` in
RATIONALE on its OWN LINE, formatted exactly as
  <nugget_id>: [strengthens-by-negation]
Count it as supporting. If zero verified_supporting + >=1 verified_infirming
that rule out near-misses, you MUST emit verdict=sufficient.

Aggregation: count distinct entity-slots across verified_supporting nuggets;
if < expected_count, verdict=insufficient and MISSING="need N more <slot>".

Output — the last 6 lines MUST match this shape exactly:

===VERDICT===
<one of: sufficient | insufficient | contradicted | unresolvable>
===MISSING===
<<=200 chars; empty string if sufficient>
===RATIONALE===
<<=400 chars; cite nugget ids you relied on>
"""


_S4B_WRITE_PROMPT = """\
Apply the decision from the previous reasoning. Call set_supportability once
with the provided fields. Do not re-reason. Do not call any other tool.
"""


_S5A_REASON_PROMPT = """\
You rewrite the draft answer using only the verified nuggets. Think step by
step (if thinking is enabled), then emit a structured block.

Per question_type, pick ONE section:

[simple]
  1-2 sentences. Name the entity, state the fact. No hedging.

[aggregation]
  Bulleted list of exactly expected_count items, one distinct entity-slot per
  bullet. If you have fewer supported items, list what you have AND note
  MISSING; do not fabricate.

[comparison]
  One short paragraph with symmetric structure: entity A => evidence; entity B
  => evidence; verdict clause. Every comparison claim cites a nugget.

[multi_hop]
  Explicit chain: "A -> B (nugget n_xx) -> C (nugget n_yy) -> answer". Each hop
  backed by a nugget id.

Infirming-nugget handling: for each verified_infirming nugget, tag exactly ONE:
- [weakens]                    — negates a draft claim; rewrite to remove/soften
- [strengthens-by-negation]    — rules out a near-miss; keep draft, optionally
                                 mention "(unlike X, which ...)"

Rewrite iff any of:
- supportability != sufficient
- any [weakens] tag
- aggregation count mismatch
- missing multi-hop hop
- draft names an entity absent from all verified_supporting nuggets
- draft supported ONLY by [strengthens-by-negation] (no direct support) — in
  that case rewrite to make the elimination explicit.

Otherwise CONFIRM (reuse draft verbatim).

Output — last 6 lines MUST match exactly:

===DRAFT===
<final answer text>
===TAGS===
<nid>: <weakens|strengthens-by-negation>   (one line per infirming nugget;
                                            empty line if none)
===NOTES===
<<=200 chars; what changed, or "confirmed">
"""


_S5B_WRITE_PROMPT = """\
Write the final answer. Call update_answer once, then stop. Do not call other
tools, do not re-reason. The confidence value will be post-processed by the
orchestrator.
"""


def _format_user_prompt(
    db: PipelineDB,
    *,
    step_id: str,
    iteration: int = 0,
    missing_hint: str = "",
    scope_pids: list[str] | None = None,
) -> str:
    """Render the per-step user prompt with current DB state.

    Fields: question / question_type / draft / iteration, then
    missing_from_prev_s4 (S1 only) or scope hints (S2 delta).
    """
    a = db.answer
    parts: list[str] = [
        f"question: {db.question}",
        f"question_type: {db.question_type or 'simple'}",
        f"draft_answer: {a.draft_answer}",
        f"iteration: {iteration}",
    ]
    if step_id == "s1":
        parts.append(f"missing_from_prev_s4: {missing_hint}")
        parts.append('\nBegin by calling list_passages(status="unclassified").')
    elif step_id == "s2":
        if scope_pids is not None:
            parts.append(
                f"scope_pids ({len(scope_pids)} new on this iteration): "
                f"{','.join(scope_pids) if scope_pids else '(none)'}"
            )
            parts.append(
                "\nExtract nuggets ONLY from the scope_pids listed above "
                "(newly accepted on this iteration). Other accepted/counter "
                "passages were already nugget-extracted."
            )
        else:
            parts.append(
                "\nExtract nuggets from every accepted / counter_evidence passage."
            )
    elif step_id == "s3":
        parts.append(
            "\nVerify every unclassified nugget via the NLI oracle and mark it."
        )
    elif step_id == "s4_reason":
        if a.expected_count is not None:
            parts.append(f"expected_count: {a.expected_count}")
        if a.counting_note:
            parts.append(f"counting_note: {a.counting_note}")
        verified_support = sum(
            1 for n in db.nuggets.values() if n.status == "verified_supporting"
        )
        verified_infirm = sum(
            1 for n in db.nuggets.values() if n.status == "verified_infirming"
        )
        parts.append(
            f"verified_supporting: {verified_support}  "
            f"verified_infirming: {verified_infirm}"
        )
    elif step_id == "s4_write":
        parts.append(f"supportability_decision: {a.supportability or '(pending)'}")
    elif step_id == "s5_reason":
        if a.expected_count is not None:
            parts.append(f"expected_count: {a.expected_count}")
    elif step_id == "s5_write":
        parts.append("Call update_answer once with the final draft.")
    return "\n".join(parts)


# OpenAI tool schemas and per-step allow-lists


def _tool_schema(name: str, description: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


_TOOL_SCHEMAS: dict[str, dict] = {
    "list_passages": _tool_schema(
        "list_passages",
        "Return passages filtered by status.",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "unclassified",
                        "accepted",
                        "rejected",
                        "counter_evidence",
                    ],
                },
            },
        },
    ),
    "read_passage": _tool_schema(
        "read_passage",
        "Return full passage text for a pid.",
        {
            "type": "object",
            "properties": {"pid": {"type": "string"}},
            "required": ["pid"],
        },
    ),
    "mark_passage": _tool_schema(
        "mark_passage",
        "Commit a terminal classification for a passage.",
        {
            "type": "object",
            "properties": {
                "pid": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["accepted", "rejected", "counter_evidence"],
                },
                "reason": {"type": "string"},
                "counter_for_answer": {"type": "string"},
            },
            "required": ["pid", "status", "reason"],
        },
    ),
    "list_nuggets": _tool_schema(
        "list_nuggets",
        "Return nuggets filtered by status or passage.",
        {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "passage_id": {"type": "string"},
            },
        },
    ),
    "read_nugget": _tool_schema(
        "read_nugget",
        "Return a nugget plus its linked passage snippet.",
        {
            "type": "object",
            "properties": {"nid": {"type": "string"}},
            "required": ["nid"],
        },
    ),
    "propose_nugget": _tool_schema(
        "propose_nugget",
        "Create a new unclassified nugget anchored to an accepted pid.",
        {
            "type": "object",
            "properties": {
                "pid": {"type": "string"},
                "text": {"type": "string"},
                "stance": {"type": "string", "enum": ["supporting", "infirming"]},
            },
            "required": ["pid", "text", "stance"],
        },
    ),
    "mark_nugget": _tool_schema(
        "mark_nugget",
        "Commit a terminal classification for a nugget. Requires a prior "
        "check_entailment(pid,nid) call this turn. The server uses the "
        "cached NLI score — no need to pass nli_score.",
        {
            "type": "object",
            "properties": {
                "nid": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["verified_supporting", "verified_infirming", "bogus"],
                },
                "reason": {"type": "string"},
                "nli_label": {
                    "type": "string",
                    "enum": ["entails", "contradicts", "neutral"],
                },
            },
            "required": ["nid", "status", "reason", "nli_label"],
        },
    ),
    "check_entailment": _tool_schema(
        "check_entailment",
        "Query the NLI oracle for passage-entails-nugget. Returns "
        "{label, score, verdict}.",
        {
            "type": "object",
            "properties": {
                "passage_id": {"type": "string"},
                "nugget_id": {"type": "string"},
            },
            "required": ["passage_id", "nugget_id"],
        },
    ),
    "read_answer": _tool_schema(
        "read_answer",
        "Return the current AnswerState.",
        {"type": "object", "properties": {}},
    ),
    "set_supportability": _tool_schema(
        "set_supportability",
        "S4b WRITE — commit supportability verdict.",
        {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": [
                        "sufficient",
                        "insufficient",
                        "contradicted",
                        "unresolvable",
                    ],
                },
                "missing": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["state", "missing", "rationale"],
        },
    ),
    "update_answer": _tool_schema(
        "update_answer",
        "S5b WRITE — commit final answer text.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "conf": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["text", "conf", "reason"],
        },
    ),
    "finish": _tool_schema(
        "finish",
        "Terminal — server validates all invariants before accepting.",
        {
            "type": "object",
            "properties": {
                "exit_reason": {
                    "type": "string",
                    "enum": [
                        "high_confidence",
                        "low_confidence_refusal",
                        "max_iter",
                        "error",
                    ],
                },
            },
            "required": ["exit_reason"],
        },
    ),
}


_STEP_ALLOWED_TOOLS: dict[str, list[str]] = {
    "s1": ["list_passages", "read_passage", "mark_passage"],
    "s2": ["list_passages", "read_passage", "propose_nugget", "list_nuggets"],
    "s3": [
        "list_nuggets",
        "read_nugget",
        "read_passage",
        "check_entailment",
        "mark_nugget",
    ],
    "s4_write": ["set_supportability"],
    "s5_write": ["update_answer"],
}


def tool_schemas_for_step(step_id: str) -> list[dict]:
    """Return the OpenAI-format tool list for a given step_id."""
    names = _STEP_ALLOWED_TOOLS.get(step_id, [])
    return [_TOOL_SCHEMAS[n] for n in names if n in _TOOL_SCHEMAS]


# Structured-block regex parsers (DOTALL, last match wins)


_S4_BLOCK_RE = re.compile(
    r"===VERDICT===[ \t]*\n(sufficient|insufficient|contradicted|unresolvable)"
    r"[ \t]*\n+"
    r"===MISSING===[ \t]*\n(.*?)\n+"
    r"===RATIONALE===[ \t]*\n(.*?)(?=\n===|\Z)",
    re.DOTALL,
)

_S5_BLOCK_RE = re.compile(
    r"===DRAFT===[ \t]*\n(.*?)\n+"
    r"===TAGS===[ \t]*\n(.*?)\n+"
    r"===NOTES===[ \t]*\n(.*?)(?=\n===|\Z)",
    re.DOTALL,
)


def parse_s4_reason(text: str) -> tuple[str, str, str] | None:
    """Parse S4a REASON output. Returns (verdict, missing, rationale) or None.

    Last match wins (DOTALL).
    """
    if not text:
        return None
    matches = list(_S4_BLOCK_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


def parse_s5_reason(text: str) -> tuple[str, str, str] | None:
    """Parse S5a REASON output. Returns (draft, tags, notes) or None."""
    if not text:
        return None
    matches = list(_S5_BLOCK_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


# Deterministic confidence formula (orchestrator-side)


# Used only when R2C2_DECOUPLED_SCORING=0; not read by _compute_confidence_decoupled.
_CONF_BASE = {
    "sufficient": 0.85,
    "insufficient": 0.55,
    "contradicted": 0.25,
    "unresolvable": 0.10,
}


def _count_weakens_tags(rationale: str) -> int:
    """Count nuggets tagged [weakens] in answer.reasoning_trace / rationale."""
    if not rationale:
        return 0
    return len(re.findall(r"(?m)^n_[0-9a-f]{8}\s*:\s*\[?weakens\]?\s*$", rationale))


def _decoupled_scoring_enabled() -> bool:
    """Feature flag: ``R2C2_DECOUPLED_SCORING=1`` switches ``compute_confidence``
    and the ``should_continue`` driver onto the decoupled formulas.

    Off by default (legacy). "0" / "false" / "" / unset count as off; any
    other truthy string enables the decoupled code path.
    """
    val = os.environ.get("R2C2_DECOUPLED_SCORING", "0").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def _compute_confidence_legacy(db: PipelineDB) -> float:
    """Additive confidence formula over passage counts.

    base = {sufficient:0.85, insufficient:0.55, contradicted:0.25,
            unresolvable:0.10}[supportability]
    conf = clip(base + 0.03·min(direct,5) + 0.015·min(s_elim,3) - 0.15·w,
                0.05, 0.97)
    direct = # distinct pids corroborating any verified_supporting nugget
    s_elim = # distinct pids on [strengthens-by-negation]-tagged
             # verified_infirming nuggets

    Elimination-only cap: if direct==0 and s_elim>=1 -> conf <= 0.90 (PCWA cap).
    Refusal override: exit_reason==low_confidence_refusal -> 0.0.

    Both counts are over passages, not nuggets: corroborating the same
    claim across several passages increases the signal without needing
    near-duplicate nugget rows.
    """
    a = db.answer
    if a.exit_reason == "low_confidence_refusal":
        return 0.0

    supportability = a.supportability or "unresolvable"
    base = _CONF_BASE.get(supportability, 0.10)

    # direct: distinct supporting pids across verified_supporting nuggets
    direct_pids: set[str] = set()
    for n in db.nuggets.values():
        if n.status != "verified_supporting":
            continue
        if getattr(n, "is_synthetic", False):
            continue
        fallback = [n.passage_id] if n.passage_id else []
        for pid in n.supporting_pids or fallback:
            if pid:
                direct_pids.add(pid)
    direct = len(direct_pids)

    # s_elim: distinct pids on tagged verified_infirming nuggets
    tagged = set(_STRENGTHEN_TAG_RE.findall(a.rationale or ""))
    elim_pids: set[str] = set()
    for n in db.nuggets.values():
        if n.status != "verified_infirming" or n.nugget_id not in tagged:
            continue
        if getattr(n, "is_synthetic", False):
            continue
        fallback = [n.passage_id] if n.passage_id else []
        for pid in n.supporting_pids or fallback:
            if pid:
                elim_pids.add(pid)
    s_elim = len(elim_pids)

    # combine tag sources: answer.rationale AND reasoning_trace
    w_rationale = _count_weakens_tags(a.rationale or "")
    w_trace = _count_weakens_tags(a.reasoning_trace or "")
    w = max(w_rationale, w_trace)  # dedup — tags may appear in both

    conf = base + 0.03 * min(direct, 5) + 0.015 * min(s_elim, 3) - 0.15 * w
    conf = max(0.05, min(0.97, conf))

    # PCWA cap (Partial Closed-World Assumption): elimination evidence
    # alone should not drive confidence above 0.90, since it is weaker
    # than direct entailment. With automatic supportability this branch
    # is unreachable: direct=0 forces supportability="insufficient"
    # (base=0.55), so the highest confidence with direct=0 is
    # 0.55 + 0.015 × 3 = 0.595. The guard only binds when replaying
    # traces where a caller pinned supportability="sufficient"
    # (base=0.85) on a DB with zero direct supporters.
    if direct == 0 and s_elim >= 1:
        conf = min(conf, 0.90)

    return conf


def compute_confidence(db: PipelineDB) -> float:
    """Reported numeric confidence in [0, 1].

    Under the default legacy path (``R2C2_DECOUPLED_SCORING`` unset or 0),
    this is the additive formula in ``_compute_confidence_legacy``.

    Under the decoupled path (``R2C2_DECOUPLED_SCORING=1``), the
    formula is::

        clip(
            base_answer_shape(db) * evidence_quality(db) * coverage_factor(db, req)
            - weakens_penalty(db),
            0.0, 1.0,
        )

    The decoupled formula can rise above the legacy ceiling for n_eff=1
    high-NLI simple-question cases (about 0.58 under the legacy formula).
    See ``base_answer_shape``, ``evidence_quality`` and
    ``coverage_factor`` for the factors; incomplete aggregation and
    comparison answers are still penalised through the items-coverage
    gate.

    ``should_continue`` is the decoupled driver; it does not read this
    float.
    """
    a = db.answer
    if a.exit_reason == "low_confidence_refusal":
        return 0.0
    if not _decoupled_scoring_enabled():
        return _compute_confidence_legacy(db)
    return _compute_confidence_decoupled(db)


# Scoring helpers for the decoupled path


@dataclass(frozen=True)
class RequiredCoverage:
    """Per-qtype coverage floor. Deterministic. See spec §3.1 table."""

    min_distinct_supporters: int
    min_distinct_claims: int
    items_expected: int | None
    directness_floor: float


@dataclass(frozen=True)
class BudgetState:
    """Per-call execution budget snapshot. Plumbed into ``should_continue``."""

    wall_clock_sec: float
    iter_count: int
    max_iter: int
    # Not wired to the live session envelope: always 0 when built by
    # run_s6, and not read by should_continue.
    calls_used: int = 0
    calls_remaining: int = 0


@dataclass(frozen=True)
class ContinueDecision:
    """Result of ``should_continue``: stop or continue + reason + code."""

    action: Literal["continue", "stop"]
    reason: str
    stop_code: (
        Literal[
            "high_confidence",
            "max_iter",
            "wall_clock",
            "low_confidence_refusal",
            "error",
        ]
        | None
    ) = None


# Jaccard / directness primitive

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Case-folded alphanumeric token set; drops empty strings."""
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_SPLIT.split(text) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union


def _answer_overlap_ratio(answer_tokens: set[str], nugget_tokens: set[str]) -> float:
    """Fraction of answer tokens present in nugget tokens. Asymmetric.

    This is the directness proxy: when the answer "Luke Skywalker"
    appears verbatim inside the nugget "Luke Skywalker piloted Red Five
    at the Battle of Yavin", directness should be 1.0. A symmetric
    Jaccard would penalise the long informative nugget unfairly
    (intersection/union ~0.22 here).

    Guard: when the answer is empty, returns 0.0.
    """
    if not answer_tokens:
        return 0.0
    inter = len(answer_tokens & nugget_tokens)
    return inter / len(answer_tokens)


def _answer_items(draft: str) -> list[set[str]]:
    """Split an enumerated draft answer into per-item token sets.

    Aggregation / comparison questions commonly have answers like
    "asthma; diabetes; hypertension". For directness we want each item's
    token coverage against the nugget, not the whole answer -- a single
    nugget "drug X for asthma" should score directness≈1.0 against the
    "asthma" slot, even though it covers only one third of the entire
    answer.
    """
    if not draft:
        return []
    raw = re.split(r"[;,\n]|\band\b", draft, flags=re.IGNORECASE)
    items: list[set[str]] = []
    for chunk in raw:
        tok = _tokenize(chunk)
        if tok:
            items.append(tok)
    return items


def _best_answer_overlap(draft: str, nugget_tokens: set[str]) -> float:
    """Max of:
      - whole-answer containment ratio (single-item questions);
      - per-item containment ratio (enumerated answers).

    Returns the ratio most favourable to the nugget. Downstream mapping
    via ``_directness_from_ratio`` decides whether this passes the floor.
    """
    whole = _answer_overlap_ratio(_tokenize(draft), nugget_tokens)
    best = whole
    for item in _answer_items(draft):
        r = _answer_overlap_ratio(item, nugget_tokens)
        if r > best:
            best = r
    return best


def _directness_from_ratio(ratio: float) -> float:
    """Map ratio ∈ [0, 1] to directness ∈ [0.4, 1.0]:
    ≤ 0.25 → 0.4 baseline; ≥ 0.75 → 1.0; linear in between."""
    if ratio <= 0.25:
        return 0.4
    if ratio >= 0.75:
        return 1.0
    return 0.4 + (ratio - 0.25) * (1.0 - 0.4) / (0.75 - 0.25)


# Per-nugget impact model


def nugget_impact(n: Nugget, db: PipelineDB) -> float:
    """Per-nugget impact ∈ [0, 1]. Product of entailment × directness ×
    tag_weight × freshness_ok.

    Defined only for ``verified_supporting`` / ``verified_infirming`` rows;
    returns 0.0 otherwise.
    """
    if n.status not in {"verified_supporting", "verified_infirming"}:
        return 0.0

    # entailment: max across pid_nli_scores, or legacy nli_score
    ent = 0.0
    for score in n.pid_nli_scores.values():
        if score is not None and score > ent:
            ent = float(score)
    if n.nli_score is not None and float(n.nli_score) > ent:
        ent = float(n.nli_score)
    ent = max(0.0, min(1.0, ent))

    # directness via asymmetric containment (answer / answer-item ⊆ nugget)
    draft = db.answer.draft_answer or ""
    ratio = _best_answer_overlap(draft, _tokenize(n.text))
    directness = _directness_from_ratio(ratio)

    # tag_weight
    tagged_strengthen = set(_STRENGTHEN_TAG_RE.findall(db.answer.rationale or ""))
    weakens = bool(
        re.search(
            rf"(?m)^{re.escape(n.nugget_id)}\s*:\s*\[?weakens\]?\s*$",
            db.answer.rationale or "",
        )
    )
    if weakens:
        tag_weight = 0.0
    elif n.nugget_id in tagged_strengthen:
        tag_weight = 0.8
    elif n.status == "verified_supporting":
        tag_weight = 1.0 if n.nli_score and n.nli_score >= 0.90 else 0.9
    else:
        tag_weight = 0.9

    # freshness_ok: 0.5 when there is a near-duplicate sibling with
    # higher impact (same normalized text prefix); else 1.0. Keep the
    # check cheap: a token-set equality heuristic on the first 10 tokens.
    my_tok = tuple(sorted(_tokenize(n.text)))[:10]
    freshness = 1.0
    if my_tok:
        for other in db.nuggets.values():
            if other.nugget_id == n.nugget_id:
                continue
            if other.status not in {"verified_supporting", "verified_infirming"}:
                continue
            other_tok = tuple(sorted(_tokenize(other.text)))[:10]
            if other_tok == my_tok and (other.nli_score or 0.0) > (n.nli_score or 0.0):
                freshness = 0.5
                break

    return ent * directness * tag_weight * freshness


def _required_k_for_qtype(db: PipelineDB) -> int:
    """Required number of supporting nuggets per qtype. Missing supporters
    penalise the score; extras push it up until the 1.0 clamp.

      * simple                  → 1
      * simple_w_condition      → 2 (main fact + condition)
      * multi_hop               → 2 (both hops)
      * comparison              → 2 (both sides)
      * post_processing         → 1
      * aggregation scalar ans  → 1 (e.g. "1 movie")
      * aggregation list ans    → expected_count (from classifier),
                                  fallback 3 when missing.
    """
    qtype_raw = (db.question_type or "").strip().lower()
    qtype = qtype_raw if qtype_raw in _VALID_QTYPES else "simple"

    if qtype == "aggregation":
        if _is_scalar_count_answer(db.answer.draft_answer):
            return 1
        exp = db.answer.expected_count
        if exp is None:
            try:
                _, exp, _ = question_parser(db.question, qtype)
            except Exception:
                exp = None
        return int(exp) if exp and exp > 0 else 3

    return {
        "simple": 1,
        "simple_w_condition": 2,
        "multi_hop": 2,
        "comparison": 2,
        "post_processing": 1,
    }.get(qtype, 1)


def evidence_quality(db: PipelineDB) -> float:
    """Sum of supporter impacts normalised by the required nugget count::

        k = _required_k_for_qtype(db)
        evidence_quality = min(sum(impacts across supporters) / k, 1.0)

    Properties:
      * Simple Q (k=1) with 1 perfect supporter   → 1.0 (can hit the
        ceiling, no shape cap).
      * Simple Q (k=1) with 2 supporters @ 0.95+0.70 → min(1.65, 1.0) =
        1.0: extras push the score up.
      * Multi-hop Q (k=2) with only 1 supporter @ 0.95 → 0.475,
        structurally penalised because the second hop is unverified.
      * Multi-hop Q (k=2) with 3 supporters @ 0.90/0.85/0.80 → min(2.55
        / 2, 1.0) = 1.0: enough corroboration to max out.
      * Monotonicity: adding any verified_supporting nugget cannot lower
        the score (the sum is monotone non-decreasing in the supporter
        set; the 1.0 clamp is also monotone).

    When there are no verified_supporting nuggets but at least one
    ``[strengthens-by-negation]``-tagged verified_infirming, those elim
    nuggets contribute at 0.8× their impact (PCWA discount).
    """
    impacts = [
        nugget_impact(n, db)
        for n in db.nuggets.values()
        if n.status == "verified_supporting" and not getattr(n, "is_synthetic", False)
    ]
    if not impacts:
        tagged = set(_STRENGTHEN_TAG_RE.findall(db.answer.rationale or ""))
        impacts = [
            0.8 * nugget_impact(n, db)
            for n in db.nuggets.values()
            if n.status == "verified_infirming"
            and n.nugget_id in tagged
            and not getattr(n, "is_synthetic", False)
        ]
    impacts = [s for s in impacts if s > 0]
    if not impacts:
        return 0.0

    k = max(1, _required_k_for_qtype(db))
    return min(sum(impacts) / k, 1.0)


def max_directness(db: PipelineDB) -> float:
    """Max per-nugget directness across verified_supporting. Same
    asymmetric containment metric as ``nugget_impact`` (whole-answer or
    best per-item match). Synthetic placeholder nuggets are excluded."""
    best = 0.0
    draft = db.answer.draft_answer or ""
    for n in db.nuggets.values():
        if n.status != "verified_supporting":
            continue
        if getattr(n, "is_synthetic", False):
            continue
        ratio = _best_answer_overlap(draft, _tokenize(n.text))
        d = _directness_from_ratio(ratio)
        if d > best:
            best = d
    return best


# Question-complexity classifier


_VALID_QTYPES = {
    "simple",
    "simple_w_condition",
    "comparison",
    "aggregation",
    "multi_hop",
    "post_processing",
}

# Scalar-count answer pattern: a bare integer, optionally followed by unit
# words ("1", "1 movie", "1: title", "one"). These answer a "how many X" style
# aggregation question with a single scalar count rather than an enumerated
# list. The `_count_distinct_items` heuristic cannot resolve these because
# "1" does not token-overlap supporting nugget text, so they are treated as
# simple for coverage purposes.
_SCALAR_COUNT_ANSWER_RE = re.compile(
    r"""^\s*
        (?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)
        \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_scalar_count_answer(draft_answer: str | None) -> bool:
    """Detect a scalar-count answer like "1", "1 movie", "one", "1: title".

    A draft answer is classified as scalar-count when it starts with a bare
    integer (or a small number word) followed by a word boundary. Callers use
    this to bypass list-style coverage penalties for count-type aggregation
    questions where the answer is a single number, not an enumerated list.
    """
    if not draft_answer:
        return False
    return bool(_SCALAR_COUNT_ANSWER_RE.match(draft_answer.strip()))


def required_coverage(db: PipelineDB) -> RequiredCoverage:
    """Deterministic per-qtype coverage table.

    Count-answer exception: when the question is tagged ``aggregation``
    but the draft answer is a scalar count (e.g. "1", "one", "1: title"),
    we return a simple-Q-like coverage spec. The items-coverage heuristic
    cannot match a bare integer against narrative supporting nugget text,
    which otherwise collapses the coverage factor to ~1/N and wipes the
    confidence of correct count answers.
    """
    qtype_raw = (db.question_type or "").strip().lower()
    qtype = qtype_raw if qtype_raw in _VALID_QTYPES else "simple"

    # Pull expected_count off the parsed answer state where available, else
    # parse on demand (seed_db_from_trace already populated it for real DBs).
    expected_count: int | None = db.answer.expected_count
    if expected_count is None:
        try:
            _, expected_count, _ = question_parser(db.question, qtype)
        except Exception:
            expected_count = None

    # Count-answer exception: a scalar integer draft on an aggregation Q
    # collapses to a simple-Q coverage spec. Preserves genuine list-type
    # aggregation semantics ("asthma; diabetes; hypertension" still enforces
    # item coverage).
    if qtype == "aggregation" and _is_scalar_count_answer(db.answer.draft_answer):
        return RequiredCoverage(
            min_distinct_supporters=1,
            min_distinct_claims=1,
            items_expected=None,
            directness_floor=0.60,
        )

    if qtype == "simple":
        return RequiredCoverage(
            min_distinct_supporters=1,
            min_distinct_claims=1,
            items_expected=None,
            directness_floor=0.85,
        )
    if qtype == "simple_w_condition":
        return RequiredCoverage(
            min_distinct_supporters=1,
            min_distinct_claims=1,
            items_expected=None,
            directness_floor=0.80,
        )
    if qtype == "comparison":
        return RequiredCoverage(
            min_distinct_supporters=2,
            min_distinct_claims=2,
            items_expected=2,
            directness_floor=0.70,
        )
    if qtype == "aggregation":
        n = int(expected_count) if expected_count else 3
        min_supporters = max(2, math.ceil(n / 2))
        return RequiredCoverage(
            min_distinct_supporters=min_supporters,
            min_distinct_claims=n,
            items_expected=n,
            directness_floor=0.60,
        )
    if qtype == "multi_hop":
        return RequiredCoverage(
            min_distinct_supporters=2,
            min_distinct_claims=2,
            items_expected=None,
            directness_floor=0.70,
        )
    # post_processing
    return RequiredCoverage(
        min_distinct_supporters=2,
        min_distinct_claims=2,
        items_expected=expected_count,
        directness_floor=0.70,
    )


def _count_distinct_items(db: PipelineDB, items_expected: int | None) -> int:
    """Count distinct items enumerated in the draft answer that are also
    present in the supporting-nugget corpus.

    Simple heuristic: split draft_answer on ``;`` / ``,`` / ``\\n`` /
    ``and`` and count the non-empty trimmed segments whose token set
    overlaps with some verified_supporting nugget text.
    """
    if items_expected is None:
        return 0
    draft = (db.answer.draft_answer or "").strip()
    if not draft:
        return 0
    raw_items = re.split(r"[;,\n]|\band\b", draft, flags=re.IGNORECASE)
    items = [it.strip() for it in raw_items if it.strip()]
    supporting_texts = [
        _tokenize(n.text)
        for n in db.nuggets.values()
        if n.status == "verified_supporting"
    ]
    matched = 0
    for item in items:
        itok = _tokenize(item)
        if not itok:
            continue
        for s_tok in supporting_texts:
            if itok & s_tok:
                matched += 1
                break
    return matched


def coverage_sufficient(db: PipelineDB, req: RequiredCoverage) -> bool:
    """Boolean predicate: does the DB meet the required coverage?"""
    n_pids = len(effective_supporting_pids(db))
    n_claims = len(effective_supporting_ids(db))
    direct_max = max_directness(db)
    if n_pids < req.min_distinct_supporters:
        return False
    if n_claims < req.min_distinct_claims:
        return False
    if direct_max < req.directness_floor:
        return False
    if req.items_expected is not None:
        items_covered = _count_distinct_items(db, req.items_expected)
        if items_covered < req.items_expected:
            return False
    return True


# base_answer_shape / coverage_factor / weakens_penalty


def base_answer_shape(db: PipelineDB) -> float:
    """Shape prior ∈ [0.5, 1.0]. Proxy for "does the answer have the
    right structural shape for the question".

    Heuristics:
        - empty draft → 0.5;
        - aggregation, items >= expected → 1.0; >=2 items → 0.7;
          otherwise 0.6 (incomplete-list penalty);
        - comparison, >=2 items → 1.0; otherwise 0.7;
        - simple / simple_w_condition / multi_hop / post_processing →
          1.0. The per-qtype required-k gate lives in
          ``evidence_quality`` (sum(impacts)/k clamped to 1.0), so the
          shape multiplier does not cap the ceiling.

    The 1.0 baseline for non-list qtypes lets a perfectly-cited simple-Q
    rollout with enough supporters reach conf=1.0. Incomplete
    aggregation/comparison answers are still penalised via this shape
    prior and via ``coverage_factor``'s items-coverage gate.
    """
    draft = (db.answer.draft_answer or "").strip()
    if not draft:
        return 0.5
    qtype = (db.question_type or "").strip().lower()
    if qtype == "aggregation":
        raw_items = re.split(r"[;,\n]|\band\b", draft, flags=re.IGNORECASE)
        items = [it.strip() for it in raw_items if it.strip()]
        exp = db.answer.expected_count
        if exp is not None and len(items) >= exp:
            return 1.0
        return 0.7 if len(items) >= 2 else 0.6
    if qtype == "comparison":
        raw_items = re.split(r"[;,\n]|\band\b", draft, flags=re.IGNORECASE)
        items = [it.strip() for it in raw_items if it.strip()]
        return 1.0 if len(items) >= 2 else 0.7
    return 1.0  # simple / simple_w_condition / multi_hop / post_processing


def coverage_factor(db: PipelineDB, req: RequiredCoverage) -> float:
    """Ratio of observed coverage to required. Clipped to [0, 1].

    Each of (supporters, claims, directness, items) contributes a
    sub-ratio; the result is the minimum, since the driver is bottlenecked
    by its weakest dimension.
    """
    n_pids = len(effective_supporting_pids(db))
    n_claims = len(effective_supporting_ids(db))
    direct_max = max_directness(db)

    sup_ratio = (
        1.0
        if req.min_distinct_supporters == 0
        else min(1.0, n_pids / req.min_distinct_supporters)
    )
    claim_ratio = (
        1.0
        if req.min_distinct_claims == 0
        else min(1.0, n_claims / req.min_distinct_claims)
    )
    dir_ratio = (
        1.0
        if req.directness_floor == 0
        else min(1.0, direct_max / req.directness_floor)
    )
    if req.items_expected is None:
        items_ratio = 1.0
    else:
        items_covered = _count_distinct_items(db, req.items_expected)
        items_ratio = min(1.0, items_covered / max(1, req.items_expected))
    return max(0.0, min(sup_ratio, claim_ratio, dir_ratio, items_ratio))


def _weakens_penalty(db: PipelineDB) -> float:
    """``0.15 × min(w, 3)``, the same penalty as the legacy formula."""
    a = db.answer
    w_rationale = _count_weakens_tags(a.rationale or "")
    w_trace = _count_weakens_tags(a.reasoning_trace or "")
    w = max(w_rationale, w_trace)
    return 0.15 * min(w, 3)


def _compute_confidence_decoupled(db: PipelineDB) -> float:
    """Decoupled formula. See the ``compute_confidence`` docstring."""
    supporters = [n for n in db.nuggets.values() if n.status == "verified_supporting"]
    tagged_elim = set(_STRENGTHEN_TAG_RE.findall(db.answer.rationale or ""))
    elim_supporters = [
        n
        for n in db.nuggets.values()
        if n.status == "verified_infirming" and n.nugget_id in tagged_elim
    ]
    if not supporters and not elim_supporters:
        return 0.0

    req = required_coverage(db)
    shape = base_answer_shape(db)
    quality = evidence_quality(db)
    cov = coverage_factor(db, req)
    penalty = _weakens_penalty(db)
    conf = shape * quality * cov - penalty
    return max(0.0, min(1.0, conf))


# Research-driver decoupling: should_continue


def should_continue(db: PipelineDB, budget: BudgetState) -> ContinueDecision:
    """Loop continue/stop signal. Pure function of (db, budget).

    Takes the place of ``run_s6`` rule (a) under
    ``R2C2_DECOUPLED_SCORING=1``. Rules (b)-(g) carry the same semantics
    as in the legacy ``run_s6`` body. The body must not read the reported
    confidence score; a static lint test enforces this.
    """
    a = db.answer

    # Rule f — wall-clock hard kill (15 min)
    if budget.wall_clock_sec > 15 * 60:
        return ContinueDecision(
            action="stop", reason="wall_clock exceeded", stop_code="error"
        )

    # Rule g — unresolvable (explicit S4 verdict)
    if a.supportability == "unresolvable":
        return ContinueDecision(
            action="stop",
            reason="supportability=unresolvable",
            stop_code="low_confidence_refusal",
        )

    n_effective = len(effective_supporting_ids(db))
    all_real_classified = all(
        p.status != "unclassified"
        for pid, p in db.passages.items()
        if pid != REFUSAL_PID
    )
    all_nuggets_classified = all(
        n.status != "unclassified" for n in db.nuggets.values()
    )

    # Rule a: coverage met, in place of conf>=0.8. The reported confidence
    # is not read here.
    req = required_coverage(db)
    if (
        a.supportability == "sufficient"
        and coverage_sufficient(db, req)
        and n_effective >= 1
        and all_real_classified
        and all_nuggets_classified
    ):
        return ContinueDecision(
            action="stop", reason="coverage_met", stop_code="high_confidence"
        )

    # Rule c — max_iter
    if budget.iter_count >= budget.max_iter:
        return ContinueDecision(
            action="stop", reason="max_iter reached", stop_code="max_iter"
        )

    # Rule d — same missing twice
    cur = re.sub(r"\s+", " ", (a.missing or "")).strip().lower()
    prev = re.sub(r"\s+", " ", (a.prev_missing or "")).strip().lower()
    if budget.iter_count >= 2 and cur and cur == prev:
        return ContinueDecision(
            action="stop",
            reason="missing unchanged across 2 iters",
            stop_code="low_confidence_refusal",
        )

    # Rule b — all classified, zero supporting, iter>=1
    if (
        all_real_classified
        and all_nuggets_classified
        and n_effective == 0
        and budget.iter_count >= 1
    ):
        return ContinueDecision(
            action="stop",
            reason="no supporting evidence after classification",
            stop_code="low_confidence_refusal",
        )

    # Rule e — accepted count stalled across 2 iterations
    accepted_now = sum(
        1
        for pid, p in db.passages.items()
        if pid != REFUSAL_PID and p.status in {"accepted", "counter_evidence"}
    )
    if (
        budget.iter_count >= 2
        and accepted_now == a.prev_accepted_pid_count
        and a.supportability != "sufficient"
    ):
        return ContinueDecision(
            action="stop",
            reason="accepted pid count stalled",
            stop_code="low_confidence_refusal",
        )

    return ContinueDecision(action="continue", reason="loop_s1", stop_code=None)


# Refusal synthesis (idempotent, guarantees >=1 nugget)


def synthesize_refusal(db: PipelineDB) -> None:
    """Guarantee invariant: exactly 1 nugget status='refusal' + conf=0.0.

    Idempotent — safe to call multiple times. Anchored to the virtual
    __refusal__ passage (seeded at DB init). Sets canonical draft and
    appends a TransitionEvent for replay.
    """
    if REFUSAL_PID not in db.passages:
        _seed_virtual_refusal_passage(db)

    if not any(n.status == "refusal" for n in db.nuggets.values()):
        n = Nugget(
            passage_id=REFUSAL_PID,
            text=REFUSAL_PASSAGE_TEXT,
            proposed_stance="supporting",
            source_step="s6_refusal_synthesis",
            source_tool_call_id="system",
            status="refusal",
            status_reason="no supporting evidence after exhaustive review",
            nli_label="entails",
            nli_score=1.0,
        )
        db.nuggets[n.nugget_id] = n
        ev = TransitionEvent(
            step_id="s6_refusal_synthesis",
            iteration=db.answer.iteration_count,
            actor="system",
            from_status=None,
            to_status="refusal",
            reason="guarantee >=1 nugget (user constraint, §4.6)",
        )
        db.events.append(ev)
        n.history.append(ev)

    db.answer.draft_answer = REFUSAL_ANSWER_TEXT
    db.answer.confidence = 0.0
    marker = f"[refusal synthesized at iter={db.answer.iteration_count}]"
    if marker not in (db.answer.reasoning_trace or ""):
        db.answer.reasoning_trace = (
            (db.answer.reasoning_trace or "") + "\n" + marker
        ).strip()


# Advance rule checks (per-step post-conditions)


# Return values: either a next-step id ("s2", "s3", "s4_reason", "s5_reason",
# "s6", "s1"), a finish instruction ("finish:<exit_reason>") or "block".
AdvanceResult = str


def advance_after_s1(db: PipelineDB) -> AdvanceResult:
    """S1 -> S2 or finish(refusal) on empty corpus at iter 0."""
    unclassified = [
        p
        for pid, p in db.passages.items()
        if pid != REFUSAL_PID and p.status == "unclassified"
    ]
    if unclassified:
        return "block"  # S1 not done — controller retries or errors
    accepted_or_counter = sum(
        1
        for pid, p in db.passages.items()
        if pid != REFUSAL_PID and p.status in {"accepted", "counter_evidence"}
    )
    if accepted_or_counter == 0 and db.answer.iteration_count == 0:
        return "finish:low_confidence_refusal"
    return "s2"


def advance_after_s2(db: PipelineDB) -> AdvanceResult:
    """S2 -> S3 once every accepted/counter pid has >=1 nugget (or quota hit)."""
    for pid, p in db.passages.items():
        if pid == REFUSAL_PID:
            continue
        if p.status not in {"accepted", "counter_evidence"}:
            continue
        nug_count = sum(1 for n in db.nuggets.values() if n.passage_id == pid)
        if nug_count == 0:
            return "block"
    return "s3"


def advance_after_s3(db: PipelineDB) -> AdvanceResult:
    """S3 -> S4 once all nuggets classified.

    S3 -> S6-unresolvable if count(verified_*)==0 AND len(nuggets)>=3 (all bogus).
    """
    if any(n.status == "unclassified" for n in db.nuggets.values()):
        return "block"
    real_nuggets = [n for n in db.nuggets.values() if n.status != "refusal"]
    verified = [
        n
        for n in real_nuggets
        if n.status in {"verified_supporting", "verified_infirming"}
    ]
    if len(verified) == 0 and len(real_nuggets) >= 3:
        db.answer.supportability = "unresolvable"
        return "s6"
    return "s4_reason"


def advance_after_s4(db: PipelineDB) -> AdvanceResult:
    """S4 -> S5 | loop_s1 | finish(refusal).

    sufficient + effective_supporting>=1 + no unclassified nuggets -> s5
    insufficient|contradicted + iter<max_iter + new missing -> s1 (loopback)
    unresolvable -> finish(refusal)
    """
    a = db.answer
    if any(n.status == "unclassified" for n in db.nuggets.values()):
        return "block"
    if a.supportability == "unresolvable":
        return "finish:low_confidence_refusal"
    if a.supportability == "sufficient":
        if len(effective_supporting_ids(db)) >= 1:
            return "s5_reason"
        # sufficient claim but no effective-supporting nuggets — invalid;
        # fall through to insufficient semantics.
        a.supportability = "insufficient"
    if a.supportability in {"insufficient", "contradicted"}:
        if a.iteration_count >= db.max_iter:
            return "finish:max_iter"
        # Normalize for comparison: collapse whitespace, lowercase
        cur = re.sub(r"\s+", " ", (a.missing or "")).strip().lower()
        prev = re.sub(r"\s+", " ", (a.prev_missing or "")).strip().lower()
        if cur and cur != prev:
            return "s1"
        # same missing OR empty missing on insufficient -> stuck -> refusal
        return "finish:low_confidence_refusal"
    return "block"


def advance_after_s5(db: PipelineDB) -> AdvanceResult:
    """S5 -> S6 always."""
    return "s6"


# S6: deterministic state machine over escape rules a-g


def _stop_code_to_exit_reason(code: str | None) -> str:
    """Translate ``ContinueDecision.stop_code`` to ``run_s6``'s legacy
    exit_reason string. Keeps downstream consumers (which switch on the
    tuple's second element) oblivious to the decoupling."""
    if code == "high_confidence":
        return "high_confidence"
    if code == "max_iter":
        return "max_iter"
    if code == "low_confidence_refusal":
        return "low_confidence_refusal"
    if code in ("wall_clock", "error"):
        return "error"
    return ""


def run_s6(db: PipelineDB, *, wall_clock_sec: float = 0.0) -> tuple[str, str]:
    """Evaluate escape rules a-g in order; return (action, exit_reason).

    Under the legacy path (``R2C2_DECOUPLED_SCORING`` unset or 0):
      a. conf>=0.8 AND supportability=sufficient AND invariants -> high_confidence
      b. all_classified AND effective_supporting==0 AND iter>=1 -> refusal
      c. iter>=max_iter -> max_iter (controller-issued)
      d. missing==prev_missing AND iter>=2 -> refusal
      e. count(accepted) unchanged 2 iters in a row -> refusal
      f. wall_clock > 15min hard kill -> error
      g. supportability==unresolvable -> refusal

    Under the decoupled path (``R2C2_DECOUPLED_SCORING=1``):
      The whole body is delegated to ``should_continue``. Rule (a)'s
      ``conf >= 0.8`` is replaced by ``coverage_sufficient(db, required_coverage(db))``.

    Returns ("finish", "<exit_reason>") on terminal; ("loop_s1", "")
    otherwise. Tuple shape preserved for back-compat with callers in
    ``run_v2_pipeline`` and ``pipeline_handlers.py``.
    """
    a = db.answer
    # Derive the confidence before consulting it. The controller is expected
    # to call compute_confidence upstream; deriving it again here guards
    # against that being forgotten. Under the decoupled path the driver does
    # not read this float, but it is still written for the submission envelope.
    derived_conf = compute_confidence(db)
    if abs(derived_conf - a.confidence) > 1e-6 and a.exit_reason is None:
        a.confidence = derived_conf

    if _decoupled_scoring_enabled():
        budget = BudgetState(
            wall_clock_sec=wall_clock_sec,
            iter_count=a.iteration_count,
            max_iter=db.max_iter,
            calls_used=0,
            calls_remaining=0,
        )
        dec = should_continue(db, budget)
        if dec.action == "stop":
            return "finish", _stop_code_to_exit_reason(dec.stop_code)
        return "loop_s1", ""

    n_effective = len(effective_supporting_ids(db))
    all_real_classified = all(
        p.status != "unclassified"
        for pid, p in db.passages.items()
        if pid != REFUSAL_PID
    )
    all_nuggets_classified = all(
        n.status != "unclassified" for n in db.nuggets.values()
    )

    # Rule f — wall-clock hard kill (15 min)
    if wall_clock_sec > 15 * 60:
        return "finish", "error"

    # Rule g — unresolvable (explicit S4 verdict)
    if a.supportability == "unresolvable":
        return "finish", "low_confidence_refusal"

    # Rule a — high_confidence
    if (
        a.supportability == "sufficient"
        and a.confidence >= 0.8
        and n_effective >= 1
        and all_real_classified
        and all_nuggets_classified
    ):
        return "finish", "high_confidence"

    # Rule c — max_iter
    if a.iteration_count >= db.max_iter:
        return "finish", "max_iter"

    # Rule d — same missing twice
    cur = re.sub(r"\s+", " ", (a.missing or "")).strip().lower()
    prev = re.sub(r"\s+", " ", (a.prev_missing or "")).strip().lower()
    if a.iteration_count >= 2 and cur and cur == prev:
        return "finish", "low_confidence_refusal"

    # Rule b — all classified, zero supporting, iter>=1
    if (
        all_real_classified
        and all_nuggets_classified
        and n_effective == 0
        and a.iteration_count >= 1
    ):
        return "finish", "low_confidence_refusal"

    # Rule e — accepted count stalled across 2 iterations
    accepted_now = sum(
        1
        for pid, p in db.passages.items()
        if pid != REFUSAL_PID and p.status in {"accepted", "counter_evidence"}
    )
    if (
        a.iteration_count >= 2
        and accepted_now == a.prev_accepted_pid_count
        and a.supportability != "sufficient"
    ):
        return "finish", "low_confidence_refusal"

    # Otherwise loop back to S1 for re-review
    return "loop_s1", ""


# Length-watchdog retry policy


# Call-type semantics for finish_reason=="length" retry cap:
#   "reason": S4a / S5a — retry with max_tokens += 4096
#   "tool":   S1 / S2 / S3 tool-ful turn — retry with ceil(prev * 1.5)
#   "write":  S4b / S5b — retry with max_tokens = 2048 (already at cap)
_LEN_RETRY_REASON_BUMP = 4096
_LEN_RETRY_TOOL_MULT = 1.5
_LEN_RETRY_WRITE_CAP = 2048
_LEN_RETRY_MAX_ATTEMPTS = 1  # one retry per call-type per step


def length_retry_max_tokens(call_type: str, prev_cap: int) -> int:
    """Return the new max_tokens to use on a length-truncation retry."""
    if call_type == "reason":
        return prev_cap + _LEN_RETRY_REASON_BUMP
    if call_type == "tool":
        return int(prev_cap * _LEN_RETRY_TOOL_MULT + 0.5)
    if call_type == "write":
        return max(prev_cap, _LEN_RETRY_WRITE_CAP)
    raise ValueError(f"unknown call_type {call_type!r}")


# ReAct-style inner loop driver


class ReactResult:
    """Outcome of a single-step ReAct session."""

    __slots__ = (
        "text",
        "stop_reason",
        "tool_calls_made",
        "length_retries",
        "messages",
    )

    def __init__(
        self,
        *,
        text: str,
        stop_reason: str,
        tool_calls_made: int,
        length_retries: int,
        messages: list[dict],
    ) -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.tool_calls_made = tool_calls_made
        self.length_retries = length_retries
        self.messages = messages


async def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    thinking: bool = False,
    temperature: float = 0.0,
) -> Any:
    """Thin wrapper around chat.completions.create honoring thinking/tools."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        },
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return await client.chat.completions.create(**kwargs)


def _tool_calls_from_message(msg: Any) -> list[Any]:
    """Safely pull tool_calls off an OpenAI message, return [] if none."""
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return []
    return list(tool_calls)


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    """Parse the JSON `arguments` string off a tool_call; return {} on error."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass
    return {}


async def run_step_react(
    client: Any,
    model: str,
    ctx: ToolContext,
    *,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: list[dict] | None = None,
    max_tokens: int = 4096,
    max_tool_calls: int = 15,
    thinking: bool = False,
    call_type: str = "tool",
) -> ReactResult:
    """Single agentic session with a tool-call inner loop.

    - Opens one chat session (system + user).
    - For each assistant message: if tool_calls -> dispatch via dispatch_tool,
      append tool messages, continue; else stop.
    - Enforces per-step tool-call cap via max_tool_calls. The global cap lives
      in dispatch_tool itself (checked against ctx.db.max_tool_calls_global).
    - finish_reason=="length" triggers one retry.

    Returns a ReactResult with the final assistant text + stop reason.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_calls_made = 0
    length_retries = 0
    current_cap = max_tokens
    length_attempted = False
    llm_turns = 0
    # Independent bound on LLM API round-trips. Even if each turn emits 0
    # tool calls (pathological), the step cannot burn more than this many
    # LLM calls.
    max_llm_turns = max(max_tool_calls // 2 + 5, 20)

    while tool_calls_made < max_tool_calls and llm_turns < max_llm_turns:
        llm_turns += 1
        resp = await _chat_completion(
            client,
            model=model,
            messages=messages,
            tools=allowed_tools,
            max_tokens=current_cap,
            thinking=thinking,
        )
        choice = resp.choices[0]
        msg = choice.message
        finish_reason = getattr(choice, "finish_reason", "stop")

        # Append the assistant turn (preserve reasoning if present).
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(msg, "content", "") or "",
        }
        reasoning = getattr(msg, "reasoning", None)
        if reasoning:
            assistant_entry["reasoning"] = reasoning
        raw_tool_calls = _tool_calls_from_message(msg)
        if raw_tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": getattr(tc, "id", f"tc_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]
        messages.append(assistant_entry)

        # Length-truncation recovery
        if finish_reason == "length" and not length_attempted:
            length_attempted = True
            length_retries += 1
            current_cap = length_retry_max_tokens(call_type, current_cap)
            # Tool calls already parsed from the truncated message are
            # dispatched, and so committed to the DB, before retrying.
            for tc in raw_tool_calls:
                ctx.tool_call_id = getattr(tc, "id", "tc_unknown")
                args = _parse_tool_args(tc.function.arguments)
                result = dispatch_tool(ctx, tc.function.name, args)
                tool_calls_made += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": ctx.tool_call_id,
                        "name": tc.function.name,
                        "content": json.dumps(result),
                    }
                )
            continue  # retry

        if not raw_tool_calls:
            # No tool calls — step ends (ReAct natural termination).
            return ReactResult(
                text=assistant_entry["content"],
                stop_reason=finish_reason,
                tool_calls_made=tool_calls_made,
                length_retries=length_retries,
                messages=messages,
            )

        # Dispatch each tool call; append results as tool messages.
        for tc in raw_tool_calls:
            if tool_calls_made >= max_tool_calls:
                # Inject step-budget-exhausted error and exit.
                err = _err(
                    "step_tool_cap_exhausted",
                    f"per-step tool-call cap ({max_tool_calls}) reached; finalize now",
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", "tc_unknown"),
                        "name": tc.function.name,
                        "content": json.dumps(err),
                    }
                )
                return ReactResult(
                    text=assistant_entry["content"],
                    stop_reason="step_tool_cap",
                    tool_calls_made=tool_calls_made,
                    length_retries=length_retries,
                    messages=messages,
                )
            ctx.tool_call_id = getattr(tc, "id", f"tc_{uuid.uuid4().hex[:8]}")
            args = _parse_tool_args(tc.function.arguments)
            result = dispatch_tool(ctx, tc.function.name, args)
            tool_calls_made += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ctx.tool_call_id,
                    "name": tc.function.name,
                    "content": json.dumps(result),
                }
            )

        # Reset length-retry availability on a successful non-length turn so
        # the next turn gets its own retry budget (per-call, not per-session).
        if finish_reason != "length":
            length_attempted = False

    # Budget exhausted without natural termination — `stop_reason` tells the
    # caller which cap fired.
    exhausted_reason = (
        "step_llm_turn_cap" if llm_turns >= max_llm_turns else "step_tool_cap"
    )
    return ReactResult(
        text="",
        stop_reason=exhausted_reason,
        tool_calls_made=tool_calls_made,
        length_retries=length_retries,
        messages=messages,
    )


async def run_reason_call(
    client: Any,
    model: str,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 12288,
    thinking: bool = True,
    call_type: str = "reason",
) -> tuple[str, str, int]:
    """Split-mode REASON call: no tools, thinking on by default.

    Returns (text, stop_reason, length_retries). One retry on finish_reason
    =="length".
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    current_cap = max_tokens
    length_retries = 0
    for attempt in range(2):
        resp = await _chat_completion(
            client,
            model=model,
            messages=messages,
            tools=None,
            max_tokens=current_cap,
            thinking=thinking,
        )
        choice = resp.choices[0]
        msg = choice.message
        finish_reason = getattr(choice, "finish_reason", "stop")
        content = getattr(msg, "content", "") or ""
        if finish_reason != "length" or attempt == 1:
            return content, finish_reason, length_retries
        length_retries += 1
        current_cap = length_retry_max_tokens(call_type, current_cap)
        # Retry with assistant-prefill hint already baked into the system prompt.
    return "", "length", length_retries


# Exports: step controllers, prompts, confidence and driver


__all__ += [  # noqa: F821
    # Prompts + schemas (exported for the step runners)
    "_S1_SYSTEM_PROMPT",
    "_S2_SYSTEM_PROMPT",
    "_S3_SYSTEM_PROMPT",
    "_S4A_REASON_PROMPT",
    "_S4B_WRITE_PROMPT",
    "_S5A_REASON_PROMPT",
    "_S5B_WRITE_PROMPT",
    "_format_user_prompt",
    "tool_schemas_for_step",
    # Parsers
    "parse_s4_reason",
    "parse_s5_reason",
    # Confidence + refusal
    "compute_confidence",
    "synthesize_refusal",
    # Scoring/driver decoupling
    "RequiredCoverage",
    "BudgetState",
    "ContinueDecision",
    "required_coverage",
    "coverage_sufficient",
    "coverage_factor",
    "base_answer_shape",
    "nugget_impact",
    "evidence_quality",
    "max_directness",
    "should_continue",
    # Advance rules
    "advance_after_s1",
    "advance_after_s2",
    "advance_after_s3",
    "advance_after_s4",
    "advance_after_s5",
    # S6 state machine
    "run_s6",
    # ReAct driver
    "ReactResult",
    "run_step_react",
    "run_reason_call",
    "length_retry_max_tokens",
]


# v2 pipeline: MMR-greedy adapter, top-level driver, legacy-shape output and
# feature-flag dispatch.
# The adapter runs candidate filter → MMR-greedy truncation → refusal
# fallback → legacy tuple. The step runners for S1-S5 wire the ReAct driver
# into the 6-step loop, and `reconcile_nuggets_and_passages()` branches on the
# feature flag.


POSTTRACE_PIPELINE_ENV = "POSTTRACE_PIPELINE"
POSTTRACE_PIPELINE_V2 = "v2_db_tools"
POSTTRACE_PIPELINE_LEGACY = "legacy"


# Tunables for the MMR adapter.
_MMR_LAMBDA = 0.7
_MMR_SIM_THRESHOLD = 0.75


# Embedder (lazy-loaded; ~22 MB, CPU-only)


_EMBEDDER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder_cache: list[Any] = []  # holds at most one entry


def _get_embedder() -> Any | None:
    """Lazy-load sentence-transformers/all-MiniLM-L6-v2.

    Returns None if sentence-transformers isn't available or fails to load;
    the adapter then falls back to score-only ranking.
    """
    if _embedder_cache:
        return _embedder_cache[0]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("sentence-transformers not available; MMR falls back to score-only")
        _embedder_cache.append(None)
        return None
    try:
        model = SentenceTransformer(_EMBEDDER_MODEL_NAME, device="cpu")
        _embedder_cache.append(model)
        return model
    except Exception:
        log.exception("embedder load failed; MMR falls back to score-only")
        _embedder_cache.append(None)
        return None


# Candidate filter


def adapter_candidate_filter(db: PipelineDB) -> list[Nugget]:
    """Candidate filter for the adapter.

    Include:
      - every `verified_supporting` nugget
      - every `verified_infirming` nugget tagged `[strengthens-by-negation]`
        (these corroborate the draft by ruling out a near-miss alternative)

    Drop:
      - any `verified_infirming` nugget tagged `[weakens]`
      - any other status (bogus, unclassified, refusal; refusal is handled
        separately by the refusal-fallback branch)

    Tags are parsed from `db.answer.rationale` and `db.answer.reasoning_trace`
    (S4 emits into rationale; S5 emits into reasoning_trace). Nuggets
    without an explicit tag default to include.
    """
    # Gather explicit tags
    strengthen: set[str] = set()
    weaken: set[str] = set()
    sources = (db.answer.rationale or "") + "\n" + (db.answer.reasoning_trace or "")
    for m in _STRENGTHEN_TAG_RE.finditer(sources):
        strengthen.add(m.group(1))
    for m in re.finditer(r"(?m)^(n_[0-9a-f]{8})\s*:\s*\[?weakens\]?\s*$", sources):
        weaken.add(m.group(1))

    kept: list[Nugget] = []
    for n in db.nuggets.values():
        # Synthetic placeholders (force_deliver_synth_nugget) are
        # bookkeeping artefacts: they entail the picked answer by
        # construction but were not recorded against retrieved corpus
        # evidence. Excluding them here keeps the emitted
        # ``reconciled_nuggets`` list to nuggets the agent verified
        # during retrieval.
        if getattr(n, "is_synthetic", False):
            continue
        if n.status == "verified_supporting":
            if n.nugget_id in weaken:
                continue  # explicit weaken drop
            kept.append(n)
        elif n.status == "verified_infirming":
            if n.nugget_id in strengthen:
                kept.append(n)
            # infirming without elim tag is dropped from the adapter output
        # bogus / unclassified / refusal — not included here
    return kept


# MMR-greedy selection


def select_nuggets_mmr(
    candidates: list[Nugget],
    *,
    target: int,
    nli_threshold: float = 0.5,
    sim_threshold: float = _MMR_SIM_THRESHOLD,
    lambda_: float = _MMR_LAMBDA,
    embedder: Any | None = None,
) -> list[Nugget]:
    """MMR-greedy selection (Carbonell & Goldstein 1998).

    Returns up to `target` nuggets; shrinks rather than pads on near-dup.
    Falls back to score-only ranking when no embedder is available.
    """
    cands = [
        n
        for n in candidates
        if n.nli_score is not None and n.nli_score >= nli_threshold
    ]
    if not cands:
        return []
    if len(cands) <= target:
        return sorted(cands, key=lambda n: -(n.nli_score or 0.0))

    if embedder is None:
        embedder = _get_embedder()

    if embedder is None:
        # Fallback: score-only truncation.
        return sorted(cands, key=lambda n: -(n.nli_score or 0.0))[:target]

    # Batched embedding (~15ms for N=30 on CPU).
    try:
        import numpy as np

        embeds = embedder.encode(
            [n.text for n in cands],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeds = np.asarray(embeds)
    except Exception:
        log.exception("embedder encode failed; falling back to score-only")
        return sorted(cands, key=lambda n: -(n.nli_score or 0.0))[:target]

    # Seed with highest-nli_score nugget.
    scores = [c.nli_score or 0.0 for c in cands]
    seed_idx = max(range(len(cands)), key=lambda i: scores[i])
    selected: list[int] = [seed_idx]

    while len(selected) < target:
        remaining = [i for i in range(len(cands)) if i not in selected]
        if not remaining:
            break

        def mmr_score(i: int) -> float:
            rel = scores[i]
            div = max(float(embeds[i] @ embeds[j]) for j in selected)
            return lambda_ * rel - (1 - lambda_) * div

        best = max(remaining, key=mmr_score)
        best_sim = max(float(embeds[best] @ embeds[j]) for j in selected)

        # Early stop: if the best remaining is a near-duplicate and we already
        # have min(3, target) items, shrink rather than pad.
        if best_sim > sim_threshold and len(selected) >= min(3, target):
            break
        selected.append(best)

    return [cands[i] for i in selected]


# Target nugget count per question type


def _nugget_target_for_db(db: PipelineDB) -> int:
    """Per-question-type target nugget count.

    Reuses legacy `_NUGGET_TARGET_BY_TYPE` if available; else a sensible default.
    """
    try:
        table = _NUGGET_TARGET_BY_TYPE  # type: ignore[name-defined] # noqa: F821
        default = _NUGGET_TARGET_DEFAULT  # type: ignore[name-defined] # noqa: F821
    except NameError:
        # Forward-declaration fallback (legacy block defines this below).
        table = {
            "simple": 3,
            "simple_w_condition": 3,
            "aggregation": 8,
            "comparison": 5,
            "multi_hop": 5,
            "post_processing": 4,
        }
        default = 5
    return int(table.get(db.question_type or "", default))


# Legacy-shape output builder


def _nugget_to_legacy_dict(n: Nugget, passage: Passage | None) -> dict:
    """Convert a v2 Nugget to the legacy dict shape consumed by
    generate_submission / validate_submission downstream.

    Legacy shape:
      {
        "nugget_text": str,
        "passage_id": str,
        "primary_passage_id": str,          # for passage-prune / rank tie-break
        "stance": "supporting" | "infirming",
        "relevance_score": float,           # derived from NLI score
        "nli_label": str,
        "nli_score": float,
      }
    """
    pid = n.passage_id
    stance = (
        "supporting" if n.status in {"verified_supporting", "refusal"} else "infirming"
    )
    return {
        "nugget_text": n.text,
        "passage_id": pid,
        "primary_passage_id": pid,
        "stance": stance,
        "relevance_score": float(n.nli_score or 0.0),
        "nli_label": n.nli_label or "entails",
        "nli_score": float(n.nli_score or 0.0),
        "source_passage_title": passage.title if passage is not None else None,
    }


def _passage_to_legacy_dict(p: Passage) -> dict:
    """Convert a v2 Passage to the legacy dict shape.

    Emits both `text` (v2-native) and `passage_text` / `doc_id` (legacy
    worker.py projection keys) so the final submission carries the
    actual passage content.
    """
    pid = p.passage_id
    doc_id = pid.rsplit("__", 1)[0] if "__" in pid else pid
    return {
        "passage_id": pid,
        "doc_id": doc_id,
        "text": p.text,
        "passage_text": p.text,
        "passage_text_with_title": (f"{p.title}: {p.text}" if p.title else p.text),
        "title": p.title or "",
        "section": "",
        "score": float(p.score or 0.0),
        "status": p.status,
    }


def _build_ranked_passages(
    db: PipelineDB, is_refusal: bool, cap: int = 20
) -> list[dict]:
    """PR ranking: normal vs refusal.

    Refusal: rank 1 = virtual __refusal__, 2..cap = top-(cap-1) rejected by score.
    Normal:  ranked by verified-supporting count per pid (tiebreak by score).
    """
    passages_by_pid = db.passages

    if is_refusal:
        out: list[dict] = []
        if REFUSAL_PID in passages_by_pid:
            out.append(_passage_to_legacy_dict(passages_by_pid[REFUSAL_PID]))
        rejected = [
            p
            for pid, p in passages_by_pid.items()
            if pid != REFUSAL_PID and p.status == "rejected"
        ]
        rejected.sort(key=lambda p: -(p.score or 0.0))
        out.extend(_passage_to_legacy_dict(p) for p in rejected[: cap - 1])
        return out

    # Normal path: rank by verified_supporting count, tiebreak by MCP score
    supp_count: dict[str, int] = {}
    for n in db.nuggets.values():
        if n.status == "verified_supporting":
            supp_count[n.passage_id] = supp_count.get(n.passage_id, 0) + 1

    real_pids = [pid for pid in passages_by_pid.keys() if pid != REFUSAL_PID]
    accepted_or_counter = [
        pid
        for pid in real_pids
        if passages_by_pid[pid].status in {"accepted", "counter_evidence"}
    ]
    accepted_or_counter.sort(
        key=lambda pid: (
            -supp_count.get(pid, 0),
            -(passages_by_pid[pid].score or 0.0),
        )
    )
    out = [_passage_to_legacy_dict(passages_by_pid[pid]) for pid in accepted_or_counter]

    # Pad remaining (rejected, unclassified) by score so downstream PR has 20
    remaining = [pid for pid in real_pids if pid not in set(accepted_or_counter)]
    remaining.sort(key=lambda pid: -(passages_by_pid[pid].score or 0.0))
    out.extend(_passage_to_legacy_dict(passages_by_pid[pid]) for pid in remaining)
    return out[:cap]


# Top-level adapter


def apply_adapter(
    db: PipelineDB, *, pr_cap: int = 20, nugget_cap: int = 10
) -> tuple[list[dict], list[dict], dict]:
    """Adapter: flatten PipelineDB to the legacy-shape tuple.

    Returns (ranked_passages, clean_nuggets, stats).

    Refusal path: emits exactly 1 nugget (the synthesized refusal-nugget),
    rank 1 is the virtual __refusal__ passage. Invariant: never return 0
    nuggets — if candidates empty and no refusal synthesized, raises
    InvariantError (should never happen — controller is expected to
    synthesize before calling the adapter).
    """
    target = min(_nugget_target_for_db(db), nugget_cap)

    # 1. Candidate filter
    cands = adapter_candidate_filter(db)

    # 2. MMR-greedy (or score-only if embedder unavailable)
    selected = select_nuggets_mmr(
        cands,
        target=target,
        nli_threshold=0.5,
    )

    # 3. Refusal fallback (guaranteed ≥1 nugget). Prefer a non-synthetic
    # refusal nugget; if only ``force_deliver_synth_nugget`` left a
    # synthetic placeholder, materialise the canonical refusal nugget
    # via ``synthesize_refusal`` so the emitted ``reconciled_nuggets``
    # carries the real refusal text.
    is_refusal = False
    if not selected:
        refusal = next(
            (
                n
                for n in db.nuggets.values()
                if n.status == "refusal" and not getattr(n, "is_synthetic", False)
            ),
            None,
        )
        if refusal is None:
            # Safety net: the controller did not synthesize, or only a
            # synthetic refusal exists. Materialise the real one.
            synthesize_refusal(db)
            refusal = next(
                (
                    n
                    for n in db.nuggets.values()
                    if n.status == "refusal" and not getattr(n, "is_synthetic", False)
                ),
                None,
            )
        if refusal is None:
            raise RuntimeError(
                "adapter: refusal path reached but refusal nugget not in DB"
            )
        selected = [refusal]
        is_refusal = True

    # 4. Build ranked passages
    ranked_passages = _build_ranked_passages(db, is_refusal, cap=pr_cap)

    # 5. Build clean nuggets list
    passages_by_pid = db.passages
    clean_nuggets = [
        _nugget_to_legacy_dict(n, passages_by_pid.get(n.passage_id))
        for n in selected[:nugget_cap]
    ]

    # 6. Stats
    tool_calls_by_name: dict[str, int] = {}
    for ev in db.events:
        name = (ev.reason or "").split()[0] if ev.reason else ""
        if ev.actor == "llm_tool" and name:
            tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
    stats = {
        "exit_reason": db.answer.exit_reason or "",
        "iteration_count": db.answer.iteration_count,
        "n_passages_accepted": sum(
            1
            for pid, p in passages_by_pid.items()
            if pid != REFUSAL_PID and p.status == "accepted"
        ),
        "n_passages_counter": sum(
            1
            for pid, p in passages_by_pid.items()
            if pid != REFUSAL_PID and p.status == "counter_evidence"
        ),
        "n_nuggets_verified_supporting": sum(
            1
            for n in db.nuggets.values()
            if n.status == "verified_supporting"
            and not getattr(n, "is_synthetic", False)
        ),
        "n_nuggets_verified_infirming": sum(
            1
            for n in db.nuggets.values()
            if n.status == "verified_infirming"
            and not getattr(n, "is_synthetic", False)
        ),
        "n_nuggets_synthetic": sum(
            1 for n in db.nuggets.values() if getattr(n, "is_synthetic", False)
        ),
        "n_effective_supporting": len(effective_supporting_ids(db)),
        "n_candidates": len(cands),
        "n_selected": len(selected),
        "is_refusal": is_refusal,
        "confidence": db.answer.confidence,
        "supportability": db.answer.supportability,
        "mmr_available": _get_embedder() is not None,
        # Legacy-compatible `steps` dict (tool-call counts keyed by handler).
        "steps": tool_calls_by_name,
    }
    return ranked_passages, clean_nuggets, stats


# Top-level v2 pipeline driver (seed → S1-S6 loop → adapter)


async def _run_s1(
    client: Any,
    model: str,
    db: PipelineDB,
    *,
    nli_server_url: str,
    missing_hint: str = "",
) -> ReactResult:
    """S1 — Evidence Gathering (tools: list/read/mark_passage)."""
    ctx = ToolContext(
        db=db,
        step_id="s1",
        iteration=db.answer.iteration_count,
        nli_server_url=nli_server_url,
    )
    user = _format_user_prompt(
        db,
        step_id="s1",
        iteration=db.answer.iteration_count,
        missing_hint=missing_hint,
    )
    return await run_step_react(
        client,
        model,
        ctx,
        system_prompt=_S1_SYSTEM_PROMPT,
        user_prompt=user,
        allowed_tools=tool_schemas_for_step("s1"),
        max_tokens=4096,
        max_tool_calls=120,  # 2 tools × up to 50 passages + slack
        thinking=False,
        call_type="tool",
    )


async def _run_s2(
    client: Any,
    model: str,
    db: PipelineDB,
    *,
    nli_server_url: str,
    scope_pids: list[str] | None = None,
) -> ReactResult:
    ctx = ToolContext(
        db=db,
        step_id="s2",
        iteration=db.answer.iteration_count,
        nli_server_url=nli_server_url,
    )
    user = _format_user_prompt(
        db,
        step_id="s2",
        iteration=db.answer.iteration_count,
        scope_pids=scope_pids,
    )
    return await run_step_react(
        client,
        model,
        ctx,
        system_prompt=_S2_SYSTEM_PROMPT,
        user_prompt=user,
        allowed_tools=tool_schemas_for_step("s2"),
        max_tokens=2048,
        max_tool_calls=80,  # 2 tools × up to 20 accepted pids × ~2 rounds
        thinking=False,
        call_type="tool",
    )


async def _run_s3(
    client: Any,
    model: str,
    db: PipelineDB,
    *,
    nli_server_url: str,
) -> ReactResult:
    ctx = ToolContext(
        db=db,
        step_id="s3",
        iteration=db.answer.iteration_count,
        nli_server_url=nli_server_url,
    )
    user = _format_user_prompt(
        db,
        step_id="s3",
        iteration=db.answer.iteration_count,
    )
    return await run_step_react(
        client,
        model,
        ctx,
        system_prompt=_S3_SYSTEM_PROMPT,
        user_prompt=user,
        allowed_tools=tool_schemas_for_step("s3"),
        max_tokens=4096,
        max_tool_calls=160,  # 4 tools × up to 40 nuggets
        thinking=False,
        call_type="tool",
    )


async def _run_s4(
    client: Any,
    model: str,
    db: PipelineDB,
    *,
    nli_server_url: str,
) -> None:
    """S4: single REASON-only call; parse the VERDICT block; the controller
    writes `supportability`/`missing`/`rationale` directly.

    A split-mode variant would add an explicit set_supportability tool
    call; writing the DB from the controller is equivalent and avoids a
    round-trip.
    """
    user = _format_user_prompt(
        db,
        step_id="s4_reason",
        iteration=db.answer.iteration_count,
    )
    text, stop, _ = await run_reason_call(
        client,
        model,
        system_prompt=_S4A_REASON_PROMPT,
        user_prompt=user,
        max_tokens=12288,
        thinking=False,
        call_type="reason",
    )
    parsed = parse_s4_reason(text)
    if parsed is None:
        # Second truncation or parse failure → unresolvable.
        db.answer.supportability = "unresolvable"
        db.answer.missing = ""
        db.answer.rationale = "S4a parse failed; supportability=unresolvable"
    else:
        verdict, missing, rationale = parsed
        db.answer.supportability = verdict  # type: ignore[assignment]
        db.answer.missing = missing
        db.answer.rationale = rationale

    # Record the S4 REASON event for replay
    db.events.append(
        TransitionEvent(
            step_id="s4_reason",
            iteration=db.answer.iteration_count,
            actor="llm_tool",
            to_status=f"supportability={db.answer.supportability}",
            reason=f"S4a stop={stop}",
        )
    )


async def _run_s5(
    client: Any,
    model: str,
    db: PipelineDB,
    *,
    nli_server_url: str,
) -> None:
    """S5: split mode for multi_hop/comparison (thinking on for REASON);
    single-call otherwise. Controller writes reasoning_trace from TAGS+NOTES.
    """
    qtype = (db.question_type or "simple").lower()
    split_mode = qtype in {"multi_hop", "comparison"}

    user = _format_user_prompt(
        db,
        step_id="s5_reason",
        iteration=db.answer.iteration_count,
    )
    text, stop, _ = await run_reason_call(
        client,
        model,
        system_prompt=_S5A_REASON_PROMPT,
        user_prompt=user,
        max_tokens=12288 if split_mode else 2048,
        thinking=split_mode,
        call_type="reason",
    )
    parsed = parse_s5_reason(text)
    if parsed is None:
        # Keep existing draft (confirm-without-rewrite fallback).
        draft = db.answer.draft_answer
        tags = ""
        notes = "S5a parse failed; draft confirmed"
    else:
        draft, tags, notes = parsed

    db.answer.draft_answer = draft
    # Tags enter reasoning_trace so the weakens/elimination terms of the
    # confidence formula apply.
    db.answer.reasoning_trace = (
        (db.answer.reasoning_trace or "") + "\n" + tags + "\n" + notes
    ).strip()
    db.events.append(
        TransitionEvent(
            step_id="s5_reason",
            iteration=db.answer.iteration_count,
            actor="llm_tool",
            to_status="draft_updated",
            reason=f"S5a stop={stop}",
        )
    )

    db.answer.confidence = compute_confidence(db)


async def run_v2_pipeline(
    client: Any,
    model: str,
    db: PipelineDB,
    *,
    nli_server_url: str = "http://127.0.0.1:18500",
    max_iter: int | None = None,
) -> dict[str, Any]:
    """Execute the 6-step loop. Mutates `db` in place.

    Returns a summary dict (exit_reason, iterations, step_wall_ms).
    """
    max_iter = max_iter or db.max_iter
    wall_start = time.time()
    step_wall_ms: dict[str, float] = {}
    # Snapshot accepted/counter pids per iteration so S2 can scope itself to
    # "newly accepted" on loop iterations.
    accepted_pids_prev: set[str] = set()

    def _tick(step: str, t0: float) -> None:
        step_wall_ms[step] = step_wall_ms.get(step, 0.0) + (time.time() - t0) * 1000

    def _current_accepted_or_counter(db: PipelineDB) -> set[str]:
        return {
            pid
            for pid, p in db.passages.items()
            if pid != REFUSAL_PID and p.status in {"accepted", "counter_evidence"}
        }

    for _ in range(max_iter + 1):
        # S1
        t0 = time.time()
        await _run_s1(
            client,
            model,
            db,
            nli_server_url=nli_server_url,
            missing_hint=db.answer.missing,
        )
        _tick("s1", t0)

        adv = advance_after_s1(db)
        if adv.startswith("finish:"):
            reason = adv.split(":", 1)[1]
            if reason == "low_confidence_refusal":
                synthesize_refusal(db)
            db.answer.exit_reason = reason  # type: ignore[assignment]
            break
        if adv == "block":
            db.answer.exit_reason = "error"  # type: ignore[assignment]
            break

        # S2
        # On iter>0, scope S2 to newly accepted/counter pids: S2 only
        # extracts nuggets from pids promoted this iteration; pids already
        # nugget-extracted on a prior iteration are untouched. On iter 0,
        # scope is None (extract from everything accepted).
        accepted_now = _current_accepted_or_counter(db)
        s2_scope: list[str] | None = None
        if db.answer.iteration_count > 0:
            delta = accepted_now - accepted_pids_prev
            s2_scope = sorted(delta) if delta else []
        # Skip the S2 LLM call entirely when there's nothing new to extract
        # (empty delta on loopback — saves a round-trip).
        if s2_scope == []:
            _tick("s2", time.time())
        else:
            t0 = time.time()
            await _run_s2(
                client,
                model,
                db,
                nli_server_url=nli_server_url,
                scope_pids=s2_scope,
            )
            _tick("s2", t0)
        accepted_pids_prev = accepted_now
        if advance_after_s2(db) != "s3":
            db.answer.exit_reason = "error"  # type: ignore[assignment]
            break

        # S3
        t0 = time.time()
        await _run_s3(client, model, db, nli_server_url=nli_server_url)
        _tick("s3", t0)
        adv = advance_after_s3(db)
        if adv == "s6":
            # S3-all-bogus shortcut → unresolvable → refusal
            pass  # drop through to S6
        elif adv == "block":
            db.answer.exit_reason = "error"  # type: ignore[assignment]
            break
        else:
            # S4
            t0 = time.time()
            await _run_s4(client, model, db, nli_server_url=nli_server_url)
            _tick("s4", t0)
            adv = advance_after_s4(db)
            if adv == "s1":
                # Loopback. Record prev state, bump iter, re-enter S1.
                db.answer.prev_missing = db.answer.missing
                db.answer.prev_accepted_pid_count = sum(
                    1
                    for pid, p in db.passages.items()
                    if pid != REFUSAL_PID
                    and p.status in {"accepted", "counter_evidence"}
                )
                db.answer.iteration_count += 1
                continue
            if adv.startswith("finish:"):
                reason = adv.split(":", 1)[1]
                if reason == "low_confidence_refusal":
                    synthesize_refusal(db)
                db.answer.exit_reason = reason  # type: ignore[assignment]
                break

            # S5
            t0 = time.time()
            await _run_s5(client, model, db, nli_server_url=nli_server_url)
            _tick("s5", t0)

        # S6
        t0 = time.time()
        action, reason = run_s6(db, wall_clock_sec=time.time() - wall_start)
        _tick("s6", t0)
        if action == "finish":
            if reason == "low_confidence_refusal":
                synthesize_refusal(db)
            db.answer.exit_reason = reason  # type: ignore[assignment]
            break
        # loop_s1
        db.answer.prev_missing = db.answer.missing
        db.answer.prev_accepted_pid_count = sum(
            1
            for pid, p in db.passages.items()
            if pid != REFUSAL_PID and p.status in {"accepted", "counter_evidence"}
        )
        db.answer.iteration_count += 1
    else:
        # Loop exhausted without explicit break → max_iter.
        db.answer.exit_reason = "max_iter"  # type: ignore[assignment]
        if len(effective_supporting_ids(db)) == 0:
            synthesize_refusal(db)

    return {
        "exit_reason": db.answer.exit_reason or "",
        "iteration_count": db.answer.iteration_count,
        "step_wall_ms": step_wall_ms,
        "total_wall_sec": time.time() - wall_start,
    }


# Feature-flag dispatch: v2 entry point for `reconcile_nuggets_and_passages`


def _use_v2_pipeline() -> bool:
    """Return True iff env var requests the v2 pipeline."""
    return (
        os.environ.get(POSTTRACE_PIPELINE_ENV, POSTTRACE_PIPELINE_LEGACY).lower()
        == POSTTRACE_PIPELINE_V2
    )


def _seed_db_from_passage_list(
    question: str,
    draft_answer: str,
    passages: list[dict],
    question_type: str,
    qid: str = "q",
) -> PipelineDB:
    """Seed a PipelineDB from the worker's already-extracted passage list.

    Cousin of `seed_db_from_trace` — used when the caller already ran
    `extract_passages_from_trace` and does not want to re-walk messages.
    """
    db = PipelineDB(
        question_id=qid,
        question=question,
        question_type=question_type or "",
    )
    _seed_virtual_refusal_passage(db)
    for i, p in enumerate(passages[: db.max_passages]):
        pid = p.get("passage_id") or f"p{i}"
        if pid in db.passages:
            continue
        # The worker's `extract_passages_from_trace` populates `passage_text`
        # (legacy key); accept both spellings.
        ptext = p.get("passage_text") or p.get("text") or p.get("passage") or ""
        db.passages[pid] = Passage(
            passage_id=pid,
            text=ptext,
            title=p.get("title"),
            score=p.get("score"),
            source_tool_call_id=p.get("tool_call_id") or "seed",
            source_msg_idx=p.get("msg_idx", i),
            first_seen_iter=0,
            last_reviewed_iter=0,
            status="unclassified",
        )
    # Parse question-level metadata
    entities, expected_count, counting_note = question_parser(
        question, question_type or ""
    )
    db.answer.draft_answer = draft_answer or ""
    db.answer.question_entities = entities
    db.answer.expected_count = expected_count
    db.answer.counting_note = counting_note
    return db


async def _reconcile_v2(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    passages: list[dict],
    question_type: str,
    mcp_client: Any = None,  # noqa: ARG001 — reserved for a retrieve tool
) -> tuple[list[dict], list[dict], dict]:
    """v2 pipeline: seed DB → run loop → apply adapter → legacy tuple."""
    db = _seed_db_from_passage_list(
        question=question,
        draft_answer=answer,
        passages=passages,
        question_type=question_type,
    )
    nli_url = os.environ.get(
        "NLI_SERVER_URL", f"http://127.0.0.1:{os.environ.get('NLI_PORT', '18500')}"
    )
    summary = await run_v2_pipeline(client, model, db, nli_server_url=nli_url)
    ranked_passages, clean_nuggets, stats = apply_adapter(db)
    stats.update(summary)
    stats["pipeline"] = POSTTRACE_PIPELINE_V2
    return ranked_passages, clean_nuggets, stats


# Exports: feature flag, adapter and driver


__all__ += [  # noqa: F821
    # Feature flag
    "POSTTRACE_PIPELINE_ENV",
    "POSTTRACE_PIPELINE_V2",
    "POSTTRACE_PIPELINE_LEGACY",
    # Adapter
    "adapter_candidate_filter",
    "select_nuggets_mmr",
    "apply_adapter",
    # Driver
    "run_v2_pipeline",
]


_CAST_LIST_RE = re.compile(r";\s*\w[\w\s]+ as \w", re.IGNORECASE)


def _is_structured_data(text: str) -> bool:
    """Detect structured/tabular data (filmography tables, cast lists).

    Two detection methods:
    1. Dense semicolons (≥5 AND >3% density) — filmography tables
    2. Cast-list pattern ("Name as Role ;") — catches lower-density cast lists
    """
    if not text:
        return False
    count = text.count(";")
    # Cast-list pattern: "Name as Role ; Name as Role ;"
    if count >= 3 and _CAST_LIST_RE.search(text):
        return True
    # Dense semicolon table
    if count < 5:
        return False
    density = count / max(len(text), 1) * 100
    return density > 3.0


# Question-type classification

_CLASSIFY_PROMPT = """\
Classify this question into exactly one of 6 types. Use both the question \
and the expected answer to decide.

## Types

1. **simple** — Direct single-fact lookup about one entity and one property.
   Example: Q: "What is Veronica Lang's nationality?" → A: "Australian"

2. **aggregation** — Asks for a LIST of items matching a criterion. The \
answer is a counted list: "N: item1, item2, ...".
   Example: Q: "What films did Ronald Mann direct?" → A: "6: Imagine the Sound, Comic Book Confidential, ..."

3. **comparison** — Names TWO entities, asks which has more/less/earlier/later \
of a property. Answer gives values for both: "entity1: value1, entity2: value2".
   Example: Q: "Which was released first, Imagine the Sound or Comic Book Confidential?" → A: "Imagine the Sound: 1981, Comic Book Confidential: 1988"

4. **simple_w_condition** — Like simple, but includes a qualifying clause \
(budget, runtime, year, network) that narrows which entity is meant.
   Example: Q: "Who reviewed the 86-minute 2015 film The Lazarus Effect?" → A: "Dowd"

5. **multi_hop** — Requires finding an intermediate entity first. Chains \
two relationships.
   Example: Q: "Which Pope was portrayed by the actor who starred in Guilty Conscience?" → A: "pope benedict xvi"

6. **post_processing** — Asks about dates, years, or ratings for editions \
of the same work.
   Example: Q: "What are the release dates for the 1997 film Lolita?" → A: "1999"

## Priority Rules

- Answer starts with "N: " (digit + colon + space) → **aggregation**
- Answer has "entity: value, entity: value" structure → **comparison**
- Question asks about dates/years/ratings of editions → **post_processing**
- Question has a qualifying clause with a specific number/date narrowing \
the target → **simple_w_condition**
- Question requires finding entity A to look up entity B → **multi_hop**
- None of the above → **simple**

Question: {question}
Answer: {answer}

Output ONLY: {{"question_type": "<type>"}}"""


async def _classify_question_type(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    caller_type: str = "",
) -> tuple[str, dict]:
    """Classify question type via LLM (1 call) or use caller-provided type.

    Args:
        client: LLM client.
        model: Model name.
        question: Question text.
        answer: Answer text.
        caller_type: Type from caller (e.g., bundle metadata). If non-empty, used directly.

    Returns:
        (question_type, stats)
    """
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "source": "caller",
        "caller_type": caller_type,
        "classified_type": "",
    }

    # If caller provides a valid type, use it
    if caller_type and caller_type in _VALID_TYPES:
        stats["classified_type"] = caller_type
        return caller_type, stats

    # Pre-check: counting-question heuristic.
    # Counting questions phrased as "how many X", "list all Y", etc. are
    # frequently misclassified as simple by the LLM when the answer is a
    # bare digit (answer "3" looks like a simple scalar). Run the heuristic
    # first; if it decisively says aggregation, skip the LLM.
    heuristic_pre = _infer_question_type(question)
    if heuristic_pre == "aggregation":
        stats["source"] = "heuristic"
        stats["classified_type"] = "aggregation"
        log.info("Pre-check classified '%s' as aggregation", question[:50])
        return "aggregation", stats

    # LLM classification
    try:
        prompt = _CLASSIFY_PROMPT.format(question=question, answer=answer)
        content = await _llm_call(client, model, prompt)
        stats["llm_calls"] = 1
        result = _parse_json(content)
        if isinstance(result, dict):
            qtype = result.get("question_type", "").strip().lower()
            if qtype in _VALID_TYPES:
                stats["source"] = "llm"
                stats["classified_type"] = qtype
                log.info("Classified %s as '%s'", question[:50], qtype)
                return qtype, stats
            log.warning("LLM returned invalid type '%s', falling back", qtype)
    except Exception:
        log.exception("Question type classification failed, falling back")

    # Fallback to heuristic
    fallback = _infer_question_type(question)
    stats["source"] = "fallback"
    stats["classified_type"] = fallback
    log.info("Classification fallback: '%s' for %s", fallback, question[:50])
    return fallback, stats


# Prompts

_ANSWER_REFORMAT_PROMPT = """\
You are a QA answer formatter for the R2C2 competition.

The answer below has formatting issues. Rewrite it as a single plain-text \
sentence or short phrase. No markdown, no lists, no newlines.

Question: {question}
Raw answer: {answer}

Output ONLY the reformulated answer (nothing else)."""

_PASSAGE_SELECT_PROMPT = """\
You are a passage relevance judge for the R2C2 QA competition \
(movie/entertainment domain).

Question: {question}
Answer: {answer}

Below are passages retrieved during the agent's search. Select the ones \
that are relevant to answering the question, and for each, provide an \
extract call to pull the 1-2 most important sentences.

PASSAGE QUALITY RULES:
- PREFER passages with narrative prose (biographical text, plot summaries, descriptions).
- REJECT passages that are raw structured data: filmography tables, cast lists, \
semicolon-separated rows (e.g. "Year: 2002 ; Title: 8 Mile ; Role: ..."). \
These cannot produce useful nuggets.
- If a passage contains both prose and a table, keep it but anchor your extract \
call on the prose portion.

Tool: extract(pattern, passage_id, pattern_to, context_sentences)
  - pattern: start anchor — word/phrase to find (case-insensitive). \
STRATEGY: always try the answer text (or key words from it) as the first \
pattern. If the answer phrase appears in the passage, anchor on it. \
Fall back to topic keywords only if the answer phrase is absent.
  - passage_id: which passage to extract from
  - pattern_to: (optional) end anchor — extract from pattern's sentence \
through pattern_to's sentence
  - context_sentences: 0-2 sentences of padding around the match. Default 1.

NOTE: Multiple extract calls on the same passage will be merged \
(deduplicated). You can safely make multiple calls to capture different \
facts from the same passage.

Passages:
{passages}

Output ONLY a JSON array, one entry per passage:
[
  {{"passage_id": "Q511554__00007", "action": "keep", "calls": [
    {{"tool": "extract", "args": {{"pattern": "Manchurian", \
"passage_id": "Q511554__00007", "context_sentences": 1}}}}
  ]}},
  {{"passage_id": "Q18407657__00054", "action": "drop"}}
]

IMPORTANT: Every passage with action "keep" MUST include at least one \
extract call. Keep without calls will be discarded."""

_NUGGET_GEN_PROMPT = """\
You are a nugget extraction system for the R2C2 QA competition \
(movie/entertainment domain).

Question: {question}
Answer: {answer}

Extract atomic facts from the passage below that support or verify the answer. \
Each nugget must:
- State one atomic fact as a single sentence
- Be grounded in the passage text (directly entailed, not inferred). \
Every fact that the passage EXPLICITLY names, dates, or describes is fair \
to extract. Do NOT add names, numbers, titles, or dates that the passage \
never mentions.
- Extract ALL distinct facts that meet the grounding bar. Do NOT stop early \
because the question seems simple — if a passage explicitly names 4 grounded \
facts and the cap is 5, emit 4.
- Be relevant to answering the question
- Replace semicolons with commas
- NOT be the entity name or article title alone (e.g., if the passage is \
titled "Anthony Mackie", do not emit "Anthony Mackie is a person.")
- NOT be meta-commentary (e.g., "The passage states..." — just state the fact)
{enum_hint}
Passage [{passage_id}]:
{passage_text}

Extract up to {max_nuggets} nuggets. Output ONLY a JSON array:
[{{"nugget_text": "..."}}]"""

_ENTAILMENT_PROMPT = """\
You are an entailment checker. Does the passage below entail the claim?

Passage:
{passage_text}

Claim: "{nugget_text}"

A claim is entailed ONLY if:
- The passage explicitly states it (verbatim or close paraphrase), OR
- It is the direct, unavoidable logical consequence of the passage text alone — \
no external knowledge required.

A claim is NOT entailed if:
- It adds specific names, quotes, or numbers not present in the passage.
- It requires world knowledge beyond what the passage provides.
- The passage only mentions a topic generally, without stating the specific claim.

Example: Passage says "he appeared in several MCU films." Claim: \
"He played Captain America." → NOT entailed (specific role not in passage).

Output ONLY a JSON object:
{{"entailed": true}} or {{"entailed": false}}"""

# Helpers


def _fmt_passages(passages: list[dict], max_chars: int = 400) -> str:
    """Format passages for prompt inclusion.

    Uses passage_text_with_title when available (after extraction), otherwise
    falls back to passage_text with title/section context in parentheses.
    """
    lines = []
    for p in passages:
        titled = p.get("passage_text_with_title", "")
        if titled:
            lines.append(f"- {p['passage_id']}: {titled[:max_chars]}")
        else:
            title = p.get("title", "")
            section = p.get("section", "")
            context = " > ".join(s for s in [title, section] if s)
            text = p.get("passage_text", "")[:max_chars]
            if context:
                lines.append(f"- {p['passage_id']} ({context}): {text}")
            else:
                lines.append(f"- {p['passage_id']}: {text}")
    return "\n".join(lines)


def _parse_json(text: str) -> Any:
    """Extract JSON from LLM response, handling code blocks and trailing text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```[\s\S]*$", "", text)
    # Find the outermost JSON structure (array or object)
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching close bracket, counting nesting
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    return json.loads(text)


def _clean_nugget_text(text: str) -> str:
    """Strip markdown and sanitize nugget text."""
    for marker in ("**", "__", "*", "_", "`"):
        text = text.replace(marker, "")
    text = text.strip().replace(";", ",").replace("\n", " ")
    return text


async def _llm_call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    enable_thinking: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Single LLM call. Thinking disabled by default.

    enable_thinking=True is meant for the sequential steps (5, 7b, 9) of
    long unattended runs. Do not enable it on the parallel fan-outs
    (2, 3, 6, 10): latency explodes.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _merge_matches(data: dict) -> str:
    """Merge match entries from an extract/grep response into a single text.

    Deduplicates overlapping sentences (from adjacent context windows).
    """
    matches = data.get("matches", [])
    if not matches:
        return ""
    seen: set[str] = set()
    parts: list[str] = []
    for m in matches:
        cb = m.get("context_before", "")
        if isinstance(cb, list):
            cb = " ".join(cb)
        sentence = m.get("sentence", "")
        ca = m.get("context_after", "")
        if isinstance(ca, list):
            ca = " ".join(ca)
        for s in [cb, sentence, ca]:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                parts.append(s)
    return " ".join(parts)


VALID_REFINE_TOOLS = {"extract", "grep"}


async def _execute_extract_call(
    mcp_client: Any,
    tool: str,
    args: dict,
) -> dict | None:
    """Execute a single MCP extract call.

    Executes twice: once without include_title (raw text for PR submission)
    and once with include_title=True (titled text for internal reasoning).

    Returns a passage dict with passage_id, passage_text,
    passage_text_with_title, title, section, doc_id — or None on failure.
    """
    if tool in ("grep", "grep_document"):
        tool = "extract"
        if "context_before" in args or "context_after" in args:
            ctx = max(args.pop("context_before", 0), args.pop("context_after", 0))
            args.setdefault("context_sentences", ctx)
    if tool not in VALID_REFINE_TOOLS:
        log.warning("EXTRACT: invalid tool %r, skipping", tool)
        return None

    def _parse_result(result: Any) -> dict:
        if isinstance(result, dict):
            return result
        raw = result.content[0].text if hasattr(result, "content") else str(result)
        return json.loads(raw)

    try:
        # Raw text (for PR submission)
        args_raw = {k: v for k, v in args.items() if k != "include_title"}
        result_raw = _parse_result(await mcp_client.call_tool(tool, args_raw))

        text_raw = _merge_matches(result_raw)
        if not text_raw:
            return None

        orig_pid = args.get("passage_id", "")
        if not orig_pid:
            matches = result_raw.get("matches", [])
            if matches:
                orig_pid = matches[0].get("passage_id", "")

        # Titled text (for internal reasoning)
        args_titled = {**args_raw, "include_title": True}
        result_titled = _parse_result(await mcp_client.call_tool(tool, args_titled))
        text_titled = _merge_matches(result_titled) or text_raw

        return {
            "passage_id": orig_pid,
            "passage_text": text_raw,
            "passage_text_with_title": text_titled,
            "title": result_raw.get("title", ""),
            "section": result_raw.get("section", ""),
            "doc_id": result_raw.get("source_id", ""),
        }
    except Exception:
        log.exception("EXTRACT: tool call %s(%s) failed", tool, args)
        return None


# Step 0: Answer format verification

# Patterns that indicate bad answer formatting
_BAD_ANSWER_PATTERNS = [
    re.compile(r"\n"),  # newlines
    re.compile(r"^[-*+]\s", re.M),  # unordered list items (- item, * item)
    re.compile(r"^\d+\.\s", re.M),  # ordered list items (1. item)
    re.compile(r"[#*]{2,}"),  # markdown headers/bold
    re.compile(r"```"),  # code blocks
    re.compile(r"\|.*\|"),  # tables
]


async def _step_0_answer_verify(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
) -> tuple[str, dict]:
    """Step 0: Verify answer format. Reformulate if needed.

    Checks for newlines, lists, markdown, >150 chars. If any flag triggers,
    uses 1 LLM call to reformulate. Otherwise passes through at 0 cost.

    Returns:
        (clean_answer, stats)
    """
    stats: dict[str, Any] = {"llm_calls": 0, "reformulated": False}

    if not answer:
        return answer, stats

    # Check for formatting issues
    needs_reformat = len(answer) > ANSWER_MAX_CHARS
    if not needs_reformat:
        for pat in _BAD_ANSWER_PATTERNS:
            if pat.search(answer):
                needs_reformat = True
                break

    if not needs_reformat:
        return answer, stats

    log.info("Step 0: answer needs reformulation (len=%d)", len(answer))

    try:
        prompt = _ANSWER_REFORMAT_PROMPT.format(question=question, answer=answer)
        content = await _llm_call(client, model, prompt)
        stats["llm_calls"] = 1
        stats["reformulated"] = True

        # Basic sanity: non-empty, shorter, no markdown
        clean = content.strip().strip('"').strip("'")
        if clean and len(clean) < len(answer) * 2:
            log.info("Step 0: reformulated %d→%d chars", len(answer), len(clean))
            return clean, stats
        log.warning("Step 0: reformulation suspicious, keeping original")
        return answer, stats
    except Exception:
        log.exception("Step 0: reformulation failed, keeping original")
        return answer, stats


# Step 1: Relevant passage extraction


async def _step_1_passage_extract(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    passages: list[dict],
    mcp_client: Any,
) -> tuple[list[dict], dict]:
    """Step 1: Select relevant passages and extract focused text.

    Single LLM call to judge relevance + specify extract calls, then
    parallel MCP extract execution.

    Returns:
        (extracted_passages, stats)
    """
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "mcp_calls": 0,
        "input_passages": len(passages),
        "output_passages": 0,
    }

    if not passages or mcp_client is None:
        stats["output_passages"] = len(passages)
        return passages, stats

    # Cap to top-50 by retrieval score to avoid LLM attention degradation
    _MAX_STEP1_INPUT = 50
    if len(passages) > _MAX_STEP1_INPUT:
        # Sort by score descending (None scores go last)
        scored = sorted(
            passages,
            key=lambda p: float(p.get("score", -1) or -1),
            reverse=True,
        )
        passages = scored[:_MAX_STEP1_INPUT]
        log.info("Step 1: capped %d→%d passages by score", len(scored), len(passages))

    # Batch into chunks of 25 to avoid lost-in-the-middle effect
    _BATCH_SIZE = 25
    all_recipes: list[dict] = []
    if len(passages) > _BATCH_SIZE:
        for batch_start in range(0, len(passages), _BATCH_SIZE):
            batch = passages[batch_start : batch_start + _BATCH_SIZE]
            prompt = _PASSAGE_SELECT_PROMPT.format(
                question=question,
                answer=answer,
                passages=_fmt_passages(batch, max_chars=800),
            )
            content = await _llm_call(client, model, prompt)
            stats["llm_calls"] += 1
            try:
                batch_recipes = _parse_json(content)
                if isinstance(batch_recipes, list):
                    all_recipes.extend(batch_recipes)
            except Exception:
                log.exception("Step 1: failed to parse batch response")
        recipes = all_recipes
    else:
        prompt = _PASSAGE_SELECT_PROMPT.format(
            question=question,
            answer=answer,
            passages=_fmt_passages(passages, max_chars=800),
        )
        content = await _llm_call(client, model, prompt)
        stats["llm_calls"] = 1
        try:
            recipes = _parse_json(content)
        except Exception:
            log.exception("Step 1: failed to parse LLM response")
            stats["output_passages"] = len(passages)
            return passages, stats

    try:
        if not isinstance(recipes, list):
            log.warning("Step 1: expected list, got %s", type(recipes).__name__)
            stats["output_passages"] = len(passages)
            return passages, stats
    except Exception:
        log.exception("Step 1: failed to parse LLM response")
        stats["output_passages"] = len(passages)
        return passages, stats

    # Execute extract calls in parallel
    # Build lookup for structured-data detection on original passages
    orig_by_pid = {p["passage_id"]: p for p in passages}

    async def _process_recipe(recipe: dict) -> tuple[list[dict], int]:
        pid = recipe.get("passage_id", "")
        action = recipe.get("action", "drop")
        if action == "drop":
            return [], 0

        # Post-hoc filter: reject structured data even if LLM said "keep"
        # — unless the answer text appears verbatim in the passage, in which
        # case the structured block is likely the direct evidence (e.g. a
        # filmography/cast-list row containing the answer entity).
        orig = orig_by_pid.get(pid, {})
        orig_text = orig.get("passage_text", "") or orig.get("text", "")
        if _is_structured_data(orig_text):
            if answer and answer.strip() and answer.lower() in orig_text.lower():
                log.info(
                    "Step 1: passage %s is structured but contains answer — kept",
                    pid,
                )
            else:
                log.info("Step 1: passage %s rejected (structured data)", pid)
                return [], 0

        calls = recipe.get("calls", [])
        if not calls:
            log.warning("Step 1: passage %s keep with no calls — dropped", pid)
            return [], 0

        results = []
        call_count = 0
        for call in calls:
            tool = call.get("tool", "extract")
            args = call.get("args", {})
            result = await _execute_extract_call(mcp_client, tool, args)
            call_count += 1
            if result and result.get("passage_text"):
                result["passage_id"] = pid  # preserve recipe's pid
                results.append(result)
        return results, call_count

    all_results = await asyncio.gather(*[_process_recipe(r) for r in recipes])

    # Tally MCP calls after gather (avoids concurrent mutation)
    stats["mcp_calls"] = sum(count for _, count in all_results)

    # Record per-passage decisions for UI
    kept_pids = set()
    for result_list, _ in all_results:
        for r in result_list:
            kept_pids.add(r["passage_id"])
    details = []
    for recipe in recipes:
        pid = recipe.get("passage_id", "")
        action = recipe.get("action", "drop")
        details.append({"passage_id": pid, "action": action, "kept": pid in kept_pids})
    stats["details"] = details

    # Flatten and merge by passage_id
    extracted: list[dict] = []
    for result_list, _ in all_results:
        extracted.extend(result_list)

    if not extracted:
        log.info("Step 1: all passages dropped, keeping originals")
        stats["output_passages"] = len(passages)
        return passages, stats

    # Merge passages with the same passage_id (multiple extract calls
    # on the same passage produce separate results — deduplicate sentences)
    merged_by_pid: dict[str, dict] = {}
    for p in extracted:
        pid = p["passage_id"]
        if pid in merged_by_pid:
            existing = merged_by_pid[pid]
            for field in ("passage_text", "passage_text_with_title"):
                existing_sents = set(existing.get(field, "").split(". "))
                new_sents = p.get(field, "").split(". ")
                for s in new_sents:
                    if s.strip() and s not in existing_sents:
                        existing[field] = existing.get(field, "") + " " + s
                        existing_sents.add(s)
        else:
            merged_by_pid[pid] = dict(p)
    extracted = list(merged_by_pid.values())

    stats["output_passages"] = len(extracted)
    log.info(
        "Step 1: %d→%d passages (%d MCP calls)",
        len(passages),
        len(extracted),
        stats["mcp_calls"],
    )
    return extracted, stats


# Step 2: Parallel nugget generation


async def _step_2_nugget_gen(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    passages: list[dict],
    question_type: str = "",
) -> tuple[list[dict], dict]:
    """Step 2: Generate nuggets from passages in parallel.

    1 LLM call per passage. Max nuggets per passage depends on question type.
    Global cap of 15 nuggets total.

    Returns:
        (nuggets, stats) where each nugget has passage_id + nugget_text.
    """
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "nuggets_generated": 0,
    }

    if not passages:
        return [], stats

    # Blank type falls back to "unknown" (hits _NUGGET_CAP_DEFAULT=4) rather
    # than "simple" (which caps at 3) — avoids silently under-extracting on
    # unclassified questions.
    effective_type = question_type or "unknown"
    max_per_passage = _NUGGET_CAP_BY_TYPE.get(effective_type, _NUGGET_CAP_DEFAULT)

    # Enumeration hint — only injected for counting/chain-reasoning types
    # where each distinct item named in the passage is a separate nugget.
    if effective_type in ("aggregation", "multi_hop", "post_processing"):
        enum_hint = (
            "\nThis is a counting/enumeration/multi-step question. Each "
            "distinct item (film, person, event, date) that the passage "
            "explicitly names and relates to the subject is a SEPARATE nugget. "
            "Prefer more nuggets over fewer.\n"
        )
    else:
        enum_hint = ""

    # Skip step 2 entirely for corpus-absent refusal answers — the model
    # will rationally produce nothing; the calls are pure waste.
    _REFUSAL_PATTERNS = (
        "information not available",
        "not available in the corpus",
        "not found in the corpus",
    )
    if any(p in answer.lower() for p in _REFUSAL_PATTERNS):
        stats["nuggets_generated"] = 0
        stats["llm_calls"] = 0
        stats["skipped_refusal"] = True
        log.info("Step 2: skipped — corpus-absent refusal answer")
        return [], stats

    async def _gen_for_passage(p: dict) -> list[dict]:
        pid = p["passage_id"]
        # Use titled text for generation (more context)
        text = p.get("passage_text_with_title", "") or p.get("passage_text", "")
        if not text:
            return []

        prompt = _NUGGET_GEN_PROMPT.format(
            question=question,
            answer=answer,
            passage_id=pid,
            passage_text=text[:_ENTAILMENT_CHAR_LIMIT],
            max_nuggets=max_per_passage,
            enum_hint=enum_hint,
        )
        try:
            content = await _llm_call(client, model, prompt)
            result = _parse_json(content)
            if not isinstance(result, list):
                return []
            nuggets = []
            for n in result[:max_per_passage]:
                ntext = _clean_nugget_text(n.get("nugget_text", ""))
                if ntext:
                    nuggets.append({"passage_id": pid, "nugget_text": ntext})
            return nuggets
        except Exception:
            log.exception("Step 2: nugget gen failed for %s", pid)
            return []

    all_results = await asyncio.gather(*[_gen_for_passage(p) for p in passages])
    stats["llm_calls"] = len(passages)

    # Flatten and apply global cap
    all_nuggets: list[dict] = []
    for result_list in all_results:
        all_nuggets.extend(result_list)

    if len(all_nuggets) > _NUGGET_GLOBAL_CAP:
        log.info(
            "Step 2: global cap %d→%d nuggets",
            len(all_nuggets),
            _NUGGET_GLOBAL_CAP,
        )
        all_nuggets = all_nuggets[:_NUGGET_GLOBAL_CAP]

    stats["nuggets_generated"] = len(all_nuggets)
    stats["details"] = [
        {"passage_id": n["passage_id"], "nugget_text": n["nugget_text"]}
        for n in all_nuggets
    ]
    log.info(
        "Step 2: generated %d nuggets from %d passages",
        len(all_nuggets),
        len(passages),
    )
    return all_nuggets, stats


# Step 3: Parallel bogus verification


async def _step_3_verify(
    client: AsyncOpenAI,
    model: str,
    nuggets: list[dict],
    passages: list[dict],
) -> tuple[list[dict], dict]:
    """Step 3: Verify each nugget is entailed by its passage.

    1 LLM call per nugget, all parallel. Minimal context: only passage text
    + nugget text (no question, no answer). Fail-closed: on error, drop.

    Returns:
        (verified_nuggets, stats)
    """
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "input_nuggets": len(nuggets),
        "verified": 0,
        "dropped": 0,
    }

    if not nuggets:
        return [], stats

    passage_by_pid: dict[str, dict] = {p["passage_id"]: p for p in passages}

    async def _verify_one(nugget: dict) -> dict | None:
        pid = nugget["passage_id"]
        p = passage_by_pid.get(pid)
        if not p:
            log.warning("Step 3: no passage for pid=%s, dropping", pid)
            return None

        # Use titled text for verification
        text = p.get("passage_text_with_title", "") or p.get("passage_text", "")
        if not text:
            return None

        # Cheap pre-check: if the nugget mentions a proper-noun entity the
        # passage does not contain, skip the LLM call and reject. Catches
        # appositive injection ("Thor's hammer Mjolnir" when passage says
        # only "Mjolnir") without needing an NLI model.
        missing_entities = _missing_proper_nouns(
            text[:_ENTAILMENT_CHAR_LIMIT], nugget["nugget_text"]
        )
        if missing_entities:
            log.info(
                "Step 3: nugget '%s' rejected — missing entities %s",
                nugget["nugget_text"][:60],
                missing_entities,
            )
            return None

        prompt = _ENTAILMENT_PROMPT.format(
            passage_text=text[:_ENTAILMENT_CHAR_LIMIT],
            nugget_text=nugget["nugget_text"],
        )
        try:
            content = await _llm_call(client, model, prompt)
            result = _parse_json(content)
            if isinstance(result, dict) and result.get("entailed", False):
                return nugget
            return None  # not entailed → drop
        except Exception:
            log.exception(
                "Step 3: verification failed for nugget '%s', dropping",
                nugget["nugget_text"][:50],
            )
            return None  # fail-closed

    results = await asyncio.gather(*[_verify_one(n) for n in nuggets])
    stats["llm_calls"] = len(nuggets)

    verified = [r for r in results if r is not None]
    verified_texts = {r["nugget_text"] for r in verified}
    stats["verified"] = len(verified)
    stats["dropped"] = len(nuggets) - len(verified)
    stats["details"] = [
        {
            "passage_id": n["passage_id"],
            "nugget_text": n["nugget_text"],
            "entailed": n["nugget_text"] in verified_texts,
        }
        for n in nuggets
    ]

    log.info(
        "Step 3: %d/%d nuggets verified (dropped %d bogus)",
        len(verified),
        len(nuggets),
        stats["dropped"],
    )
    return verified, stats


# Step 5: Deduplication with multi-passage support

_DEDUP_PROMPT = """\
You are a nugget deduplication system for the R2C2 QA competition \
(movie/entertainment domain).

Question: {question}
Question type: {question_type}

Below are nuggets, each grounded in a specific passage. Cluster nuggets \
that express the same fact. For each cluster:
- Merge into ONE nugget with the most precise wording that is entailed \
by ALL passages in the cluster (conservative intersection).
- List ALL passage_ids that supported any nugget in the cluster.
- Set primary_passage_id to the passage that supports the FEWEST other \
nuggets (to spread citations).

Rules:
- If two nuggets say the same thing differently, merge them.
- If a general nugget is covered by a more specific one, keep only the specific.
- Do NOT merge nuggets that contain genuinely different information.
- Do NOT invent new facts — the merged text must be entailed by each passage.
- CRITICAL — counting/enumeration: If the question asks "how many", "which ones", \
"name all", or the type is "aggregation", "post_processing", or "comparison", then nuggets about DIFFERENT items in \
the count (different movies, characters, episodes, etc.) MUST remain separate. \
Never collapse distinct enumeration entries into one generic nugget.

Nuggets:
{nuggets}

Nugget-to-passage count (for primary selection):
{passage_nugget_counts}

Output ONLY a JSON array:
[{{"passage_ids": ["pid1", "pid2"], "primary_passage_id": "pid1", \
"nugget_text": "merged text", "source_indices": [0, 3]}}]

source_indices = indices of the original nuggets that were merged."""

_COMPLETENESS_PROMPT = """\
You are evaluating answer completeness for the R2C2 QA competition \
(movie/entertainment domain).

Question: {question}
Answer: {answer}
Question type: {question_type}

Supporting nuggets:
{nuggets}

Can the answer "{answer}" be fully justified using ONLY the nuggets above? \
Every key claim in the answer must be supported by at least one nugget.

{counting_note}


Output ONLY a JSON object:
- If complete: {{"complete": true}}
- If incomplete: {{"complete": false, "severity": "critical" or "minor", \
"missing": "brief description of what evidence is missing"}}"""

_GAP_FILL_PROMPT = """\
You are a nugget extraction system for the R2C2 QA competition \
(movie/entertainment domain).

Question: {question}
Answer: {answer}

Existing nuggets:
{existing_nuggets}

Missing evidence: {missing}

Extract up to 5 NEW nuggets from the passages below to fill the gap. \
Each nugget must:
- State one atomic fact as a single sentence
- Be grounded in a specific passage_id
- Add information NOT already covered by existing nuggets
- Replace semicolons with commas

Passages:
{passages}

Output ONLY a JSON array:
[{{"passage_id": "...", "nugget_text": "..."}}]"""

_NUGGET_SELECT_PROMPT = """\
You are a nugget selection system for the R2C2 QA competition.

Question: {question}
Answer: {answer}
Question type: {question_type}

Select the {target} most important nuggets from the candidates below. \
Prioritize:
1. {enumeration_rule}
2. Nuggets that form the core evidence chain from question to answer
3. Nuggets with diverse information (not near-duplicates)
4. Precision matters: fewer confident nuggets > many uncertain ones

Candidates:
{nuggets}

Output ONLY a JSON array of integer indices of the selected nuggets:
[0, 3, 5, ...]"""

_ENUM_RULES: dict[str, str] = {
    "aggregation": "ENUMERATION OVER SUMMARY — each individual item must appear in its own nugget (a count alone is insufficient)",
    "post_processing": "ENUMERATION OVER SUMMARY — each date/year/rating must appear in its own nugget",
    "multi_hop": "CHAIN COMPLETENESS — keep every link in the reasoning chain, even if some links look redundant",
    "comparison": "BALANCED COVERAGE — keep nuggets about BOTH compared entities, not just one",
}
_ENUM_DEFAULT = (
    "SUMMARY OVER ENUMERATION — prefer 1 summary nugget over many individual ones"
)

_RELEVANCE_PROMPT = """\
You are a relevance judge for the R2C2 QA competition.

Question: {question}
Answer: {answer}

Passage the nugget came from:
{passage_text}

Does this nugget help verify or support the answer?

Nugget: "{nugget_text}"

A nugget is RELEVANT if it:
- Directly names or identifies the answer entity
- Provides a fact that logically connects two entities in the question \
(e.g., links a person to a film mentioned in the question)
- Confirms or supports a key claim in the answer
- Establishes the existence or identity of any item that belongs to the \
counted set (for counting questions where the answer is a number)

A nugget is NOT relevant ONLY if it would be equally true and equally \
informative for a completely different question — i.e., it has zero \
specificity to the question or answer domain.

{counting_note}

Output ONLY: {{"relevant": true}} or {{"relevant": false}}"""


def _fmt_nuggets(nuggets: list[dict]) -> str:
    """Format nuggets for prompt inclusion."""
    lines = []
    for i, n in enumerate(nuggets):
        pid = n.get("primary_passage_id", n.get("passage_id", ""))
        lines.append(f"[{i}] ({pid}): {n['nugget_text']}")
    return "\n".join(lines) if lines else "(none)"


def _enrich_nuggets(nuggets: list[dict]) -> list[dict]:
    """Return copies of nuggets with multi-passage fields for downstream steps."""
    enriched = []
    for n in nuggets:
        e = dict(n)
        e.setdefault("passage_ids", [e["passage_id"]])
        e.setdefault("primary_passage_id", e["passage_id"])
        e.setdefault(
            "original_nuggets",
            [{"passage_id": e["passage_id"], "nugget_text": e["nugget_text"]}],
        )
        enriched.append(e)
    return enriched


async def _step_5_dedup(
    client: AsyncOpenAI,
    model: str,
    nuggets: list[dict],
    question: str = "",
    question_type: str = "",
) -> tuple[list[dict], dict]:
    """Step 5: Deduplicate nuggets with multi-passage support.

    Single LLM call. Clusters identical facts, merges conservatively.
    Output nuggets carry passage_ids[], primary_passage_id, original_nuggets.

    Returns:
        (deduped_nuggets, stats)
    """
    effective_type = question_type or "simple"  # pre-resolved by classifier
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "input": len(nuggets),
        "output": 0,
        "effective_type": effective_type,
    }

    if len(nuggets) <= 1:
        nuggets = _enrich_nuggets(nuggets)
        stats["output"] = len(nuggets)
        return nuggets, stats

    # Count how many nuggets each passage supports (for primary selection)
    pid_counts: dict[str, int] = {}
    for n in nuggets:
        pid = n["passage_id"]
        pid_counts[pid] = pid_counts.get(pid, 0) + 1
    counts_str = ", ".join(f"{pid}: {c}" for pid, c in pid_counts.items())

    prompt = _DEDUP_PROMPT.format(
        question=question,
        question_type=effective_type,
        nuggets=_fmt_nuggets(nuggets),
        passage_nugget_counts=counts_str,
    )

    try:
        content = await _llm_call(client, model, prompt)
        stats["llm_calls"] = 1
        result = _parse_json(content)
        if not isinstance(result, list):
            stats["output"] = len(nuggets)
            return nuggets, stats

        valid_pids = {n["passage_id"] for n in nuggets}
        deduped: list[dict] = []
        for item in result:
            ntext = _clean_nugget_text(item.get("nugget_text", ""))
            if not ntext:
                continue
            pids = [p for p in item.get("passage_ids", []) if p in valid_pids]
            if not pids:
                continue
            primary = item.get("primary_passage_id", pids[0])
            if primary not in valid_pids:
                primary = pids[0]

            # Collect originals from source_indices
            source_indices = item.get("source_indices", [])
            originals = []
            for idx in source_indices:
                if 0 <= idx < len(nuggets):
                    originals.append(dict(nuggets[idx]))
            # Validate completeness: if LLM returned partial indices,
            # also include any nuggets sharing the same passage_ids
            if len(originals) < len(pids):
                orig_texts = {o["nugget_text"] for o in originals}
                for n in nuggets:
                    if n["passage_id"] in pids and n["nugget_text"] not in orig_texts:
                        originals.append(dict(n))
                        orig_texts.add(n["nugget_text"])
            if not originals:
                originals = [{"passage_id": primary, "nugget_text": ntext}]

            deduped.append(
                {
                    "passage_id": primary,  # backward compat
                    "passage_ids": pids,
                    "primary_passage_id": primary,
                    "nugget_text": ntext,
                    "original_nuggets": originals,
                }
            )

        if not deduped:
            nuggets = _enrich_nuggets(nuggets)
            stats["output"] = len(nuggets)
            return nuggets, stats

        stats["output"] = len(deduped)
        stats["details"] = [
            {
                "nugget_text": d["nugget_text"],
                "primary_passage_id": d["primary_passage_id"],
                "merged_from": len(d.get("original_nuggets", [])),
                "passage_ids": d.get("passage_ids", []),
            }
            for d in deduped
        ]
        log.info("Step 5: %d→%d nuggets after dedup", len(nuggets), len(deduped))
        return deduped, stats
    except Exception:
        log.exception("Step 5: dedup failed, keeping all")
        nuggets = _enrich_nuggets(nuggets)
        stats["output"] = len(nuggets)
        return nuggets, stats


# Step 6: Re-verification post-dedup


async def _step_6_reverify(
    client: AsyncOpenAI,
    model: str,
    nuggets: list[dict],
    passages: list[dict],
) -> tuple[list[dict], dict]:
    """Step 6: Re-verify merged nuggets against their primary passage.

    Same entailment check as step 3. If fails on primary_passage_id, tries
    next passage in passage_ids. If all fail, reverts to original nugget text.

    Returns:
        (verified_nuggets, stats)
    """
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "verified_direct": 0,
        "verified_fallback_passage": 0,
        "reverted_to_original": 0,
    }

    if not nuggets:
        return [], stats

    passage_by_pid: dict[str, dict] = {p["passage_id"]: p for p in passages}

    async def _verify_merged(nugget: dict) -> tuple[list[dict], dict]:
        """Returns (nugget_list, local_stats) to avoid shared-dict mutation.

        Returns a list because revert may expand a merged nugget back to
        multiple originals.
        """
        local = {
            "llm_calls": 0,
            "verified_direct": 0,
            "verified_fallback_passage": 0,
            "reverted_to_original": 0,
        }
        pids = nugget.get("passage_ids", [nugget["passage_id"]])
        primary = nugget.get("primary_passage_id", nugget["passage_id"])

        # Try primary first, then others, bounded to 3 attempts in total to
        # cap worst-case per-nugget LLM calls.
        ordered_pids = ([primary] + [p for p in pids if p != primary])[:3]

        for pid in ordered_pids:
            p = passage_by_pid.get(pid)
            if not p:
                continue
            text = p.get("passage_text_with_title", "") or p.get("passage_text", "")
            if not text:
                continue

            # Cheap proper-noun pre-check: skips the LLM when the nugget
            # mentions entities absent from the passage.
            missing_entities = _missing_proper_nouns(
                text[:_ENTAILMENT_CHAR_LIMIT], nugget["nugget_text"]
            )
            if missing_entities:
                log.info(
                    "Step 6: pid=%s rejected for '%s' — missing entities %s",
                    pid,
                    nugget["nugget_text"][:60],
                    missing_entities,
                )
                continue

            prompt = _ENTAILMENT_PROMPT.format(
                passage_text=text[:_ENTAILMENT_CHAR_LIMIT],
                nugget_text=nugget["nugget_text"],
            )
            try:
                content = await _llm_call(client, model, prompt)
                local["llm_calls"] += 1
                result = _parse_json(content)
                if isinstance(result, dict) and result.get("entailed", False):
                    if pid == primary:
                        local["verified_direct"] = 1
                    else:
                        local["verified_fallback_passage"] = 1
                        nugget["primary_passage_id"] = pid
                        nugget["passage_id"] = pid
                    return [nugget], local
            except Exception:
                local["llm_calls"] += 1
                continue

        # All passages failed: revert to originals, but dedup by text
        # similarity to collapse near-duplicate pre-merge artifacts while
        # preserving distinct facts (e.g. 3 different films from the same
        # filmography passage). Expanding unconditionally re-introduces
        # exact duplicates, and a pid-only dedup is too aggressive and
        # loses enumerated items.
        originals = nugget.get("original_nuggets", [])
        if originals:
            local["reverted_to_original"] = 1
            # Collapse exact-text + near-dup originals, preserve distinct facts.
            accepted: list[dict] = []
            accepted_texts: list[str] = []
            for orig in originals:
                otext = orig.get("nugget_text", "")
                if not otext:
                    continue
                if _is_near_duplicate(otext, accepted_texts, threshold=0.85):
                    continue
                accepted.append(orig)
                accepted_texts.append(otext)
            log.info(
                "Step 6: reverting merged nugget to %d originals (deduped from %d)",
                len(accepted),
                len(originals),
            )
            reverted = []
            for orig in accepted:
                n = dict(nugget)
                n["nugget_text"] = orig["nugget_text"]
                n["primary_passage_id"] = orig["passage_id"]
                n["passage_id"] = orig["passage_id"]
                n["passage_ids"] = [orig["passage_id"]]
                reverted.append(n)
            return reverted, local

        return [nugget], local  # keep as-is if no originals

    gather_results = await asyncio.gather(*[_verify_merged(n) for n in nuggets])

    # Aggregate stats from each coroutine (no shared mutation)
    results = []
    details = []
    for nugget_list, local in gather_results:
        results.extend(nugget_list)
        for k in stats:
            stats[k] += local.get(k, 0)
        outcome = (
            "direct"
            if local.get("verified_direct")
            else (
                "fallback"
                if local.get("verified_fallback_passage")
                else ("reverted" if local.get("reverted_to_original") else "kept")
            )
        )
        for nugget in nugget_list:
            details.append(
                {
                    "nugget_text": nugget["nugget_text"],
                    "primary_passage_id": nugget.get(
                        "primary_passage_id", nugget.get("passage_id", "")
                    ),
                    "outcome": outcome,
                }
            )
    stats["details"] = details

    log.info(
        "Step 6: %d direct, %d fallback-passage, %d reverted",
        stats["verified_direct"],
        stats["verified_fallback_passage"],
        stats["reverted_to_original"],
    )
    return list(results), stats


# Step 7: Completeness check


_WORD_RE = re.compile(r"[a-z0-9]+")

# Proper-noun-like tokens: length>=4, starting uppercase, mostly letters.
# Deliberately skips 2-3 letter acronyms to avoid false positives on
# common short capitalized words at sentence starts.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z][A-Za-z'\-]{2,}\b")
# Common sentence-start words that match proper-noun regex but aren't entities.
_STOP_PROPER = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Their",
        "They",
        "When",
        "While",
        "Where",
        "Which",
        "What",
        "Who",
        "Why",
        "How",
        "Before",
        "After",
        "During",
        "Between",
        "Among",
        "Over",
        "Under",
        "Some",
        "Many",
        "Most",
        "Several",
        "Each",
        "Every",
        "Both",
        "Neither",
        "However",
        "Therefore",
        "Also",
        "Thus",
        "Hence",
        "Still",
        "Yet",
        "She",
        "He's",
        "She's",
        "It's",
        "They're",
        "There's",
        "First",
        "Second",
        "Third",
        "Fourth",
        "Fifth",
        "Last",
    }
)


def _missing_proper_nouns(passage: str, nugget: str) -> list[str]:
    """Return list of proper-noun-like tokens present in nugget but not in passage.

    Used as a cheap pre-check before LLM entailment. If the list is
    non-empty, the nugget makes a claim about an entity the passage does
    not mention, so it is unlikely to be entailed.
    """
    passage_lower = passage.lower()
    missing = []
    seen: set[str] = set()
    for m in _PROPER_NOUN_RE.finditer(nugget):
        tok = m.group()
        if tok in _STOP_PROPER or tok in seen:
            continue
        seen.add(tok)
        if tok.lower() not in passage_lower:
            missing.append(tok)
    return missing


def _nugget_word_set(text: str) -> set[str]:
    """Case-folded word set for near-duplicate detection."""
    return set(_WORD_RE.findall(text.lower()))


def _is_near_duplicate(
    text: str, existing_texts: list[str], threshold: float = 0.85
) -> bool:
    """True if Jaccard similarity to any existing text exceeds threshold."""
    new_tokens = _nugget_word_set(text)
    if not new_tokens:
        return True  # empty nugget, reject
    for other in existing_texts:
        other_tokens = _nugget_word_set(other)
        if not other_tokens:
            continue
        inter = len(new_tokens & other_tokens)
        union = len(new_tokens | other_tokens)
        if union == 0:
            continue
        if inter / union >= threshold:
            return True
        # Also check containment: if shorter nugget is 85%+ inside longer one
        shorter, longer = (
            (new_tokens, other_tokens)
            if len(new_tokens) <= len(other_tokens)
            else (other_tokens, new_tokens)
        )
        if len(shorter) > 0 and len(shorter & longer) / len(shorter) >= threshold:
            return True
    return False


def _dedup_by_similarity(
    nuggets: list[dict], threshold: float = 0.80
) -> tuple[list[dict], int]:
    """Greedy intra-set near-duplicate filter using _is_near_duplicate.

    Catches nuggets that are semantically near-identical but textually
    distinct (e.g. "X played Nick Fury" vs "X portrays Nick Fury") that
    step 5's LLM dedup misses. Threshold 0.80 is slightly below the
    step-7 gap-fill default (0.85) because step 5's LLM is expected to
    have already caught the verbatim duplicates; residuals here are
    paraphrases that tighter threshold catches.

    Returns:
        (filtered_nuggets, n_dropped)
    """
    accepted: list[dict] = []
    accepted_texts: list[str] = []
    dropped = 0
    for nugget in nuggets:
        text = nugget.get("nugget_text", "")
        if _is_near_duplicate(text, accepted_texts, threshold=threshold):
            dropped += 1
            continue
        accepted.append(nugget)
        accepted_texts.append(text)
    return accepted, dropped


async def _step_7_completeness(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    nuggets: list[dict],
    passages: list[dict],
    question_type: str = "",
) -> tuple[list[dict], dict]:
    """Step 7: Check if nuggets fully support the answer.

    1 LLM call. If incomplete + severity=critical: 1 gap-fill retry from
    existing passages. Max 1 retry then accept.

    When called with nuggets=[] but passages+answer present, skips the
    completeness check and jumps straight to gap-fill — the zero-nugget
    state is implicitly "missing everything".

    Returns:
        (nuggets_possibly_augmented, stats)
    """
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "is_complete": False,
        "gap_fill_attempted": False,
        "gap_fill_added": 0,
    }

    # Zero-nugget recovery path: jump straight to gap-fill.
    zero_nugget_recovery = not nuggets
    if zero_nugget_recovery and (not passages or not answer):
        return nuggets, stats

    effective_type = question_type or "simple"  # pre-resolved by classifier
    if effective_type in ("aggregation", "post_processing"):
        counting_note = (
            "CRITICAL — this is an ENUMERATION question. Steps:\n"
            "1. List every individual item the answer counts or enumerates.\n"
            "2. Check that EACH item is individually named as the PRIMARY SUBJECT "
            "of at least one nugget.\n"
            "3. A nugget mentioning an ordinal ('fourth installment') counts only for "
            "that single item.\n"
            "4. A nugget that REFERENCES an item as context (e.g., 'direct sequel to X') "
            "does NOT count as naming X — X must be the primary subject of its own nugget.\n"
            "5. If even ONE item is missing, output complete=false, severity=critical."
        )
    elif effective_type == "comparison":
        counting_note = (
            "This is a COMPARISON question. Both compared entities must have "
            "supporting nuggets. If only one entity has evidence, output "
            "complete=false, severity=critical."
        )
    elif effective_type == "multi_hop":
        counting_note = (
            "This is a MULTI-HOP question requiring a reasoning chain. "
            "Each reasoning step must be supported by its own nugget. "
            "If any link in the chain is missing, output complete=false, "
            "severity=critical."
        )
    else:
        counting_note = ""

    try:
        if zero_nugget_recovery:
            # Skip completeness check — nothing to assess. Synthesize the
            # same state the LLM would report: critical + missing-everything.
            severity = "critical"
            missing = f"No supporting nugget found for answer '{answer}'"
            log.info("Step 7: zero-nugget recovery — forcing gap-fill")
        else:
            prompt = _COMPLETENESS_PROMPT.format(
                question=question,
                answer=answer,
                question_type=effective_type,
                counting_note=counting_note,
                nuggets=_fmt_nuggets(nuggets),
            )
            content = await _llm_call(client, model, prompt)
            stats["llm_calls"] = 1
            result = _parse_json(content)
            if not isinstance(result, dict):
                stats["is_complete"] = True
                return nuggets, stats

            if result.get("complete", True):
                # Enum-gate: for counting/chain questions, verify we have enough
                # nuggets to match the type target. The LLM verdict is lenient
                # on enumeration — it accepts any answer-justifying nugget set.
                if effective_type in ("aggregation", "multi_hop", "post_processing"):
                    target = _NUGGET_TARGET_BY_TYPE.get(
                        effective_type, _NUGGET_TARGET_DEFAULT
                    )
                    if len(nuggets) < target:
                        severity = "critical"
                        missing = (
                            f"Enumeration incomplete: {len(nuggets)} nuggets "
                            f"present, need {target} for {effective_type} "
                            "questions. Extract one nugget per distinct item."
                        )
                        log.info(
                            "Step 7: enum-gate override — %d/%d, forcing gap-fill",
                            len(nuggets),
                            target,
                        )
                        # fall through to gap-fill
                    else:
                        stats["is_complete"] = True
                        log.info("Step 7: answer is complete (enum target met)")
                        return nuggets, stats
                else:
                    stats["is_complete"] = True
                    log.info("Step 7: answer is complete")
                    return nuggets, stats
            else:
                severity = result.get("severity", "minor")
                missing = result.get("missing", "")
                log.info(
                    "Step 7: incomplete (severity=%s, missing=%s)",
                    severity,
                    missing[:80],
                )

                # Only gap-fill on critical
                if severity != "critical" or not missing:
                    return nuggets, stats

        # Gap-fill: up to 2 passes for enumeration types. Thinking is on for
        # the gap-fill generation call (a single sequential call per pass),
        # where reasoning helps.
        stats["gap_fill_attempted"] = True
        valid_pids = {p["passage_id"] for p in passages}
        passage_by_pid = {p["passage_id"]: p for p in passages}

        async def _verify_gap(cand: dict) -> dict | None:
            p = passage_by_pid.get(cand["passage_id"])
            if not p:
                return None
            text = p.get("passage_text_with_title", "") or p.get("passage_text", "")
            if not text:
                return None
            prompt = _ENTAILMENT_PROMPT.format(
                passage_text=text[:_ENTAILMENT_CHAR_LIMIT],
                nugget_text=cand["nugget_text"],
            )
            try:
                content = await _llm_call(client, model, prompt)
                result = _parse_json(content)
                if isinstance(result, dict) and result.get("entailed", False):
                    return cand
                return None
            except Exception:
                return None  # fail-closed

        async def _run_gap_fill_pass(missing_text: str) -> int:
            """One gap-fill attempt. Returns number of nuggets added."""
            gap_prompt = _GAP_FILL_PROMPT.format(
                question=question,
                answer=answer,
                existing_nuggets=_fmt_nuggets(nuggets) if nuggets else "(none)",
                missing=missing_text,
                passages=_fmt_passages(passages, max_chars=800),
            )
            # Thinking is on for gap-fill generation: it helps cross-passage
            # synthesis, and the latency of one sequential call is tolerable.
            # max_tokens=2048 caps long-tail thinking blowups.
            gap_content = await _llm_call(
                client,
                model,
                gap_prompt,
                enable_thinking=True,
                max_tokens=2048,
            )
            stats["llm_calls"] += 1

            gap_result = _parse_json(gap_content)
            if not isinstance(gap_result, list):
                return 0

            existing_texts = [n["nugget_text"] for n in nuggets]
            candidates = []
            for n in gap_result[:5]:
                pid = n.get("passage_id", "")
                ntext = _clean_nugget_text(n.get("nugget_text", ""))
                if not pid or not ntext or pid not in valid_pids:
                    continue
                if _is_near_duplicate(ntext, existing_texts):
                    continue
                candidates.append({"passage_id": pid, "nugget_text": ntext})

            if not candidates:
                return 0
            verified = await asyncio.gather(*[_verify_gap(c) for c in candidates])
            stats["llm_calls"] += len(candidates)

            added = 0
            accepted_texts = list(existing_texts)
            for v in verified:
                if v is None:
                    continue
                if _is_near_duplicate(v["nugget_text"], accepted_texts):
                    continue
                pid = v["passage_id"]
                nuggets.append(
                    {
                        "passage_id": pid,
                        "passage_ids": [pid],
                        "primary_passage_id": pid,
                        "nugget_text": v["nugget_text"],
                        "original_nuggets": [
                            {"passage_id": pid, "nugget_text": v["nugget_text"]}
                        ],
                    }
                )
                accepted_texts.append(v["nugget_text"])
                added += 1
            return added

        # Pass 1
        added_1 = await _run_gap_fill_pass(missing)
        stats["gap_fill_added"] += added_1
        log.info("Step 7: gap-fill pass 1 added %d nuggets", added_1)

        # Pass 2 — only for enumeration types, if pass 1 added something,
        # and we're still under target. Caps at 2 passes total.
        if added_1 > 0 and effective_type in (
            "aggregation",
            "multi_hop",
            "post_processing",
        ):
            target = _NUGGET_TARGET_BY_TYPE.get(effective_type, _NUGGET_TARGET_DEFAULT)
            if len(nuggets) < target:
                missing_2 = (
                    f"Still short of target ({len(nuggets)}/{target}). Find "
                    "additional distinct items or reasoning-chain links not "
                    "already covered by the nuggets above."
                )
                added_2 = await _run_gap_fill_pass(missing_2)
                stats["gap_fill_added"] += added_2
                log.info("Step 7: gap-fill pass 2 added %d nuggets", added_2)

        return nuggets, stats

    except Exception:
        log.exception("Step 7: completeness check failed")
        return nuggets, stats


# Step 8: Passage pruning (>20, deterministic)


def _step_8_passage_prune(
    nuggets: list[dict],
    passages: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Step 8: Prune passages to MAX_PASSAGES using greedy set cover.

    Deterministic, no LLM. Picks passages maximizing nugget coverage first,
    then fills remaining slots with uncited passages (for other teams' nDCG).
    Orphaned nuggets (whose only passage was dropped) are removed.

    Returns:
        (pruned_passages, pruned_nuggets, stats)
    """
    stats: dict[str, Any] = {
        "input_passages": len(passages),
        "input_nuggets": len(nuggets),
        "output_passages": 0,
        "output_nuggets": 0,
        "orphaned_nuggets": 0,
    }

    if len(passages) <= MAX_PASSAGES:
        stats["output_passages"] = len(passages)
        stats["output_nuggets"] = len(nuggets)
        return passages, nuggets, stats

    # Build nugget coverage map: passage_id → set of nugget indices.
    # Use all passage_ids (not just primary) so set cover considers
    # every passage that can support each nugget.
    pid_to_nugget_idx: dict[str, set[int]] = {}
    for i, n in enumerate(nuggets):
        pids = n.get(
            "passage_ids", [n.get("primary_passage_id", n.get("passage_id", ""))]
        )
        for pid in pids:
            pid_to_nugget_idx.setdefault(pid, set()).add(i)

    passage_by_pid = {p["passage_id"]: p for p in passages}

    # Greedy set cover: pick passage covering most uncovered nuggets
    covered_nuggets: set[int] = set()
    selected_pids: list[str] = []

    while len(selected_pids) < MAX_PASSAGES and len(covered_nuggets) < len(nuggets):
        best_pid = ""
        best_gain = 0
        for pid, nidxs in pid_to_nugget_idx.items():
            if pid in selected_pids or pid not in passage_by_pid:
                continue
            gain = len(nidxs - covered_nuggets)
            if gain > best_gain:
                best_gain = gain
                best_pid = pid
        if not best_pid or best_gain == 0:
            break
        selected_pids.append(best_pid)
        covered_nuggets |= pid_to_nugget_idx[best_pid]

    # Fill remaining slots with uncited passages (preserve original order)
    remaining_slots = MAX_PASSAGES - len(selected_pids)
    if remaining_slots > 0:
        selected_set = set(selected_pids)
        for p in passages:
            if p["passage_id"] not in selected_set:
                selected_pids.append(p["passage_id"])
                if len(selected_pids) >= MAX_PASSAGES:
                    break

    # Build output
    kept_pids = set(selected_pids)
    pruned_passages = [
        passage_by_pid[pid] for pid in selected_pids if pid in passage_by_pid
    ]

    # Remove orphaned nuggets (a nugget survives if any of its passages is kept)
    pruned_nuggets = []
    orphaned = 0
    for n in nuggets:
        n_pids = n.get(
            "passage_ids", [n.get("primary_passage_id", n.get("passage_id", ""))]
        )
        if any(pid in kept_pids for pid in n_pids):
            # Update primary to a kept passage if current primary was pruned
            primary = n.get("primary_passage_id", n.get("passage_id", ""))
            if primary not in kept_pids:
                for pid in n_pids:
                    if pid in kept_pids:
                        n["primary_passage_id"] = pid
                        n["passage_id"] = pid
                        break
            pruned_nuggets.append(n)
        else:
            orphaned += 1

    stats["output_passages"] = len(pruned_passages)
    stats["output_nuggets"] = len(pruned_nuggets)
    stats["orphaned_nuggets"] = orphaned

    log.info(
        "Step 8: %d→%d passages, %d orphaned nuggets removed",
        len(passages),
        len(pruned_passages),
        orphaned,
    )
    return pruned_passages, pruned_nuggets, stats


# Step 9: Nugget pruning (>10)

# Target nugget counts by question type
_NUGGET_TARGET_BY_TYPE: dict[str, int] = {
    "simple": 3,
    "simple_w_condition": 3,
    "aggregation": 8,
    "comparison": 5,
    "multi_hop": 6,
    "post_processing": 6,
}
_NUGGET_TARGET_DEFAULT = 5


def _infer_question_type(question: str) -> str:
    """Heuristic type inference when question_type is empty.

    Conservative fallback — only detects aggregation with high confidence.
    All other types default to simple (safest caps and rules).
    """
    q = question.lower()
    if any(
        kw in q for kw in ("how many", "list all", "enumerate", "count", "name all")
    ):
        return "aggregation"
    # Don't guess multi_hop from " and " — too many false positives.
    # The caller should provide question_type from bundle metadata.
    log.debug("_infer_question_type: no type provided, defaulting to simple")
    return "simple"


async def _step_9_nugget_prune(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    nuggets: list[dict],
    question_type: str = "",
) -> tuple[list[dict], dict]:
    """Step 9: Prune nuggets to MAX_NUGGETS.

    Phase 1: Algorithmic pre-filter to ≤30 (diversity + brevity scoring).
    Phase 2: 1 LLM call to select target count.

    Returns:
        (pruned_nuggets, stats)
    """
    effective_type = question_type or "simple"  # pre-resolved by classifier
    target = _NUGGET_TARGET_BY_TYPE.get(effective_type, _NUGGET_TARGET_DEFAULT)
    target = min(target, MAX_NUGGETS)

    stats: dict[str, Any] = {
        "llm_calls": 0,
        "target": target,
        "input": len(nuggets),
        "output": 0,
    }

    if len(nuggets) <= target:
        stats["output"] = len(nuggets)
        return nuggets, stats

    # Phase 1: algorithmic pre-filter to ≤30
    candidates = nuggets
    if len(candidates) > 30:
        # Score by brevity (shorter = better, less likely to be compound)
        # and passage diversity (spread across passages)
        scored = []
        pid_seen: dict[str, int] = {}
        for n in candidates:
            pid = n.get("primary_passage_id", n.get("passage_id", ""))
            pid_seen[pid] = pid_seen.get(pid, 0) + 1
            # Lower score = better: brevity + diversity penalty
            score = len(n["nugget_text"]) + (pid_seen[pid] - 1) * 100
            scored.append((score, n))
        scored.sort(key=lambda x: x[0])
        candidates = [n for _, n in scored[:30]]

    # Phase 2: LLM selection
    enumeration_rule = _ENUM_RULES.get(effective_type, _ENUM_DEFAULT)
    prompt = _NUGGET_SELECT_PROMPT.format(
        question=question,
        answer=answer,
        question_type=effective_type,
        target=target,
        enumeration_rule=enumeration_rule,
        nuggets=_fmt_nuggets(candidates),
    )
    try:
        content = await _llm_call(client, model, prompt)
        stats["llm_calls"] = 1
        result = _parse_json(content)
        if not isinstance(result, list):
            stats["output"] = len(candidates[:target])
            return candidates[:target], stats

        selected = []
        for idx in result:
            if isinstance(idx, (int, float)):
                idx = int(idx)
            else:
                continue
            if 0 <= idx < len(candidates):
                selected.append(candidates[idx])
            if len(selected) >= target:
                break

        if not selected:
            stats["output"] = len(candidates[:target])
            return candidates[:target], stats

        stats["output"] = len(selected)
        log.info(
            "Step 9: %d→%d nuggets (target=%d)", len(nuggets), len(selected), target
        )
        return selected, stats
    except Exception:
        log.exception("Step 9: LLM selection failed, using algorithmic top-%d", target)
        stats["output"] = len(candidates[:target])
        return candidates[:target], stats


# Step 10: Relevance scoring


async def _step_10_relevance(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    nuggets: list[dict],
    passages: list[dict],
    question_type: str = "",
) -> tuple[list[dict], dict]:
    """Step 10: Score each nugget for relevance and drop nonrelevant.

    1 LLM call per nugget, all parallel. Mirrors evaluator's judgment.

    Returns:
        (relevant_nuggets, stats)
    """
    effective_type = question_type or "simple"  # pre-resolved by classifier
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "input": len(nuggets),
        "relevant": 0,
        "dropped": 0,
    }

    if not nuggets:
        return [], stats

    if effective_type == "aggregation":
        counting_note = (
            "Note: This is a COUNTING question. Every nugget that establishes "
            "the existence or identity of any item in the counted set IS relevant, "
            "even if it does not contain the numeric answer."
        )
    else:
        counting_note = (
            "Note: For questions that span two entities or events, facts about "
            "either entity in the chain ARE relevant."
        )

    passage_by_pid: dict[str, dict] = {p["passage_id"]: p for p in passages}

    async def _score_one(nugget: dict) -> dict | None:
        pid = nugget.get("primary_passage_id", nugget.get("passage_id", ""))
        p = passage_by_pid.get(pid, {})
        ptext = p.get("passage_text_with_title", "") or p.get("passage_text", "")
        prompt = _RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
            passage_text=ptext[:800] if ptext else "(not available)",
            nugget_text=nugget["nugget_text"],
            counting_note=counting_note,
        )
        try:
            content = await _llm_call(client, model, prompt)
            result = _parse_json(content)
            if isinstance(result, dict) and result.get("relevant", False):
                nugget["relevant"] = True
                return nugget
            return None
        except Exception:
            # Fail-open for relevance (already verified as non-bogus)
            nugget["relevant"] = True
            return nugget

    results = await asyncio.gather(*[_score_one(n) for n in nuggets])
    stats["llm_calls"] = len(nuggets)

    relevant = [r for r in results if r is not None]
    # Safety floor: never return 0 nuggets if we had any
    if not relevant and nuggets:
        log.warning("Step 10: all nuggets dropped — keeping first as safety floor")
        nuggets[0]["relevant"] = True
        relevant = [nuggets[0]]
    relevant_texts = {r["nugget_text"] for r in relevant}
    stats["relevant"] = len(relevant)
    stats["dropped"] = len(nuggets) - len(relevant)
    stats["details"] = [
        {
            "nugget_text": n["nugget_text"],
            "passage_id": n.get("primary_passage_id", n.get("passage_id", "")),
            "relevant": n["nugget_text"] in relevant_texts,
        }
        for n in nuggets
    ]

    log.info(
        "Step 10: %d/%d nuggets relevant (dropped %d)",
        len(relevant),
        len(nuggets),
        stats["dropped"],
    )
    return relevant, stats


# Public API


async def reconcile_nuggets_and_passages(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    raw_nuggets: list[dict],
    passages: list[dict],
    mcp_client: Any = None,
    question_type: str = "",
) -> tuple[list[dict], list[dict], dict]:
    """Run the post-answer pipeline.

    Args:
        client: Async OpenAI-compatible client (reuses the worker's vLLM).
        model: Model name.
        question: Question text.
        answer: Extracted answer string.
        raw_nuggets: Nuggets from regex extraction (ignored — step 2 generates fresh).
        passages: All passages from the trace (any ranking).
        mcp_client: MCP HTTP client for extract calls. Can be None.
        question_type: Question type string (simple, aggregation, multi_hop, etc.).

    Returns:
        (ranked_passages, clean_nuggets, stats) — mutually consistent.
        Passages ranked by nugget link count, no orphans on either side.
        stats: dict with per-step breakdown.
    """
    # Feature-flag dispatch (v2 path bypasses the 10-step JSON pipeline).
    if _use_v2_pipeline():
        if not passages:
            return (
                passages,
                raw_nuggets,
                {
                    "pipeline": POSTTRACE_PIPELINE_V2,
                    "skipped": "no_passages",
                },
            )
        return await _reconcile_v2(
            client=client,
            model=model,
            question=question,
            answer=answer,
            passages=passages,
            question_type=question_type,
            mcp_client=mcp_client,
        )

    stats: dict[str, Any] = {
        "llm_calls": 0,
        "mcp_calls": 0,
        "steps": {},
    }

    if not passages:
        return passages, raw_nuggets, stats

    # Classify question type (if not provided)
    question_type, classify_stats = await _classify_question_type(
        client,
        model,
        question,
        answer,
        question_type,
    )
    stats["llm_calls"] += classify_stats["llm_calls"]
    stats["steps"]["classify"] = classify_stats

    # Step 0: Answer format verification
    answer, step0_stats = await _step_0_answer_verify(
        client,
        model,
        question,
        answer,
    )
    stats["llm_calls"] += step0_stats["llm_calls"]
    stats["steps"]["0_answer_verify"] = step0_stats

    # Step 1: Relevant passage extraction
    passages, step1_stats = await _step_1_passage_extract(
        client,
        model,
        question,
        answer,
        passages,
        mcp_client,
    )
    stats["llm_calls"] += step1_stats["llm_calls"]
    stats["mcp_calls"] += step1_stats["mcp_calls"]
    stats["steps"]["1_passage_extract"] = step1_stats

    # Step 2: Parallel nugget generation
    nuggets, step2_stats = await _step_2_nugget_gen(
        client,
        model,
        question,
        answer,
        passages,
        question_type,
    )
    stats["llm_calls"] += step2_stats["llm_calls"]
    stats["steps"]["2_nugget_gen"] = step2_stats

    # Step 3: Parallel bogus verification
    nuggets, step3_stats = await _step_3_verify(
        client,
        model,
        nuggets,
        passages,
    )
    stats["llm_calls"] += step3_stats["llm_calls"]
    stats["steps"]["3_verify"] = step3_stats

    # Step 5: Deduplication with multi-passage support
    nuggets, step5_stats = await _step_5_dedup(
        client,
        model,
        nuggets,
        question,
        question_type,
    )
    stats["llm_calls"] += step5_stats["llm_calls"]
    stats["steps"]["5_dedup"] = step5_stats

    # Step 5b: Intra-set near-duplicate filter
    # Deterministic pass catching paraphrase-level duplicates that step 5's
    # LLM dedup preserved (e.g. "X played Y" vs "X portrays Y"). Zero LLM cost.
    nuggets, sim_dropped = _dedup_by_similarity(nuggets, threshold=0.80)
    stats["steps"]["5b_similarity_dedup"] = {
        "input": len(nuggets) + sim_dropped,
        "output": len(nuggets),
        "dropped": sim_dropped,
    }
    if sim_dropped:
        log.info("Step 5b: similarity dedup dropped %d nuggets", sim_dropped)

    # Step 6: Re-verification post-dedup
    nuggets, step6_stats = await _step_6_reverify(
        client,
        model,
        nuggets,
        passages,
    )
    stats["llm_calls"] += step6_stats["llm_calls"]
    stats["steps"]["6_reverify"] = step6_stats

    # Step 7: Completeness check
    nuggets, step7_stats = await _step_7_completeness(
        client,
        model,
        question,
        answer,
        nuggets,
        passages,
        question_type,
    )
    stats["llm_calls"] += step7_stats["llm_calls"]
    stats["steps"]["7_completeness"] = step7_stats

    # Step 8: Passage pruning (>20, deterministic)
    passages, nuggets, step8_stats = _step_8_passage_prune(nuggets, passages)
    stats["steps"]["8_passage_prune"] = step8_stats

    # Step 9: Nugget pruning (>10)
    nuggets, step9_stats = await _step_9_nugget_prune(
        client,
        model,
        question,
        answer,
        nuggets,
        question_type,
    )
    stats["llm_calls"] += step9_stats["llm_calls"]
    stats["steps"]["9_nugget_prune"] = step9_stats

    # Step 9b: Final near-duplicate filter before relevance
    # Step 6 can re-expand merged nuggets to their pre-merge originals
    # (bypassing 5b), and step 7 gap-fill can add near-duplicates.
    # Run the similarity pass again to catch revert-expansions and
    # gap-fill dupes. Zero LLM cost.
    nuggets, sim_dropped_9b = _dedup_by_similarity(nuggets, threshold=0.80)
    stats["steps"]["9b_similarity_dedup"] = {
        "input": len(nuggets) + sim_dropped_9b,
        "output": len(nuggets),
        "dropped": sim_dropped_9b,
    }
    if sim_dropped_9b:
        log.info("Step 9b: similarity dedup dropped %d nuggets", sim_dropped_9b)

    # Step 10: Relevance scoring
    nuggets, step10_stats = await _step_10_relevance(
        client,
        model,
        question,
        answer,
        nuggets,
        passages,
        question_type,
    )
    stats["llm_calls"] += step10_stats["llm_calls"]
    stats["steps"]["10_relevance"] = step10_stats

    # Final assembly: rank passages by nugget citation count
    pid_count: dict[str, int] = {}
    for n in nuggets:
        pid = n.get("primary_passage_id", n.get("passage_id", ""))
        pid_count[pid] = pid_count.get(pid, 0) + 1

    passage_by_pid = {p["passage_id"]: p for p in passages}

    # Passages cited by nuggets, ranked by citation count
    ranked_passages = [
        passage_by_pid[pid]
        for pid in sorted(pid_count, key=lambda p: pid_count[p], reverse=True)
        if pid in passage_by_pid
    ]
    # Append remaining passages (not cited) for PR nDCG coverage
    cited_pids = set(pid_count.keys())
    for p in passages:
        if p["passage_id"] not in cited_pids:
            ranked_passages.append(p)

    # Final cap safety (step 8 should already enforce this)
    ranked_passages = ranked_passages[:MAX_PASSAGES]
    kept_pids = {p["passage_id"] for p in ranked_passages}
    nuggets = [
        n
        for n in nuggets
        if n.get("primary_passage_id", n.get("passage_id", "")) in kept_pids
    ]
    nuggets = nuggets[:MAX_NUGGETS]

    log.info(
        "Pipeline done: %d passages, %d nuggets [%d LLM, %d MCP calls]",
        len(ranked_passages),
        len(nuggets),
        stats["llm_calls"],
        stats["mcp_calls"],
    )
    return ranked_passages, nuggets, stats
