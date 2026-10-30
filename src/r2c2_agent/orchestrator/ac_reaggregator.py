"""Post-hoc AC re-aggregator for the Phase 1 frozen R2C2 PR submission.

Reads the FROZEN ``BITEM-PG-{1,2}`` (PR) and ``BITEM-AC-{1,2}`` (original AC)
files plus the snapshot ``iter_*/test_traces.jsonl`` rollouts. Emits v2 AC
files (``BITEM-AC-1.v2`` + ``BITEM-AC-2.v2``) and a ``BITEM-AC.v2.zip``
bundle. The frozen PR ZIP is NEVER touched.

v2 AC selection rules (informed by Phase 2 audits 2.1-2.4):

* **R1 (audit 2.1)** — the retired ladder tier fired 0/236 in the PR run; this branch
  is treated as dead. Nugget weighting is termination-aware: nuggets
  produced under T1 (prior-assistant scrape, no NLI) get no NLI bonus
  unless they carry an explicit ``nli_score > 0``.
* **R2 (audit 2.2)** — S1 verifier acceptance falls 59% → 47% across
  iter_0 → iter_2. Earlier-iteration nuggets get a +1.0 weight; iter_2/3
  nuggets get +0.7. Cross-rollout co-occurrence adds +0.1 per extra
  rollout.
* **R3 (audit 2.3, biggest lever)** — refusal-qid rescue. When a qid's
  frozen AC carries the canonical refusal answer AND no rollout produced
  a verified-supporting nugget that maps to a frozen-PR rank, synthesise
  AC nugget records from the frozen-PR passage text itself (first
  sentence ≤ 200 chars). Bounded by ``MAX_NUGGETS_PER_QID``. The frozen
  PR roster is the authoritative passage set, so this only widens
  AC-side evidence within the PR-roster envelope (subset invariant
  preserved).
* **R4 (audit 2.4)** — retrieval recall is 97.96%; misses are
  retrieval-side. We do NOT re-do retrieval; v2 only re-picks nuggets
  from what the PR run already retrieved.

Hard invariants:

1. Every v2 AC ``PassageRank`` MUST be a valid rank in the frozen PR
   for the same ``(qid, run_name)`` pair. Hard-fail otherwise.
2. PG-2 path (``first_iter_only=False``) preserves the frozen
   ``Answer;Confidence`` line verbatim.
3. PG-1 path (``first_iter_only=True``) applies the trace-redesign
   strategy: when
   the frozen AC carries the canonical refusal AND the iter_0 trace
   contains a non-refusal ``submit_answer`` attempt, surface that
   attempt's text + confidence (clamped to the frozen-PR conf as a
   floor, ``;`` sanitised). Otherwise the frozen line is kept.
4. R2C2 spec validation must report 8/8 PASS (7 spec + 1 source-coverage).

Reuse: ``r2c2_agent.agent.postprocess.format_ac_result``/``format_ac_file``
emit the AC text; ``r2c2_agent.agent.submission_validator.validate_ac_file``
+ ``scripts.validation.validate_submission`` perform the format checks;
``ensemble_aggregator._extract_nuggets_from_trace`` /
``_extract_pids_from_trace`` parse the trace records.
"""

from __future__ import annotations

import argparse
import os
import collections
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Path bootstrap, as in ensemble_aggregator.

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Trace extraction and format checks come from the existing aggregator.
from r2c2_agent.agent.interfaces.types import (  # noqa: E402
    ACResult,
    Nugget,
    QuestionTrace,
)
from r2c2_agent.agent.postprocess import (  # noqa: E402
    format_ac_file,
    format_ac_result,
)
from r2c2_agent.agent.submission_validator import (  # noqa: E402
    validate_ac_file,
    validate_pr_file,
)
from r2c2_agent.orchestrator.ensemble_aggregator import (  # noqa: E402
    REFUSAL_RE,
    _extract_nuggets_from_trace,
    _extract_pids_from_trace,
    _sanitize,
    check_source_coverage,
)
from r2c2_agent.validation.validate_submission import (  # noqa: E402
    check_ac_answer_confidence,
    check_ac_nugget_format,
    check_ac_nugget_num,
    check_ac_passage_refs,
    check_ac_xml_tags,
    check_pr_max_passages,
    check_pr_passage_rank,
)

log = logging.getLogger(__name__)

__all__ = [
    "FrozenPR",
    "FrozenAC",
    "build_pid_index",
    "load_frozen_pr",
    "load_frozen_ac",
    "load_rollouts_for_run",
    "select_v2_nuggets",
    "build_v2_ac_text",
    "run_validation",
    "main",
]

# Constants

MAX_NUGGETS_PER_QID = 10  # R2C2 cap

# Phase A nugget scoring
PHASE_A_NLI_BONUS = 1.0
PHASE_A_EARLY_ITER_WEIGHT = 1.0  # earliest_ens ≤ 1
PHASE_A_LATE_ITER_WEIGHT = 0.7  # earliest_ens ≥ 2 (verifier acceptance is lower)
PHASE_A_EARLY_ITER_THRESHOLD = 1
PHASE_A_CO_OCCURRENCE_BONUS = 0.1  # per extra rollout beyond the first

# Phase B rescue
RESCUE_NUGGET_TEXT_MAX = 200
RESCUE_NUGGET_MIN_LEN = 20  # don't bother synthesising near-empty nuggets

# Frozen-PR roster file naming convention
PR_FILE_PG1 = "BITEM-PG-1"
PR_FILE_PG2 = "BITEM-PG-2"
AC_FILE_PG1 = "BITEM-AC-1"
AC_FILE_PG2 = "BITEM-AC-2"

# Run-name pairing: PR file -> AC v2 output file.
# Keep the AC file naming convention from the R2C2 spec (no `.AC` suffix).
RUN_NAME_PG1 = "BITEM-PG-1"
RUN_NAME_PG2 = "BITEM-PG-2"

# AC re-aggregation strategies.

STRATEGY_TRACE_REDESIGN = "trace-redesign"  # AC-1: PG-1 per-iter trace pick
STRATEGY_BEST_ROLLOUT = "best-rollout"  # earlier AC-2 design, kept for backward compat
STRATEGY_PLURALITY = "plurality"  # earlier AC-3 design, kept for backward compat
STRATEGY_LLM_JUDGE = "llm-judge"  # AC-4: LLM-as-judge
STRATEGY_MAX_CONF = "max-conf"  # AC-2: order-statistic variance reduction
STRATEGY_CONF_SUM_CLUSTER = "conf-sum-cluster"  # AC-3: evidence pooling

ALLOWED_STRATEGIES = frozenset(
    {
        STRATEGY_TRACE_REDESIGN,
        STRATEGY_BEST_ROLLOUT,
        STRATEGY_PLURALITY,
        STRATEGY_LLM_JUDGE,
        STRATEGY_MAX_CONF,
        STRATEGY_CONF_SUM_CLUSTER,
    }
)

# Best-rollout scoring
BEST_ROLLOUT_NON_REFUSAL_BONUS = 10.0

# Plurality voting
PLURALITY_CLASS_WEIGHT_OK = 1.0
PLURALITY_CLASS_WEIGHT_HARDGATE = 0.4
PLURALITY_CLASS_WEIGHT_FORMAT = 0.1
PLURALITY_MIN_CLUSTER_SCORE = 0.5
REFUSAL_FLOOR_CONF = 25  # confidence of the canonical-refusal fallback


# Datatypes


@dataclass(frozen=True)
class FrozenPR:
    """Parsed frozen PR file: ``qid -> [(rank, doc_id, sanitized_passage_text)]``."""

    run_name: str
    rows_by_qid: dict[str, list[tuple[int, str, str]]] = field(default_factory=dict)

    def qids(self) -> set[str]:
        return set(self.rows_by_qid)

    def rank_set(self, qid: str) -> set[int]:
        return {rank for rank, _, _ in self.rows_by_qid.get(qid, [])}


@dataclass(frozen=True)
class FrozenAC:
    """Parsed frozen AC file: ``qid -> (answer, confidence)``.

    Nugget records from the frozen AC are intentionally discarded — v2
    AC re-aggregation rebuilds the nuggets from scratch using the rules
    above. Only the ``Answer;Confidence`` line is preserved verbatim.
    """

    run_name: str
    answers_by_qid: dict[str, tuple[str, int]] = field(default_factory=dict)


@dataclass
class _NuggetCandidate:
    """A single nugget candidate scored for v2 selection."""

    passage_id: str
    text: str
    nli_score: float
    earliest_ens: int
    co_occurrence: int
    pr_rank: int  # frozen-PR rank for the resolved pid
    is_synthetic: bool = False  # True for Phase B rescue nuggets


# Frozen-file parsers


def load_frozen_pr(pr_path: Path, run_name: str) -> FrozenPR:
    """Parse a frozen ``BITEM-PG-{1,2}`` PR file.

    Each line is ``QID;Rank;DocID;PassageText`` (rsplit on the third
    semicolon — PassageText is allowed to contain ``;``).

    Args:
        pr_path: Frozen PR file path. Read-only.
        run_name: Logical run name, e.g. ``"BITEM-PG-1"``.

    Returns:
        ``FrozenPR`` with ``rows_by_qid`` populated.

    Raises:
        FileNotFoundError: if ``pr_path`` is missing.
        ValueError: if any line is malformed.
    """
    if not pr_path.exists():
        raise FileNotFoundError(f"frozen PR file missing: {pr_path}")
    rows_by_qid: dict[str, list[tuple[int, str, str]]] = collections.defaultdict(list)
    for line_num, raw in enumerate(pr_path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        # QID;Rank;DocID;PassageText — at least 4 fields; PassageText may
        # itself contain ';' (organiser PO files do). split with
        # maxsplit=3.
        parts = line.split(";", 3)
        if len(parts) < 4:
            raise ValueError(
                f"frozen PR {pr_path.name} line {line_num}: expected 4 "
                f"fields, got {len(parts)}: {line[:80]!r}"
            )
        qid, rank_str, doc_id, passage_text = parts
        try:
            rank = int(rank_str)
        except ValueError as exc:  # noqa: PERF203
            raise ValueError(
                f"frozen PR {pr_path.name} line {line_num}: bad rank {rank_str!r}"
            ) from exc
        # PR text is already sanitized by the live aggregator — keep it
        # as-is so the (qid, doc_id, text) join key is stable.
        rows_by_qid[qid].append((rank, doc_id, passage_text))
    # Sort each qid's rows by rank ascending for deterministic iteration.
    for qid in rows_by_qid:
        rows_by_qid[qid].sort(key=lambda row: row[0])
    return FrozenPR(run_name=run_name, rows_by_qid=dict(rows_by_qid))


_AC_OPEN_TAG_RE = re.compile(r"^<([^/][^>]*)>$")
_AC_CLOSE_TAG_RE = re.compile(r"^</([^>]+)>$")


def load_frozen_ac(ac_path: Path, run_name: str) -> FrozenAC:
    """Parse a frozen ``BITEM-AC-{1,2}`` AC file for ``Answer;Confidence`` only.

    Nugget lines inside each block are intentionally discarded — v2
    nuggets are rebuilt from the trace rollouts.

    Args:
        ac_path: Frozen AC file path. Read-only.
        run_name: Logical run name, e.g. ``"BITEM-PG-1"``.

    Returns:
        ``FrozenAC`` with ``answers_by_qid`` populated.

    Raises:
        FileNotFoundError: if ``ac_path`` is missing.
        ValueError: if any block is malformed.
    """
    if not ac_path.exists():
        raise FileNotFoundError(f"frozen AC file missing: {ac_path}")
    answers_by_qid: dict[str, tuple[str, int]] = {}
    current_qid: str | None = None
    answer_seen = False
    for line_num, raw in enumerate(ac_path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        open_m = _AC_OPEN_TAG_RE.match(line)
        close_m = _AC_CLOSE_TAG_RE.match(line)
        if open_m and current_qid is None:
            current_qid = open_m.group(1)
            answer_seen = False
            continue
        if close_m:
            if current_qid is None:
                raise ValueError(
                    f"frozen AC {ac_path.name} line {line_num}: stray close "
                    f"tag {line!r}"
                )
            current_qid = None
            continue
        if current_qid is None:
            raise ValueError(
                f"frozen AC {ac_path.name} line {line_num}: content outside "
                f"block: {line[:80]!r}"
            )
        if not answer_seen:
            # AnswerString;ConfidenceScore — answer can contain ';' (e.g. the
            # canonical refusal), so split on the last ';'.
            parts = line.rsplit(";", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"frozen AC {ac_path.name} line {line_num}: answer line "
                    f"missing confidence: {line!r}"
                )
            answer, conf_str = parts[0], parts[1].strip()
            try:
                confidence = int(conf_str)
            except ValueError as exc:
                raise ValueError(
                    f"frozen AC {ac_path.name} line {line_num}: bad "
                    f"confidence {conf_str!r}"
                ) from exc
            answers_by_qid[current_qid] = (answer, confidence)
            answer_seen = True
            continue
        # Nugget lines are skipped — v2 rebuilds nuggets from scratch.
    if current_qid is not None:
        raise ValueError(
            f"frozen AC {ac_path.name}: unclosed block <{current_qid}> at EOF"
        )
    return FrozenAC(run_name=run_name, answers_by_qid=answers_by_qid)


# Trace ingestion + (qid, pid) → frozen-PR rank join


@dataclass
class _Rollout:
    """Slimmed rollout view tailored to the v2 reaggregator.

    We do NOT reuse ``ensemble_aggregator._Rollout`` directly because
    we keep the iter index alongside passages for cross-rollout
    co-occurrence scoring (Phase A).
    """

    qid: str
    ensemble_idx: int
    iter_dir: int  # iter_<n> dir number
    termination: str
    final_answer: str
    nuggets: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)
    # Raw trace record, read only by the trace-redesign strategy. The
    # Phase-A/Phase-B nugget pipeline does not use it.
    raw_trace: dict[str, Any] = field(default_factory=dict, repr=False)


# Trace-redesign helpers.
#
# These power the PG-1 ("first_iter_only") AC strategy: when iter_0's
# live pipeline forced a canonical refusal, surface the model's actual
# submit_answer attempt(s) instead. The PG-2 path is unaffected.

# Classification of one submit_answer attempt.
ATTEMPT_OK = "OK"  # committed=True AND args.answer non-refusal
ATTEMPT_REJ_HARDGATE = "REJ_HARDGATE"  # committed=False, low_confidence_or_…
ATTEMPT_REJ_FORMAT = "REJ_FORMAT"  # committed=False, error=invalid_rationale
ATTEMPT_REFUSAL_TEXT = "REFUSAL_TEXT"  # args.answer matches the refusal pattern

# Tier-A* synthetic-NLI floor; ensures Tier-A* nuggets can win Phase-A
# scoring when Phase-A real nuggets are missing.
TIER_A_STAR_NLI_BONUS = 1.0


@dataclass
class _SubmitAttempt:
    """One submit_answer tool_call paired with its tool-result message."""

    answer_text: str
    rationale: str
    citation_pids: list[str]
    committed: bool
    reason: str | None
    error: str | None
    confidence: float | None
    n_eff: int | None
    ts_idx: int  # ordinal position in messages[] for stable tie-breaking
    classification: str = ""

    @property
    def class_priority(self) -> int:
        # Smaller is preferred by the picker.
        return {
            ATTEMPT_OK: 0,
            ATTEMPT_REJ_HARDGATE: 1,
            ATTEMPT_REJ_FORMAT: 2,
            ATTEMPT_REFUSAL_TEXT: 3,
        }.get(self.classification, 4)


def _is_canonical_refusal_text(text: str) -> bool:
    """Match the live pipeline's REFUSAL_RE against an answer string.

    Detects both the canonical "The corpus does not support…" form and
    near-paraphrases the model emits when refusing on its own.
    """
    if not text:
        return False
    return bool(REFUSAL_RE.search(text))


def _extract_submit_attempts_from_trace(rec: dict[str, Any]) -> list[_SubmitAttempt]:
    """Walk ``rec['messages']`` and pair each ``submit_answer`` tool_call
    with its matching ``tool_call_id``-keyed result message.

    Reads ``answer_text`` from ``args['answer']`` (NOT from the
    post-sanitised tool-result ``answer`` field — force_deliver may have
    rewritten the visible answer to the canonical refusal even when
    ``committed=True``; the model's true answer is in args).

    Args:
        rec: One trace record (one element of ``test_traces.jsonl``).

    Returns:
        List of attempts, in trace order. Empty if no ``submit_answer``
        tool_call appears.
    """
    msgs = rec.get("messages") or []
    if not isinstance(msgs, list):
        return []

    # Pair each call with the first later tool message carrying its
    # tool_call_id; the live runtime pairs strictly in order.
    out: list[_SubmitAttempt] = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        tcs = m.get("tool_calls") or []
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = (tc.get("function") or {}).get("name")
            if fn != "submit_answer":
                continue
            args_field = (tc.get("function") or {}).get("arguments")
            args: dict[str, Any] = {}
            if isinstance(args_field, dict):
                args = args_field
            elif isinstance(args_field, str):
                try:
                    args = json.loads(args_field)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            answer_text = (
                (args.get("answer") or "").strip() if isinstance(args, dict) else ""
            )
            rationale = (args.get("rationale") or "") if isinstance(args, dict) else ""
            citation_pids_raw = (
                args.get("citation_pids") if isinstance(args, dict) else None
            )
            citation_pids: list[str] = []
            if isinstance(citation_pids_raw, list):
                citation_pids = [str(p) for p in citation_pids_raw if p]

            tcid = tc.get("id")
            result: dict[str, Any] = {}
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if not isinstance(cand, dict):
                    continue
                if cand.get("role") != "tool":
                    continue
                if cand.get("tool_call_id") != tcid:
                    continue
                content = cand.get("content")
                if isinstance(content, dict):
                    result = content
                elif isinstance(content, str):
                    try:
                        result = json.loads(content)
                    except (json.JSONDecodeError, ValueError):
                        result = {}
                break

            committed = bool(result.get("committed"))
            reason = result.get("reason")
            error = result.get("error")
            conf_raw = result.get("confidence")
            confidence: float | None = None
            if isinstance(conf_raw, (int, float)):
                confidence = float(conf_raw)
            n_eff_raw = result.get("n_effective_supporting")
            n_eff: int | None = None
            if isinstance(n_eff_raw, int):
                n_eff = n_eff_raw

            attempt = _SubmitAttempt(
                answer_text=answer_text,
                rationale=rationale,
                citation_pids=citation_pids,
                committed=committed,
                reason=reason if isinstance(reason, str) else None,
                error=error if isinstance(error, str) else None,
                confidence=confidence,
                n_eff=n_eff,
                ts_idx=i,
            )
            attempt.classification = _classify_attempt(attempt)
            out.append(attempt)
    return out


def _classify_attempt(a: _SubmitAttempt) -> str:
    """Classify one attempt into one of the four redesign classes."""
    if _is_canonical_refusal_text(a.answer_text):
        return ATTEMPT_REFUSAL_TEXT
    if a.committed:
        # committed AND args.answer is not a refusal pattern → OK class.
        # (force_deliver may have rewritten the public-facing answer to
        # the canonical refusal; we read args.answer not result.answer.)
        return ATTEMPT_OK
    # not committed → either hardgate or format gate
    if a.error == "invalid_rationale":
        return ATTEMPT_REJ_FORMAT
    if a.reason == "low_confidence_or_insufficient_evidence":
        return ATTEMPT_REJ_HARDGATE
    # Unknown rejection → treat as REJ_FORMAT (least preferred non-refusal).
    return ATTEMPT_REJ_FORMAT


def _sanitize_ac_answer(text: str) -> str:
    """Sanitise model answers for the AC ``Answer;Confidence`` line.

    Operations (in order):
        1. Replace literal ``;`` with ``,`` (the R2C2 line uses ``;``
           as a delimiter, so the answer field cannot contain raw ``;``).
        2. Strip surrounding ASCII single/double quotes the model
           sometimes adds.
        3. Collapse runs of whitespace to one space.
    """
    if not text:
        return ""
    out = text.replace(";", ",")
    out = out.strip()
    while len(out) >= 2 and (
        (out.startswith('"') and out.endswith('"'))
        or (out.startswith("'") and out.endswith("'"))
    ):
        out = out[1:-1].strip()
    # Collapse extra whitespace introduced by stripping or by source text.
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _pick_per_iter_answer(
    attempts: list[_SubmitAttempt],
    frozen_pr_answer: str,
    frozen_pr_conf: int,
    pr_rank_map: dict[str, int],
    qid: str,  # noqa: ARG001 — kept for symmetry / future logging
) -> tuple[str, int, list[str]] | None:
    """Pick the best non-refusal submit_answer attempt per §2.4.

    Args:
        attempts: All submit_answer attempts for one rollout.
        frozen_pr_answer: The frozen-AC answer line for ``qid``. Used
            ONLY for the confidence floor (§2 + Open Item AC2-O2).
        frozen_pr_conf: Frozen-AC confidence as the lower bound. The
            new strategy emits ``max(tool_conf × 100, frozen_pr_conf)``
            so a re-aggregation never downgrades a confident answer.
        pr_rank_map: ``{pid: rank}`` for the qid; used to break ties in
            favour of attempts whose citations land in the PR roster.
        qid: Question id (reserved for future logging).

    Returns:
        ``(answer, confidence, citation_pids)`` if a non-refusal attempt
        survives the ranking, else ``None`` (caller falls back to the
        existing canonical-refusal path).
    """
    if not attempts:
        return None

    # Drop REFUSAL_TEXT attempts: a refusal is never preferred, whether
    # or not it was committed.
    candidates = [a for a in attempts if a.classification != ATTEMPT_REFUSAL_TEXT]
    if not candidates:
        return None

    pr_pids = set(pr_rank_map.keys())

    def _sort_key(a: _SubmitAttempt) -> tuple[Any, ...]:
        # 1. Class priority (smaller = better).
        # 2. Descending confidence — None at the back.
        conf = a.confidence if a.confidence is not None else -1.0
        # 3. Descending n_eff — None treated as 0.
        n_eff = a.n_eff if a.n_eff is not None else 0
        # 4. Descending count of citation_pids that resolve to PR roster.
        pr_hits = sum(1 for p in a.citation_pids if p in pr_pids)
        # 5. Stable order — earliest message index wins ties.
        return (
            a.class_priority,
            -conf,
            -n_eff,
            -pr_hits,
            a.ts_idx,
        )

    candidates.sort(key=_sort_key)
    chosen = candidates[0]

    # Take the max with the frozen-PR confidence so re-aggregation never
    # lowers a confidence.
    tool_conf_int = int(round((chosen.confidence or 0.0) * 100))
    confidence = max(tool_conf_int, int(frozen_pr_conf))
    confidence = max(0, min(100, confidence))

    answer = _sanitize_ac_answer(chosen.answer_text)
    if not answer:
        # Sanitisation produced an empty string — treat as refusal-fall-through.
        return None
    return answer, confidence, list(chosen.citation_pids)


def _iter_dir_num(path: Path) -> int:
    """Extract the integer ``n`` from a path that contains ``iter_<n>``."""
    m = re.search(r"iter_(\d+)", str(path))
    return int(m.group(1)) if m else -1


def load_rollouts_for_run(
    traces_dir: Path,
    *,
    first_iter_only: bool,
) -> dict[str, list[_Rollout]]:
    """Load rollouts grouped by qid for one PR run.

    Args:
        traces_dir: Snapshot dir containing ``iter_<n>/test_traces.jsonl``
            files (e.g. the original run directory). Only iters 0..3
            (the K=4 ensemble dispatched to the PR) are considered; later
            iters in the snapshot are scratch and skipped.
        first_iter_only: ``True`` for the PG-1 (single-pass) run — keep
            only iter_0 rollouts. ``False`` for PG-2 (full ensemble).

    Returns:
        Mapping ``qid -> [_Rollout, ...]`` sorted by ensemble_idx.
    """
    by_qid: dict[str, list[_Rollout]] = collections.defaultdict(list)
    iter_paths = sorted(traces_dir.glob("iter_*/test_traces.jsonl"))
    for path in iter_paths:
        iter_n = _iter_dir_num(path.parent)
        if iter_n < 0 or iter_n > 3:
            # Snapshot retains iter_4..9 as scratch; only iter_0..3 were
            # dispatched to the K=4 PR aggregator.
            continue
        if first_iter_only and iter_n != 0:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping malformed trace line in %s", path)
                    continue
                if int(rec.get("iterations") or 0) <= 0:
                    # iters=0 placeholder (dead-endpoint dispatch)
                    continue
                qid = rec.get("question_id") or ""
                if not qid:
                    continue
                rollout = _Rollout(
                    qid=qid,
                    ensemble_idx=int(rec.get("ensemble_idx") or 0),
                    iter_dir=iter_n,
                    termination=rec.get("termination") or "",
                    final_answer=(rec.get("final_answer") or "").strip(),
                    nuggets=_extract_nuggets_from_trace(rec),
                    passages=_extract_pids_from_trace(rec),
                    raw_trace=rec,
                )
                # For the PG-1 (single-pass) build, we restrict to
                # iter_0 rollouts. Mirroring the live aggregator's
                # ``first_iter_only`` rescue (splice passages from later
                # iters when iter_0 is empty) is unnecessary here:
                # passage selection is locked to the frozen PR roster,
                # not derived from rollouts.
                by_qid[qid].append(rollout)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda r: r.ensemble_idx)
    return dict(by_qid)


def build_pid_index(
    rollouts_by_qid: dict[str, list[_Rollout]],
) -> dict[tuple[str, str, str], str]:
    """Build a ``(qid, doc_id, sanitized_text) -> pid`` lookup.

    Joins the frozen-PR roster (which lacks ``passage_id``) back to
    real passage_ids by string-matching on ``(doc_id, sanitized_text)``
    within the qid. Sources both ``reconciled_passages`` and the
    ``passages_db`` rescue path used by ``_extract_pids_from_trace``.

    Args:
        rollouts_by_qid: Output of ``load_rollouts_for_run``.

    Returns:
        Mapping with the first observed pid for each
        ``(qid, doc_id, sanitized_text)`` triple.
    """
    index: dict[tuple[str, str, str], str] = {}
    for qid, rollouts in rollouts_by_qid.items():
        for r in rollouts:
            for p in r.passages:
                pid = p.get("passage_id") or ""
                if not pid or pid.startswith("__refusal"):
                    continue
                doc_id = p.get("doc_id") or ""
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                key = (qid, doc_id, _sanitize(text))
                index.setdefault(key, pid)
    return index


def map_pr_rows_to_pids(
    pr: FrozenPR,
    pid_index: dict[tuple[str, str, str], str],
) -> dict[str, dict[str, int]]:
    """Build per-qid ``{pid: rank}`` map by joining frozen-PR rows with pid_index.

    Args:
        pr: Parsed frozen PR.
        pid_index: Output of ``build_pid_index``.

    Returns:
        ``{qid: {pid: rank}}``. Rows that fail to resolve to a pid are
        dropped from the per-qid map (their ranks are still in
        ``pr.rank_set(qid)`` for the subset invariant).
    """
    out: dict[str, dict[str, int]] = {}
    for qid, rows in pr.rows_by_qid.items():
        rank_map: dict[str, int] = {}
        for rank, doc_id, passage_text in rows:
            key = (qid, doc_id, passage_text)
            pid = pid_index.get(key)
            if pid is None:
                # Try with our own re-sanitization (PR text was sanitized
                # at write time, but if a producer ever changed the
                # sanitisation rule we still want to match).
                key2 = (qid, doc_id, _sanitize(passage_text))
                pid = pid_index.get(key2)
            if pid is not None:
                rank_map[pid] = rank
        out[qid] = rank_map
    return out


# v2 nugget selection


def _is_refusal_answer(answer: str) -> bool:
    """Use the same regex the live aggregator uses for cluster classification."""
    if not answer:
        return False
    return bool(REFUSAL_RE.search(answer))


def _phase_a_score(cand: _NuggetCandidate) -> float:
    """v2 score for a real (non-synthetic) nugget candidate.

    Formula: ``iter_weight + nli_bonus + co_bonus``.
    Higher is better. See module docstring R1/R2.
    """
    iter_weight = (
        PHASE_A_EARLY_ITER_WEIGHT
        if cand.earliest_ens <= PHASE_A_EARLY_ITER_THRESHOLD
        else PHASE_A_LATE_ITER_WEIGHT
    )
    nli_bonus = PHASE_A_NLI_BONUS if cand.nli_score > 0 else 0.0
    co_bonus = PHASE_A_CO_OCCURRENCE_BONUS * max(0, cand.co_occurrence - 1)
    return iter_weight + nli_bonus + co_bonus


def _norm_text_key(text: str) -> str:
    """Cluster key for nugget de-duplication (lower-case, whitespace-collapsed)."""
    return re.sub(r"\s+", " ", text.strip().lower())[:120]


def _phase_a_candidates(
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
) -> list[_NuggetCandidate]:
    """Collect Phase A (real-nugget) candidates whose pid maps to frozen PR.

    Nuggets are merged on ``(passage_id, normalised_text_prefix)`` so
    that the same nugget surfacing in multiple rollouts contributes to
    co-occurrence rather than counting as separate slots.
    """
    grouped: dict[tuple[str, str], _NuggetCandidate] = {}
    for r in rollouts:
        for n in r.nuggets:
            primary_pid = n.get("passage_id") or ""
            # Prefer the primary pid; fall back to any supporting pid that
            # is in the rank_map.
            pid: str | None = None
            if primary_pid in rank_map:
                pid = primary_pid
            else:
                for spid in n.get("supporting_pids") or []:
                    if spid in rank_map:
                        pid = spid
                        break
            if pid is None:
                continue
            text = (n.get("text") or "").strip()
            if not text:
                continue
            key = (pid, _norm_text_key(text))
            cand = grouped.get(key)
            nli_score = float(n.get("nli_score") or 0.0)
            for s in (n.get("pid_nli_scores") or {}).values():
                if s is not None:
                    nli_score = max(nli_score, float(s))
            if cand is None:
                grouped[key] = _NuggetCandidate(
                    passage_id=pid,
                    text=text,
                    nli_score=nli_score,
                    earliest_ens=r.ensemble_idx,
                    co_occurrence=1,
                    pr_rank=rank_map[pid],
                    is_synthetic=False,
                )
            else:
                cand.co_occurrence += 1
                if r.ensemble_idx < cand.earliest_ens:
                    cand.earliest_ens = r.ensemble_idx
                if nli_score > cand.nli_score:
                    cand.nli_score = nli_score
    return list(grouped.values())


def _synthesize_rescue_text(passage_text: str) -> str:
    """Pick the first sentence (≤ 200 chars) of a passage as a rescue nugget.

    Sanitised the same way ``postprocess._sanitize_passage_text`` does
    for PR rows so the AC text round-trips cleanly through the validator.

    Self-closing HTML void tags surviving in source PR rows (``<br>``,
    ``<br/>``, ``<hr>``) would trip ``check_ac_xml_tags`` (which parses
    any ``<X>`` as an XML-style block tag with no closing pair). These
    must be stripped here. Balanced HTML pairs (``<small>...</small>``,
    ``<i>...</i>``) pass through unchanged — the validator accepts them
    as balanced sub-tags.
    """
    text = _sanitize(passage_text)
    if not text:
        return ""
    # Strip self-closing HTML void tags only — leave balanced pairs alone
    # to preserve the AC-1 frozen byte-identity (which ships ``<small>``).
    text = re.sub(
        r"<\s*(?:br|hr|wbr|img|input)\b[^>]*/?\s*>", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # Prefer the first sentence (period+space+capital boundary).
    boundary = re.search(r"\.\s+(?=[A-Z(])", text[: RESCUE_NUGGET_TEXT_MAX + 60])
    if boundary and boundary.start() + 1 <= RESCUE_NUGGET_TEXT_MAX:
        return text[: boundary.start() + 1].strip()
    if len(text) <= RESCUE_NUGGET_TEXT_MAX:
        return text
    truncated = text[:RESCUE_NUGGET_TEXT_MAX]
    last_space = truncated.rfind(" ")
    if last_space > RESCUE_NUGGET_TEXT_MAX // 2:
        return truncated[:last_space].rstrip()
    return truncated.strip()


def _phase_b_candidates(
    qid: str,
    pr: FrozenPR,
    rank_map: dict[str, int],
) -> list[_NuggetCandidate]:
    """Synthesise refusal-rescue nuggets directly from frozen-PR rows.

    Walks frozen-PR rows in rank order; for each, synthesise a nugget
    from the row's passage text. The pid is resolved via ``rank_map``
    (reverse lookup); rows whose text never appeared in any rollout's
    ``reconciled_passages`` or ``passages_db`` (extreme edge case) get
    a synthetic ``__pr_rank_<rank>`` pid that is later threaded through
    a synthetic rank_map entry — see ``select_v2_nuggets``.
    """
    rank_to_pid = {rank: pid for pid, rank in rank_map.items()}
    out: list[_NuggetCandidate] = []
    for rank, _doc_id, passage_text in pr.rows_by_qid.get(qid, []):
        text = _synthesize_rescue_text(passage_text)
        if not text or len(text) < RESCUE_NUGGET_MIN_LEN:
            continue
        pid = rank_to_pid.get(rank) or f"__pr_rank_{rank}__{qid}"
        out.append(
            _NuggetCandidate(
                passage_id=pid,
                text=text,
                nli_score=0.0,
                earliest_ens=999,  # synthetic — never beats Phase A
                co_occurrence=0,
                pr_rank=rank,
                is_synthetic=True,
            )
        )
        if len(out) >= MAX_NUGGETS_PER_QID:
            break
    return out


def _tier_a_star_candidates(
    qid: str,
    seed_pids: list[str],
    rank_map: dict[str, int],
    pr: FrozenPR,
) -> list[_NuggetCandidate]:
    """Synthesise Tier-A* nugget candidates from the chosen attempt's citation_pids.

    For each pid in ``seed_pids`` that resolves to a frozen-PR rank,
    emit a nugget whose text is the first sentence (≤200 chars) of the
    matching frozen-PR row's passage text. Pids without a PR-rank entry
    are dropped (subset invariant safety).

    Args:
        qid: Question id (selects the PR roster slice).
        seed_pids: ``citation_pids`` from the chosen submit_answer
            attempt. Order is preserved.
        rank_map: ``{pid: pr_rank}`` for the qid.
        pr: Parsed frozen PR.

    Returns:
        List of ``_NuggetCandidate`` records. Caller deduplicates against
        Phase-A candidates by ``(passage_id, _norm_text_key(text))``.
    """
    out: list[_NuggetCandidate] = []
    if not seed_pids:
        return out
    rank_to_text: dict[int, str] = {
        rank: passage_text
        for rank, _doc_id, passage_text in pr.rows_by_qid.get(qid, [])
    }
    seen_ranks: set[int] = set()
    for pid in seed_pids:
        rank = rank_map.get(pid)
        if rank is None or rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        passage_text = rank_to_text.get(rank, "")
        text = _synthesize_rescue_text(passage_text)
        if not text or len(text) < RESCUE_NUGGET_MIN_LEN:
            continue
        out.append(
            _NuggetCandidate(
                passage_id=pid,
                text=text,
                # Tier-A* gets the standard NLI bonus so it can win
                # Phase-A scoring on its own merits.
                nli_score=TIER_A_STAR_NLI_BONUS,
                # Place in the early-iter weight bucket for parity with
                # iter_0-sourced Phase-A nuggets.
                earliest_ens=0,
                co_occurrence=1,
                pr_rank=rank,
                is_synthetic=False,
            )
        )
    return out


def select_v2_nuggets(
    qid: str,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    pr: FrozenPR,
    answer: str,
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
    seed_citation_pids: list[str] | None = None,
) -> tuple[list[_NuggetCandidate], list[str]]:
    """Apply v2 selection rules and return ranked nugget candidates.

    Args:
        qid: Question id.
        rollouts: All rollouts for ``qid`` for the run being rebuilt.
        rank_map: ``{pid: frozen_pr_rank}`` for ``qid``.
        pr: Frozen PR roster (used by Phase B rescue).
        answer: The frozen AC answer string for ``qid`` (drives the
            refusal-rescue gate).
        max_nuggets: R2C2 cap (default 10).
        seed_citation_pids: Optional list of ``citation_pids`` from a
            chosen ``submit_answer`` attempt (Tier-A* per the trace
            redesign §3). Only honoured when Phase-A produced no
            real-nugget candidates; pids without a PR-rank entry are
            dropped.

    Returns:
        ``(candidates, warnings)``: ranked candidates capped at
        ``max_nuggets``; warnings is a list of human-readable tags
        (``"phase_a_only"``, ``"phase_b_rescue"``, ``"empty"`` …).
    """
    warnings: list[str] = []
    a_cands = _phase_a_candidates(rollouts, rank_map)
    a_cands.sort(
        key=lambda c: (
            -_phase_a_score(c),
            -c.nli_score,
            -c.co_occurrence,
            c.earliest_ens,
            c.pr_rank,
        )
    )

    if a_cands:
        warnings.append("phase_a_only")
        return a_cands[:max_nuggets], warnings

    # Tier-A* — synthesise from chosen submit_answer's citation_pids
    # before falling through to the refusal-rescue branch.
    if seed_citation_pids:
        a_star = _tier_a_star_candidates(qid, seed_citation_pids, rank_map, pr)
        if a_star:
            a_star.sort(
                key=lambda c: (
                    -_phase_a_score(c),
                    -c.nli_score,
                    c.pr_rank,
                )
            )
            warnings.append("tier_a_star")
            return a_star[:max_nuggets], warnings

    # Phase A produced nothing — branch on whether the answer is the
    # canonical refusal.
    if _is_refusal_answer(answer):
        b_cands = _phase_b_candidates(qid, pr, rank_map)
        if b_cands:
            warnings.append("phase_b_rescue")
            return b_cands[:max_nuggets], warnings

    # Nothing to surface — emit an empty AC body. The validator
    # accepts empty blocks; ``require_nuggets`` is opt-in.
    warnings.append("empty")
    return [], warnings


# AC text emission


def _build_ac_result(
    qid: str,
    answer: str,
    confidence: int,
    candidates: list[_NuggetCandidate],
    rank_map: dict[str, int],
    run_name: str,
    pr: FrozenPR,
) -> ACResult:
    """Build an ``ACResult`` from candidates, threading them through ``format_ac_result``.

    Phase B rescue candidates whose ``pid`` is not present in the
    ``rank_map`` (extreme corner case where the frozen PR row has no
    matching rollout passage_id) are threaded via a synthetic rank_map
    entry — this keeps the call into ``format_ac_result`` honest while
    preserving the subset invariant (the rank itself is always a
    frozen-PR rank).
    """
    threaded_rank_map = dict(rank_map)
    nuggets: list[Nugget] = []
    for cand in candidates:
        if cand.passage_id not in threaded_rank_map:
            threaded_rank_map[cand.passage_id] = cand.pr_rank
        nuggets.append(Nugget(passage_id=cand.passage_id, text=cand.text))
    trace = QuestionTrace(
        question_id=qid,
        draft_answer=answer,
        final_confidence=confidence,
    )
    trace.nuggets.extend(nuggets)
    return format_ac_result(
        trace,
        run_name,
        threaded_rank_map,
        max_nuggets=MAX_NUGGETS_PER_QID,
    )


# Shared strategy helpers for AC-2, AC-3 and AC-4.


# Canonical refusal text used by AC-3 / AC-4 fallback paths. Matches the
# live runtime's force_deliver template.
CANONICAL_REFUSAL_TEXT = (
    "The corpus does not support an answer to this question; "
    "insufficient evidence to respond."
)


def _normalise_answer(text: str) -> str:
    """Normalise an answer string for cluster keying (AC-3 / AC-4).

    Distinct from ``_norm_text_key`` (which caps at 120 chars for nugget
    de-duplication). This helper returns sentinels for empty / refusal
    strings, strips Unicode/ASCII punctuation, collapses whitespace, and
    does NOT truncate — the qid 0022 lesson is that long enumerated
    answers must round-trip without surface-form collisions.

    Returns:
        ``"__empty__"`` if ``text`` is falsy after stripping,
        ``"__refusal__"`` if ``text`` matches the canonical refusal,
        otherwise a lower-cased, punctuation-stripped, whitespace-collapsed
        version of the input.
    """
    if not text:
        return "__empty__"
    stripped = text.strip()
    if not stripped:
        return "__empty__"
    if _is_canonical_refusal_text(stripped):
        return "__refusal__"
    out = stripped.lower()
    out = re.sub(r"[^\w\s]", " ", out, flags=re.UNICODE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _pool_attempts_across_iters(
    rollouts: list[_Rollout],
) -> list[tuple[_Rollout, _SubmitAttempt]]:
    """Pool every submit_answer attempt across iter_0..3 for one qid.

    Walks each rollout, runs ``_extract_submit_attempts_from_trace``, and
    yields ``(rollout, attempt)`` pairs in stable iter_dir/ensemble_idx
    order. Used by AC-3 (plurality) and AC-4 (LLM judge candidate build).
    """
    out: list[tuple[_Rollout, _SubmitAttempt]] = []
    for r in rollouts:
        attempts = _extract_submit_attempts_from_trace(r.raw_trace)
        for a in attempts:
            out.append((r, a))
    return out


def _count_verified_supporting_nuggets(rollout: _Rollout) -> int:
    """Count verified-supporting nuggets attached to a rollout (AC-2 scoring).

    ``rollout.nuggets`` is populated at load time by
    ``_extract_nuggets_from_trace``, which already filters synthetic +
    non-supporting + ``__refusal__`` pids. Returning ``len(...)`` is the
    canonical AC-2 nugget count.
    """
    return len(rollout.nuggets)


def _class_weight(attempt: _SubmitAttempt) -> float:
    """Map an attempt classification to its plurality cluster weight.

    AC-3 §3.1 calibration:
        - OK            → 1.0
        - REJ_HARDGATE  → 0.4
        - REJ_FORMAT    → 0.1
        - REFUSAL_TEXT  → 0.0 (filtered upstream)
        - unknown       → 0.0
    """
    return {
        ATTEMPT_OK: PLURALITY_CLASS_WEIGHT_OK,
        ATTEMPT_REJ_HARDGATE: PLURALITY_CLASS_WEIGHT_HARDGATE,
        ATTEMPT_REJ_FORMAT: PLURALITY_CLASS_WEIGHT_FORMAT,
    }.get(attempt.classification, 0.0)


def _dedupe_nuggets_by_pid_text(
    cands: list[_NuggetCandidate],
) -> list[_NuggetCandidate]:
    """Stable-dedupe a list of nugget candidates by ``(pid, text-key)``.

    Keeps the first occurrence. Text key reuses ``_norm_text_key`` (120-char
    cap, lower-cased, whitespace-collapsed).
    """
    seen: set[tuple[str, str]] = set()
    out: list[_NuggetCandidate] = []
    for c in cands:
        key = (c.passage_id, _norm_text_key(c.text))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _apply_per_iter_to_chosen_iter(
    chosen: _Rollout,
    frozen_answer: str,
    frozen_conf: int,
    rank_map: dict[str, int],
    qid: str,
) -> tuple[str, int, list[str] | None]:
    """Run the AC-1 per-iter trace builder on a chosen rollout.

    Returns ``(emit_answer, emit_confidence, seed_citation_pids)``.
    Falls back to ``(frozen_answer, frozen_conf, None)`` when the chosen
    iter has no committable non-refusal ``submit_answer`` attempt. The
    confidence floor (``max(tool_conf*100, frozen_conf)``) is enforced
    inside ``_pick_per_iter_answer``.

    This is the load-bearing shared helper for AC-1 / AC-2 / AC-4. It
    must remain byte-identical to the previous in-place block in
    ``build_v2_ac_text`` so the shipped AC-1 SHA does not drift.
    """
    attempts = _extract_submit_attempts_from_trace(chosen.raw_trace)
    picked = _pick_per_iter_answer(
        attempts,
        frozen_pr_answer=frozen_answer,
        frozen_pr_conf=int(frozen_conf),
        pr_rank_map=rank_map,
        qid=qid,
    )
    if picked is None:
        return frozen_answer, int(frozen_conf), None
    new_answer, new_confidence, citation_pids = picked
    return new_answer, new_confidence, list(citation_pids)


# Best-rollout helpers.


def _score_rollout(rollout: _Rollout) -> float:
    """AC-2 score for one rollout (additive: non_refusal_bonus + ok_sum + nug_n).

    Formula::

        score = (10 if final_answer is non-refusal else 0)
              + sum_{a in OK attempts}(a.confidence * a.n_eff)
              + count_of_verified_supporting_nuggets

    Missing ``a.confidence`` → 0.0; missing ``a.n_eff`` → 0. Reuses
    ``_is_canonical_refusal_text`` for refusal detection and
    ``_extract_submit_attempts_from_trace`` for attempt extraction.
    """
    non_refusal_bonus = (
        BEST_ROLLOUT_NON_REFUSAL_BONUS
        if not _is_canonical_refusal_text(rollout.final_answer)
        else 0.0
    )
    attempts = _extract_submit_attempts_from_trace(rollout.raw_trace)
    ok_sum = 0.0
    for a in attempts:
        if a.classification != ATTEMPT_OK:
            continue
        conf = a.confidence if a.confidence is not None else 0.0
        n_eff = a.n_eff if a.n_eff is not None else 0
        ok_sum += conf * n_eff
    nug_n = _count_verified_supporting_nuggets(rollout)
    return non_refusal_bonus + ok_sum + nug_n


def _pick_best_iter(
    rollouts: list[_Rollout],
    *,
    score_fn=_score_rollout,
) -> _Rollout | None:
    """Pick the highest-scoring rollout per AC-2.

    Tie-break: lowest ``iter_dir`` first, then lowest ``ensemble_idx``.
    Returns ``None`` if ``rollouts`` is empty or every rollout scores
    exactly 0.0 (the all-refusal-no-evidence corner case).

    The ``score_fn`` injection point lets AC-3 reuse this picker with a
    cluster-score function. Default = ``_score_rollout``.
    """
    if not rollouts:
        return None
    scored = [(score_fn(r), r.iter_dir, r.ensemble_idx, r) for r in rollouts]
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    top_score = scored[0][0]
    if top_score <= 0.0:
        return None
    return scored[0][3]


def _strategy_best_rollout_qid(
    qid: str,
    pr: FrozenPR,
    ac: FrozenAC,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> tuple[str, int, list[_NuggetCandidate], list[str]]:
    """Emit one qid's AC line under the AC-2 best-rollout strategy.

    Algorithm (plan §4.2 of ac2-best-rollout-plan.md):
      1. ``chosen = _pick_best_iter(rollouts)``.
      2. If ``chosen is None`` (empty rollouts OR all-zero score), fall
         back to the frozen ``(answer, confidence)`` and let
         ``select_v2_nuggets`` rescue via Phase B.
      3. Else apply ``_apply_per_iter_to_chosen_iter`` and restrict nugget
         selection to ``[chosen]`` — Phase-A scoring only sees the chosen
         iter's nuggets.

    Returns ``(emit_answer, emit_confidence, candidates, warnings)``.
    """
    frozen_answer, frozen_conf = ac.answers_by_qid.get(qid, ("", 0))
    warnings: list[str] = []

    if not rollouts:
        warnings.append("no_rollouts")
        candidates, w2 = select_v2_nuggets(
            qid,
            [],
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    chosen = _pick_best_iter(rollouts)
    if chosen is None:
        warnings.append("all_zero_score")
        warnings.append("frozen_fallback")
        candidates, w2 = select_v2_nuggets(
            qid,
            [],
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    emit_answer, emit_conf, seed_pids = _apply_per_iter_to_chosen_iter(
        chosen,
        frozen_answer,
        int(frozen_conf),
        rank_map,
        qid,
    )
    warnings.append(f"best_rollout_iter_{chosen.iter_dir}")

    candidates, w2 = select_v2_nuggets(
        qid,
        [chosen],
        rank_map,
        pr,
        emit_answer,
        seed_citation_pids=seed_pids,
        max_nuggets=max_nuggets,
    )
    warnings.extend(w2)
    return emit_answer, emit_conf, candidates, warnings


# Plurality helpers.


def _score_attempt(attempt: _SubmitAttempt) -> float:
    """Score one submit_answer attempt for plurality clustering.

    Formula: ``class_weight × n_eff_with_floor × tool_conf``.

    ``n_eff_with_floor`` policy: ``n_eff=None`` → 1 for OK attempts only
    (model self-asserts evidence), 0 for everything else. Documented in
    AC-3 §5.2 — load-bearing for qid 0022 (the all-OK-no-n_eff cluster).
    """
    if attempt.n_eff is not None:
        n = attempt.n_eff
    elif attempt.classification == ATTEMPT_OK:
        n = 1
    else:
        n = 0
    conf = attempt.confidence if attempt.confidence is not None else 0.0
    return _class_weight(attempt) * n * conf


def _cluster_attempts(
    pairs: list[tuple[_Rollout, _SubmitAttempt]],
) -> dict[str, list[tuple[_Rollout, _SubmitAttempt]]]:
    """Group ``(rollout, attempt)`` pairs by normalised answer text.

    Drops the ``"__refusal__"`` and ``"__empty__"`` clusters — those
    should never win. Returns the surviving groups keyed by the
    ``_normalise_answer`` output.
    """
    out: dict[str, list[tuple[_Rollout, _SubmitAttempt]]] = {}
    for r, a in pairs:
        key = _normalise_answer(a.answer_text)
        if key in ("__refusal__", "__empty__"):
            continue
        out.setdefault(key, []).append((r, a))
    return out


def _score_cluster(cluster: list[tuple[_Rollout, _SubmitAttempt]]) -> float:
    """Sum ``_score_attempt`` over a cluster's members."""
    return sum(_score_attempt(a) for _, a in cluster)


def _pick_cluster_representative(
    cluster: list[tuple[_Rollout, _SubmitAttempt]],
) -> tuple[_Rollout, _SubmitAttempt]:
    """Pick the canonical (rollout, attempt) representative of a cluster.

    Tie-break: highest ``tool_conf``, then highest ``n_eff``, then lowest
    ``iter_dir``, then lowest ``ts_idx``. Documented in AC-3 §3.2.
    """

    def _key(t: tuple[_Rollout, _SubmitAttempt]) -> tuple[Any, ...]:
        r, a = t
        conf = a.confidence if a.confidence is not None else -1.0
        n_eff = a.n_eff if a.n_eff is not None else -1
        return (-conf, -n_eff, r.iter_dir, a.ts_idx)

    return sorted(cluster, key=_key)[0]


def _aggregate_cluster_confidence(
    cluster: list[tuple[_Rollout, _SubmitAttempt]],
    frozen_pr_conf: int,
) -> int:
    """Aggregate cluster confidence per AC-3 §4.4.

    Returns ``min(100, max(round(Σ 100·tool_conf·class_weight), frozen_pr_conf))``.
    The min(100,) is the safety clamp for high-K agreement; the
    max(.., frozen_pr_conf) preserves the frozen-AC floor.
    """
    raw = 0.0
    for _, a in cluster:
        conf = a.confidence if a.confidence is not None else 0.0
        raw += 100.0 * conf * _class_weight(a)
    summed = int(round(raw))
    out = max(summed, int(frozen_pr_conf))
    return max(0, min(100, out))


def _pool_cluster_nuggets(
    cluster: list[tuple[_Rollout, _SubmitAttempt]],
    rank_map: dict[str, int],
    pr: FrozenPR,
    qid: str,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> list[_NuggetCandidate]:
    """Pool nuggets from the winning cluster's rollouts + their citation_pids.

    Generalises ``select_v2_nuggets`` (which gates Tier-A* behind "Phase
    A empty"): for AC-3 we always pool both because the winning cluster's
    citation_pids encode the model's voted evidence. Dedup on
    ``(pid, _norm_text_key(text))``. Sort by Phase-A score and cap at
    ``max_nuggets``.
    """
    cluster_rollouts = list({id(r): r for r, _ in cluster}.values())
    phase_a = _phase_a_candidates(cluster_rollouts, rank_map)

    seed_pids: list[str] = []
    seen_pids: set[str] = set()
    for _, a in cluster:
        for pid in a.citation_pids:
            if pid in rank_map and pid not in seen_pids:
                seed_pids.append(pid)
                seen_pids.add(pid)
    tier_a_star = _tier_a_star_candidates(qid, seed_pids, rank_map, pr)

    combined = phase_a + tier_a_star
    deduped = _dedupe_nuggets_by_pid_text(combined)
    deduped.sort(
        key=lambda c: (
            -_phase_a_score(c),
            -c.nli_score,
            -c.co_occurrence,
            c.earliest_ens,
            c.pr_rank,
        )
    )
    return deduped[:max_nuggets]


def _refusal_fallback_emit(
    qid: str,
    pr: FrozenPR,
    rank_map: dict[str, int],
    frozen_answer: str,
    frozen_conf: int,
    warning_tag: str,
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> tuple[str, int, list[_NuggetCandidate], list[str]]:
    """Emit the canonical-refusal fallback line per AC-3 §4.6.

    When the frozen AC already carries a refusal, preserve frozen
    answer + conf; otherwise emit the canonical refusal text at
    ``REFUSAL_FLOOR_CONF`` (25). Nuggets come from Phase B (frozen-PR
    first-sentence rescue).
    """
    warnings: list[str] = [warning_tag]
    if _is_refusal_answer(frozen_answer):
        emit_answer = frozen_answer
        emit_conf = int(frozen_conf)
    else:
        emit_answer = CANONICAL_REFUSAL_TEXT
        emit_conf = REFUSAL_FLOOR_CONF
    candidates, w2 = select_v2_nuggets(
        qid,
        [],
        rank_map,
        pr,
        emit_answer,
        seed_citation_pids=None,
        max_nuggets=max_nuggets,
    )
    warnings.extend(w2)
    return emit_answer, emit_conf, candidates, warnings


def _strategy_plurality_qid(
    qid: str,
    pr: FrozenPR,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    frozen_answer: str,
    frozen_conf: int,
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> tuple[str, int, list[_NuggetCandidate], list[str]]:
    """Emit one qid's AC line under the AC-3 plurality strategy.

    Algorithm (plan §4.1):
      1. Pool all submit_answer attempts across iter_0..3.
      2. Filter out canonical-refusal attempts.
      3. Cluster by ``_normalise_answer``.
      4. Score each cluster; sort desc.
      5. If top score < ``PLURALITY_MIN_CLUSTER_SCORE`` → refusal fallback.
      6. Pick representative; emit ``(answer, conf, candidates, warnings)``.

    Returns ``(emit_answer, emit_confidence, candidates, warnings)``.
    """
    pairs = _pool_attempts_across_iters(rollouts)
    if not pairs:
        return _refusal_fallback_emit(
            qid,
            pr,
            rank_map,
            frozen_answer,
            frozen_conf,
            "no_attempts",
            max_nuggets=max_nuggets,
        )

    non_refusal = [(r, a) for r, a in pairs if a.classification != ATTEMPT_REFUSAL_TEXT]
    if not non_refusal:
        return _refusal_fallback_emit(
            qid,
            pr,
            rank_map,
            frozen_answer,
            frozen_conf,
            "all_refusal",
            max_nuggets=max_nuggets,
        )

    clusters = _cluster_attempts(non_refusal)
    if not clusters:
        return _refusal_fallback_emit(
            qid,
            pr,
            rank_map,
            frozen_answer,
            frozen_conf,
            "empty_after_normalise",
            max_nuggets=max_nuggets,
        )

    # Score + tie-break order: -score, -max(tool_conf), -max(n_eff), key.
    scored: list[
        tuple[float, float, int, str, list[tuple[_Rollout, _SubmitAttempt]]]
    ] = []
    for key, members in clusters.items():
        score = _score_cluster(members)
        max_conf = max(
            (a.confidence if a.confidence is not None else 0.0) for _, a in members
        )
        max_neff = max((a.n_eff if a.n_eff is not None else 0) for _, a in members)
        scored.append((score, max_conf, max_neff, key, members))
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))

    top_score, _max_conf, _max_neff, top_key, top_members = scored[0]

    if top_score < PLURALITY_MIN_CLUSTER_SCORE:
        warning_tag = f"below_threshold:{top_score:.2f}"
        return _refusal_fallback_emit(
            qid,
            pr,
            rank_map,
            frozen_answer,
            frozen_conf,
            warning_tag,
            max_nuggets=max_nuggets,
        )

    _rep_r, rep_a = _pick_cluster_representative(top_members)
    answer = _sanitize_ac_answer(rep_a.answer_text)
    if not answer:
        # Try fallback clusters in score order.
        chosen_members: list[tuple[_Rollout, _SubmitAttempt]] | None = None
        for entry in scored[1:]:
            sscore, _smc, _smn, _skey, smembers = entry
            if sscore < PLURALITY_MIN_CLUSTER_SCORE:
                break
            _sr, sa = _pick_cluster_representative(smembers)
            cand_ans = _sanitize_ac_answer(sa.answer_text)
            if cand_ans:
                answer = cand_ans
                chosen_members = smembers
                break
        if not answer:
            return _refusal_fallback_emit(
                qid,
                pr,
                rank_map,
                frozen_answer,
                frozen_conf,
                "sanitised_empty",
                max_nuggets=max_nuggets,
            )
        if chosen_members is not None:
            top_members = chosen_members

    confidence = _aggregate_cluster_confidence(top_members, int(frozen_conf))
    candidates = _pool_cluster_nuggets(
        top_members,
        rank_map,
        pr,
        qid,
        max_nuggets=max_nuggets,
    )

    warnings = [
        f"cluster_winner:{top_key[:40]}",
        f"score:{top_score:.2f}",
        f"n_members:{len(top_members)}",
    ]
    # Tier-A* / Phase-A reuse already accounts for subset filter; no extra warns.
    return answer, confidence, candidates, warnings


# Max-conf (AC-2): order-statistic variance reduction.
#
# Across all K rollouts, pick the single highest-tool_conf non-refusal
# submit_answer attempt. Sanitise the answer, restrict citations to the
# frozen PR roster, emit confidence = round(tool_conf * 100). When no
# non-refusal attempt exists across any iter, fall back to frozen
# (answer, conf) and let Phase B rescue the nuggets.
#
# Motivation (closed-form, parameter-free): for a noisy estimator
# (sampling-based LLM decoding) drawn K times, the max-confidence sample
# is the standard order-statistic point estimate. It uses no hand-tuned
# class weights and no n_eff floor; the model's self-reported tool_conf
# is treated as the only ranking signal among non-refusal attempts.


def _strategy_run1_max_conf_qid(
    qid: str,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    pr: FrozenPR,
    frozen_answer: str,
    frozen_conf: int,
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> tuple[str, int, list[_NuggetCandidate], list[str]]:
    """Run 1 — order-statistic variance reduction (replaces legacy AC-2).

    Algorithm:
      1. Pool every ``submit_answer`` attempt across iter_0..K-1.
      2. Filter out canonical-refusal attempts.
      3. ``argmax(attempts, key=tool_conf)`` — break ties by message order.
      4. Sanitise answer text via ``_sanitize_ac_answer`` (``;`` → ``,``).
      5. Confidence = ``round(tool_conf × 100)`` clamped to ``[0, 100]``.
         No frozen-floor.
      6. Citations are the chosen attempt's ``citation_pids`` intersected
         with the frozen PR-2 roster (subset invariant).
      7. If no non-refusal attempt exists in any iter → fall back to
         ``(frozen_answer, frozen_conf)`` and Phase-B rescue nuggets.

    Returns ``(emit_answer, emit_confidence, candidates, warnings)``.
    """
    warnings: list[str] = []
    pairs = _pool_attempts_across_iters(rollouts)
    non_refusal = [
        (_r, a) for _r, a in pairs if a.classification != ATTEMPT_REFUSAL_TEXT
    ]
    if not non_refusal:
        warnings.append("max_conf_no_attempts")
        candidates, w2 = select_v2_nuggets(
            qid,
            rollouts,
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    def _sort_key(item: tuple[_Rollout, _SubmitAttempt]) -> tuple[Any, ...]:
        _r, a = item
        conf = a.confidence if a.confidence is not None else -1.0
        # Descending confidence; ties → earliest message index for stability.
        return (-conf, a.ts_idx)

    non_refusal_sorted = sorted(non_refusal, key=_sort_key)
    _chosen_rollout, chosen = non_refusal_sorted[0]

    answer = _sanitize_ac_answer(chosen.answer_text)
    if not answer:
        warnings.append("max_conf_sanitised_empty")
        candidates, w2 = select_v2_nuggets(
            qid,
            rollouts,
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    tool_conf = chosen.confidence if chosen.confidence is not None else 0.0
    confidence = int(round(tool_conf * 100))
    confidence = max(0, min(100, confidence))

    seed_pids = [p for p in chosen.citation_pids if p in rank_map]
    warnings.append(f"max_conf_winner_conf:{tool_conf:.2f}")

    candidates, w2 = select_v2_nuggets(
        qid,
        rollouts,
        rank_map,
        pr,
        answer,
        seed_citation_pids=seed_pids or None,
        max_nuggets=max_nuggets,
    )
    warnings.extend(w2)
    return answer, confidence, candidates, warnings


# Conf-sum-cluster (AC-3): agreement-as-evidence accumulation.
#
# Pool every submit_answer attempt across iter_0..K-1, filter refusals,
# cluster by ``_normalise_answer`` (case-fold + punct-strip), and score
# each cluster as ``Σ tool_conf × 100`` summed across all its members.
# The winning cluster's max-tool_conf attempt's sanitised text is the
# answer; confidence is ``min(100, round(cluster_score))``; citations
# are the union of cluster-member citation_pids intersected with the
# frozen PR-2 roster.
#
# Motivation (closed-form, parameter-free): under an evidence-pooling
# model, independent rollouts converging on the same answer accumulate
# log-likelihood. Summing per-rollout tool_conf is the simplest scalar
# proxy. The cap-at-100 is the natural bound for the AC confidence field
# (no calibration drift, no class weights).


def _strategy_run3_conf_sum_cluster_qid(
    qid: str,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    pr: FrozenPR,
    frozen_answer: str,
    frozen_conf: int,
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> tuple[str, int, list[_NuggetCandidate], list[str]]:
    """Run 3 — agreement-as-evidence accumulation (replaces legacy AC-3).

    Algorithm:
      1. Pool ``submit_answer`` attempts across iter_0..K-1.
      2. Filter out canonical-refusal attempts.
      3. Cluster by ``_normalise_answer`` (case-fold + strip punct).
      4. ``cluster_score = Σ (tool_conf × 100)`` across cluster members.
      5. Pick the cluster with the highest score (ties → earliest
         ``ts_idx`` of the cluster's first member, then lex key).
      6. Answer = sanitised text of the cluster's max-tool_conf attempt.
      7. Confidence = ``min(100, round(cluster_score))`` — no frozen-floor.
      8. Citations = union of cluster-member ``citation_pids`` ∩
         frozen PR roster (preserves subset invariant).

    Falls back to ``(frozen_answer, frozen_conf)`` + Phase-B rescue
    nuggets when no non-refusal attempt exists across any iter.

    Returns ``(emit_answer, emit_confidence, candidates, warnings)``.
    """
    warnings: list[str] = []
    pairs = _pool_attempts_across_iters(rollouts)
    non_refusal = [
        (_r, a) for _r, a in pairs if a.classification != ATTEMPT_REFUSAL_TEXT
    ]
    if not non_refusal:
        warnings.append("conf_sum_no_attempts")
        candidates, w2 = select_v2_nuggets(
            qid,
            rollouts,
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    # Group by normalised answer key; drop empty/refusal sentinels.
    buckets: dict[str, list[tuple[_Rollout, _SubmitAttempt]]] = collections.defaultdict(
        list
    )
    for r, a in non_refusal:
        key = _normalise_answer(a.answer_text)
        if key in ("__refusal__", "__empty__"):
            continue
        buckets[key].append((r, a))

    if not buckets:
        warnings.append("conf_sum_empty_after_normalise")
        candidates, w2 = select_v2_nuggets(
            qid,
            rollouts,
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    def _cluster_score(members: list[tuple[_Rollout, _SubmitAttempt]]) -> float:
        # R2C2_DEDUPE_ATTEMPTS=1 counts one attempt per rollout per cluster, the highest-confidence
        # one, instead of every attempt the rollout made. Without it a rollout that had a submission
        # refused by the answer gate and then resubmitted the same string contributes twice, so the
        # gate's own rejection adds to the score of the answer it rejected. Measured on the filed
        # trace bundle: question 0029's `rock hammer` scores 77.1 + 100.0 = 177.1 against `movie
        # poster`'s 100.0, and the whole of that margin is the refused attempt.
        if os.environ.get("R2C2_DEDUPE_ATTEMPTS") == "1":
            best: dict[int, _SubmitAttempt] = {}
            for _r, a in members:
                k = id(_r)
                c = a.confidence if a.confidence is not None else -1.0
                prev = best.get(k)
                pc = (prev.confidence if prev and prev.confidence is not None else -1.0)
                if prev is None or c > pc:
                    best[k] = a
            return sum(
                100.0 * (a.confidence if a.confidence is not None else 0.0)
                for a in best.values()
            )
        return sum(
            100.0 * (a.confidence if a.confidence is not None else 0.0)
            for _r, a in members
        )

    scored = [
        (key, members, _cluster_score(members)) for key, members in buckets.items()
    ]
    # Sort: desc score; tie-break on earliest member ts_idx; then lex key.
    scored.sort(key=lambda t: (-t[2], min(a.ts_idx for _r, a in t[1]), t[0]))

    top_key, top_members, top_score = scored[0]

    def _rep_key(item: tuple[_Rollout, _SubmitAttempt]) -> tuple[Any, ...]:
        _r, a = item
        conf = a.confidence if a.confidence is not None else -1.0
        return (-conf, a.ts_idx)

    members_sorted = sorted(top_members, key=_rep_key)
    _rep_rollout, rep_attempt = members_sorted[0]
    answer = _sanitize_ac_answer(rep_attempt.answer_text)
    if not answer:
        # Try other clusters in score order until one yields a sanitised
        # answer; otherwise fall back to frozen.
        answer = ""
        for key, members, _score in scored[1:]:
            for _r, a in sorted(members, key=_rep_key):
                cand_ans = _sanitize_ac_answer(a.answer_text)
                if cand_ans:
                    answer = cand_ans
                    top_key = key
                    top_members = members
                    top_score = _score
                    break
            if answer:
                break
        if not answer:
            warnings.append("conf_sum_sanitised_empty")
            candidates, w2 = select_v2_nuggets(
                qid,
                rollouts,
                rank_map,
                pr,
                frozen_answer,
                seed_citation_pids=None,
                max_nuggets=max_nuggets,
            )
            warnings.extend(w2)
            return frozen_answer, int(frozen_conf), candidates, warnings

    confidence = int(round(top_score))
    confidence = max(0, min(100, confidence))

    # Union of citation_pids across cluster members, preserving order
    # of first appearance and restricted to the frozen PR roster.
    seed_pids: list[str] = []
    seen_pids: set[str] = set()
    for _r, a in top_members:
        for pid in a.citation_pids:
            if pid in rank_map and pid not in seen_pids:
                seed_pids.append(pid)
                seen_pids.add(pid)

    warnings.append(f"conf_sum_cluster:{top_key[:40]}")
    warnings.append(f"score:{top_score:.2f}")
    warnings.append(f"n_members:{len(top_members)}")

    candidates, w2 = select_v2_nuggets(
        qid,
        rollouts,
        rank_map,
        pr,
        answer,
        seed_citation_pids=seed_pids or None,
        max_nuggets=max_nuggets,
    )
    warnings.extend(w2)
    return answer, confidence, candidates, warnings


# LLM-judge helpers (AC-4).


def _trace_redesign_emit(
    frozen_answer: str,
    frozen_confidence: int,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    qid: str,
    *,
    apply: bool,
) -> tuple[str, int, list[str] | None]:
    """Per-qid PG-1 trace-redesign emit (extracted from build_v2_ac_text).

    Returns ``(emit_answer, emit_confidence, seed_citation_pids)``. When
    ``apply=False`` (PG-2 / legacy strategy), the input frozen values are
    passed through verbatim.

    Refactor target for the AC-1 byte-identity invariant: this helper
    MUST produce ``(frozen_answer, int(frozen_confidence), None)`` in the
    same cases the old inline block did.
    """
    if not apply:
        return frozen_answer, int(frozen_confidence), None
    if not _is_refusal_answer(frozen_answer) or not rollouts:
        return frozen_answer, int(frozen_confidence), None
    # PG-1 has K=1 (iter_0 only) — primary = lowest (iter_dir, ensemble_idx).
    primary = min(rollouts, key=lambda r: (r.iter_dir, r.ensemble_idx))
    new_answer, new_conf, seed_pids = _apply_per_iter_to_chosen_iter(
        primary,
        frozen_answer,
        int(frozen_confidence),
        rank_map,
        qid,
    )
    # When _pick_per_iter_answer returns None the helper hands back
    # (frozen_answer, int(frozen_conf), None); pass it through unchanged so
    # the emitted line stays byte-identical.
    if seed_pids is None and new_answer == frozen_answer:
        return frozen_answer, int(frozen_confidence), None
    return new_answer, new_conf, seed_pids


def _strategy_llm_judge_qid(
    qid: str,
    pr: FrozenPR,
    rollouts: list[_Rollout],
    rank_map: dict[str, int],
    frozen_answer: str,
    frozen_conf: int,
    decision,  # type-erased — see r2c2_agent.orchestrator.llm_judge.JudgeDecision
    *,
    max_nuggets: int = MAX_NUGGETS_PER_QID,
) -> tuple[str, int, list[_NuggetCandidate], list[str]]:
    """Emit one qid's AC line from a pre-computed LLM judge decision.

    Algorithm:
      - If no decision or ``decision.parse_failed`` → frozen fallback.
        The PG-2 guardrail at ``build_v2_ac_text`` rejects this case
        before we ever get here, but the fallback preserves shape for
        the development-only ``allow_incomplete_judge`` path.
      - Else use ``decision.final_answer`` + ``decision.final_confidence``
        as the emitted answer and seed nugget selection with
        ``decision.final_citation_pids``. The redesigned judge always
        emits its own ``(answer, confidence, evidence_pids)`` via a
        tool call — no enum decoding is needed.
    """
    warnings: list[str] = []
    if decision is None or getattr(decision, "parse_failed", False):
        warnings.append("judge_fallback_to_frozen")
        candidates, w2 = select_v2_nuggets(
            qid,
            rollouts,
            rank_map,
            pr,
            frozen_answer,
            seed_citation_pids=None,
            max_nuggets=max_nuggets,
        )
        warnings.extend(w2)
        return frozen_answer, int(frozen_conf), candidates, warnings

    emit_answer = decision.final_answer or frozen_answer
    emit_conf = int(decision.final_confidence)
    seed_pids = (
        list(decision.final_citation_pids) if decision.final_citation_pids else None
    )
    final_choice = getattr(decision, "final_choice", "") or "tool_call"
    warnings.append(f"judge_choice_{final_choice}")
    candidates, w2 = select_v2_nuggets(
        qid,
        rollouts,
        rank_map,
        pr,
        emit_answer,
        seed_citation_pids=seed_pids,
        max_nuggets=max_nuggets,
    )
    warnings.extend(w2)
    return emit_answer, emit_conf, candidates, warnings


def build_v2_ac_text(
    pr: FrozenPR,
    ac: FrozenAC,
    rollouts_by_qid: dict[str, list[_Rollout]],
    *,
    apply_trace_redesign: bool = False,
    strategy: str = STRATEGY_TRACE_REDESIGN,
    judge_decisions: dict[str, Any] | None = None,
    allow_incomplete_judge: bool = False,
) -> tuple[str, dict[str, list[_NuggetCandidate]], dict[str, list[str]]]:
    """Top-level: emit v2 AC content + per-qid candidate list + warnings.

    Args:
        pr: Parsed frozen PR.
        ac: Parsed frozen AC (used for ``Answer;Confidence`` line).
        rollouts_by_qid: Output of ``load_rollouts_for_run``.
        apply_trace_redesign: When True (PG-1 only), surface the model's
            actual ``submit_answer`` attempt for refusal-qids.
        strategy: One of ``ALLOWED_STRATEGIES``. Default
            ``trace-redesign`` preserves the AC-1 ship behaviour. PG-1
            (``apply_trace_redesign=True``) always uses trace-redesign
            regardless of this flag — the other strategies reduce to
            trace-redesign on K=1.
        judge_decisions: Required when ``strategy=llm-judge`` — mapping
            ``qid -> JudgeDecision`` (see
            ``r2c2_agent.orchestrator.llm_judge``).
        allow_incomplete_judge: Development-only escape hatch. When
            False (default, production), ``strategy=llm-judge`` on PG-2
            hard-fails with ``SystemExit`` unless every PR qid has a
            real, non-parse-failed LLM judge decision. Never set to
            True for any AC-4 shipped to organisers.

    Returns:
        ``(ac_text, candidates_by_qid, warnings_by_qid)``.
    """
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    # PG-1 (``apply_trace_redesign=True``) always reduces to trace-redesign
    # regardless of ``strategy`` — see effective_strategy below. The guardrail
    # only applies to PG-2 (where the LLM judge actually fires per-qid).
    #
    # A "real" decision is one where the LLM actually returned a parseable
    # JSON pick — ``parse_failed=False`` and no degenerate-fallback marker on
    # ``fallback_reason``. Degenerate fallbacks (``parse_failed``,
    # ``endpoint_pending``, anything indicating "no real LLM call happened")
    # are not acceptable for a shipped AC-4.
    #
    # Note: legitimate-call fallbacks like ``invalid_choice:foo``,
    # ``candidate_<i>_unavailable``, or
    # ``refusal_overridden_by_non_refusal_default`` still count as real
    # decisions — the LLM did respond; the finaliser just re-mapped its
    # pick. Only the markers below indicate no real LLM call happened.
    if (
        strategy == STRATEGY_LLM_JUDGE
        and not apply_trace_redesign
        and not allow_incomplete_judge
    ):
        decisions = judge_decisions or {}
        all_qids = sorted(pr.qids())
        n_total = len(all_qids)
        # Markers that indicate "no real LLM call happened" — reject these.
        degenerate_fallbacks = frozenset(
            {"parse_failed", "endpoint_pending", "deferred"}
        )
        missing: list[str] = []
        for qid in all_qids:
            dec = decisions.get(qid)
            if dec is None:
                missing.append(qid)
                continue
            if getattr(dec, "parse_failed", False):
                missing.append(qid)
                continue
            fb = getattr(dec, "fallback_reason", None)
            if fb in degenerate_fallbacks:
                missing.append(qid)
                continue
        if not decisions:
            raise SystemExit(
                "[FATAL] --strategy llm-judge requires --judge-decisions with "
                "valid entries; none found. Aborting AC-4 emission rather than "
                "silently falling back to frozen PG-2 for every qid. To override "
                "for development only, set allow_incomplete_judge=True."
            )
        if missing:
            n_with_real_choice = n_total - len(missing)
            raise SystemExit(
                f"[FATAL] LLM-judge strategy requires a real LLM decision for "
                f"every qid. Have {n_with_real_choice}/{n_total}. "
                f"Missing/parse_failed: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}. "
                f"Refusing to emit a degenerate BITEM-AC-4 that duplicates "
                f"frozen PG-2 for the missing qids."
            )

    pid_index = build_pid_index(rollouts_by_qid)
    rank_maps = map_pr_rows_to_pids(pr, pid_index)

    ac_results: dict[str, ACResult] = {}
    candidates_by_qid: dict[str, list[_NuggetCandidate]] = {}
    warnings_by_qid: dict[str, list[str]] = {}

    # PG-1 (apply_trace_redesign=True) always uses trace-redesign — the
    # other strategies reduce to it on K=1 (iter_0 only).
    effective_strategy = STRATEGY_TRACE_REDESIGN if apply_trace_redesign else strategy

    for qid in sorted(pr.qids()):
        frozen_answer, frozen_confidence = ac.answers_by_qid.get(qid, ("", 0))
        rollouts = rollouts_by_qid.get(qid, [])
        rank_map = rank_maps.get(qid, {})

        if effective_strategy == STRATEGY_TRACE_REDESIGN:
            emit_answer, emit_confidence, seed_citation_pids = _trace_redesign_emit(
                frozen_answer,
                frozen_confidence,
                rollouts,
                rank_map,
                qid,
                apply=apply_trace_redesign,
            )
            candidates, warnings = select_v2_nuggets(
                qid,
                rollouts,
                rank_map,
                pr,
                emit_answer,
                seed_citation_pids=seed_citation_pids,
            )
        elif effective_strategy == STRATEGY_BEST_ROLLOUT:
            emit_answer, emit_confidence, candidates, warnings = (
                _strategy_best_rollout_qid(
                    qid,
                    pr,
                    ac,
                    rollouts,
                    rank_map,
                    max_nuggets=MAX_NUGGETS_PER_QID,
                )
            )
        elif effective_strategy == STRATEGY_PLURALITY:
            emit_answer, emit_confidence, candidates, warnings = (
                _strategy_plurality_qid(
                    qid,
                    pr,
                    rollouts,
                    rank_map,
                    frozen_answer,
                    int(frozen_confidence),
                    max_nuggets=MAX_NUGGETS_PER_QID,
                )
            )
        elif effective_strategy == STRATEGY_LLM_JUDGE:
            decision = (judge_decisions or {}).get(qid)
            emit_answer, emit_confidence, candidates, warnings = (
                _strategy_llm_judge_qid(
                    qid,
                    pr,
                    rollouts,
                    rank_map,
                    frozen_answer,
                    int(frozen_confidence),
                    decision,
                    max_nuggets=MAX_NUGGETS_PER_QID,
                )
            )
        elif effective_strategy == STRATEGY_MAX_CONF:
            emit_answer, emit_confidence, candidates, warnings = (
                _strategy_run1_max_conf_qid(
                    qid,
                    rollouts,
                    rank_map,
                    pr,
                    frozen_answer,
                    int(frozen_confidence),
                    max_nuggets=MAX_NUGGETS_PER_QID,
                )
            )
        elif effective_strategy == STRATEGY_CONF_SUM_CLUSTER:
            emit_answer, emit_confidence, candidates, warnings = (
                _strategy_run3_conf_sum_cluster_qid(
                    qid,
                    rollouts,
                    rank_map,
                    pr,
                    frozen_answer,
                    int(frozen_confidence),
                    max_nuggets=MAX_NUGGETS_PER_QID,
                )
            )
        else:  # pragma: no cover — guarded by ALLOWED_STRATEGIES at top.
            raise ValueError(f"unknown strategy {effective_strategy!r}")

        ac_results[qid] = _build_ac_result(
            qid,
            emit_answer,
            emit_confidence,
            candidates,
            rank_map,
            pr.run_name,
            pr,
        )
        candidates_by_qid[qid] = candidates
        warnings_by_qid[qid] = warnings

    ac_text = format_ac_file(ac_results)
    return ac_text, candidates_by_qid, warnings_by_qid


# Validation


@dataclass
class ValidationOutcome:
    """Result of one validation run for one (PR, AC) pair."""

    run_name: str
    spec_checks: list[tuple[str, bool, str]]
    spec_passed: int
    spec_failed: int
    pr_errors: list[str]
    ac_errors: list[str]
    subset_violations: list[str]

    @property
    def total_checks(self) -> int:
        return len(self.spec_checks)

    @property
    def all_pass(self) -> bool:
        return (
            self.spec_failed == 0
            and not self.pr_errors
            and not self.ac_errors
            and not self.subset_violations
        )


def _check_subset_invariant(
    ac_text: str,
    pr: FrozenPR,
    run_name: str,
) -> list[str]:
    """Hard invariant: every AC PassageRank exists in the frozen PR for that qid."""
    violations: list[str] = []
    cur_qid: str | None = None
    answer_seen = False
    for line_num, raw in enumerate(ac_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        open_m = _AC_OPEN_TAG_RE.match(line)
        close_m = _AC_CLOSE_TAG_RE.match(line)
        if open_m:
            cur_qid = open_m.group(1)
            answer_seen = False
            continue
        if close_m:
            cur_qid = None
            continue
        if cur_qid is None:
            continue
        if not answer_seen:
            answer_seen = True
            continue
        # nugget line: ``num;run;rank;text`` — split with maxsplit=3
        parts = line.split(";", 3)
        if len(parts) < 4:
            violations.append(
                f"line {line_num} ({cur_qid}): malformed nugget {line[:60]!r}"
            )
            continue
        run_field = parts[1]
        try:
            rank = int(parts[2])
        except ValueError:
            violations.append(
                f"line {line_num} ({cur_qid}): non-integer rank {parts[2]!r}"
            )
            continue
        if run_field != run_name:
            violations.append(
                f"line {line_num} ({cur_qid}): nugget run_name "
                f"{run_field!r} != expected {run_name!r}"
            )
        if rank not in pr.rank_set(cur_qid):
            violations.append(
                f"line {line_num} ({cur_qid}): rank {rank} not in frozen PR "
                f"(roster ranks: {sorted(pr.rank_set(cur_qid))[:5]}...)"
            )
    return violations


def run_validation(
    pr_path: Path,
    ac_v2_path: Path,
    pr: FrozenPR,
    run_name: str,
    source_qids: set[str] | None,
) -> ValidationOutcome:
    """Run the full 8/8 validation chain plus the subset invariant.

    Args:
        pr_path: Frozen PR file (read-only).
        ac_v2_path: v2 AC file (just written).
        pr: Parsed frozen PR (used by the subset check).
        run_name: PR run name for the AC ``PRrunName`` field check.
        source_qids: Set of source qids for the coverage check; pass
            ``None`` to skip the 8th check.

    Returns:
        ``ValidationOutcome`` with all check results.
    """
    pr_text = pr_path.read_text()
    ac_text = ac_v2_path.read_text()
    pr_lines = [ln for ln in pr_text.strip().split("\n") if ln.strip()]

    pr_validator_errors = validate_pr_file(pr_text, run_name)
    ac_validator_errors = validate_ac_file(
        ac_text,
        run_name,
        pr_question_ids=set(pr.rows_by_qid),
        pr_rank_map={qid: pr.rank_set(qid) for qid in pr.rows_by_qid},
        # require_nuggets=False — a qid is allowed to produce zero nuggets
        # when Phase B rescue legitimately cannot fire (e.g. passages_db
        # has no rescuable rows for that qid). Don't hard-fail the build.
        require_nuggets=False,
    )
    subset_violations = _check_subset_invariant(ac_text, pr, run_name)

    spec_checks: list[tuple[str, bool, str]] = []
    spec_runners: list[tuple[str, Any]] = [
        ("1. PR PassageRank [1-20], no ties", lambda: check_pr_passage_rank(pr_lines)),
        ("2. PR max 20 passages/question", lambda: check_pr_max_passages(pr_lines)),
        ("3. AC XML-style question tags", lambda: check_ac_xml_tags(ac_text)),
        (
            "4. AC AnswerString;ConfidenceScore",
            lambda: check_ac_answer_confidence(ac_text),
        ),
        ("5. AC nugget format (max 10)", lambda: check_ac_nugget_format(ac_text)),
        (
            "6. AC PassageKey references valid PR",
            lambda: check_ac_passage_refs(ac_text, pr_lines, run_name),
        ),
        ("7. AC NuggetNum [1-10]", lambda: check_ac_nugget_num(ac_text)),
    ]
    if source_qids is not None:
        spec_runners.append(
            (
                "8. Source coverage: every qid has answer + ≥1 passage",
                lambda: check_source_coverage(source_qids, pr_lines, ac_text),
            )
        )
    spec_passed = 0
    spec_failed = 0
    for name, runner in spec_runners:
        ok, detail = runner()
        spec_checks.append((name, ok, detail))
        if ok:
            spec_passed += 1
        else:
            spec_failed += 1

    return ValidationOutcome(
        run_name=run_name,
        spec_checks=spec_checks,
        spec_passed=spec_passed,
        spec_failed=spec_failed,
        pr_errors=pr_validator_errors,
        ac_errors=ac_validator_errors,
        subset_violations=subset_violations,
    )


def _print_outcome(outcome: ValidationOutcome) -> None:
    print(f"\n=== R2C2 Submission Validation — {outcome.run_name} ===", flush=True)
    for name, ok, detail in outcome.spec_checks:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}", flush=True)
        print(f"  {detail}", flush=True)
    print(
        f"\nResult: {outcome.spec_passed}/{outcome.total_checks} PASS, "
        f"{outcome.spec_failed}/{outcome.total_checks} FAIL",
        flush=True,
    )
    print(
        f"PR-validator errors: {len(outcome.pr_errors)}; "
        f"AC-validator errors: {len(outcome.ac_errors)}; "
        f"subset violations: {len(outcome.subset_violations)}",
        flush=True,
    )
    if outcome.subset_violations:
        for v in outcome.subset_violations[:5]:
            print(f"  SUBSET: {v}", flush=True)


# CLI


def _load_source_qids(source_path: Path) -> set[str] | None:
    """Read source-question file and return the set of question_ids.

    Returns ``None`` if the file is missing or set to ``-`` (skip flag).
    Mirrors ``ensemble_aggregator._load_source_qids``.
    """
    if str(source_path) == "-":
        return None
    if not source_path.exists():
        log.warning(
            "source-questions file %s missing — coverage check skipped",
            source_path,
        )
        return None
    qids: set[str] = set()
    with open(source_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(rec.get("question_id") or rec.get("qid") or "").strip()
            if qid:
                qids.add(qid)
    return qids


def _process_run(
    pr_dir: Path,
    traces_dir: Path,
    output_dir: Path,
    pr_filename: str,
    ac_filename: str,
    run_name: str,
    *,
    first_iter_only: bool,
    source_qids: set[str] | None,
    strategy: str = STRATEGY_TRACE_REDESIGN,
    judge_decisions: dict[str, Any] | None = None,
    output_suffix: str | None = None,
    allow_incomplete_judge: bool = False,
) -> tuple[Path, ValidationOutcome, dict[str, list[str]]]:
    """Process one (PR, AC) pair → v2 AC file + validation outcome.

    Args:
        ...: standard params.
        strategy: one of ``ALLOWED_STRATEGIES``. PG-1 (``first_iter_only=True``)
            always uses ``trace-redesign`` (K=1 makes other strategies a no-op).
        judge_decisions: required when ``strategy=llm-judge``.
        output_suffix: when set, the on-disk AC file uses
            ``BITEM-AC-<suffix>`` instead of ``<ac_filename>.v2``. Only
            applied to the PG-2 path (``first_iter_only=False``).
    """
    pr_path = pr_dir / pr_filename
    ac_path = pr_dir / ac_filename
    pr = load_frozen_pr(pr_path, run_name)
    ac = load_frozen_ac(ac_path, run_name)
    rollouts = load_rollouts_for_run(traces_dir, first_iter_only=first_iter_only)
    ac_text, _candidates, warnings = build_v2_ac_text(
        pr,
        ac,
        rollouts,
        apply_trace_redesign=first_iter_only,
        strategy=strategy,
        judge_decisions=judge_decisions,
        allow_incomplete_judge=allow_incomplete_judge,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if (not first_iter_only) and output_suffix is not None:
        ac_v2_path = output_dir / f"BITEM-AC-{output_suffix}"
    else:
        ac_v2_path = output_dir / f"{ac_filename}.v2"
    ac_v2_path.write_text(ac_text)

    outcome = run_validation(pr_path, ac_v2_path, pr, run_name, source_qids)
    return ac_v2_path, outcome, warnings


def _bundle_zip(
    output_dir: Path,
    ac_v2_files: list[Path],
    *,
    zip_name: str = "BITEM-AC.v2.zip",
) -> Path:
    """Create ``BITEM-AC.v2.zip`` containing the v2 AC files at the ZIP root.

    Per R2C2 submission spec: files inside the ZIP root use the bare
    canonical filename (``BITEM-AC-1`` / ``BITEM-AC-2``), not the
    ``.v2`` suffix. The ``.v2`` suffix is only on the on-disk artefact
    so we don't shadow the frozen artefact in case someone copies them
    side-by-side.
    """
    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in ac_v2_files:
            # Strip the trailing ``.v2`` so the ZIP root holds the
            # canonical filename.
            arcname = f.name[: -len(".v2")] if f.name.endswith(".v2") else f.name
            zf.write(f, arcname=arcname)
    return zip_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point.

    Returns:
        0 on full success (7 spec + source-coverage all PASS, no subset
        violations); non-zero on any failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing frozen BITEM-PG-{1,2} + BITEM-AC-{1,2} (read-only)"
        ),
    )
    parser.add_argument(
        "--traces-dir",
        required=True,
        type=Path,
        help="Snapshot dir containing iter_{0..3}/test_traces.jsonl (read-only)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory; will hold BITEM-AC-{1,2}.v2 + BITEM-AC.v2.zip",
    )
    parser.add_argument(
        "--source-questions",
        type=Path,
        default=Path("data/r2c2/test_questions.jsonl"),
        help=(
            "Path to source question JSONL (each row has question_id). "
            "Pass '-' to skip the source-coverage check."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if the v2 AC fails 8/8 spec, validator, or "
            "subset checks. ON by default for production use; pass "
            "--no-strict (or omit --strict) for inspection runs."
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=STRATEGY_TRACE_REDESIGN,
        choices=sorted(ALLOWED_STRATEGIES),
        help=(
            "AC re-aggregation strategy. 'trace-redesign' (default) "
            "reproduces the AC-1 ship behaviour (PG-1 per-iter trace "
            "pick; PG-2 verbatim frozen). 'max-conf' picks the single "
            "highest-tool_conf non-refusal attempt across all rollouts "
            "(AC-2). 'conf-sum-cluster' pools attempts, clusters by "
            "normalised answer, and emits the highest-sum cluster "
            "(AC-3). 'llm-judge' requires --judge-decisions (AC-4). "
            "'best-rollout' and 'plurality' are preserved for backward "
            "compatibility with the legacy AC-2 / AC-3 designs."
        ),
    )
    parser.add_argument(
        "--judge-decisions",
        type=Path,
        default=None,
        help=(
            "Path to judge_decisions.jsonl produced by "
            "`r2c2_agent llm-judge`. Required iff --strategy llm-judge."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Output AC filename suffix (e.g. '2', '3', '4'). When set, "
            "the PG-2 output is written as 'BITEM-AC-<SUFFIX>' instead "
            "of 'BITEM-AC-2.v2'. PG-1 output is never affected (always "
            "'BITEM-AC-1.v2'). Defaults: trace-redesign→none, "
            "best-rollout→'2', plurality→'3', llm-judge→'4'."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-judge",
        action="store_true",
        default=False,
        help=(
            "Development-only escape hatch. When set together with "
            "--strategy llm-judge, the reaggregator will NOT hard-fail "
            "if some qids are missing or have parse_failed=True. "
            "Production AC-4 runs MUST omit this flag — a real LLM "
            "decision is required for every qid."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.strategy == STRATEGY_LLM_JUDGE and args.judge_decisions is None:
        log.error("--strategy llm-judge requires --judge-decisions")
        return 2
    if args.judge_decisions is not None and args.strategy != STRATEGY_LLM_JUDGE:
        log.warning(
            "--judge-decisions supplied but --strategy=%s; ignoring",
            args.strategy,
        )

    judge_decisions: dict[str, Any] | None = None
    if args.strategy == STRATEGY_LLM_JUDGE:
        from r2c2_agent.orchestrator.llm_judge import load_judge_decisions

        judge_decisions = load_judge_decisions(args.judge_decisions)

    default_suffix = {
        STRATEGY_TRACE_REDESIGN: None,
        STRATEGY_BEST_ROLLOUT: "2",
        STRATEGY_PLURALITY: "3",
        STRATEGY_LLM_JUDGE: "4",
        STRATEGY_MAX_CONF: "2",
        STRATEGY_CONF_SUM_CLUSTER: "3",
    }[args.strategy]
    output_suffix = args.output_suffix or default_suffix

    source_qids = _load_source_qids(args.source_questions)

    runs = [
        (PR_FILE_PG1, AC_FILE_PG1, RUN_NAME_PG1, True),
        (PR_FILE_PG2, AC_FILE_PG2, RUN_NAME_PG2, False),
    ]

    ac_v2_files: list[Path] = []
    outcomes: list[ValidationOutcome] = []
    all_warnings: dict[str, dict[str, list[str]]] = {}

    for pr_filename, ac_filename, run_name, first_iter_only in runs:
        log.info(
            "rebuilding v2 AC for %s (first_iter_only=%s, strategy=%s)",
            run_name,
            first_iter_only,
            args.strategy,
        )
        ac_v2_path, outcome, warnings = _process_run(
            args.pr_dir,
            args.traces_dir,
            args.output,
            pr_filename,
            ac_filename,
            run_name,
            first_iter_only=first_iter_only,
            source_qids=source_qids,
            strategy=args.strategy,
            judge_decisions=judge_decisions,
            output_suffix=output_suffix,
            allow_incomplete_judge=args.allow_incomplete_judge,
        )
        ac_v2_files.append(ac_v2_path)
        outcomes.append(outcome)
        all_warnings[run_name] = warnings
        log.info(
            "%s → %s (8/8 spec passed=%d failed=%d; subset_violations=%d)",
            run_name,
            ac_v2_path,
            outcome.spec_passed,
            outcome.spec_failed,
            len(outcome.subset_violations),
        )
        _print_outcome(outcome)

    zip_label = (
        "BITEM-AC.v2.zip"
        if args.strategy == STRATEGY_TRACE_REDESIGN and args.output_suffix is None
        else f"BITEM-AC.{args.strategy}.zip"
    )
    zip_path = _bundle_zip(args.output, ac_v2_files, zip_name=zip_label)
    log.info("wrote ZIP → %s (%d files)", zip_path, len(ac_v2_files))

    # Persist a small JSON summary alongside the v2 AC files so reviewers
    # can audit the rule trace per qid without re-running the script.
    summary = {
        "runs": [
            {
                "run_name": o.run_name,
                "ac_v2_path": str(p),
                "spec_passed": o.spec_passed,
                "spec_failed": o.spec_failed,
                "total_spec_checks": o.total_checks,
                "pr_validator_errors": len(o.pr_errors),
                "ac_validator_errors": len(o.ac_errors),
                "subset_violations": len(o.subset_violations),
                "all_pass": o.all_pass,
            }
            for o, p in zip(outcomes, ac_v2_files)
        ],
        "zip": str(zip_path),
        "warnings_by_qid": all_warnings,
    }
    summary_path = args.output / "ac_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info("wrote summary → %s", summary_path)

    all_pass = all(o.all_pass for o in outcomes)
    if args.strict and not all_pass:
        log.error("strict mode: at least one run failed validation")
        return 2
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
