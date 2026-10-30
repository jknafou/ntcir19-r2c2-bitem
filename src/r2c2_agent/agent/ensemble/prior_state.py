"""Reconstruct the ensemble exclusion set from on-disk iteration records.

When all N iterations run inside a single worker HTTP call, the ensemble
accumulates ``touched_pids_set`` in memory. Per-iteration dispatch sends
one HTTP call per ensemble slot, so the worker has to rebuild the
cumulative exclusion set from the prior iter_K/test_traces.jsonl files
before running the current iteration.

This module only reads JSONL records and unions their
``ensemble_touched_delta`` fields. It imports nothing from the worker,
orchestrator, or pipeline handlers, so it can be unit-tested in isolation
and reused by offline tooling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MissingPriorIterError",
    "PriorAnswerResult",
    "PriorIterResult",
    "load_prior_rollout_answer",
    "load_touched_set",
]

log = logging.getLogger(__name__)


class MissingPriorIterError(RuntimeError):
    """Raised in strict mode when a prior iter file lacks the target record."""


@dataclass(frozen=True)
class PriorIterResult:
    """Outcome of a prior-iteration scan for a single (qid, rollout) pair.

    Attributes:
        touched: Union of ``ensemble_touched_delta`` across every prior
            iter file that contained a record for this (qid, rollout).
        iters_found: Sorted tuple of ``ensemble_idx`` values that
            contributed to ``touched``. Read from each record's
            ``ensemble_idx`` field; falls back to the path's index when
            the record does not carry one.
        missing_paths: Paths that the caller supplied but which did not
            contain a matching (qid, rollout) record. In strict mode a
            non-empty tuple raises instead of being returned.
    """

    touched: frozenset[str]
    iters_found: tuple[int, ...]
    missing_paths: tuple[str, ...]


def _read_delta_for_record(
    path: Path,
    qid: str,
    rollout_idx: int,
) -> tuple[set[str] | None, int | None]:
    """Return (delta_set, ensemble_idx) for the target record in ``path``.

    Returns ``(None, None)`` if the file does not contain a record for
    the (qid, rollout) pair. Malformed JSON lines are skipped, as the
    orchestrator's ``load_done_ids`` does.
    """
    if not path.exists():
        return None, None
    try:
        fh = path.open(encoding="utf-8")
    except OSError as exc:
        log.warning("prior_state: cannot open %s: %s", path, exc)
        return None, None

    with fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question_id") != qid:
                continue
            if int(rec.get("rollout_idx", 0)) != int(rollout_idx):
                continue
            delta = rec.get("ensemble_touched_delta") or []
            idx_raw = rec.get("ensemble_idx")
            try:
                idx = int(idx_raw) if idx_raw is not None else None
            except (TypeError, ValueError):
                idx = None
            return {str(pid) for pid in delta if pid}, idx
    return None, None


def load_touched_set(
    prior_paths: Sequence[Path | str],
    qid: str,
    rollout_idx: int,
    *,
    strict: bool = False,
) -> PriorIterResult:
    """Build the cumulative exclusion set for iter K from iter_0..K-1 records.

    Args:
        prior_paths: Ordered list of ``iter_0/test_traces.jsonl`` ..
            ``iter_{K-1}/test_traces.jsonl`` paths. Order matters only
            for the fallback ensemble_idx (when the record has none);
            the returned ``touched`` is order-independent.
        qid: Question id to match against each record's ``question_id``.
        rollout_idx: Rollout index to match (records without
            ``rollout_idx`` default to 0).
        strict: If True, raise :class:`MissingPriorIterError` when any
            prior path fails to yield a record. If False (default),
            populate ``missing_paths`` and keep going — the caller gets
            a partial union and can decide whether that is acceptable.

    Returns:
        :class:`PriorIterResult` with the union, the iter indices that
        contributed, and the list of paths missing the target record.

    Raises:
        MissingPriorIterError: Only when ``strict=True`` and at least
            one path did not yield a matching record.
    """
    touched: set[str] = set()
    iters_found: list[int] = []
    missing: list[str] = []
    for pos, raw_path in enumerate(prior_paths):
        p = Path(raw_path)
        delta, idx = _read_delta_for_record(p, qid, rollout_idx)
        if delta is None:
            missing.append(str(p))
            continue
        touched |= delta
        iters_found.append(idx if idx is not None else pos)
    iters_found.sort()
    if strict and missing:
        raise MissingPriorIterError(
            f"prior iter file(s) missing record for qid={qid} "
            f"rollout={rollout_idx}: {missing}"
        )
    return PriorIterResult(
        touched=frozenset(touched),
        iters_found=tuple(iters_found),
        missing_paths=tuple(missing),
    )


# Prior-rollout answer harvest: the last fallback of the answer-submission
# pipeline. When every earlier path has failed, the highest-confidence
# committed answer from any prior iteration becomes a candidate for
# answer_search_support.


@dataclass(frozen=True)
class PriorAnswerResult:
    """Outcome of scanning prior iter records for a committed answer.

    Attributes:
        best_answer: The highest-confidence non-refusal answer text seen
            across prior iterations, or ``None`` if no record cleared
            the ``min_confidence`` threshold.
        best_confidence: The raw_confidence (0..100) associated with
            ``best_answer``; 0 when ``best_answer is None``.
        source_iter: Iteration index (0-based) that produced
            ``best_answer``; ``None`` when no match.
        all_prior: Full list of (iter_idx, answer, confidence, termination)
            tuples for every prior record — regardless of filter — so
            downstream audit / telemetry can inspect the ensemble shape.
        missing_paths: Prior paths that did not yield a matching
            (qid, rollout) record (same semantics as
            :class:`PriorIterResult`).
    """

    best_answer: str | None
    best_confidence: int
    source_iter: int | None
    all_prior: tuple[tuple[int, str, int, str], ...]
    missing_paths: tuple[str, ...]


def _read_answer_for_record(
    path: Path,
    qid: str,
    rollout_idx: int,
) -> tuple[dict | None, int | None]:
    """Return (rec, ensemble_idx) for the target record in ``path``.

    Returns ``(None, None)`` if the file does not contain a record for
    the (qid, rollout) pair.
    """
    if not path.exists():
        return None, None
    try:
        fh = path.open(encoding="utf-8")
    except OSError as exc:
        log.warning("prior_state: cannot open %s: %s", path, exc)
        return None, None

    with fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question_id") != qid:
                continue
            if int(rec.get("rollout_idx", 0)) != int(rollout_idx):
                continue
            idx_raw = rec.get("ensemble_idx")
            try:
                idx = int(idx_raw) if idx_raw is not None else None
            except (TypeError, ValueError):
                idx = None
            return rec, idx
    return None, None


def _normalize_confidence(raw: object) -> int:
    """Normalize a confidence to int (0..100).

    Traces store ``raw_confidence`` as int 0..100, but ``force_deliver``
    envelopes may carry a float 0..1 on ``confidence``. Returns 0 on any
    parse error.
    """
    if raw is None:
        return 0
    try:
        if isinstance(raw, bool):
            return 0
        if isinstance(raw, int):
            return max(0, min(100, raw))
        if isinstance(raw, float):
            if raw <= 1.0:
                return max(0, min(100, int(round(raw * 100))))
            return max(0, min(100, int(round(raw))))
        return max(0, min(100, int(round(float(raw)))))
    except (TypeError, ValueError):
        return 0


# Termination values that mean the rollout committed an answer rather
# than a synthesised refusal. Only these contribute to the
# ``load_prior_rollout_answer`` candidate pool.
_COMMITTED_TERMINATIONS = {
    "answer_submitted",
    "force_deliver",
    "high_confidence",
}


def load_prior_rollout_answer(
    prior_paths: Sequence[Path | str],
    qid: str,
    rollout_idx: int,
    *,
    min_confidence: int = 70,
    strict: bool = False,
) -> PriorAnswerResult:
    """Scan prior iter records for the highest-confidence committed answer.

    Last fallback of the answer-submission pipeline: when every other
    path (hardgate retries, answer_search_support, educated-guess
    ladder) has failed, force_deliver can seed answer_search_support
    with the best answer a previous rollout committed.

    Args:
        prior_paths: Ordered list of ``iter_0/test_traces.jsonl`` ..
            ``iter_{K-1}/test_traces.jsonl`` paths (same contract as
            :func:`load_touched_set`).
        qid: Question id to match against each record's ``question_id``.
        rollout_idx: Rollout index to match.
        min_confidence: Minimum ``raw_confidence`` (0..100) for a record
            to be eligible as a candidate. The default of 70 matches
            the hardgate's ``conf >= 0.70`` thin-evidence floor, so a
            prior answer clearing this bar is already corpus-supported
            under its own rollout.
        strict: If True, raise :class:`MissingPriorIterError` when any
            prior path fails to yield a record.

    Returns:
        :class:`PriorAnswerResult` with the best candidate and the full
        audit trail.
    """
    all_prior: list[tuple[int, str, int, str]] = []
    missing: list[str] = []
    for pos, raw_path in enumerate(prior_paths):
        p = Path(raw_path)
        rec, idx = _read_answer_for_record(p, qid, rollout_idx)
        if rec is None:
            missing.append(str(p))
            continue
        iter_idx = idx if idx is not None else pos
        # Traces written by a run use ``final_answer`` + ``termination``;
        # some test fixtures use ``answer`` + ``termination_reason``.
        # Reading only the second pair would miss every committed answer
        # in real output.
        answer = str(rec.get("final_answer") or rec.get("answer") or "").strip()
        termination = str(
            rec.get("termination") or rec.get("termination_reason") or ""
        ).strip()
        conf = _normalize_confidence(rec.get("raw_confidence", rec.get("confidence")))
        all_prior.append((iter_idx, answer, conf, termination))

    if strict and missing:
        raise MissingPriorIterError(
            f"prior iter file(s) missing record for qid={qid} "
            f"rollout={rollout_idx}: {missing}"
        )

    candidates = [
        (iter_idx, ans, conf, term)
        for (iter_idx, ans, conf, term) in all_prior
        if ans and conf >= min_confidence and term in _COMMITTED_TERMINATIONS
    ]
    if not candidates:
        return PriorAnswerResult(
            best_answer=None,
            best_confidence=0,
            source_iter=None,
            all_prior=tuple(all_prior),
            missing_paths=tuple(missing),
        )
    # Pick highest confidence; ties → earliest iteration (more independent).
    candidates.sort(key=lambda x: (-x[2], x[0]))
    best_iter, best_ans, best_conf, _ = candidates[0]
    return PriorAnswerResult(
        best_answer=best_ans,
        best_confidence=best_conf,
        source_iter=best_iter,
        all_prior=tuple(all_prior),
        missing_paths=tuple(missing),
    )
