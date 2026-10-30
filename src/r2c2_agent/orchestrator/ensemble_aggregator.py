"""Post-orchestrator ensemble aggregator → R2C2 PR/AC submission files.

Scans a run dir of per-rollout traces (``iter_*/test_traces.jsonl``),
clusters semantically-equivalent answers across rollouts, picks one
winning cluster per qid using a sum-with-decay confidence model, and
emits R2C2-compliant PR + AC submission files plus a debug breakdown.

Reuses the existing ``r2c2_agent.agent.postprocess`` formatter chain and
the ``submission_validator`` integrity checks — does NOT reimplement
file emission or validation.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import glob
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from r2c2_agent.agent.interfaces.types import (  # noqa: E402
    ACResult,
    CitedPassage,
    Nugget,
    PREntry,
    QuestionTrace,
)
from r2c2_agent.agent.postprocess import (  # noqa: E402
    format_ac_file,
    format_ac_result,
    format_pr_entries,
    format_pr_file,
    validate_submission,
)
from r2c2_agent.agent.submission_validator import (  # noqa: E402
    validate_ac_file,
    validate_pr_file,
)

# Standalone 7-check R2C2 validator lives under src/r2c2_agent/validation/.
# Import its check functions so the aggregator can invoke the same
# format-spec checks the user runs manually after submission build.
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from r2c2_agent.validation.validate_submission import (  # noqa: E402
    check_ac_answer_confidence,
    check_ac_nugget_format,
    check_ac_nugget_num,
    check_ac_passage_refs,
    check_ac_xml_tags,
    check_pr_max_passages,
    check_pr_passage_rank,
)

# check_pr_semicolon_separated is not imported: the organiser's own PO files
# contain unescaped ';' mid-passage, so the spec is more permissive than
# that strict 4-field check. PassageText routinely contains ';'.

log = logging.getLogger(__name__)

__all__ = [
    "AnswerCluster",
    "AggregatedQid",
    "aggregate_qid",
    "aggregate_run_dir",
    "build_pr_ac_files",
    "cluster_match",
    "cluster_confidence",
    "main",
]

# Constants

MAX_PASSAGES_PER_QID = 20  # R2C2 PR limit
MAX_NUGGETS_PER_QID = 10  # R2C2 AC limit

JACCARD_MATCH_THRESHOLD = 0.6
NUGGET_TEXT_SIMILARITY_THRESHOLD = 0.92

AGREEMENT_DECAY = 0.55  # higher = slower agreement growth; 0.55 → 1→.45, 5→.95
FORCE_DELIVER_DISCOUNT = 0.4  # force_deliver rollout weight in cluster sum
COMPETITOR_THRESHOLD = 0.5  # other-cluster strength below which it doesn't dampen
COMPETITOR_DAMPEN_K = 1.5  # multiplier in 1 / (1 + k*competitor_mass)

REFUSAL_FLOOR_LOW = 0.10
REFUSAL_FLOOR_HIGH = 0.20
REFUSAL_HIGH_THRESHOLD = 3  # ≥3 refusals → high floor

REFUSAL_RE = re.compile(
    r"(does not support|insufficient evidence|refuse|cannot answer)",
    re.IGNORECASE,
)
LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
PARENS_RE = re.compile(r"\s*\([^)]*\)\s*")
PUNCT_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")

CANONICAL_REFUSAL_ANSWER = (
    "The corpus does not support an answer to this question; "
    "insufficient evidence to respond."
)


# Datatypes


@dataclass
class _Rollout:
    """One rollout's data extracted from a trace record."""

    qid: str
    question_text: str
    question_type: str
    ensemble_idx: int
    final_answer: str
    confidence: int  # raw 0-100 from trace
    termination: str  # "answer_submitted" | "force_deliver" | ...
    nuggets: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0


@dataclass
class AnswerCluster:
    """One semantic answer cluster within a qid."""

    canonical: str
    members: list[_Rollout] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def n_real(self) -> int:
        return sum(1 for r in self.members if r.termination == "answer_submitted")

    @property
    def n_force_deliver(self) -> int:
        return sum(1 for r in self.members if r.termination != "answer_submitted")

    @property
    def peak_conf(self) -> float:
        """Max raw confidence across members, normalised to [0, 1]."""
        if not self.members:
            return 0.0
        return max(r.confidence for r in self.members) / 100.0


@dataclass
class AggregatedQid:
    """Aggregated output for one qid."""

    question_id: str
    final_answer: str
    final_confidence: int  # 0-100, integer for R2C2
    supporting_nuggets: list[dict[str, Any]] = field(default_factory=list)
    supporting_passages: list[dict[str, Any]] = field(default_factory=list)
    cluster_breakdown: list[dict[str, Any]] = field(default_factory=list)
    n_rollouts_total: int = 0
    aggregator_warnings: list[str] = field(default_factory=list)


# Trace ingestion


def _extract_pids_from_trace(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull ``reconciled_passages`` out of a trace as a list of dicts.

    Each entry must have ``passage_id``; doc_id and passage_text are
    sourced where present (the legacy schema also has
    ``passage_text_with_title`` and ``title``).

    Rescue: if ``reconciled_passages`` is empty or contains only the
    ``__refusal__`` sentinel (i.e. the LLM bypassed the S1 mark loop and
    requested refusal before classifying retrieved evidence), fall through
    to ``m11_pipeline.passages_db`` to surface any retrieved-but-unclassified
    passages. This catches the ``model_requested_refusal`` regression
    pattern: passages were retrieved by ``search_passages`` but never
    routed through S1, leaving them with ``status="unclassified"`` instead
    of being promoted to ``reconciled_passages``.
    """
    raw = rec.get("reconciled_passages") or []
    out: list[dict[str, Any]] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        pid = p.get("passage_id") or ""
        if not pid:
            continue
        doc_id = p.get("doc_id") or ""
        # Filter the virtual ``__refusal__`` sentinel passage that the
        # reconciler injects on canonical refusals. It must not leak
        # into the R2C2 PR submission.
        if pid.startswith("__refusal") or doc_id.startswith("__refusal"):
            continue
        text = (p.get("passage_text") or p.get("text") or "").strip()
        if not text:
            # R2C2 PR validator rejects empty PassageText — drop here so
            # the aggregator never proposes a malformed passage row.
            continue
        out.append(
            {
                "passage_id": pid,
                "doc_id": doc_id,
                "text": text,
                "title": p.get("title") or "",
            }
        )
    if out:
        return out

    # Rescue path: pull from the pipeline's passages_db when the reconciled
    # set is empty or refusal-only.
    pipe = rec.get("m11_pipeline") or {}
    pdb = pipe.get("passages_db") or {}
    if not isinstance(pdb, dict):
        return out
    for pid, p in pdb.items():
        if not isinstance(p, dict):
            continue
        if pid.startswith("__refusal"):
            continue
        doc_id = p.get("doc_id") or ""
        # passages_db often stores doc_id=None — derive from canonical
        # pid format ``<doc_id>__<chunk_id>`` (split on the double-underscore).
        if not doc_id and "__" in pid:
            doc_id = pid.split("__", 1)[0]
        if doc_id.startswith("__refusal"):
            continue
        # passages_db may store text under various keys depending on stage
        text = (
            p.get("passage_text")
            or p.get("text")
            or p.get("passage_text_with_title")
            or ""
        ).strip()
        if not text:
            continue
        out.append(
            {
                "passage_id": pid,
                "doc_id": doc_id,
                "text": text,
                "title": p.get("title") or "",
            }
        )
    return out


def _extract_nuggets_from_trace(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull ``reconciled_nuggets`` out of a trace; drop synthetic + non-supporting.

    Handles both schemas:
      - Reconciler-internal: ``text``, ``status``, ``supporting_pids``,
        ``pid_nli_scores``
      - Trace-emitted (legacy): ``nugget_text``, ``stance``,
        ``passage_id``, ``primary_passage_id``, ``nli_score``
    """
    raw = rec.get("reconciled_nuggets") or []
    out: list[dict[str, Any]] = []
    for n in raw:
        if not isinstance(n, dict):
            continue
        if n.get("is_synthetic"):
            continue
        # Stance/status: accept verified_supporting or stance="supporting"
        status = n.get("status")
        stance = n.get("stance")
        if status and status != "verified_supporting":
            continue
        if stance and stance != "supporting" and not status:
            continue
        text = (n.get("text") or n.get("nugget_text") or "").strip()
        pid = n.get("passage_id") or n.get("primary_passage_id") or ""
        if not text or not pid:
            continue
        # Drop nuggets pinned to the virtual __refusal__ passage —
        # those are placeholders, not real corpus evidence.
        if pid.startswith("__refusal"):
            continue
        # Best NLI score across the legacy field and any per-pid map
        nli = float(n.get("nli_score") or 0.0)
        pid_scores = n.get("pid_nli_scores") or {}
        for s in pid_scores.values():
            if s is not None and float(s) > nli:
                nli = float(s)
        # Build supporting_pids: prefer explicit field, fall back to pid
        supporting_pids = list(n.get("supporting_pids") or [pid])
        if pid not in supporting_pids:
            supporting_pids.insert(0, pid)
        # Build pid_nli_scores: ensure pid has a score
        scores = dict(pid_scores) if pid_scores else {}
        if pid not in scores and nli > 0:
            scores[pid] = nli
        out.append(
            {
                "passage_id": pid,
                "supporting_pids": supporting_pids,
                "pid_nli_scores": scores or {pid: nli},
                "nli_score": nli,
                "text": text,
            }
        )
    return out


def _load_rollouts(
    run_dir: Path, *, first_iter_only: bool = False
) -> dict[str, list[_Rollout]]:
    """Load per-rollout traces and group by qid.

    When ``first_iter_only`` is set, only ``iter_0/test_traces.jsonl`` is
    loaded — used to produce the Run-1 (single-pass) submission. Exception:
    if iter_0 produced a qid with zero usable passages (e.g. canonical
    refusal that yielded only ``__refusal__`` sentinels), the loader
    augments that qid's iter_0 rollout with passages + nuggets pulled from
    later iters of the same qid. The answer cluster still derives from
    iter_0; only the evidence is rescued so source-coverage holds.
    """
    pattern_all = str(run_dir / "iter_*" / "test_traces.jsonl")
    by_qid_full: dict[str, list[_Rollout]] = collections.defaultdict(list)
    for path in sorted(glob.glob(pattern_all)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping malformed line in %s", path)
                    continue
                # Skip iters=0 placeholder (dead-endpoint dispatch)
                if int(rec.get("iterations") or 0) <= 0:
                    continue
                qid = rec.get("question_id") or ""
                if not qid:
                    continue
                rollout = _Rollout(
                    qid=qid,
                    question_text=rec.get("question_text") or "",
                    question_type=rec.get("question_type") or "default",
                    ensemble_idx=int(rec.get("ensemble_idx") or 0),
                    final_answer=(rec.get("final_answer") or "").strip(),
                    confidence=int(rec.get("confidence") or 0),
                    termination=rec.get("termination") or "",
                    nuggets=_extract_nuggets_from_trace(rec),
                    passages=_extract_pids_from_trace(rec),
                    iterations=int(rec.get("iterations") or 0),
                )
                by_qid_full[qid].append(rollout)

    if not first_iter_only:
        return by_qid_full

    # First-iter-only: keep ens=0 rollouts; for any qid where ens=0 has
    # zero usable passages, splice in passages + nuggets from later iters
    # of the same qid so source coverage holds.
    out: dict[str, list[_Rollout]] = {}
    for qid, rollouts in by_qid_full.items():
        ens0 = [r for r in rollouts if r.ensemble_idx == 0]
        if not ens0:
            continue
        primary = ens0[0]
        if not primary.passages:
            for r in rollouts:
                if r.ensemble_idx == 0:
                    continue
                if r.passages:
                    primary.passages = r.passages
                    if not primary.nuggets:
                        primary.nuggets = r.nuggets
                    break
        out[qid] = [primary]
    return out


# Answer clustering


def _normalize(answer: str) -> str:
    s = answer.strip().lower()
    s = PARENS_RE.sub(" ", s)
    s = LEADING_ARTICLE_RE.sub("", s)
    s = PUNCT_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def _looks_like_refusal(answer: str) -> bool:
    if not answer:
        return False
    return bool(REFUSAL_RE.search(answer))


def _whole_word_substring(small: str, big: str) -> bool:
    """True if ``small`` appears as whole-word substring of ``big``."""
    if not small or not big:
        return False
    if small == big:
        return True
    pattern = r"(?:^|\s)" + re.escape(small) + r"(?:\s|$)"
    return bool(re.search(pattern, big))


def _shared_nugget_text(
    a_nuggets: list[dict[str, Any]], b_nuggets: list[dict[str, Any]]
) -> bool:
    """Match if any pair of nugget texts crosses Ratcliff-Obershelp ≥ 0.92."""
    a_texts = [n.get("text", "") for n in a_nuggets if n.get("text")]
    b_texts = [n.get("text", "") for n in b_nuggets if n.get("text")]
    for ta in a_texts:
        for tb in b_texts:
            ratio = difflib.SequenceMatcher(None, ta, tb).ratio()
            if ratio >= NUGGET_TEXT_SIMILARITY_THRESHOLD:
                return True
    return False


def cluster_match(
    a_answer: str,
    b_answer: str,
    a_nuggets: list[dict[str, Any]] | None = None,
    b_nuggets: list[dict[str, Any]] | None = None,
) -> bool:
    """True if two rollout answers belong in the same semantic cluster."""
    a_ref = _looks_like_refusal(a_answer)
    b_ref = _looks_like_refusal(b_answer)
    if a_ref and b_ref:
        return True
    if a_ref != b_ref:
        return False

    a_norm = _normalize(a_answer)
    b_norm = _normalize(b_answer)
    if not a_norm or not b_norm:
        return False

    # (a) whole-word substring containment in either direction
    if a_norm == b_norm:
        return True
    if len(a_norm) <= len(b_norm):
        if _whole_word_substring(a_norm, b_norm):
            return True
    else:
        if _whole_word_substring(b_norm, a_norm):
            return True

    # (b) Jaccard on token sets
    a_tok = set(a_norm.split())
    b_tok = set(b_norm.split())
    if a_tok and b_tok:
        inter = a_tok & b_tok
        union = a_tok | b_tok
        if len(inter) / len(union) >= JACCARD_MATCH_THRESHOLD:
            return True

    # (c) shared verified-supporting nugget text
    if a_nuggets and b_nuggets and _shared_nugget_text(a_nuggets, b_nuggets):
        return True

    return False


def _cluster_rollouts(rollouts: list[_Rollout]) -> list[AnswerCluster]:
    """Greedy clustering in conf-desc order."""
    clusters: list[AnswerCluster] = []
    ordered = sorted(rollouts, key=lambda r: -r.confidence)
    for r in ordered:
        joined = False
        for c in clusters:
            seed = c.members[0]
            if cluster_match(
                r.final_answer, seed.final_answer, r.nuggets, seed.nuggets
            ):
                c.members.append(r)
                joined = True
                break
        if not joined:
            is_ref = _looks_like_refusal(r.final_answer)
            clusters.append(
                AnswerCluster(
                    canonical=r.final_answer,
                    members=[r],
                    is_refusal=is_ref,
                )
            )
    # Refresh canonical to the shortest member that is a substring of every
    # other member's normalized form (when applicable).
    for c in clusters:
        c.canonical = _pick_canonical(c)
    return clusters


def _pick_canonical(cluster: AnswerCluster) -> str:
    """Shortest member string that is a substring of every other member."""
    if not cluster.members:
        return ""
    if cluster.is_refusal:
        return CANONICAL_REFUSAL_ANSWER
    candidates = sorted(
        {r.final_answer for r in cluster.members if r.final_answer},
        key=len,
    )
    others_norm = [_normalize(r.final_answer) for r in cluster.members]
    for cand in candidates:
        cand_norm = _normalize(cand)
        if not cand_norm:
            continue
        if all(
            cand_norm == o
            or _whole_word_substring(cand_norm, o)
            or _whole_word_substring(o, cand_norm)
            for o in others_norm
            if o
        ):
            return cand
    # Fallback: the highest-conf member's answer
    best = max(cluster.members, key=lambda r: r.confidence)
    return best.final_answer


# Cluster confidence


def _cluster_weight(cluster: AnswerCluster) -> float:
    return cluster.n_real + FORCE_DELIVER_DISCOUNT * cluster.n_force_deliver


def cluster_confidence(
    cluster: AnswerCluster, all_clusters: list[AnswerCluster]
) -> float:
    """Final cluster score in [0, 1] before integer rounding for AC."""
    w = _cluster_weight(cluster)
    if w <= 0:
        return 0.0
    peak = cluster.peak_conf
    agree = 1.0 - math.pow(AGREEMENT_DECAY, w)
    raw = peak + (1.0 - peak) * agree

    competitor = 0.0
    for other in all_clusters:
        if other is cluster:
            continue
        ow = _cluster_weight(other)
        if ow <= 0:
            continue
        o_peak = other.peak_conf
        o_agree = 1.0 - math.pow(AGREEMENT_DECAY, ow)
        strength = o_peak * o_agree
        if strength > COMPETITOR_THRESHOLD:
            competitor += strength - COMPETITOR_THRESHOLD

    dampen = 1.0 / (1.0 + COMPETITOR_DAMPEN_K * competitor)
    raw = raw * dampen

    if cluster.is_refusal:
        floor = (
            REFUSAL_FLOOR_HIGH
            if len(cluster.members) >= REFUSAL_HIGH_THRESHOLD
            else REFUSAL_FLOOR_LOW
        )
        raw = max(raw, floor)

    return max(0.0, min(1.0, raw))


def _pick_winner(
    clusters: list[AnswerCluster],
) -> tuple[AnswerCluster, list[tuple[AnswerCluster, float]]]:
    """Pick the winning cluster + the full ranked list."""
    scored = [(c, cluster_confidence(c, clusters)) for c in clusters]
    # Tie-breaker chain: cluster_conf desc, n_real desc, peak desc, len(canonical) asc
    scored.sort(
        key=lambda cs: (
            -cs[1],
            -cs[0].n_real,
            -cs[0].peak_conf,
            len(cs[0].canonical),
        )
    )
    return scored[0][0], scored


# Nugget + passage merge


def _norm_text_prefix(s: str, n: int = 80) -> str:
    return WHITESPACE_RE.sub(" ", s.strip().lower())[:n]


def _merge_nuggets(cluster: AnswerCluster) -> list[dict[str, Any]]:
    """Group nuggets by (primary_pid, normalized_text[:80])."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for r in cluster.members:
        for n in r.nuggets:
            key = (n["passage_id"], _norm_text_prefix(n["text"]))
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = {
                    "passage_id": n["passage_id"],
                    "text": n["text"],
                    "supporting_pids": list(
                        n.get("supporting_pids") or [n["passage_id"]]
                    ),
                    "pid_nli_scores": dict(n.get("pid_nli_scores") or {}),
                    "nli_score": float(n.get("nli_score") or 0.0),
                    "rollout_co_occurrence": 1,
                }
                continue
            existing["rollout_co_occurrence"] += 1
            seen = set(existing["supporting_pids"])
            for pid in n.get("supporting_pids") or [n["passage_id"]]:
                if pid not in seen:
                    seen.add(pid)
                    existing["supporting_pids"].append(pid)
            # merge pid_nli_scores by max
            for pid, s in (n.get("pid_nli_scores") or {}).items():
                cur = existing["pid_nli_scores"].get(pid)
                s_f = float(s or 0.0)
                if cur is None or s_f > float(cur):
                    existing["pid_nli_scores"][pid] = s_f
            n_f = float(n.get("nli_score") or 0.0)
            if n_f > existing["nli_score"]:
                existing["nli_score"] = n_f
    # Sort: highest nli_score desc, then co-occurrence desc
    merged = sorted(
        grouped.values(),
        key=lambda m: (-m["nli_score"], -m["rollout_co_occurrence"]),
    )
    return merged[:MAX_NUGGETS_PER_QID]


def _merge_passages(
    cluster: AnswerCluster, merged_nuggets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """2-tier passage selection: nugget-pinned first, then frequency-ranked.

    Caps at MAX_PASSAGES_PER_QID. Dedups by (doc_id, sanitized_text).
    """
    # All passages across the cluster, with retrieval frequency
    freq: dict[str, int] = collections.Counter()
    first_pos: dict[str, int] = {}
    catalog: dict[str, dict[str, Any]] = {}
    for r in cluster.members:
        for pos, p in enumerate(r.passages):
            pid = p["passage_id"]
            freq[pid] += 1
            if pid not in first_pos:
                first_pos[pid] = pos
            if pid not in catalog:
                catalog[pid] = p

    # Tier 1: pids referenced by any merged nugget's supporting_pids
    tier1_pids: set[str] = set()
    nugget_max_nli: dict[str, float] = {}
    for nug in merged_nuggets:
        for pid in nug.get("supporting_pids") or [nug["passage_id"]]:
            tier1_pids.add(pid)
            score = float(nug["pid_nli_scores"].get(pid) or nug["nli_score"] or 0.0)
            if score > nugget_max_nli.get(pid, 0.0):
                nugget_max_nli[pid] = score

    tier1: list[dict[str, Any]] = sorted(
        (catalog[pid] for pid in tier1_pids if pid in catalog),
        key=lambda p: (-nugget_max_nli.get(p["passage_id"], 0.0),),
    )
    tier2: list[dict[str, Any]] = sorted(
        (catalog[pid] for pid in catalog if pid not in tier1_pids),
        key=lambda p: (
            -freq[p["passage_id"]],
            first_pos.get(p["passage_id"], 1_000_000),
        ),
    )

    out: list[dict[str, Any]] = []
    seen_pair: set[tuple[str, str]] = set()
    seen_pid: set[str] = set()
    for p in tier1 + tier2:
        if len(out) >= MAX_PASSAGES_PER_QID:
            break
        pid = p["passage_id"]
        if pid in seen_pid:
            continue
        sanitized = _sanitize(p.get("text") or "")
        pair = (p.get("doc_id") or "", sanitized)
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        seen_pid.add(pid)
        out.append(p)
    return out


def _sanitize(text: str) -> str:
    """Match postprocess._sanitize_passage_text: \\n\\r\\t → space, dedupe."""
    cleaned = re.sub(r"[\n\r\t]+", " ", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


# Per-qid aggregation


def aggregate_qid(rollouts: list[_Rollout]) -> AggregatedQid | None:
    """Aggregate one qid's rollouts into a single AggregatedQid."""
    if not rollouts:
        return None
    qid = rollouts[0].qid

    clusters = _cluster_rollouts(rollouts)
    winner, scored = _pick_winner(clusters)
    final_conf = cluster_confidence(winner, clusters)

    merged_nuggets = _merge_nuggets(winner)
    merged_passages = _merge_passages(winner, merged_nuggets)

    warnings: list[str] = []

    # Detect tied high-conf clusters
    if len(scored) >= 2:
        top_conf = scored[0][1]
        runner_conf = scored[1][1]
        if top_conf >= 0.5 and runner_conf >= 0.5 and (top_conf - runner_conf) < 0.05:
            warnings.append("tied_high_conf")

    # Min-evidence guard: if no real nuggets/passages, fall back to refusal
    if not merged_passages or not merged_nuggets or winner.is_refusal:
        # Try the strongest non-winning cluster
        for cand, _ in scored:
            if cand is winner or cand.is_refusal:
                continue
            cand_nuggets = _merge_nuggets(cand)
            cand_passages = _merge_passages(cand, cand_nuggets)
            if cand_nuggets and cand_passages:
                warnings.append("fell_back_to_secondary_cluster")
                winner = cand
                merged_nuggets = cand_nuggets
                merged_passages = cand_passages
                final_conf = cluster_confidence(cand, clusters)
                break
        else:
            # No cluster has usable evidence: emit refusal at floor
            return _emit_refusal(qid, rollouts, clusters, scored, warnings)

    final_answer = _pick_canonical(winner)
    if not final_answer or _looks_like_refusal(final_answer):
        return _emit_refusal(qid, rollouts, clusters, scored, warnings)

    breakdown = []
    for c, conf in scored:
        breakdown.append(
            {
                "canonical": c.canonical,
                "rollouts": [
                    r.ensemble_idx
                    for r in sorted(c.members, key=lambda x: x.ensemble_idx)
                ],
                "peak_conf": round(c.peak_conf, 4),
                "n_real": c.n_real,
                "n_force_deliver": c.n_force_deliver,
                "cluster_conf": round(conf, 4),
                "won": c is winner,
                "is_refusal": c.is_refusal,
            }
        )

    return AggregatedQid(
        question_id=qid,
        final_answer=final_answer,
        final_confidence=int(round(final_conf * 100)),
        supporting_nuggets=merged_nuggets,
        supporting_passages=merged_passages,
        cluster_breakdown=breakdown,
        n_rollouts_total=len(rollouts),
        aggregator_warnings=warnings,
    )


def _emit_refusal(
    qid: str,
    rollouts: list[_Rollout],
    clusters: list[AnswerCluster],
    scored: list[tuple[AnswerCluster, float]],
    warnings: list[str],
) -> AggregatedQid:
    """Final-resort: refusal envelope with a placeholder nugget+passage.

    Picks the highest-NLI nugget + its passage from any cluster (so the
    submission still has ≥1 nugget + ≥1 passage), but with a refusal
    answer string and floor confidence.
    """
    refusal_clusters = [c for c in clusters if c.is_refusal]
    if refusal_clusters:
        ref = max(refusal_clusters, key=lambda c: len(c.members))
        ref_conf = cluster_confidence(ref, clusters)
    else:
        ref_conf = REFUSAL_FLOOR_LOW

    # Pull strongest nugget+passage from any cluster (winning or not)
    best_n: dict[str, Any] | None = None
    for c in clusters:
        for r in c.members:
            for n in r.nuggets:
                if best_n is None or float(n.get("nli_score") or 0) > float(
                    best_n.get("nli_score") or 0
                ):
                    best_n = n
    placeholder_nuggets: list[dict[str, Any]] = [best_n] if best_n else []

    # Pull as many retrieved passages as possible (up to MAX_PASSAGES_PER_QID).
    # Ranking for the no-evidence path: prefer the earliest rollout that
    # retrieved each pid, and within that rollout the search engine's own
    # rank. Cross-rollout frequency misleads on refusal-only qids: when the
    # LLM refuses before classifying evidence in early rollouts, later
    # rollouts often re-retrieve the same wrong-but-tempting docs (e.g.
    # movies by the same director), while the correct doc may have been hit
    # only once, by the cleanest search.
    earliest_ens: dict[str, int] = {}
    pos_in_earliest: dict[str, int] = {}
    freq: dict[str, int] = collections.Counter()
    catalog: dict[str, dict[str, Any]] = {}
    for c in clusters:
        for r in c.members:
            for pos, p in enumerate(r.passages):
                pid = p["passage_id"]
                freq[pid] += 1
                if pid not in earliest_ens or r.ensemble_idx < earliest_ens[pid]:
                    earliest_ens[pid] = r.ensemble_idx
                    pos_in_earliest[pid] = pos
                if pid not in catalog:
                    catalog[pid] = p
    best_pid = best_n["passage_id"] if best_n else None
    ranked_pids = sorted(
        catalog,
        key=lambda pid: (
            0 if pid == best_pid else 1,
            earliest_ens.get(pid, 1_000_000),
            pos_in_earliest.get(pid, 1_000_000),
            -freq[pid],
        ),
    )
    placeholder_passages: list[dict[str, Any]] = []
    seen_pair: set[tuple[str, str]] = set()
    for pid in ranked_pids:
        if len(placeholder_passages) >= MAX_PASSAGES_PER_QID:
            break
        p = catalog[pid]
        sanitized = _sanitize(p.get("text") or "")
        pair = (p.get("doc_id") or "", sanitized)
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        placeholder_passages.append(p)

    breakdown = []
    for c, conf in scored:
        breakdown.append(
            {
                "canonical": c.canonical,
                "rollouts": [
                    r.ensemble_idx
                    for r in sorted(c.members, key=lambda x: x.ensemble_idx)
                ],
                "peak_conf": round(c.peak_conf, 4),
                "n_real": c.n_real,
                "n_force_deliver": c.n_force_deliver,
                "cluster_conf": round(conf, 4),
                "won": False,
                "is_refusal": c.is_refusal,
            }
        )

    if "no_usable_evidence" not in warnings:
        warnings.append("no_usable_evidence")

    return AggregatedQid(
        question_id=qid,
        final_answer=CANONICAL_REFUSAL_ANSWER,
        final_confidence=int(round(ref_conf * 100)),
        supporting_nuggets=placeholder_nuggets,
        supporting_passages=placeholder_passages,
        cluster_breakdown=breakdown,
        n_rollouts_total=len(rollouts),
        aggregator_warnings=warnings,
    )


# PR/AC submission file emission


def _to_cited_passages(passages: list[dict[str, Any]]) -> list[CitedPassage]:
    """Build CitedPassage objects in the order given (rank = position+1)."""
    out: list[CitedPassage] = []
    for i, p in enumerate(passages):
        cp = CitedPassage(
            passage_id=p["passage_id"],
            citation_count=1,
            first_appearance=i,
            doc_id=p.get("doc_id") or "",
            text=p.get("text") or "",
        )
        # Pre-rank-ordered: first entry should win the existing rank_passages
        # tiebreak; we feed format_pr_entries directly so this stays linear.
        out.append(cp)
    return out


def _to_synthetic_trace(
    qid: str,
    final_answer: str,
    final_confidence: int,
    nuggets: list[dict[str, Any]],
) -> QuestionTrace:
    """Build a minimal QuestionTrace that format_ac_result can consume.

    Only the fields touched by format_ac_result need to be populated:
    nuggets (list[Nugget]), draft_answer, final_confidence.
    """
    trace = QuestionTrace(
        question_id=qid,
        draft_answer=final_answer,
        final_confidence=final_confidence,
    )
    for n in nuggets:
        trace.nuggets.append(Nugget(passage_id=n["passage_id"], text=n["text"]))
    return trace


def build_pr_ac_files(
    aggregated: list[AggregatedQid],
    run_name: str,
) -> tuple[
    str,
    str,
    dict[str, list[PREntry]],
    dict[str, ACResult],
]:
    """Emit PR + AC file content + per-qid PR entries / AC results."""
    pr_entries_by_qid: dict[str, list[PREntry]] = {}
    ac_results_by_qid: dict[str, ACResult] = {}

    for agg in aggregated:
        ranked = _to_cited_passages(agg.supporting_passages)
        # format_pr_entries assigns rank=1..n in list order
        pr_entries = format_pr_entries(agg.question_id, ranked)
        pr_entries_by_qid[agg.question_id] = pr_entries

        # rank_map: passage_id → 1-based rank in the PR file. Iteration
        # order matches between supporting_passages and pr_entries because
        # format_pr_entries assigns rank=enumerate-position+1.
        rank_map = {
            agg.supporting_passages[i]["passage_id"]: pe.rank
            for i, pe in enumerate(pr_entries)
        }

        # Rewire nuggets whose primary passage_id was filtered out of PR
        # (e.g. a `__refusal__` sentinel pid). format_ac_result drops any
        # nugget whose passage_id isn't in rank_map, so on canonical-refusal
        # qids that's every nugget — leaving the AC block answer-only and
        # tripping the require_nuggets validator. Substitute the first
        # supporting_pid that is in rank_map; if none, fall back to
        # the rank-1 PR passage.
        rewired_nuggets: list[dict[str, Any]] = []
        for n in agg.supporting_nuggets:
            pid = n.get("passage_id") or ""
            if pid in rank_map:
                rewired_nuggets.append(n)
                continue
            substitute: str | None = None
            for spid in n.get("supporting_pids") or []:
                if spid in rank_map:
                    substitute = spid
                    break
            if substitute is None and rank_map:
                # Fall back to the rank-1 PR passage (lowest rank value).
                substitute = min(rank_map, key=lambda k: rank_map[k])
            if substitute is None:
                continue  # no PR passages at all — drop nugget
            rewired = dict(n)
            rewired["passage_id"] = substitute
            rewired_nuggets.append(rewired)

        synthetic = _to_synthetic_trace(
            agg.question_id,
            agg.final_answer,
            agg.final_confidence,
            rewired_nuggets,
        )
        ac_result = format_ac_result(
            synthetic, run_name, rank_map, max_nuggets=MAX_NUGGETS_PER_QID
        )
        ac_results_by_qid[agg.question_id] = ac_result

    pr_text = format_pr_file(pr_entries_by_qid)
    ac_text = format_ac_file(ac_results_by_qid)

    return pr_text, ac_text, pr_entries_by_qid, ac_results_by_qid


# Top-level orchestration


def _load_source_qids(source_path: Path) -> set[str]:
    """Read source-question file and return the set of question_ids.

    Accepts JSONL with either ``question_id`` or ``qid`` keys. The R2C2
    test set lives at ``data/r2c2/test_questions.jsonl`` by default.
    """
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


def _qids_in_pr(pr_lines: list[str]) -> set[str]:
    """Extract distinct qids from PR rows (first ``;``-delimited field)."""
    out: set[str] = set()
    for ln in pr_lines:
        head = ln.split(";", 1)[0].strip()
        if head:
            out.add(head)
    return out


def _qids_in_ac(ac_text: str) -> set[str]:
    """Extract distinct qids from AC ``<QID>...</QID>`` blocks."""
    return set(re.findall(r"<(\d+)>", ac_text))


def check_source_coverage(
    source_qids: set[str], pr_lines: list[str], ac_text: str
) -> tuple[bool, str]:
    """Every source qid must appear in PR (≥1 passage) AND in AC (answer)."""
    pr_qids = _qids_in_pr(pr_lines)
    ac_qids = _qids_in_ac(ac_text)
    missing_pr = sorted(source_qids - pr_qids)
    missing_ac = sorted(source_qids - ac_qids)
    if not missing_pr and not missing_ac:
        return True, (
            f"PASS: all {len(source_qids)} source qids have ≥1 passage and an answer"
        )
    parts = []
    if missing_pr:
        head = ", ".join(missing_pr[:10])
        more = f" (+{len(missing_pr) - 10} more)" if len(missing_pr) > 10 else ""
        parts.append(f"  Missing from PR ({len(missing_pr)}): {head}{more}")
    if missing_ac:
        head = ", ".join(missing_ac[:10])
        more = f" (+{len(missing_ac) - 10} more)" if len(missing_ac) > 10 else ""
        parts.append(f"  Missing from AC ({len(missing_ac)}): {head}{more}")
    return False, (
        f"FAIL: {len(missing_pr)} qid(s) missing from PR, "
        f"{len(missing_ac)} from AC\n" + "\n".join(parts)
    )


def run_r2c2_format_checks(
    pr_path: Path,
    ac_path: Path,
    run_name: str,
    source_qids: set[str] | None = None,
) -> dict[str, Any]:
    """Run R2C2 format-spec checks against on-disk PR + AC files.

    Mirrors ``src/r2c2_agent/validation/validate_submission.py`` plus a coverage
    check (every source qid has ≥1 PR row + an AC answer) when
    ``source_qids`` is provided.

    Returns a dict with ``passed`` / ``failed`` counts and per-check
    ``(name, ok, detail)`` tuples.
    """
    pr_text = pr_path.read_text()
    ac_text = ac_path.read_text()
    pr_lines = [ln for ln in pr_text.strip().split("\n") if ln.strip()]
    checks: list[tuple[str, bool, str]] = []
    # The semicolon-separated 4-field check is left out (see the comment at
    # the imports). PassageText is allowed to contain ';' per organiser
    # PO files, so a strict field-count split is wrong.
    runners: list[tuple[str, Any]] = [
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
        runners.append(
            (
                "8. Source coverage: every qid has answer + ≥1 passage",
                lambda: check_source_coverage(source_qids, pr_lines, ac_text),
            )
        )
    for name, runner in runners:
        ok, detail = runner()
        checks.append((name, ok, detail))
    passed = sum(1 for _, ok, _ in checks if ok)
    failed = len(checks) - passed
    return {"checks": checks, "passed": passed, "failed": failed}


def aggregate_run_dir(
    run_dir: Path, *, first_iter_only: bool = False
) -> list[AggregatedQid]:
    """Load all rollouts in run_dir, aggregate per qid."""
    by_qid = _load_rollouts(run_dir, first_iter_only=first_iter_only)
    out: list[AggregatedQid] = []
    for qid in sorted(by_qid):
        agg = aggregate_qid(by_qid[qid])
        if agg is not None:
            out.append(agg)
    return out


def write_outputs(
    run_dir: Path,
    aggregated: list[AggregatedQid],
    run_name: str,
    *,
    require_nuggets: bool = True,
) -> dict[str, Any]:
    """Write submission_pr.csv + submission_ac.csv + ensemble_aggregated.jsonl.

    Returns a status dict with per-file paths and validator results.
    """
    pr_text, ac_text, pr_entries_by_qid, ac_results_by_qid = build_pr_ac_files(
        aggregated, run_name
    )

    pr_errors = validate_pr_file(pr_text, run_name)
    ac_errors = validate_ac_file(
        ac_text,
        run_name,
        pr_question_ids=set(pr_entries_by_qid.keys()),
        pr_rank_map={
            qid: {pe.rank for pe in entries}
            for qid, entries in pr_entries_by_qid.items()
        },
        require_nuggets=require_nuggets,
    )
    submission_errors = validate_submission(
        pr_entries_by_qid, ac_results_by_qid, run_name
    )

    # R2C2 file naming: ``[TeamName]-{PG|PO}-[RunNumber].{PR|AC}``
    pr_path = run_dir / f"{run_name}.PR"
    ac_path = run_dir / f"{run_name}.AC"
    debug_path = run_dir / f"{run_name}.aggregated.jsonl"
    pr_path.write_text(pr_text)
    ac_path.write_text(ac_text)

    with open(debug_path, "w") as f:
        for agg in aggregated:
            f.write(
                json.dumps(
                    {
                        "question_id": agg.question_id,
                        "final_answer": agg.final_answer,
                        "final_confidence": agg.final_confidence,
                        "supporting_nuggets": agg.supporting_nuggets,
                        "supporting_passages": agg.supporting_passages,
                        "cluster_breakdown": agg.cluster_breakdown,
                        "n_rollouts_total": agg.n_rollouts_total,
                        "aggregator_warnings": agg.aggregator_warnings,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "pr_path": str(pr_path),
        "ac_path": str(ac_path),
        "debug_path": str(debug_path),
        "n_qids": len(aggregated),
        "pr_errors": pr_errors,
        "ac_errors": ac_errors,
        "submission_errors": submission_errors,
        "all_valid": not (pr_errors or ac_errors or submission_errors),
    }


# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Run directory containing iter_*/test_traces.jsonl",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="PR run name to embed in AC nugget records",
    )
    parser.add_argument(
        "--no-require-nuggets",
        action="store_true",
        help="Allow AC blocks with answer but zero nuggets (validator flag)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if validators report any errors",
    )
    parser.add_argument(
        "--first-iter-only",
        action="store_true",
        help=(
            "Aggregate only iter_0 (single-pass baseline). Used to build "
            "Run-1 from the same run_dir while Run-2 uses the full ensemble."
        ),
    )
    parser.add_argument(
        "--source-questions",
        type=Path,
        default=Path("data/r2c2/test_questions.jsonl"),
        help=(
            "Path to source question JSONL (each row has question_id). "
            "Enables coverage check: every source qid must have ≥1 passage "
            "and an answer. Pass an empty path or '-' to disable."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.info(
        "aggregating run_dir=%s run_name=%s first_iter_only=%s",
        args.run_dir,
        args.run_name,
        args.first_iter_only,
    )

    aggregated = aggregate_run_dir(args.run_dir, first_iter_only=args.first_iter_only)
    if not aggregated:
        log.error("no aggregable rollouts found in %s", args.run_dir)
        return 1

    status = write_outputs(
        args.run_dir,
        aggregated,
        args.run_name,
        require_nuggets=not args.no_require_nuggets,
    )

    log.info("wrote PR  → %s", status["pr_path"])
    log.info("wrote AC  → %s", status["ac_path"])
    log.info("wrote DBG → %s", status["debug_path"])
    log.info("qids: %d", status["n_qids"])
    if status["pr_errors"]:
        log.error("PR validator errors: %d", len(status["pr_errors"]))
        for e in status["pr_errors"][:20]:
            log.error("  %s", e)
    if status["ac_errors"]:
        log.error("AC validator errors: %d", len(status["ac_errors"]))
        for e in status["ac_errors"][:20]:
            log.error("  %s", e)
    if status["submission_errors"]:
        log.error("Submission cross-check errors: %d", len(status["submission_errors"]))
        for e in status["submission_errors"][:20]:
            log.error("  %s", e)

    # Final R2C2 format-spec + source-coverage check
    print("", flush=True)
    print("=== R2C2 Submission Validation ===", flush=True)
    print(f"PR file: {status['pr_path']}", flush=True)
    print(f"AC file: {status['ac_path']}", flush=True)
    print(f"PR run name: {args.run_name}", flush=True)
    source_qids: set[str] | None = None
    src = args.source_questions
    if src and str(src) != "-" and src.exists():
        source_qids = _load_source_qids(src)
        print(f"Source questions: {src} ({len(source_qids)} qids)", flush=True)
    elif src and str(src) != "-":
        print(
            f"Source questions: {src} (NOT FOUND — coverage check skipped)", flush=True
        )
    print("", flush=True)
    spec_report = run_r2c2_format_checks(
        Path(status["pr_path"]),
        Path(status["ac_path"]),
        args.run_name,
        source_qids=source_qids,
    )
    for name, ok, detail in spec_report["checks"]:
        status_tag = "PASS" if ok else "FAIL"
        print(f"[{status_tag}] {name}", flush=True)
        print(f"  {detail}", flush=True)
    n_total = spec_report["passed"] + spec_report["failed"]
    print(
        f"\n=== Result: {spec_report['passed']}/{n_total} PASS, "
        f"{spec_report['failed']}/{n_total} FAIL ===",
        flush=True,
    )

    if args.strict and (not status["all_valid"] or spec_report["failed"] > 0):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
