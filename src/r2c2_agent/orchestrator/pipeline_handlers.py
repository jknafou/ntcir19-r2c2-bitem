"""Tool handlers for the single-session agentic pipeline.

Three server-side tool handlers and their helpers:

- ``handle_record_evidence`` commits a stance and up to three nuggets for a
  passage, each verified by NLI in three bands.
- ``handle_submit_answer`` emits the final answer, computes its confidence
  and terminates the loop.
- ``wrap_tool_response`` decorates every tool response with the ``budget``
  envelope, ``already_recorded`` flags and deterministic ``reflect`` nudges
  (priority-ordered, 2-turn cooldown).

Also the seeder ``seed_db_empty`` (an empty PipelineDB holding only the
__refusal__ virtual pid) and ``diagnose_nli_failure`` (a heuristic
classifier of NLI rejections).

Reuses from nugget_reconciler.py:
    - PipelineDB / Passage / Nugget / TransitionEvent / AnswerState
    - effective_supporting_ids, append_event
    - compute_confidence, synthesize_refusal, validate_finish
    - REFUSAL_PID, _seed_virtual_refusal_passage

Design choices that hold throughout:
    - every tool response carries the budget;
    - the server counts calls before committing, and its count is the
      authority;
    - ``already_recorded`` is a within-session watermark;
    - at most one reflect field per response, with a 2-turn cooldown;
    - NLI runs on a 2-worker async executor;
    - quoted spans match by Ratcliff-Obershelp at 0.92 with a 20-char
      window margin;
    - NLI scores fall into three bands;
    - the __refusal__ virtual pid is kept.

``nugget_reconciler.py`` is also the fallback path behind
``POSTTRACE_PIPELINE=legacy``.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, Literal

import httpx
from pydantic import Field

from r2c2_agent.orchestrator.nugget_reconciler import (
    Nugget,
    Passage,
    PipelineDB,
    TransitionEvent,
    _seed_virtual_refusal_passage,
    append_event,
    compute_confidence,
    effective_supporting_pids,
    max_support_nli_score,
    synthesize_refusal,
    validate_finish,
)

__all__ = [
    "PipelineState",
    "build_phase2_db_yaml",
    "diagnose_nli_failure",
    "force_deliver",
    "handle_batch_record_stance",
    "handle_flag_nugget_bogus",
    "handle_list_nuggets",
    "handle_record_evidence",
    "handle_submit_answer",
    "parse_regex_nugget_fallback",
    "seed_db_empty",
    "wrap_tool_response",
    # Test-facing helpers
    "quoted_span_match",
    "fire_reflect_nudge",
    "extract_question_entities",
    "CANONICAL_REFUSAL_ANSWER",
    "_coerce_answer_for_emit",
    "REGEX_NUGGET_PATTERN",
    "STANCE_TO_PASSAGE_STATUS",
    "PHASE2_MAX_PASSAGES_INLINED",
    "BATCH_RECORD_STANCE_MAX",
    # Finalizer educated-guess ladder.
    "TIER4_FRESH_SESSION_VERIFIED",
    "TIER5_REDUCED_TRUST_CONFIDENCE",
    "VERIFY_ENTAIL_THRESHOLD",
    "VERIFY_CONTRA_THRESHOLD_UPPER",
    "VERIFY_ENTAIL_MARGIN_MIN",
    "MAX_FINALIZER_CALLS",
    "FINALIZER_LLM_TIMEOUT_S",
    "FINALIZER_SEARCH_TIMEOUT_S",
    "FINALIZER_MAX_TOKENS",
    "FINALIZER_TEMPERATURE",
    "FINALIZER_GUIDED_REGEX",
    "_question_shape_classifier",
    "_answer_shape_check",
    "_strip_finalizer_prefixes",
    "set_finalizer_clients",
    "clear_finalizer_clients",
    "set_finalizer_session_messages",
    "set_finalizer_tools",
    "_run_coro_sync",
    # Answer-submission pipeline.
    "answer_search_support",
    "ANSWER_SEARCH_SUPPORT_MAX_ITERS",
    "AUTO_COMMIT_NLI_THRESHOLD",
    "AUTO_COMMIT_CONTRA_MAX",
    "SOFT_INJECT_NLI_MIN",
    "SUBMIT_RETRY_CAP",
    # Recursive submission cycle.
    "TRY_SUBMIT_RECURSION_CAP",
    "try_submit",
    "_synthesize_record_evidence_turn",
    "_reenter_agentic_loop",
    "_ass_recursive_submit_enabled",
]

log = logging.getLogger(__name__)

# Structured-log channel. Emits one JSON line per semantic event to a
# dedicated logger so operators can `jq` the post-run stream without
# regex-parsing free text. The plain-text ``log.info`` calls stay —
# terminal readability is the other half of the instrumentation budget.
_events_log = logging.getLogger("pipeline.events")


def _emit_event(event: str, **fields: Any) -> None:
    """Emit a JSON line to the ``pipeline.events`` logger.

    Kept local (rather than imported from ``r2c2_agent.agent.obs``) so the
    handlers module stays importable without the r2c2_agent package on
    sys.path — tests and the orchestrator both import this as a standalone
    script. ``obs.emit_event`` is the driver-side twin.
    """
    payload: dict[str, Any] = {"event": event}
    payload.update(fields)
    try:
        _events_log.info(json.dumps(payload, default=str, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        _events_log.exception("pipeline.events emit_event failed for %s", event)


# Constants

# Canonical refusal answer text surfaced on submit_answer(refusal=True).
# Phrasing chosen to match the ``REFUSAL_REGEX`` trigger words
# ("insufficient"/"cannot"/"no evidence"/"unable"/"refuse"/"nothing has/was
# been found") so ``validate_finish(low_confidence_refusal)`` accepts it.
#
# Single source of truth: ``r2c2_agent.agent.pipeline._r2c2_rules.CANONICAL_REFUSAL_TEXT``,
# aliased here under the name the call sites use.
# ``validate_r2c2_answer(CANONICAL_REFUSAL_ANSWER)`` must pass.
from r2c2_agent.agent.pipeline.answer_format import (  # noqa: E402
    ANSWER_MAX_LEN as _ANSWER_MAX_LEN,
)
from r2c2_agent.agent.pipeline.answer_format import (  # noqa: E402
    CANONICAL_REFUSAL_TEXT as _CANONICAL_REFUSAL_TEXT,
)
from r2c2_agent.agent.pipeline.answer_format import (  # noqa: E402
    validate_r2c2_answer as _validate_r2c2_answer,
)

CANONICAL_REFUSAL_ANSWER = _CANONICAL_REFUSAL_TEXT


def _coerce_answer_for_emit(
    raw_answer: str | None,
    qtype: str | None = None,
    *,
    site: str = "unknown",
) -> tuple[str, str]:
    """Central "no right to fail" answer-emission gate.

    Every Tier-2+ emission site in this module routes its candidate
    answer through this helper before stamping it onto
    ``db.answer.draft_answer`` / the force_deliver envelope. Contract:

      * On ``ok=True`` → return ``(cleaned, "")``. The cleaned string is
        safe to emit verbatim (whitespace normalized, no ``\\n\\r\\t``,
        no ``;``, no sentinel leak, within length cap).
      * On ``ok=False`` → substitute ``CANONICAL_REFUSAL_ANSWER`` at
        the caller's floor confidence and log a WARN with the failure
        reason. Return ``(CANONICAL_REFUSAL_ANSWER, reason)``.

    Never raises. The caller uses the returned ``reason`` as a telemetry
    tag (empty on success).

    Args:
        raw_answer: Candidate answer string pulled from the tier's
            extraction path (or None).
        qtype: Optional question type, passed through to the validator.
        site: Short string identifying the emission site, used only in
            the WARN log on substitution (e.g. ``"tier4"``,
            ``"force_deliver:tier1"``). Purely telemetry.

    Returns:
        ``(answer_to_emit, reason)`` — emit ``answer_to_emit`` verbatim.
        ``reason`` is empty on success, or a short tag ("contains_semicolon",
        "sentinel_leak", "too_long", "empty") on substitution.
    """
    result = _validate_r2c2_answer(raw_answer or "", qtype)
    if result.ok:
        return result.cleaned, ""
    log.warning(
        "pipeline answer_format qid=? site=%s substitute_refusal reason=%s "
        "cleaned_preview=%r",
        site,
        result.reason,
        (result.cleaned or "")[:120],
    )
    return CANONICAL_REFUSAL_ANSWER, result.reason


# Pre-compiled splitter for ``_extract_first_short_token`` — peels the
# first short token off a malformed answer (e.g. "6: Memento (2000),
# ..." → "6"). Splits on the first ``: ; , ( - —`` or newline, whichever
# comes first. Tabs / carriage returns are already squashed by
# ``_validate_r2c2_answer`` cleanup; we keep them in the regex anyway
# for robustness against direct callers.
_FIRST_TOKEN_SPLIT_RE = re.compile(r"[:;,()\n\r\t—–]|(?:\s-\s)")


def _extract_first_short_token(raw: str | None) -> str | None:
    """Peel the leading short token off a malformed answer.

    Used by the force_deliver format-gate rescue. When the picked
    answer fails ``_validate_r2c2_answer`` (typically ``too_long`` for
    a model reply like ``"6: Memento (2000), Inception (2010), ..."``),
    the first delimiter-bounded token is often a valid R2C2 answer
    (``"6"``). The caller validates the returned token through the
    same R2C2 gate and feeds it to ``_best_non_empty_answer`` so the
    NLI verifier gets a chance.

    Returns:
        The leading token, stripped, ``≤ANSWER_MAX_LEN`` chars, or
        ``None`` when ``raw`` is empty or the leading token is itself
        empty after stripping. Returns the input unchanged when no
        delimiter is found and the input already fits the cap (caller
        will detect rescue == picked_answer and abort the rescue).
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    parts = _FIRST_TOKEN_SPLIT_RE.split(text, maxsplit=1)
    head = (parts[0] if parts else text).strip()
    head = re.sub(r"\s+", " ", head).strip()
    if not head:
        return None
    if len(head) > _ANSWER_MAX_LEN:
        return None
    return head


# Budget caps for the unified loop.
# MAX_CALLS_TOTAL leaves slack for the exhaustion gate in the refusal
# branch of handle_submit_answer: with MIN_CALLS_BEFORE_REFUSAL=12 and two
# possible rejection cycles, the agent needs 20-24 calls after a rejection
# to satisfy the gate, plus room for a final submit_answer.
MAX_CALLS_TOTAL = 30  # total tool calls per qid
MAX_NUGGETS_HARD = 10  # R2C2 contract hard cap

# Per-call record_evidence nugget cap.
MAX_NUGGETS_PER_CALL = 3

# Exhaustion-gate thresholds: the refusal path in handle_submit_answer
# stays closed until the agent has gathered evidence across the 4 MCP
# retrieval tools.
MIN_CALLS_BEFORE_REFUSAL = 12
MIN_UNIQUE_QUERIES = 6
MIN_GET_PASSAGE_CALLS = 2
MIN_EXTRACT_OR_TOPCHUNK_READ = 1
MAX_JACCARD_CONSECUTIVE_QUERIES = 0.70
MAX_REFUSAL_REJECTIONS = 2

# Every terminal exit emits a non-empty answer with confidence >=
# CONFIDENCE_FLOOR. Under R2C2 Brier scoring, a guess at conf=0.05 that is
# right 1-in-20 scores strictly better than a zero-confidence refusal.
CONFIDENCE_FLOOR = 0.05
# Tier-4 (any real passage exists, no supporting nugget) — calibrated low
# confidence. Picked so repeated guesses net positive under Brier when
# accuracy >= ~0.05.
FORCE_DELIVER_TIER4_CONFIDENCE = 0.10
# Tier-5 (no real passage ever retrieved) — the last resort. Kept distinct
# so the audit trail can tell a corpus-silent refusal from a didn't-commit
# one.
FORCE_DELIVER_TIER5_CONFIDENCE = 0.05

# Finalizer educated-guess ladder.
#
# Tier-4 (fresh-session, closed-book) runs on the same vLLM instance as the
# main agentic loop. It builds a fresh [{system}, {user}] list with no tool
# history and no passages, and needs no separate verifier model.
#
# Verified-guess emit confidence. A guess that is unverified, fails the
# shape check, is UNKNOWN or times out does not emit: it falls through, and
# so does a guess refuted by NLI (p_contra above threshold). When tier-4
# falls through, force_deliver lands at the tier-5 canonical refusal.
# Passage titles are not an answer source under this ladder.
TIER4_FRESH_SESSION_VERIFIED = 0.15
# Tier-5 reduced-trust: the prior rollout's answer passed the hardgate, but
# an infrastructure failure (timeout, budget, exception) prevented
# re-verification in this rollout. It emits below seed-rescue (0.15), which
# has fresh corpus corroboration, and above canonical refusal (0.05),
# because a verified prior still stands behind it. The value is flat and
# not a function of prior_rollout_confidence: compute_confidence's
# {base=0.85 + 0.03 × direct_pids, clipped 0.97} forces
# prior_rollout_confidence into the 3-point set {91, 94, 97} under the ≥70
# gate, so any continuous function over the admitted range collapses to a
# constant under the cap. That holds for the additive formula; with
# R2C2_DECOUPLED_SCORING=1 the formula is multiplicative, and the same
# constant remains a sensible floor.
TIER5_REDUCED_TRUST_CONFIDENCE = 0.10

# DeBERTa-NLI verification thresholds. verified = (p_entail >= entail_thr)
# and (p_contra < contra_thr_upper). Separate `refuted` signal = (p_contra
# >= 0.70) is logged for telemetry but treated identically to
# "not verified" — fall through.
VERIFY_ENTAIL_THRESHOLD = 0.60
VERIFY_CONTRA_THRESHOLD_UPPER = 0.40
# Minimum gap between max-entail and max-contra for a span to count
# as verified. Without this, a span at pe=0.60, pc=0.39 (margin 0.21)
# slips through despite a 39% DeBERTa contradiction signal — the two
# maxima are taken independently across up to 3 spans, so the sum can
# exceed 1.0 and both can be non-trivial. 0.25 excludes the narrowest
# same-span both-non-trivial cases while preserving every clearly-
# verified span (margin >= 0.25 passes any pe >= 0.60 against
# pc <= pe - 0.25, which for pe=0.60 means pc <= 0.35).
VERIFY_ENTAIL_MARGIN_MIN = 0.25

# Finalizer call budget, separate from the 30-call agent budget. Enforced
# via db.finalizer_llm_calls_used + db.finalizer_search_calls_used. 4 total
# covers the worst case: 2 LLM generations and 2 search calls (one verify
# per generation).
MAX_FINALIZER_CALLS = 4

# Per-call wall-clock timeouts; on timeout the tier falls through.
FINALIZER_LLM_TIMEOUT_S = 60.0
# Finalizer MCP search timeout. MCP search latency measured on this corpus
# was min=17.5s, avg=42.3s, max=217.8s; a 5s budget made 96% of finalizer
# verify calls time out, which defeated the ladder. 60s covers the observed
# p95 while still bounding the worst case.
FINALIZER_SEARCH_TIMEOUT_S = 60.0

# Generation parameters for the guess calls. Every tier-2+ answer has one
# shape (R2C2 guidelines, section 4.2: "a word, phrase, sentence, quotation
# from a passage, etc."), so one budget applies to all of them.
#
# FINALIZER_MAX_TOKENS=120 covers a ~20-word English sentence under a
# BPE-32K tokenizer (~30 tokens) with 4× headroom for multi-sentence
# aggregation/post-processing answers. FINALIZER_GUIDED_REGEX caps output
# at 200 chars regardless, so 120 tokens is a soft ceiling that prevents
# runaway generation without cutting off legitimate sentences.
#
# FINALIZER_TEMPERATURE=0.4 allows slight exploration, which helps the
# model escape pattern-completion of refusal stems. The guided regex
# already constrains the output space, so the temperature is safe.
FINALIZER_MAX_TOKENS = 120
FINALIZER_TEMPERATURE = 0.4
FINALIZER_TOP_P = 0.9
FINALIZER_STOP_SEQUENCES: tuple[str, ...] = ("\n\n", "<tool_call", "<|")

# Structured-output regex for tier-2+ generation. Forces the model to emit
# either ``Answer: <payload>`` (1..200 chars, no ``\n`` or ``;`` in the
# payload, so the downstream validator's char-set rules are enforced at
# generation time) or the literal ``UNKNOWN`` escape hatch. The caller
# strips the ``Answer: `` prefix with ``_strip_finalizer_prefixes``, which
# also covers output from a client that ignores the regex. vLLM takes it
# via ``extra_body["guided_regex"]`` on the OpenAI-compatible request.
FINALIZER_GUIDED_REGEX = r"^(Answer: [^\n;]{1,200}|UNKNOWN)$"

# answer_search_support (ASS)
# ASS has no shared per-qid cap: the recursive cascade runs as many ASS
# invocations as the tier structure requires (tier 2 on the model's answer,
# the tier-4 rescues, the tier-5 seed). Each invocation is internally
# capped at ANSWER_SEARCH_SUPPORT_MAX_ITERS=3 reformulations. The
# ``answer_search_support_calls_used`` counter is telemetry and does not
# gate execution.
# Path 4: minimum prior-rollout confidence to consider seeding.
# Matches ensemble/prior_state.py::load_prior_rollout_answer(min_confidence=70).
PRIOR_ROLLOUT_MIN_CONFIDENCE = 70
# 3 reformulated queries per candidate answer: question+answer ->
# entities+answer -> answer-only.
ANSWER_SEARCH_SUPPORT_MAX_ITERS = 3
# Auto-commit a verified_supporting nugget when a retrieved span entails
# the (question, candidate_answer) claim above this threshold with low
# contradiction. Tight bounds — counterfeit-nugget prevention.
AUTO_COMMIT_NLI_THRESHOLD = 0.85
AUTO_COMMIT_CONTRA_MAX = 0.25
# Softer band: surface the passage to the model as a reflect hint without
# committing a nugget. Below SOFT_INJECT_NLI_MIN we silently drop.
SOFT_INJECT_NLI_MIN = 0.60

# The 3-retry budget lets the model self-correct before the server
# intervenes: only after three rejected submits is answer_search_support
# budget spent.
SUBMIT_RETRY_CAP = 5

# Thin-evidence hardgate escape.
# Under R2C2_DECOUPLED_SCORING=0: the primary admission criterion for
#   n_eff=1 commits. The compute_confidence formula caps at base+0.03=0.58
#   (supportability auto-set to 'insufficient' for n_eff<2), so without this
#   escape n_eff=1 would never clear the hardgate's `conf >= 0.50` check.
# Under R2C2_DECOUPLED_SCORING=1: simple questions with high NLI land at
#   conf ≈ 0.77, so the hardgate's `conf < 0.50` trap rarely fires. The
#   escape still rescues the narrow case of high-NLI evidence with low
#   directness (e.g., small token overlap between answer and nugget) where
#   the multiplicative formula produces conf < 0.50 despite strong entailment.
# Kept under both flags as a second line of defence.
THIN_EVIDENCE_NLI_THRESHOLD = 0.90
THIN_EVIDENCE_CONF_FLOOR = 0.55


def _thin_evidence_clears(pipeline_db: Any, n_eff: int, conf: float) -> bool:
    """Return True if a thin-evidence commit is allowed.

    Thin evidence = exactly one supporting passage. Permitted to commit
    only when the NLI score on that passage is high enough
    (``max_support_nli_score >= THIN_EVIDENCE_NLI_THRESHOLD``, i.e.
    0.90) to offset the lack of corroboration.

    The ``conf >= THIN_EVIDENCE_CONF_FLOOR`` check looks like a broad
    gate but is narrowly scoped. Under ``handle_submit_answer`` the
    supportability is auto-pinned to "insufficient" whenever n_eff<2,
    which forces ``compute_confidence`` base=0.55. So conf starts at
    0.55 exactly (plus 0.03 per direct pid, plus 0.015 per s_elim,
    minus 0.15 per weakens tag). The floor is only effective when a
    ``[weakens]`` tag in the rationale drags conf below 0.55 — a rare
    path (<5% of thin-evidence attempts). In the common case the floor
    is trivially satisfied.

    Why keep it: the weakens-tag case is a legitimate safety valve —
    a model that self-reports an objection to its own claim shouldn't
    then get a thin-evidence commit. Dropping the floor would lose
    that safety net. Raising it would introduce a new arithmetic
    boundary with no semantic improvement.
    """
    if n_eff != 1:
        return False
    if conf < THIN_EVIDENCE_CONF_FLOOR:
        return False
    return max_support_nli_score(pipeline_db) >= THIN_EVIDENCE_NLI_THRESHOLD


def _route_to_search_support_enabled() -> bool:
    """Feature flag: route hardgate-exhausted submits through
    ``answer_search_support`` (Path 2) before falling to ``force_deliver``.

    Default on. Set ``ROUTE_TO_SEARCH_SUPPORT=0`` to restore the legacy
    path (retries-exhausted → force_deliver directly) for A/B or
    emergency rollback without a code change.
    """
    val = os.environ.get("ROUTE_TO_SEARCH_SUPPORT", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def _refusal_routes_through_force_deliver() -> bool:
    """Whether a model refusal is routed through ``force_deliver``.

    Returns True (default) when the env var ``R2C2_NO_BARE_REFUSAL`` is
    unset or truthy: every gate-passing model refusal is routed through
    ``force_deliver`` and emits at a low calibrated confidence instead of a
    bare conf=0 commit.

    Set ``R2C2_NO_BARE_REFUSAL=0`` for a bare ``CANONICAL_REFUSAL_ANSWER``
    at conf=0 on model refusal. The switch exists for A/B comparisons and
    emergency rollback; under R2C2 Brier scoring the default is strictly
    better.
    """
    val = os.environ.get("R2C2_NO_BARE_REFUSAL", "1").strip().lower()
    return val not in {"0", "false", "no", "off", ""}


def _finalizer_disabled_by_env() -> bool:
    """Rollback switch for the educated-guess ladder.

    When ``R2C2_FINALIZER_DISABLED=1`` is set, ``set_finalizer_clients``
    keeps the registry at ``(None, None)`` and ``force_deliver`` skips
    tier-4 entirely: any force-delivered exit with no committed answer
    ends in the tier-5 canonical refusal.
    """
    val = os.environ.get("R2C2_FINALIZER_DISABLED", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


# Finalizer client registry and async/sync bridge
# ``force_deliver`` is synchronous and called from both sync (tests) and
# async (loop.py under ``await``) contexts. The clients + session
# messages are stored as ``contextvars.ContextVar`` so each asyncio
# task (= each concurrent request handled by the worker) sees its own
# values. Module-level globals would race under aiohttp's default
# concurrent-request handling.
import contextvars as _cv  # noqa: E402

_FINALIZER_LLM_CLIENT_VAR: _cv.ContextVar[Any] = _cv.ContextVar(
    "_finalizer_llm_client", default=None
)
_FINALIZER_MCP_CLIENT_VAR: _cv.ContextVar[Any] = _cv.ContextVar(
    "_finalizer_mcp_client", default=None
)
# Stored as a tuple so pydantic / pickle / repr can't accidentally
# mutate the shared default.
_FINALIZER_SESSION_MESSAGES_VAR: _cv.ContextVar[tuple[dict[str, Any], ...]] = (
    _cv.ContextVar("_finalizer_session_messages", default=())
)
# Tool catalogue (OpenAI function-calling JSON) stashed per-task so
# ``_reenter_agentic_loop`` can forward it on the recursive submission
# re-entry call. Required for ``tool_choice="required"`` to bind on vLLM
# — OpenAI/vLLM reject ``tool_choice="required"`` when ``tools`` is empty.
_FINALIZER_TOOLS_VAR: _cv.ContextVar[tuple[dict[str, Any], ...]] = _cv.ContextVar(
    "_finalizer_tools", default=()
)


def set_finalizer_clients(llm_client: Any | None, mcp_client: Any | None) -> None:
    """Wire the finalizer educated-guess ladder clients for the current
    asyncio task (or sync thread).

    Called at the top of ``answer_question_pipeline`` and the Phase-2
    loop. Each request runs as its own asyncio task, so contextvar
    values are isolated — two concurrent requests cannot observe each
    other's clients.

    No-op when ``R2C2_FINALIZER_DISABLED`` is truthy — the registry
    stays ``(None, None)`` regardless of what is passed.
    """
    if _finalizer_disabled_by_env():
        _FINALIZER_LLM_CLIENT_VAR.set(None)
        _FINALIZER_MCP_CLIENT_VAR.set(None)
        return
    _FINALIZER_LLM_CLIENT_VAR.set(llm_client)
    _FINALIZER_MCP_CLIENT_VAR.set(mcp_client)


def clear_finalizer_clients() -> None:
    """Reset the context-local registry to (None, None) and drop the
    cached session messages for the current task/thread."""
    _FINALIZER_LLM_CLIENT_VAR.set(None)
    _FINALIZER_MCP_CLIENT_VAR.set(None)
    _FINALIZER_SESSION_MESSAGES_VAR.set(())
    _FINALIZER_TOOLS_VAR.set(())


def _get_finalizer_clients() -> tuple[Any, Any]:
    """Snapshot accessor for tests and the force_deliver body."""
    return _FINALIZER_LLM_CLIENT_VAR.get(), _FINALIZER_MCP_CLIENT_VAR.get()


def set_finalizer_session_messages(messages: list[dict[str, Any]]) -> None:
    """Cache the current chat message list for the tier-4 fresh-session
    guess. Stored per task/thread via ``ContextVar`` — concurrent
    requests never overwrite each other's snapshots.

    ``None`` / empty-list inputs clear the cache.
    """
    _FINALIZER_SESSION_MESSAGES_VAR.set(tuple(messages) if messages else ())


def _get_finalizer_session_messages() -> list[dict[str, Any]]:
    return list(_FINALIZER_SESSION_MESSAGES_VAR.get())


def set_finalizer_tools(tools: list[dict[str, Any]] | None) -> None:
    """Cache the OpenAI-format tool catalogue for the recursive
    submission re-entry. Required so ``_reenter_agentic_loop`` can pass
    ``tools=[...]`` alongside ``tool_choice="required"`` — vLLM rejects
    the constraint without a catalogue.

    ``None`` / empty-list inputs clear the cache. Stored per
    task/thread via ``ContextVar``: concurrent requests never overwrite
    each other's snapshots.
    """
    _FINALIZER_TOOLS_VAR.set(tuple(tools) if tools else ())


def _get_finalizer_tools() -> list[dict[str, Any]]:
    return list(_FINALIZER_TOOLS_VAR.get())


def _run_coro_sync(coro: Any, timeout: float) -> Any:
    """Run an async coroutine from a synchronous context with a hard
    wall-clock cap. Propagates return values, exceptions and the
    current ``contextvars`` state to the child thread so finalizer
    contextvar lookups inside the coroutine see the caller's clients
    / session messages (not a fresh-thread default of None).

    The caller is synchronous ``force_deliver``, which may itself be
    executing on a thread that has a live event loop (handler stack
    invoked from ``await _dispatch_pipeline_tool``). To avoid "there is no
    current event loop" / "this event loop is already running" errors,
    we spawn a short-lived daemon thread with its own brand-new event
    loop, run the coroutine there with ``asyncio.wait_for``, and join
    the thread with a small grace beyond the timeout.

    On timeout: raises ``TimeoutError`` (built-in). On any other
    exception in the coroutine: re-raises the exception unchanged.
    """
    result: dict[str, Any] = {}
    # Snapshot of the caller's contextvars (finalizer clients + session
    # messages). Propagated to the child-thread event loop via
    # ``ctx.run`` so concurrent requests don't race on module state.
    ctx = _cv.copy_context()

    def _runner() -> None:
        def _do_work() -> Any:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:  # noqa: BLE001
                    pass
                try:
                    loop.close()
                except Exception:  # noqa: BLE001
                    pass

        try:
            result["value"] = ctx.run(_do_work)
        except TimeoutError as exc:
            result["error"] = TimeoutError(str(exc) or f"timeout after {timeout}s")
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=_runner, daemon=True, name="finalizer-coro")
    t.start()
    # Grace = timeout + 2s. If the runner hangs past this, the daemon
    # thread leaks but force_deliver still returns: its outer wrapper
    # catches the TimeoutError raised here.
    t.join(timeout=timeout + 2.0)
    if t.is_alive():
        raise TimeoutError(
            f"finalizer coroutine did not finish within {timeout + 2.0}s "
            "(thread leaked)"
        )
    err = result.get("error")
    if err is not None:
        raise err
    return result.get("value")


# Ratcliff-Obershelp fuzzy-match parameters.
FUZZY_SPAN_THRESHOLD = 0.92
FUZZY_SPAN_WINDOW_MARGIN = 20

# Three-band NLI boundaries applied to the combined hybrid score
# (see ``_hybrid_score`` below), not the raw MiniCheck output.
#
# The hybrid cascade layers paraphrase-shy overrides and hard caps on top
# of raw MiniCheck. A TRUST boundary of 0.80 keeps the default gate strict,
# while the override path (score >= 0.85 when overlap >= 0.85 and no cap
# fired) recovers MiniCheck's paraphrase false negatives without loosening
# the adversarial surface.
NLI_TRUST_THRESHOLD = 0.80
NLI_REFLECT_THRESHOLD = 0.55

# 5-gram window used by ``_claim_5gram_substring_in_span``.
_HYBRID_SUBSTRING_NGRAM = 5
# Jaccard threshold that triggers the paraphrase-shy override.
_HYBRID_OVERLAP_THRESHOLD = 0.85
# Hybrid-cascade trigger range — overrides and LLM-judge fire only when
# raw MiniCheck lands in this uncertain band.
_HYBRID_UNCERTAIN_LO = 0.40
_HYBRID_UNCERTAIN_HI = 0.70
# Lift target for the paraphrase-shy override (score bumped to at least
# this value when the override fires). Chosen above TRUST so the
# override lands in the TRUST band.
_HYBRID_OVERRIDE_LIFT = 0.85
# Adversarial hard caps — applied as ``min(base, CAP)``.
_HYBRID_POLARITY_CAP = 0.20
_HYBRID_ENTITY_CAP = 0.40
_HYBRID_QUANT_CAP = 0.50
# DeBERTa-probability thresholds that trigger the hard / soft reject caps.
# ``p_wrong`` is DeBERTa p_contradiction for affirming claims,
# p_entailment for infirming claims. Exposed as constants so a calibration
# run can tune them without touching the cascade shape.
_HYBRID_HARD_REJECT_P = 0.80
_HYBRID_SOFT_REJECT_P = 0.50
# Uncertain-band tiebreak, served by a DeBERTa-v3-MNLI head.
#
# Runtime selection:
#   - ``DEBERTA_NLI_URL`` env var points at the DeBERTa NLI server. When
#     set, the cascade calls the server for any nugget whose raw MiniCheck
#     score lands in [_HYBRID_UNCERTAIN_LO, _HYBRID_UNCERTAIN_HI) and for
#     which no hard cap or paraphrase-shy override has fired.
#   - When unset (default), the cascade falls through to
#     ``minicheck_default``: deterministic-only scoring, no judge call.
#   - Network / HTTP errors are swallowed and treated as "judge
#     unavailable" → fall-through; a wedged DeBERTa server never
#     blocks record_evidence progress.
DEBERTA_URL = os.environ.get("DEBERTA_NLI_URL", "").rstrip("/")
# CPU-bound DeBERTa-v3-large inference on 1500-char passages takes ~11 s
# on the worker nodes, and up to ~16 s at the 8000-char server truncation
# cap. A 5 s timeout starves the client on nearly every real passage: the
# server keeps computing but the socket closes, aiohttp logs "200 0", and
# the cascade silently falls through to MiniCheck. Override via
# DEBERTA_NLI_TIMEOUT_S if the worker CPU tier changes.
_DEBERTA_TIMEOUT_S = float(os.environ.get("DEBERTA_NLI_TIMEOUT_S", "30"))
# Per-call asyncio timeout for a single _deberta_label_probs invocation
# inside _score_spans_with_nli. Defaults to 2× the HTTP timeout so a wedged
# DeBERTa HTTP call still times out without triggering on a legitimately
# slow inference. Override via DEBERTA_CALL_TIMEOUT_S env var.
_DEBERTA_CALL_TIMEOUT_S = float(
    os.environ.get("DEBERTA_CALL_TIMEOUT_S", str(_DEBERTA_TIMEOUT_S * 2))
)

# No per-call NLI read timeout. A wedged NLI server is bounded by the
# orchestrator's per-qid wall-clock, not by a cascading per-call timeout.
# The network connect timeout stays at 30 s (in the httpx.AsyncClient
# below) so a dead service fails fast, while the read phase waits
# indefinitely so that legitimately slow responses (loaded GPU, queued
# batch) are not cancelled.
# NLI batches run on 2 concurrent workers.
NLI_EXECUTOR_WORKERS = 2

# Reflect cooldown.
REFLECT_COOLDOWN_TURNS = 2

# Periodic reminder nudge to push the model toward submit_answer when it
# has the evidence but hasn't committed. Fires every N calls (modulo) as
# an additional source feeding ``fire_reflect_nudge`` — the 2-turn
# cooldown still dominates, so back-to-back spam is prevented.
REMINDER_NUDGE_EVERY_N_CALLS = 10

# Rationale / claim length guards.
MAX_RATIONALE_CHARS = 400
MAX_CLAIM_CHARS = 400
MIN_CLAIM_CHARS = 1

# Stance → Passage.status mapping. The Pydantic Passage.status Literal only
# accepts the legacy states; the stances "supporting"/"irrelevant"/"context"
# map onto them. "context" leaves the passage accepted for downstream
# anchoring but by convention triggers no nugget extraction (the LLM
# records zero nuggets on a context stance).
STANCE_TO_PASSAGE_STATUS: dict[str, str] = {
    "supporting": "accepted",
    "counter_evidence": "counter_evidence",
    "irrelevant": "rejected",
    "context": "accepted",
}

# Valid stance literals for arg validation.
_VALID_STANCES = frozenset(STANCE_TO_PASSAGE_STATUS.keys())

# Valid polarity literals.
_VALID_POLARITIES = frozenset({"affirming", "infirming"})

# Capitalized-entity regex (for the no_answer_keyword_search nudge and diagnose_nli_failure).
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Negation tokens used by diagnose_nli_failure negation_miss heuristic.
_NEGATION_TOKENS = frozenset({"not", "no", "never", "cannot", "n't"})

# Universal-quantifier tokens used by scope_broadening heuristic.
_UNIVERSAL_TOKENS = frozenset({"all", "every", "always", "any"})

# Year / date regex for temporal_drift heuristic.
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")

# Digit-sequence regex for quantifier_drift heuristic.
_DIGIT_SEQ_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Answer-shape classifier for the finalizer educated-guess ladder
# Catches a person-name title such as "Kang Min-ah" emitted as an
# "occupation" answer. _answer_shape_check runs after the guess LLM
# returns and before NLI verification; shape-mismatched guesses do not
# emit and do not consume the verification budget.
_SHAPE_PATTERNS: dict[str, re.Pattern[str]] = {
    # Covers "Luke Skywalker", "Kang Min-ah", "Woo Min-ho", "Kim Kang-woo",
    # "Isaac Asimov". The `[A-Z]?` after hyphen admits hyphenated second
    # syllables that are lowercase-only in Korean/Vietnamese romanization.
    "person_name": re.compile(r"^[A-Z][a-z]+(?:[- ][A-Z]?[a-z]+){1,3}$"),
    "count_number": re.compile(r"^\d{1,4}$"),
    "count_written": re.compile(
        r"^(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven"
        r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen"
        r"|nineteen|twenty)$",
        re.IGNORECASE,
    ),
    "year": re.compile(r"^(1[5-9]\d\d|20\d\d|21\d\d)$"),
    "date": re.compile(
        r"^(January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+\d{1,2},?\s+\d{4}$"
    ),
    "noun_phrase": re.compile(r"^[A-Za-z][A-Za-z0-9 ,.'\-]{1,60}$"),
    "list_csv": re.compile(r"^([A-Z][^,]{1,40})(, [A-Z][^,]{1,40}){1,9}$"),
    # "sentence" accepts long-form prose answers (e.g. the full statement of
    # a rule or law). Requires 6+ whitespace-separated tokens so a person
    # name ("Kang Min-ah", "Isaac Asimov") cannot pass as a sentence, and
    # 20-300 chars overall so we don't admit a single long token or a
    # runaway paragraph.
    "sentence": re.compile(r"^[A-Z][A-Za-z0-9 ,.;:'\-()/?!\"]{20,300}$"),
}

# Per-qtype allowed shapes. The negative rule in _answer_shape_check
# further strips "person_name" when the occupation-question pattern
# matches. Unknown qtype falls back to the "simple" set.
_QTYPE_TO_EXPECTED_SHAPES: dict[str, frozenset[str]] = {
    "simple": frozenset({"person_name", "noun_phrase", "sentence", "year", "date"}),
    "multi_hop": frozenset({"person_name", "noun_phrase", "sentence"}),
    "aggregation": frozenset({"count_number", "count_written", "list_csv"}),
    "comparison": frozenset({"person_name", "noun_phrase", "sentence"}),
    "simple_w_condition": frozenset(
        {"person_name", "noun_phrase", "sentence", "year", "date"}
    ),
    "post_processing": frozenset({"noun_phrase", "count_number", "year", "date"}),
}

# Occupation-question detector. On match, _answer_shape_check removes
# "person_name" from the allowed-shapes set so a title like "Kang Min-ah"
# cannot be emitted as the main character's occupation.
_OCCUPATION_Q_RE = re.compile(
    r"\b(occupation|profession|job|role|career|what\s+does\s+\S+\s+do\s+for)",
    re.IGNORECASE,
)

# NUGGET[pid]: claim [stance] regex fallback. Scrapes free-text nuggets the
# LLM typed in prose instead of calling record_evidence. The pid pattern is
# loose ([A-Za-z0-9_]+) because corpus pids include forms like
# "Q14320__00003"; membership is validated against db.passages before
# committing.
REGEX_NUGGET_PATTERN = re.compile(
    r"NUGGET\[(?P<pid>[A-Za-z0-9_]+)\]\s*:\s*"
    r"(?P<text>[^\n]{10,400}?)"
    r"(?:\s*\[(?P<stance>supporting|counter|context)\])?"
    r"(?:\n|$)"
)


# Extended PipelineDB


class PipelineState(PipelineDB):
    """PipelineDB subclass carrying the budget and termination state of the pipeline.

    Subclass (not monkey-patch) so every handler can type-annotate against
    ``PipelineDB`` and still see the extra fields via ``isinstance``.
    """

    calls_used: int = 0
    calls_total: int = MAX_CALLS_TOTAL
    nuggets_recorded: int = 0
    recorded_pids: list[str] = Field(default_factory=list)
    current_turn: int = 0
    last_reflect_turn: int = -10
    terminated: bool = False
    answer_submit_retries: int = 0
    # counts every refusal-branch validate_finish rejection; the
    # force-deliver trigger fires at >=1 (refusal has no retry design).
    refusal_validator_rejects: int = 0
    # memoised force_deliver envelope — re-calls on an already
    # force-delivered DB return this instead of re-doing the work.
    force_deliver_envelope: dict[str, Any] = Field(default_factory=dict)
    # Per-pid record_evidence rejection history (for the nli_repeat_fail nudge).
    pid_reject_streak: dict[str, int] = Field(default_factory=dict)
    # search_passages query log (for the no_answer_keyword_search nudge).
    search_queries: list[str] = Field(default_factory=list)
    # Per-handler call counts — used by nudges and telemetry.
    calls_used_on_record_evidence: int = 0
    # Per-tool call counters feed the exhaustion_check gate.
    calls_used_on_search_passages: int = 0
    calls_used_on_get_passage: int = 0
    calls_used_on_extract_from_passage: int = 0
    calls_used_on_search_entities: int = 0
    # Exhaustion-gate state: how many refusal attempts the gate has
    # rejected for this qid. After MAX_REFUSAL_REJECTIONS the server
    # routes to force_deliver(reason="exhaustion_safety_valve") rather
    # than emit a zero-confidence refusal.
    refusal_exhaustion_rejects: int = 0
    # Reflect-nudge state: rank-1 pids ever seen in a search_passages
    # response, and pids the agent has actually fetched via get_passage.
    # The untouched_top_chunk nudge fires when seen - read is non-empty.
    seen_top_pids: set[str] = Field(default_factory=set)
    top_chunk_read_pids: set[str] = Field(default_factory=set)
    # Phase 2: set while inside a synthesis session (see loop_phase2).
    # Toggles relaxed refusal hard-gate + gates the `created_this_session`
    # list_nuggets filter.
    phase2_active: bool = False
    phase2_session_start_turn: int = 0
    # Ensemble feed-forward: snapshot of the pids this run has explicitly
    # judged (accepted / rejected / counter_evidence / force-delivered as
    # citation). Captured pre-submit so the post-submit auto-reject of
    # unreviewed pids does not leak into the next iteration's exclusion
    # set.
    ensemble_touched_snapshot: set[str] = Field(default_factory=set)

    # Finalizer educated-guess ladder: populated by force_deliver only.
    # All default to zero/empty so older traces replay cleanly.
    finalizer_llm_calls_used: int = 0
    finalizer_search_calls_used: int = 0
    finalizer_tiers_attempted: list[int] = Field(default_factory=list)
    finalizer_tier_emitted: int = 0
    # Stamped as soon as the ladder picks a path, so that a later
    # exception in ``_force_deliver_impl`` can still persist the tag on
    # the emitted envelope.
    finalizer_ladder_path: str | None = None
    tier4_guess_raw: str = ""
    tier4_guess_shape_pass: bool = False
    tier4_support_p_entail: float = 0.0
    tier4_support_p_contra: float = 0.0
    tier4_verified: bool = False

    # Answer-submission pipeline.
    # All default-safe so older traces replay cleanly.
    answer_search_support_calls_used: int = 0
    answer_search_support_iter_count: int = 0
    answer_search_support_auto_nuggets: list[str] = Field(default_factory=list)
    answer_search_support_decisions: list[str] = Field(default_factory=list)
    pending_reflects: list[str] = Field(default_factory=list)
    prior_rollout_answer: str | None = None
    prior_rollout_confidence: int = 0
    prior_rollout_source_iter: int | None = None
    # Every submit_answer invocation (commit or rejection) appends an entry.
    # Final fallback picks the max-confidence entry if all paths fail.
    submit_history: list[dict[str, Any]] = Field(default_factory=list)
    # Per-db NLI score cache. Key = (pid, claim_hash), value = probs
    # dict ({"entailment": float, "neutral": float, "contradiction": float}).
    # Consulted by _score_spans_with_nli before a DeBERTa round-trip; filled
    # on miss. Avoids re-scoring when the same (passage, claim) pair reaches
    # several _score_spans_with_nli calls on the same qid, e.g. a tier-4
    # verify followed by a Path-3 rescue against the same claim. Within a
    # single ASS call the scored_pids dedup covers repeats across
    # iterations; this cache covers the cross-call case. Pydantic cannot
    # serialise tuple-keyed dicts, so this field is marked exclude=True.
    nli_score_cache: dict[tuple[str, str], dict[str, float]] = Field(
        default_factory=dict, exclude=True
    )


# Seeder


def seed_db_empty(
    qid: str,
    question_text: str,
    qtype: str = "",
    *,
    calls_total: int = MAX_CALLS_TOTAL,
) -> PipelineState:
    """Build a fresh DB with only the __refusal__ virtual passage seeded.

    No trace, no retrieved passages: the unified loop builds the DB as it
    runs. Every record_evidence call that references a pid not yet in the DB
    must either come from a get_passage/search_passages response that was
    pre-seeded, or the handler returns unknown_pid. The orchestrator-side
    driver is responsible for seeding passages on MCP tool calls before
    the LLM can record evidence on them.

    Args:
        qid: Question id.
        question_text: Natural-language question.
        qtype: Question type (simple | aggregation | comparison |
            simple_w_condition | multi_hop | post_processing). "" is allowed.
        calls_total: Total tool-call budget for this qid.

    Returns:
        PipelineState typed as PipelineDB; isinstance(db, PipelineDB) holds.
    """
    db = PipelineState(
        question_id=qid,
        question=question_text,
        question_type=qtype,
        calls_total=calls_total,
    )
    _seed_virtual_refusal_passage(db)
    return db


# Quoted-span fuzzy matcher


def quoted_span_match(span: str, passage_text: str) -> bool:
    """Accept if span appears verbatim or Ratcliff-Obershelp ≥ 0.92.

    Primary: exact substring.
    Fallback: fuzzy match over a window anchored at the best offset with
    ±``FUZZY_SPAN_WINDOW_MARGIN`` char margin, using
    ``difflib.SequenceMatcher.ratio()``.
    """
    if not span or not passage_text:
        return False
    if span in passage_text:
        return True

    span_len = len(span)
    text_len = len(passage_text)
    if span_len == 0:
        return False

    # Anchor: longest matching block fixes an approximate alignment. Then
    # slide a span-length window around that anchor with ±margin offsets and
    # take the best Ratcliff-Obershelp ratio.
    sm_full = difflib.SequenceMatcher(a=span, b=passage_text, autojunk=False)
    block = sm_full.find_longest_match(0, span_len, 0, text_len)
    anchor_start = max(0, block.b - block.a) if block.size else 0

    best_ratio = 0.0
    step = max(1, FUZZY_SPAN_WINDOW_MARGIN // 4)
    for off in range(
        max(0, anchor_start - FUZZY_SPAN_WINDOW_MARGIN),
        min(text_len, anchor_start + FUZZY_SPAN_WINDOW_MARGIN) + 1,
        step,
    ):
        window_text = passage_text[off : off + span_len]
        if not window_text:
            continue
        r = difflib.SequenceMatcher(a=span, b=window_text, autojunk=False).ratio()
        if r > best_ratio:
            best_ratio = r
            if best_ratio >= FUZZY_SPAN_THRESHOLD:
                return True
    return False


# Diagnose NLI failure (heuristic)


def diagnose_nli_failure(span: str, claim: str) -> str:
    """Return a typed failure reason (heuristic, no ML).

    Priority order: quantifier_drift → negation_miss → entity_swap →
    scope_broadening → temporal_drift → generic_low_overlap. The first
    matching trigger wins; the span/claim may satisfy more than one
    heuristic, and the most informative is returned.
    """
    span = span or ""
    claim = claim or ""
    span_low = span.lower()
    claim_low = claim.lower()

    # Pre-compute year / non-year digits so a year-only mismatch routes to
    # temporal_drift ("1977" belongs in temporal_drift, not
    # quantifier_drift).
    span_years = set(_YEAR_RE.findall(span))
    claim_years = set(_YEAR_RE.findall(claim))
    span_digits_all = set(_DIGIT_SEQ_RE.findall(span))
    claim_digits_all = set(_DIGIT_SEQ_RE.findall(claim))
    span_nonyear_digits = span_digits_all - span_years
    claim_nonyear_digits = claim_digits_all - claim_years

    # 1. quantifier_drift — non-year digit sequence mismatch.
    if span_nonyear_digits.symmetric_difference(claim_nonyear_digits):
        return "quantifier_drift"

    # 2. negation_miss — one has negation, other doesn't.
    span_neg = _has_negation(span_low)
    claim_neg = _has_negation(claim_low)
    if span_neg != claim_neg:
        return "negation_miss"

    # 3. entity_swap — claim names a capitalized entity absent from span.
    # Decompose multi-word entities to their tokens so "Luke" ⊂ "Luke
    # Skywalker" counts as a hit (the regex returns the longest run).
    claim_entity_tokens: set[str] = set()
    for e in _ENTITY_RE.findall(claim):
        claim_entity_tokens.update(w.lower() for w in e.split())
    span_entity_tokens: set[str] = set()
    for e in _ENTITY_RE.findall(span):
        span_entity_tokens.update(w.lower() for w in e.split())
    # Ignore tokens also present as a plain word in span (handles case where
    # span has "Luke" lowercased or part of a larger entity).
    span_words_lower = set(re.findall(r"\b\w+\b", span_low))
    unique_to_claim = claim_entity_tokens - span_entity_tokens - span_words_lower
    if unique_to_claim:
        return "entity_swap"

    # 4. scope_broadening — claim uses universal quantifier, span doesn't.
    span_tokens = set(re.findall(r"\b\w+\b", span_low))
    claim_tokens = set(re.findall(r"\b\w+\b", claim_low))
    claim_has_universal = claim_tokens & _UNIVERSAL_TOKENS
    span_has_universal = span_tokens & _UNIVERSAL_TOKENS
    if claim_has_universal and not span_has_universal:
        return "scope_broadening"

    # 5. temporal_drift — one has year/date, other lacks it (years already
    # computed above to disambiguate from quantifier_drift).
    if span_years.symmetric_difference(claim_years):
        return "temporal_drift"

    return "generic_low_overlap"


def _entity_swap_details(span: str, claim: str) -> list[str]:
    """Return entity tokens present in ``claim`` but absent from ``span``.

    Mirrors the ``entity_swap`` check in :func:`diagnose_nli_failure` but
    returns the offending surface forms instead of a category tag. Used by
    :func:`handle_record_evidence` to build a cascade-aware reflect nudge
    that tells the model *which* entity needs an alias nugget first.

    Returns the entity tokens (lowercased) unique to the claim. Empty list
    if no entity swap detected.
    """
    if not span or not claim:
        return []
    claim_entity_tokens: set[str] = set()
    for e in _ENTITY_RE.findall(claim):
        claim_entity_tokens.update(w.lower() for w in e.split())
    span_entity_tokens: set[str] = set()
    for e in _ENTITY_RE.findall(span):
        span_entity_tokens.update(w.lower() for w in e.split())
    span_words_lower = set(re.findall(r"\b\w+\b", span.lower()))
    unique = claim_entity_tokens - span_entity_tokens - span_words_lower
    # Preserve first-appearance order from the claim for the reflect message.
    ordered: list[str] = []
    seen: set[str] = set()
    for e in _ENTITY_RE.findall(claim):
        for w in e.split():
            wl = w.lower()
            if wl in unique and wl not in seen:
                seen.add(wl)
                ordered.append(w)
    return ordered


def _has_negation(text_lower: str) -> bool:
    """Token-level negation presence check."""
    if "n't" in text_lower:
        return True
    tokens = set(re.findall(r"\b\w+\b", text_lower))
    return bool(tokens & (_NEGATION_TOKENS - {"n't"}))


# Hybrid-scoring helpers
#
# The MiniCheck-Flan-T5-Large head is paraphrase-shy: an LLM-as-judge
# audit measured 54% recall on clear entails at 100% precision. The
# cascade below layers cheap
# deterministic signals (token overlap, substring match, polarity /
# entity / quantifier heuristics) on top of raw MiniCheck to recover the
# false-negative band without widening the adversarial surface — the
# paraphrase-shy override only fires when no hard-cap trigger is active.


_WORD_RE = re.compile(r"\w+")


def _tokenize_lower(text: str) -> list[str]:
    """Lowercase unigram tokenizer used by the overlap + substring checks."""
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _jaccard_unigrams(a: str, b: str) -> float:
    """Token-level Jaccard overlap between two strings. Both empty → 0."""
    toks_a = set(_tokenize_lower(a))
    toks_b = set(_tokenize_lower(b))
    if not toks_a and not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(1, len(toks_a | toks_b))


# Question-keyword stopword set (small, curated), used by
# ``_keyword_coverage`` to decide which question tokens must appear in
# the retrieved-passage union before refusal is admissible. Keep short;
# false positives are acceptable — the test is "did any search surface
# this token", which is loose enough that over-filtering hurts more
# than under-filtering.
_KEYWORD_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "for",
        "to",
        "with",
        "by",
        "from",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "who",
        "what",
        "when",
        "where",
        "which",
        "how",
        "why",
        "did",
        "does",
        "do",
        "done",
        "as",
        "that",
        "this",
        "these",
        "those",
        "and",
        "or",
        "but",
        "if",
        "so",
        "has",
        "have",
        "had",
        "it",
        "its",
        "his",
        "her",
        "their",
    }
)


def _keyword_coverage(
    question: str,
    passages: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """Which question keywords appear in at least one retrieved passage.

    Used by the ``uncovered_keywords`` reflect nudge. If the agent
    has searched N times and a must-cover question token has never
    landed in a single retrieved passage, that's a signal the search
    space for that token hasn't been explored.

    Args:
        question: The user's question text.
        passages: ``db.passages`` (pid → ``Passage`` instance, or any
            object with a ``.text`` attribute or ``.get("text")`` method).
            The synthetic ``__refusal__`` pid is skipped automatically
            when it appears.

    Returns:
        ``(covered, missed)`` — both are lowercased token sets. Tokens
        that are in the question's must-cover set and also appear in at
        least one passage land in ``covered``; those absent everywhere
        land in ``missed``.
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _RID

    # 1. Must-cover token set = question tokens (minus stopwords) ∪
    #    lowercased proper-noun entities.
    q_tokens = {
        t
        for t in _tokenize_lower(question or "")
        if t and t not in _KEYWORD_STOPWORDS and len(t) >= 2
    }
    try:
        entities = extract_question_entities(question or "")
        for ent in entities or []:
            for t in _tokenize_lower(ent):
                if t and t not in _KEYWORD_STOPWORDS:
                    q_tokens.add(t)
    except Exception:  # noqa: BLE001
        # extract_question_entities is regex-only; defensive only.
        pass

    if not q_tokens:
        return set(), set()

    # 2. Union of text across all non-refusal passages, tokenized.
    seen_tokens: set[str] = set()
    for pid, p in (passages or {}).items():
        if pid == _RID:
            continue
        text = None
        if hasattr(p, "text"):
            text = getattr(p, "text", None)
        elif isinstance(p, dict):
            text = p.get("text")
        if not text:
            continue
        seen_tokens.update(_tokenize_lower(text))
        # Early exit — once every q_token is accounted for there's no
        # point tokenizing further passages. Saves time on dense runs.
        if q_tokens.issubset(seen_tokens):
            break

    covered = q_tokens & seen_tokens
    missed = q_tokens - seen_tokens
    return covered, missed


def _claim_ngram_substring_in_span(
    claim: str, span: str, n: int = _HYBRID_SUBSTRING_NGRAM
) -> bool:
    """True iff any ``n``-token window of ``claim`` appears verbatim in ``span``.

    Uses lowercased token sequences (not raw substrings) so quotation
    marks / punctuation don't defeat the match. Catches the case where
    the relevant sub-phrase sits inside a longer quoted span.
    """
    claim_toks = _tokenize_lower(claim)
    span_toks = _tokenize_lower(span)
    if len(claim_toks) < n or len(span_toks) < n:
        return False
    span_set = {tuple(span_toks[i : i + n]) for i in range(len(span_toks) - n + 1)}
    for i in range(len(claim_toks) - n + 1):
        if tuple(claim_toks[i : i + n]) in span_set:
            return True
    return False


def _hybrid_score(
    base: float,
    claim: str,
    span: str,
    polarity: str,
    *,
    deberta_probs_fn: _DebertaProbsFn | None = None,
) -> tuple[float, str, dict[str, object]]:
    """Combine MiniCheck ``base`` with DeBERTa probs + lexical statistics.

    The decision surface is:
      - Two neural NLI models (MiniCheck + DeBERTa-v3-MNLI), and
      - Token-level lexical statistics (Jaccard, 5-gram substring).

    There are no curated negation token lists and no
    ``diagnose_nli_failure`` regex heuristics in the scoring path.
    ``diagnose_nli_failure`` is called only downstream, to word the
    reflect message.

    Cascade (priority-ordered):
      1. If DeBERTa gives high-confidence "wrong polarity"
         (p_contradiction ≥0.80 for affirming; p_entailment ≥0.80 for
         infirming) → hard cap 0.20 (REJECT).
      2. If DeBERTa gives moderate-confidence wrong-polarity (≥0.50)
         → soft cap 0.50.
      3. Paraphrase-shy override: if base ∈ [0.40, 0.70) and (Jaccard
         ≥0.85 or 5-gram substring match) and steps 1-2 didn't fire
         → lift to 0.85.
      4. Dual-model blend: if DeBERTa available and 0.20 ≤ base < 0.80
         → score = 0.5·base + 0.5·p_expected (p_entail for affirming,
         p_contradict for infirming).
      5. MiniCheck pass-through otherwise.

    Returns ``(combined_score, rule_name, sub_signals)``. Rule names
    are one of: deberta_hard_reject / deberta_soft_reject /
    lexical_overlap_override / deberta_blend / minicheck_default.

    ``deberta_probs_fn`` injection point — production wires the HTTP
    client to the DeBERTa-MNLI server; tests pass a mock. Returning
    ``None`` (server unreachable) disables the DeBERTa-driven branches;
    the cascade gracefully falls back to lexical+MiniCheck-only.
    """
    overlap = _jaccard_unigrams(claim, span)
    sub5 = _claim_ngram_substring_in_span(claim, span)

    sub_signals: dict[str, object] = {
        "base": round(base, 4),
        "overlap": round(overlap, 4),
        "substring_5gram": sub5,
    }

    # Step 1/2 — DeBERTa polarity check
    deberta: dict[str, float] | None = None
    if deberta_probs_fn is not None:
        try:
            raw = deberta_probs_fn(span, claim)
        except Exception:  # noqa: BLE001
            raw = None
        if isinstance(raw, dict):
            deberta = {
                "entailment": float(raw.get("entailment", 0.0) or 0.0),
                "neutral": float(raw.get("neutral", 0.0) or 0.0),
                "contradiction": float(raw.get("contradiction", 0.0) or 0.0),
            }

    if deberta is not None:
        sub_signals["deberta_entail"] = round(deberta["entailment"], 4)
        sub_signals["deberta_neutral"] = round(deberta["neutral"], 4)
        sub_signals["deberta_contra"] = round(deberta["contradiction"], 4)

        # Polarity-aware "wrong-direction" probability.
        # - affirming claim should entail → contradiction is the flag.
        # - infirming claim should contradict → entailment is the flag
        #   (model recorded infirming but actually entailed — polarity bug).
        # Unknown polarities (shouldn't happen; validated upstream) fall
        # back to treating as affirming.
        if polarity == "infirming":
            p_wrong = deberta["entailment"]
        else:
            p_wrong = deberta["contradiction"]

        if p_wrong >= _HYBRID_HARD_REJECT_P:
            return (
                min(base, _HYBRID_POLARITY_CAP),
                "deberta_hard_reject",
                sub_signals,
            )
        if p_wrong >= _HYBRID_SOFT_REJECT_P:
            return (
                min(base, _HYBRID_QUANT_CAP),
                "deberta_soft_reject",
                sub_signals,
            )

    # Step 3 — paraphrase-shy override
    # Lexical statistic, not a handcrafted rule. Rescues MiniCheck
    # false-negatives where the claim is a verbatim sub-phrase of the
    # span. Gated to the uncertain band so confident MiniCheck decisions
    # are never touched.
    if (
        overlap >= _HYBRID_OVERLAP_THRESHOLD or sub5
    ) and _HYBRID_UNCERTAIN_LO <= base < _HYBRID_UNCERTAIN_HI:
        return (
            max(base, _HYBRID_OVERRIDE_LIFT),
            "lexical_overlap_override",
            sub_signals,
        )

    # Step 4 — dual-model blend
    # When DeBERTa is available and neither the hard/soft rejects nor
    # the paraphrase override decided, blend its polarity-aligned prob
    # with the MiniCheck base for a stronger combined signal. Gated to
    # the uncertain-ish range [0.20, 0.80) so we don't touch
    # already-decisive MiniCheck outputs at either tail.
    if deberta is not None and 0.20 <= base < 0.80:
        p_right = (
            deberta["contradiction"]
            if polarity == "infirming"
            else deberta["entailment"]
        )
        combined = 0.5 * base + 0.5 * p_right
        sub_signals["blended_with"] = round(p_right, 4)
        return (combined, "deberta_blend", sub_signals)

    # Step 5 — pass-through
    return (base, "minicheck_default", sub_signals)


# Answer-shape classifier helpers (finalizer educated-guess ladder)


def _question_shape_classifier(question: str) -> str:
    """Return a qtype-tag string for shape-check lookup.

    The returned tag is one of the six canonical qtypes
    (simple / aggregation / comparison / simple_w_condition /
    multi_hop / post_processing) — matching the keys of
    ``_QTYPE_TO_EXPECTED_SHAPES``. Falls back to "simple" when the
    question is empty or unparseable.

    This is a shallow dispatcher, not a replacement for the upstream
    pipeline's ``question_type`` field: callers pass in the stored
    ``db.question_type`` (authoritative) and use this helper only for
    questions lacking one (e.g. dry-run inputs). The helper's only
    domain knowledge is the 6-tuple of qtypes — it does not try to
    distinguish "occupation" from other simple questions; that is done
    inline by ``_answer_shape_check`` via ``_OCCUPATION_Q_RE``.
    """
    q = (question or "").strip().lower()
    if not q:
        return "simple"
    if q.startswith("how many") or q.startswith("how much") or "list all" in q:
        return "aggregation"
    if q.startswith(("which of", "who is taller", "who is older", "which is")):
        return "comparison"
    if " and " in q and (q.startswith("when") or q.startswith("what year")):
        return "simple_w_condition"
    if "->" in q or "because" in q or " and then " in q:
        return "multi_hop"
    if q.startswith(("when", "what year", "what date")):
        return "post_processing"
    return "simple"


# Patterns that signal the LLM tried to hedge/refuse instead of producing
# a concrete guess. When any of these is present the ladder aborts the
# current tier without spending a verification MCP call — otherwise we
# waste a search + NLI check on a guess that is itself the refusal.
_CANONICAL_REFUSAL_PATTERNS: tuple[str, ...] = (
    "corpus does not support",
    "corpus doesn't support",
    "corpus does not contain",
    "corpus doesn't contain",
    "corpus does not mention",
    "corpus doesn't mention",
    "corpus does not specify",
    "corpus doesn't specify",
    "not specified in the corpus",
    "not mentioned in the corpus",
    "not found in the corpus",
    "insufficient evidence",
    "cannot determine",
    "cannot be determined",
    "cannot answer",
    "unable to answer",
    "no information",
    "not enough information",
    "i don't know",
    "no answer can be provided",
    "based on the passages provided",  # often a refusal preamble
)


def _looks_like_canonical_refusal(text: str) -> bool:
    """Return True if the text carries a refusal/hedge phrase.

    Case-insensitive substring match against ``_CANONICAL_REFUSAL_PATTERNS``.
    Treated as equivalent to an empty / UNKNOWN guess by the ladder.
    """
    if not text:
        return False
    lo = text.lower()
    return any(p in lo for p in _CANONICAL_REFUSAL_PATTERNS)


_FINALIZER_ANSWER_PREFIXES: tuple[str, ...] = (
    "answer:",
    "the answer is",
    "my answer is",
    "best guess:",
    "guess:",
    "a:",
)


# Priority-ordered shape detection — stricter shapes first, so "Kang Min-ah"
# detects as person_name (not the loose noun_phrase), and "1977" detects as
# year (not noun_phrase). The detection result is a single dominant shape,
# which the caller then checks against the per-qtype allowed set.
# Year must precede count_number: a 4-digit token like "1977" matches
# both, but should be detected as a year so _answer_shape_check correctly
# accepts it for "when" questions and rejects it for "how many" questions.
_SHAPE_DETECT_ORDER: tuple[str, ...] = (
    "year",
    "date",
    "count_number",
    "count_written",
    "list_csv",
    "person_name",
    "noun_phrase",
    # ``sentence`` last — strictly longer than noun_phrase's 60-char cap so
    # it only catches prose the shorter shapes reject.
    "sentence",
)


# Trailing-period regex. Strips a trailing ``.`` only when it is not
# preceded by an uppercase letter, so a name with an initial such as
# ``Michael J. Fox.`` loses its terminal period (the character before
# it is not uppercase) while acronyms like ``Ph.D.`` / ``2001 A.D.`` keep
# theirs. Counting periods instead would leave the trailing period on a
# person name with an embedded initial.
_TRAILING_PERIOD_STRIP_RE = re.compile(r"(?<![A-Z])\.$")


def _strip_finalizer_prefixes(raw: str) -> str:
    """Strip leading prose prefixes, surrounding quotes, and a trailing
    period from a finalizer guess. Idempotent under repeated application.

    Trailing-period rule: ``_TRAILING_PERIOD_STRIP_RE`` makes
    "Michael J. Fox." / "Dr. Strange." / "A robot ... humanity." all lose
    their trailing period while "Ph.D." / "2001 A.D." keep theirs
    (an uppercase letter before the period signals an acronym / initial).

    Quote+period pre-pass: strings like ``'"Luke".'`` first lose the
    trailing period so the quote-unwrap guard (which requires equal
    first/last chars) can match on ``'"Luke"'``. Otherwise the malformed
    output ``Answer: "Michael J. Fox".`` stays stuck at
    ``"Michael J. Fox".`` because ``.`` ≠ ``"`` blocks the unwrap.

    UNKNOWN short-circuit: every caller of this helper then checks
    ``cleaned.upper() == "UNKNOWN"``. The stripped form ``"UNKNOWN."``
    keeps its trailing period (``N`` is uppercase), which that check
    would miss. Any ``UNKNOWN[\\.]*`` form is therefore collapsed to bare
    ``UNKNOWN`` so the UNKNOWN-leak guard stays universal.
    """
    if not raw:
        return ""
    s = raw.strip()
    for _ in range(6):
        before = s
        low = s.lower()
        for pfx in _FINALIZER_ANSWER_PREFIXES:
            if low.startswith(pfx):
                s = s[len(pfx) :].strip()
                break
        # Quote+period pre-pass: '"X".' -> '"X"' so the unwrap fires.
        if len(s) >= 3 and s[-1] == "." and s[-2] in ('"', "'", "`"):
            s = s[:-1].strip()
        if s.startswith(('"', "'", "`")) and len(s) >= 2 and s[0] == s[-1]:
            s = s[1:-1].strip()
        # Trailing-period strip — keeps initial/acronym period unchanged.
        if s.endswith(".") and len(s) <= 60:
            s = _TRAILING_PERIOD_STRIP_RE.sub("", s).strip()
        if s == before:
            break
    # UNKNOWN canonicalization: the guided regex emits the sentinel as
    # uppercase ``UNKNOWN``; the trailing-period rule above keeps
    # ``UNKNOWN.`` (N is uppercase), which the caller guards
    # ``cleaned.upper() == "UNKNOWN"`` would miss. Collapse uppercase
    # ``UNKNOWN[\\.]*`` to bare ``UNKNOWN`` without touching a lowercase
    # ``unknown`` noun answer, which a caller without the guided regex
    # might still surface.
    if s.strip(".") == "UNKNOWN":
        return "UNKNOWN"
    return s


def _detect_answer_shape(answer: str) -> str | None:
    """Return the single dominant shape name the answer matches, or None.

    Walks ``_SHAPE_DETECT_ORDER`` priority-first: "Kang Min-ah" returns
    "person_name" (not "noun_phrase"); "1977" returns "year"; "police
    officer" returns "noun_phrase" only because no stricter shape
    matches.

    The bare string ``"UNKNOWN"`` returns ``"noun_phrase"`` here. The
    ``noun_phrase`` regex is a loose last-resort pattern that accepts any
    short alphabetic token, so an uppercase sentinel like ``UNKNOWN``
    matches it. This function is therefore not a reliable "is the agent
    refusing?" check. The UNKNOWN-leak guard lives in
    ``_answer_shape_check``, which rejects
    ``cleaned.upper() == "UNKNOWN"`` before consulting
    ``_detect_answer_shape``. Callers of ``_detect_answer_shape`` that
    need to reject UNKNOWN must replicate that guard or call
    ``_answer_shape_check`` instead.
    """
    if not answer:
        return None
    for shape_name in _SHAPE_DETECT_ORDER:
        pat = _SHAPE_PATTERNS.get(shape_name)
        if pat is None:
            continue
        try:
            if pat.match(answer):
                return shape_name
        except Exception:  # noqa: BLE001
            continue
    return None


def _answer_shape_check(answer: str, qtype: str, question: str) -> bool:
    """True when the cleaned answer's dominant shape is in the allowed set
    for the qtype.

    Rules:
      * Strips finalizer prefixes / quotes / trailing period.
      * An empty / whitespace / "UNKNOWN" / ``<tool_call`` leak answer
        is rejected.
      * Looks up the allowed-shape set for the qtype (unknown qtype →
        "simple" set).
      * Negative rule: when the question matches ``_OCCUPATION_Q_RE``,
        ``"person_name"`` is removed from the allowed set — so a bare
        actor name like "Kang Min-ah" cannot ship as an occupation
        answer.
      * Detection is priority-based (``_detect_answer_shape``): a name
        matches "person_name" (not "noun_phrase"), so removing
        "person_name" from ``allowed`` genuinely excludes names —
        it does not silently re-admit them via the looser noun_phrase
        pattern.

    Never raises. Malformed / huge / binary input returns False.
    """
    try:
        cleaned = _strip_finalizer_prefixes(answer or "")
    except Exception:  # noqa: BLE001
        return False
    if not cleaned:
        return False
    up = cleaned.upper()
    if up == "UNKNOWN" or "<TOOL_CALL" in up or "<|" in cleaned:
        return False
    if len(cleaned) > 200:
        return False

    allowed = set(
        _QTYPE_TO_EXPECTED_SHAPES.get(qtype, _QTYPE_TO_EXPECTED_SHAPES["simple"])
    )
    if _OCCUPATION_Q_RE.search(question or ""):
        allowed.discard("person_name")
    if not allowed:
        return False

    detected = _detect_answer_shape(cleaned)
    return detected is not None and detected in allowed


# Typing shim — the injectable DeBERTa-probs callable. Production
# wires ``_deberta_label_probs``; tests inject mocks. Returning
# ``None`` means "DeBERTa unavailable" and the cascade falls back to
# MiniCheck + lexical only (graceful degradation).
from collections.abc import Callable  # noqa: E402

_DebertaProbsFn = Callable[[str, str], dict[str, float] | None]


# DeBERTa-MNLI clients — (/check_entailment) + (/label_probs)


def _deberta_entail(premise: str, hypothesis: str) -> float | None:
    """Ask the DeBERTa-MNLI server for P(entailment).

    Legacy tiebreak client — kept for external callers that only
    need a scalar. Returns ``None`` on any failure (env unset,
    connection error, HTTP error, malformed body) so the hybrid
    cascade can fall through to ``minicheck_default`` without the
    caller needing to catch.
    """
    if not DEBERTA_URL:
        return None
    import json as _json
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    payload = _json.dumps(
        {
            "passage_text": premise or "",
            "claim_text": hypothesis or "",
        }
    ).encode("utf-8")
    req = _urlreq.Request(
        f"{DEBERTA_URL}/check_entailment",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=_DEBERTA_TIMEOUT_S) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
    except (_urlerr.URLError, TimeoutError, OSError, ValueError):
        return None
    score = body.get("score")
    if not isinstance(score, (int, float)):
        return None
    return max(0.0, min(1.0, float(score)))


def _deberta_label_probs(premise: str, hypothesis: str) -> dict[str, float] | None:
    """Ask the DeBERTa-MNLI server for the full 3-class softmax.

    This is the always-on DeBERTa client: the polarity / entity /
    quantifier detector for the hybrid cascade.

    Returns a dict ``{entailment, neutral, contradiction}`` with
    probabilities summing to ~1.0. Returns ``None`` on any failure so
    the cascade gracefully falls back to MiniCheck-only mode.
    """
    if not DEBERTA_URL:
        return None
    import json as _json
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    payload = _json.dumps(
        {
            "passage_text": premise or "",
            "claim_text": hypothesis or "",
        }
    ).encode("utf-8")
    req = _urlreq.Request(
        f"{DEBERTA_URL}/label_probs",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=_DEBERTA_TIMEOUT_S) as resp:
            status = resp.status
            raw_bytes = resp.read()
    except (_urlerr.URLError, TimeoutError, OSError):
        return None
    # Detect the "server returned 200 but body is empty" failure mode
    # explicitly so it shows up in the worker log as a distinct signal
    # (not silently swallowed as generic "NLI=none").
    if not raw_bytes:
        log.warning(
            "deberta_label_probs: empty body on HTTP %d (server=%s) — "
            "falling back to minicheck_default. Check nli_verifier.py "
            "response-finalize path.",
            status,
            DEBERTA_URL,
        )
        return None
    try:
        body = _json.loads(raw_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict) or "entailment" not in body:
        return None
    try:
        return {
            "entailment": max(0.0, min(1.0, float(body["entailment"]))),
            "neutral": max(0.0, min(1.0, float(body.get("neutral", 0.0)))),
            "contradiction": max(0.0, min(1.0, float(body.get("contradiction", 0.0)))),
        }
    except (TypeError, ValueError):
        return None


# Finalizer educated-guess helpers


def _finalizer_budget_allows(
    db: PipelineState,
    *,
    needs_llm: bool = False,
    needs_search: bool = False,
) -> bool:
    """True when the projected call count stays within the finalizer
    budget (``MAX_FINALIZER_CALLS = 4``, separate from the 30-call agent
    budget).

    The caller increments counters before issuing the call, so a
    timeout still charges budget (the call was issued).
    """
    projected = (
        db.finalizer_llm_calls_used
        + db.finalizer_search_calls_used
        + (1 if needs_llm else 0)
        + (1 if needs_search else 0)
    )
    return projected <= MAX_FINALIZER_CALLS


def _build_finalizer_search_query(db: PipelineState, answer: str) -> str:
    """Construct an answer-seeded ``search_passages`` query for NLI
    verification. Mixes question entities, non-stopword ≥4-char question
    tokens, and the guess verbatim (to preserve proper-noun casing).
    Capped at 400 chars — the MCP server's practical query length.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    try:
        entities = extract_question_entities(db.question or "")
    except Exception:  # noqa: BLE001
        entities = []
    for ent in entities or []:
        for t in (ent or "").split():
            k = t.lower()
            if k and k not in seen:
                seen.add(k)
                tokens.append(t)
    for t in _tokenize_lower(db.question or ""):
        if t in _KEYWORD_STOPWORDS or len(t) < 4:
            continue
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    parts = tokens[:8]
    clean_answer = (answer or "").strip()
    if clean_answer:
        parts.append(clean_answer)
    query = " ".join(p for p in parts if p).strip()
    return query[:400]


def _claim_template_for_verification(question: str, answer: str, qtype: str) -> str:
    """Build an NLI claim string from (question, answer). Strips leading
    wh-word and a subsequent copula to produce a noun-phrase prefix,
    then appends "is <answer>". Aggregation questions get a quote-wrapped
    template to preserve the full question verbatim.

    Stripping the copula matters: "What is the zeroth law?" + answer
    "A robot may not harm humanity..." would otherwise produce "is the
    zeroth law is A robot may not harm humanity...", and DeBERTa scores
    that claim well below the entail threshold because the doubled copula
    is not natural text. With the copula removed the claim reads "the
    zeroth law is <answer>".
    """
    q_raw = (question or "").strip().rstrip("?!.").strip()
    a = (answer or "").strip()
    if not q_raw or not a:
        return f"{question or ''} {answer or ''}".strip()
    if qtype == "aggregation":
        return f"The answer to '{question}' is {a}"
    low = q_raw.lower()
    for wh in (
        "who",
        "what",
        "when",
        "where",
        "which",
        "whose",
        "how many",
        "how much",
        "how",
    ):
        if low.startswith(wh):
            rest = q_raw[len(wh) :].strip()
            if rest:
                q_raw = rest
            break
    low2 = q_raw.lower()
    for cop in ("is ", "are ", "was ", "were "):
        if low2.startswith(cop):
            q_raw = q_raw[len(cop) :].strip()
            break
    return f"{q_raw} is {a}" if q_raw else a


def _strip_refused_submit_answer_turns(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter out assistant turns that called ``submit_answer`` with
    ``refusal=True`` plus their tool-result responses.

    The session snapshot at ladder entry contains every refused
    submit_answer attempt with its rationale ("The corpus does not
    contain..."). An RLHF-trained model such as Qwen3.5 then
    pattern-completes another refusal when nudged for an educated guess.
    Removing the model's own prior refusal commitments from the replay
    preserves the retrieval trajectory (passages seen, nugget judgments)
    while letting the model reason afresh about the question.

    Safety: we never drop tool_call sequences for non-refusal tools
    (search_passages, get_passage, record_evidence, etc.). Only the
    specific assistant+tool-result pair for a refused submit_answer is
    removed. Unpaired orphans (e.g. assistant WRITE with no matching
    tool-result) are preserved as-is.
    """
    import json as _json

    refused_tool_call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if fn.get("name") != "submit_answer":
                continue
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = _json.loads(args_raw)
                except Exception:  # noqa: BLE001
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            if bool(args.get("refusal")):
                tc_id = tc.get("id")
                if tc_id:
                    refused_tool_call_ids.add(str(tc_id))

    if not refused_tool_call_ids:
        return list(messages)

    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                # Keep only non-refused tool_calls from this assistant turn;
                # drop the whole message if all of its tool_calls were refused
                # submit_answer's.
                remaining = [
                    tc
                    for tc in tool_calls
                    if str(tc.get("id")) not in refused_tool_call_ids
                ]
                if not remaining:
                    continue
                if len(remaining) != len(tool_calls):
                    new_msg = dict(msg)
                    new_msg["tool_calls"] = remaining
                    cleaned.append(new_msg)
                    continue
            cleaned.append(msg)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and str(tc_id) in refused_tool_call_ids:
                continue
            cleaned.append(msg)
        else:
            cleaned.append(msg)
    return cleaned


def _build_fresh_session_messages(db: PipelineState) -> list[dict[str, Any]]:
    """Build the 2-message list for the tier-4 fresh-session guess.

    No tool-call history, no retrieved passages, no main agentic system
    prompt — just a closed-book trivia prompt keyed by shape hint plus
    the raw question. The model answers from parametric knowledge only.
    """
    from r2c2_agent.agent.prompt import (
        FINALIZER_FRESH_SESSION_SYSTEM,
        FINALIZER_FRESH_SESSION_USER_TEMPLATE,
        SHAPE_HINT_BY_QTYPE,
    )

    qtype = db.question_type or _question_shape_classifier(db.question or "")
    shape_hint = SHAPE_HINT_BY_QTYPE.get(qtype, SHAPE_HINT_BY_QTYPE["simple"])
    return [
        {
            "role": "system",
            "content": FINALIZER_FRESH_SESSION_SYSTEM.format(shape_hint=shape_hint),
        },
        {
            "role": "user",
            "content": FINALIZER_FRESH_SESSION_USER_TEMPLATE.format(
                question=db.question or ""
            ),
        },
    ]


async def _make_educated_guess(
    db: PipelineState,
    llm_client: Any,
) -> tuple[str, dict[str, Any]]:
    """Run a single fresh-session finalizer LLM call. Returns
    ``(raw_answer, stats)`` where stats carries
    raw/stripped/elapsed/error. Never raises — any exception becomes
    ``("", {"error": str(exc)})``.

    Builds a fresh 2-message list (no tool history, no passages) and
    asks the model for a closed-book guess from parametric knowledge
    only. Used by tier-4 of the educated-guess ladder.
    """
    import time as _time

    t0 = _time.monotonic()
    stats: dict[str, Any] = {
        "raw": "",
        "stripped": "",
        "elapsed_s": 0.0,
        "error": None,
    }

    try:
        messages = _build_fresh_session_messages(db)
        log.info(
            "pipeline guess qid=%s mode=fresh messages=%d (closed-book)",
            db.question_id,
            len(messages),
        )

        user_tag = f"qid={db.question_id or ''};finalizer=tier4"
        # chat() signatures vary across concrete clients; only pass
        # ``user`` / ``max_tokens`` / ``temperature`` / ``guided_regex``
        # when the client accepts them (DummyLLMClient accepts them as
        # noops; older minimal stubs may not).
        chat_kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": None,
            "enable_thinking": False,
        }
        try:
            import inspect as _inspect

            sig = _inspect.signature(llm_client.chat)
            params = sig.parameters
            if "user" in params:
                chat_kwargs["user"] = user_tag
            # One budget for every finalizer call: FINALIZER_MAX_TOKENS /
            # FINALIZER_TEMPERATURE.
            if "max_tokens" in params:
                chat_kwargs["max_tokens"] = FINALIZER_MAX_TOKENS
            if "temperature" in params:
                chat_kwargs["temperature"] = FINALIZER_TEMPERATURE
            # Structured-output enforcement: the prompts request the
            # ``Answer: <x>`` prefix, and the regex forces the model to emit
            # either that or ``UNKNOWN``.
            if "guided_regex" in params:
                chat_kwargs["guided_regex"] = FINALIZER_GUIDED_REGEX
        except (TypeError, ValueError):
            pass

        response = await llm_client.chat(**chat_kwargs)
        raw = (getattr(response, "content", "") or "").strip()
        stats["raw"] = raw
        stripped = _strip_finalizer_prefixes(raw)
        stats["stripped"] = stripped
        stats["elapsed_s"] = _time.monotonic() - t0
        log.info(
            "pipeline guess qid=%s mode=fresh raw=%r stripped=%r dt=%.2fs",
            db.question_id,
            raw[:100],
            stripped[:100],
            stats["elapsed_s"],
        )
        return raw, stats
    except Exception as exc:  # noqa: BLE001
        stats["error"] = f"{type(exc).__name__}: {exc}"
        stats["elapsed_s"] = _time.monotonic() - t0
        log.exception("finalizer LLM call failed (mode=fresh)")
        return "", stats


# Transport-flake retry tunables. ``MCPHTTPClient.call_tool`` opens a
# fresh ``streamablehttp_client`` session per call, and the session
# teardown races with the server, so search_passages can return a
# ClosedResourceError on the in-process MCP transport. The main loop
# recovers through its own retry; these bounds give the finalizer path
# up to 2 extra attempts with small jittered backoff.
_VERIFY_MCP_RETRY_MAX = 3  # total attempts including first
_VERIFY_MCP_RETRY_BACKOFF_S = 0.4


def _verify_outer_timeout_s() -> float:
    """Outer ``_run_coro_sync`` timeout for ``_verify_guess_with_retrieval``.

    Must cover worst-case retry budget inside ``_retrieve_for_support``:

      * ``_VERIFY_MCP_RETRY_MAX`` attempts × ``FINALIZER_SEARCH_TIMEOUT_S``
      * ``_VERIFY_MCP_RETRY_MAX-1`` backoffs × ``_VERIFY_MCP_RETRY_BACKOFF_S`` * n
      * then NLI: up to 3 spans × ``_DEBERTA_TIMEOUT_S``
      * + 2s slack for argument/claim building + log emission

    Sizing it to a single attempt
    (``FINALIZER_SEARCH_TIMEOUT_S + 2.0 + _DEBERTA_TIMEOUT_S * 3``) would
    let the outer wrapper fire before a retry could finish, turning a
    recoverable transport flake into ``verify_timeout``.
    """
    # Per-attempt slack: asyncio.wait_for(5s) has been measured at ~6.65s
    # on mcp_timeout paths, because streamablehttp_client's response
    # decoder does not honour cancellation promptly. 2s of drift per
    # attempt keeps the outer wrapper from preempting a live retry.
    attempt_cancel_drift_s = 2.0
    retrieval_budget = (
        FINALIZER_SEARCH_TIMEOUT_S + attempt_cancel_drift_s
    ) * _VERIFY_MCP_RETRY_MAX
    backoff_sum = sum(
        _VERIFY_MCP_RETRY_BACKOFF_S * (i + 1) for i in range(_VERIFY_MCP_RETRY_MAX - 1)
    )
    nli_budget = _DEBERTA_TIMEOUT_S * 3
    return retrieval_budget + backoff_sum + nli_budget + 2.0


def _mcp_result_looks_transport_flake(result: Any) -> bool:
    """Detect the ``anyio.ClosedResourceError`` / parse-error pattern
    that MCPHTTPClient returns under session-teardown races.

    The client normalizes flakes into ``{"error": ..., "mcp_unhealthy":
    True}`` (see ``src/r2c2_agent/agent/mcp/client.py``) — plus a raft of
    string fingerprints that match the underlying transport exception.
    Any of those signals counts as a flake; a clean
    ``{"passages": [...]}`` response never matches.
    """
    if not isinstance(result, dict):
        return False
    if result.get("mcp_unhealthy") is True:
        return True
    err = result.get("error")
    if not isinstance(err, str) or not err:
        return False
    low = err.lower()
    return any(
        needle in low
        for needle in (
            "closedresourceerror",
            "brokenresourceerror",
            "session is closed",
            "error parsing json response",
            "unhandled errors in a taskgroup",
            "connection refused",
            "connection reset",
        )
    )


async def _retrieve_for_support(
    db: PipelineState,
    query: str,
    mcp_client: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call ``search_passages`` with the answer-seeded query.

    Unwraps any ``ExcludingMCPWrapper`` so the finalizer / answer-search-
    support loop sees the full corpus (the per-iteration exclusion list
    serves ensemble diversity, not truth). Uses the MCP tool's actual
    argument names, ``queries`` + ``max_results``; ``query`` / ``top_k``
    silently return 0 passages.

    Retries transport flakes (``ClosedResourceError`` /
    ``BrokenResourceError`` / JSON-parse errors) up to
    ``_VERIFY_MCP_RETRY_MAX`` total attempts with jittered backoff. The
    in-process MCP transport race is the dominant failure mode in the
    finalizer path; without a retry the ladder silently emits tier-5
    refusals. A clean ``{"passages": []}`` response is a real empty
    result and is not retried.

    Never raises. Returns ``(passages, meta)`` where ``meta`` carries
    ``{unwrapped, error, query, n_retrieved, attempts, retries_used}``.
    """
    verify_client = getattr(mcp_client, "unwrapped", mcp_client)
    unwrapped = verify_client is not mcp_client
    log.info(
        "pipeline verify qid=%s query=%r unwrapped=%s",
        db.question_id,
        query[:140],
        unwrapped,
    )
    meta: dict[str, Any] = {
        "unwrapped": unwrapped,
        "query": query[:200],
        "error": None,
        "n_retrieved": 0,
        "attempts": 0,
        "retries_used": 0,
    }

    result: Any = None
    last_error: str | None = None
    for attempt in range(_VERIFY_MCP_RETRY_MAX):
        meta["attempts"] = attempt + 1
        try:
            result = await asyncio.wait_for(
                verify_client.call_tool(
                    "search_passages",
                    {"queries": query, "max_results": 5, "corpus": "all"},
                ),
                timeout=FINALIZER_SEARCH_TIMEOUT_S,
            )
        except TimeoutError:
            last_error = "mcp_timeout"
            log.info(
                "pipeline verify qid=%s attempt=%d mcp_timeout",
                db.question_id,
                attempt + 1,
            )
            if attempt + 1 < _VERIFY_MCP_RETRY_MAX:
                await asyncio.sleep(_VERIFY_MCP_RETRY_BACKOFF_S * (attempt + 1))
                meta["retries_used"] = attempt + 1
                continue
            meta["error"] = last_error
            return [], meta
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            log.info(
                "pipeline verify qid=%s attempt=%d raised %s",
                db.question_id,
                attempt + 1,
                type(exc).__name__,
            )
            if attempt + 1 < _VERIFY_MCP_RETRY_MAX:
                await asyncio.sleep(_VERIFY_MCP_RETRY_BACKOFF_S * (attempt + 1))
                meta["retries_used"] = attempt + 1
                continue
            meta["error"] = last_error
            return [], meta
        # Call returned. Check if it's a transport flake normalized
        # into an error dict by MCPHTTPClient.call_tool.
        if _mcp_result_looks_transport_flake(result):
            last_error = str((result or {}).get("error") or "mcp_unhealthy")[:240]
            log.info(
                "pipeline verify qid=%s attempt=%d transport_flake %s",
                db.question_id,
                attempt + 1,
                last_error[:80],
            )
            if attempt + 1 < _VERIFY_MCP_RETRY_MAX:
                await asyncio.sleep(_VERIFY_MCP_RETRY_BACKOFF_S * (attempt + 1))
                meta["retries_used"] = attempt + 1
                continue
            meta["error"] = f"transport_flake: {last_error}"
            return [], meta
        break  # clean result

    passages_raw = (result or {}).get("passages") if isinstance(result, dict) else None
    passages: list[dict[str, Any]] = (
        [p for p in passages_raw if isinstance(p, dict)] if passages_raw else []
    )
    meta["n_retrieved"] = len(passages)
    log.info(
        "pipeline verify qid=%s mcp returned %d passages",
        db.question_id,
        len(passages),
    )
    return passages, meta


def _claim_cache_key(pid: str, claim: str) -> tuple[str, str]:
    """Build the ``(pid, claim_hash)`` key for ``db.nli_score_cache``.

    blake2s with an 8-byte digest gives a short stable hash of the claim
    (16 hex chars) — long enough to avoid collisions on the ~10 claims any
    single qid ever scores, short enough to keep the cache dict compact.
    """
    claim_hash = hashlib.blake2s(
        (claim or "").encode("utf-8", errors="ignore"), digest_size=8
    ).hexdigest()
    return (pid or "", claim_hash)


async def _score_spans_with_nli(
    db: PipelineState,
    passages: list[dict[str, Any]],
    claim: str,
    qtype: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, float]]], float, float, int]:
    """Run DeBERTa NLI against the top-3 passage spans.

    ``_deberta_label_probs`` is synchronous — wrap per call via
    ``run_in_executor``. Returns:
      - ``nli_results``: list of ``(passage, probs)`` for spans that
        yielded non-None probs, in original order.
      - ``p_entail_max``, ``p_contra_max``: max across all spans.
      - ``n_spans``: number of spans successfully scored.

    Consults ``db.nli_score_cache`` before a DeBERTa round-trip, keyed
    by ``(pid, claim_hash)``. On hit, the cached probs are reused; on miss,
    DeBERTa runs and the result is written back. This avoids re-scoring
    across tiers (e.g. a tier-4 verify, then a tier-4 rescue on the same
    claim) and complements the ``scored_pids`` guard inside a single ASS
    call.

    Logs one line per span (pid + probs) so the decision chain is
    auditable in the worker err log.
    """
    log.info(
        "pipeline verify qid=%s claim=%r qtype=%s",
        db.question_id,
        claim[:140],
        qtype,
    )
    loop = asyncio.get_running_loop()
    # Per-span tuple layout:
    #   (span_idx, passage_dict, span_text, pid, cache_key, cached_probs)
    # A non-None cached_probs means we skip the DeBERTa call for this span.
    spans: list[
        tuple[int, dict[str, Any], str, Any, tuple[str, str], dict[str, float] | None]
    ] = []
    cache = db.nli_score_cache if isinstance(db.nli_score_cache, dict) else {}
    for span_idx, p in enumerate(passages[:3]):
        span_text = (p.get("text") or "").strip() if isinstance(p, dict) else ""
        pid = p.get("passage_id") if isinstance(p, dict) else "?"
        if not span_text:
            continue
        cache_key = _claim_cache_key(str(pid) if pid is not None else "", claim)
        cached = cache.get(cache_key) if cache else None
        spans.append((span_idx, p, span_text, pid, cache_key, cached))

    def _deberta_with_timeout(span_text: str) -> dict[str, float] | None:
        """Run _deberta_label_probs in a daemon thread with a hard wall-clock
        timeout.  A daemon thread is abandoned (not killed) when the timeout
        fires, but the outer ``run_in_executor`` thread returns immediately,
        keeping ``asyncio.run()`` from blocking on executor drain.

        ``asyncio.wait_for`` on a ``run_in_executor`` future only cancels
        the coroutine — the underlying thread keeps running and ``asyncio.run()``
        still waits for it via ``loop.shutdown_executor()``.  Using a nested
        daemon thread with ``Thread.join(timeout=...)`` is the only portable way
        to enforce a hard wall-clock deadline on a synchronous blocking call.
        """
        import threading as _threading

        result: list[dict[str, float] | None] = [None]
        exc: list[BaseException | None] = [None]

        def _run() -> None:
            try:
                result[0] = _deberta_label_probs(span_text, claim)
            except Exception as _e:  # noqa: BLE001
                exc[0] = _e

        t = _threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=_DEBERTA_CALL_TIMEOUT_S)
        if t.is_alive():
            raise TimeoutError(
                f"_deberta_label_probs timed out after {_DEBERTA_CALL_TIMEOUT_S}s"
            )
        if exc[0] is not None:
            raise exc[0]  # type: ignore[misc]
        return result[0]

    async def _score_one(span_text: str) -> dict[str, float] | None:
        try:
            return await loop.run_in_executor(None, _deberta_with_timeout, span_text)
        except Exception:  # noqa: BLE001
            return None

    # Only dispatch DeBERTa for cache misses. Cache hits are threaded in
    # unchanged as their stored probs.
    miss_indices: list[int] = [i for i, s in enumerate(spans) if s[5] is None]
    miss_coros = [_score_one(spans[i][2]) for i in miss_indices]
    miss_probs_list: list[dict[str, float] | None] = []
    if miss_coros:
        miss_probs_list = list(
            await asyncio.gather(*miss_coros, return_exceptions=False)
        )
    # Rebuild an aligned probs_list matching spans order: hits reuse cached,
    # misses consume from miss_probs_list in the same order they were queued.
    probs_list: list[dict[str, float] | None] = []
    _miss_iter = iter(miss_probs_list)
    for _, _, _, _, _, cached_probs in spans:
        if cached_probs is not None:
            probs_list.append(cached_probs)
        else:
            probs_list.append(next(_miss_iter, None))

    nli_results: list[tuple[dict[str, Any], dict[str, float]]] = []
    p_entail_max = 0.0
    p_contra_max = 0.0
    n_spans = 0
    for (span_idx, p, _span_text, pid, cache_key, cached_probs), probs in zip(
        spans, probs_list, strict=False
    ):
        if not probs:
            log.info(
                "pipeline verify qid=%s span=%d pid=%s NLI=none",
                db.question_id,
                span_idx,
                pid,
            )
            continue
        pe = float(probs.get("entailment", 0.0))
        pc = float(probs.get("contradiction", 0.0))
        pn = float(probs.get("neutral", 0.0))
        normalised = {"entailment": pe, "neutral": pn, "contradiction": pc}
        nli_results.append((p, normalised))
        p_entail_max = max(p_entail_max, pe)
        p_contra_max = max(p_contra_max, pc)
        n_spans += 1
        # Write miss results back to the cache so a later call with the
        # same (pid, claim) pair reuses the probs.
        if cached_probs is None:
            try:
                db.nli_score_cache[cache_key] = normalised
            except Exception:  # noqa: BLE001 — cache write is best-effort
                pass
        log.info(
            "pipeline verify qid=%s span=%d pid=%s e=%.3f n=%.3f c=%.3f%s",
            db.question_id,
            span_idx,
            pid,
            pe,
            pn,
            pc,
            " cache=hit" if cached_probs is not None else "",
        )
    return nli_results, p_entail_max, p_contra_max, n_spans


def _decide_verification(
    p_entail_max: float,
    p_contra_max: float,
    n_spans: int,
) -> tuple[bool, str]:
    """Pure decision function: verified iff p_entail ≥ 0.60 and p_contra
    < 0.40 and (p_entail - p_contra) ≥ VERIFY_ENTAIL_MARGIN_MIN (0.25).

    Returns ``(verified, decision_tag)``. ``decision_tag`` is one of
    ``{"verified", "refuted", "no_spans", "uncertain"}``. Kept as a
    standalone pure function so the identical threshold logic applies
    to every code path (Path 2 via answer_search_support, Paths 3a/3b
    via the educated-guess ladder, Path 4 via prior-rollout seed).

    The margin guard rejects narrow-margin both-non-trivial spans. The
    maxima ``p_entail_max`` and ``p_contra_max`` can come from different
    spans in the same retrieval set (best-entail span ≠ best-contra
    span), so their sum can legitimately exceed 1.0 on contested
    questions. Without the margin, ``pe=0.60, pc=0.39`` would return
    verified despite a 39% contradiction signal.
    """
    verified = (
        p_entail_max >= VERIFY_ENTAIL_THRESHOLD
        and p_contra_max < VERIFY_CONTRA_THRESHOLD_UPPER
        and (p_entail_max - p_contra_max) >= VERIFY_ENTAIL_MARGIN_MIN
    )
    if verified:
        return True, "verified"
    if p_contra_max >= 0.70:
        return False, "refuted"
    if n_spans == 0:
        return False, "no_spans"
    return False, "uncertain"


async def _verify_guess_with_retrieval(
    db: PipelineState,
    answer: str,
    mcp_client: Any,
) -> tuple[bool, dict[str, Any]]:
    """Post-generation NLI verification — composes the 3 primitives.

    Thin glue: strip/validate the answer → build an answer-seeded query →
    ``_retrieve_for_support`` → ``_score_spans_with_nli`` →
    ``_decide_verification``. ``answer_search_support`` reuses the same
    three primitives, so the NLI and retrieval logic exists once.

    Never raises — any exception becomes
    ``(False, {"decision": "error", "error": ...})``.
    """
    cleaned = _strip_finalizer_prefixes(answer or "")
    if not cleaned or cleaned.upper() == "UNKNOWN":
        log.info(
            "pipeline verify qid=%s skip reason=empty_or_unknown_answer",
            db.question_id,
        )
        return False, {
            "decision": "skip",
            "p_entail_max": 0.0,
            "p_contra_max": 0.0,
            "n_spans": 0,
            "error": "empty_or_unknown_answer",
        }

    try:
        query = _build_finalizer_search_query(db, cleaned)
        if not query:
            log.info(
                "pipeline verify qid=%s skip reason=empty_query",
                db.question_id,
            )
            return False, {
                "decision": "skip",
                "p_entail_max": 0.0,
                "p_contra_max": 0.0,
                "n_spans": 0,
                "error": "empty_query",
            }

        passages, retrieval_meta = await _retrieve_for_support(db, query, mcp_client)
        if retrieval_meta.get("error"):
            return False, {
                "decision": "error",
                "p_entail_max": 0.0,
                "p_contra_max": 0.0,
                "n_spans": 0,
                "error": retrieval_meta["error"],
                "query": retrieval_meta.get("query", query[:200]),
            }
        if not passages:
            return False, {
                "decision": "no_spans",
                "p_entail_max": 0.0,
                "p_contra_max": 0.0,
                "n_spans": 0,
                "query": query[:200],
            }

        qtype = db.question_type or _question_shape_classifier(db.question or "")
        claim = _claim_template_for_verification(db.question or "", cleaned, qtype)
        _nli_results, p_entail_max, p_contra_max, n_spans = await _score_spans_with_nli(
            db, passages, claim, qtype
        )
        verified, decision = _decide_verification(p_entail_max, p_contra_max, n_spans)
        log.info(
            "pipeline verify qid=%s decision=%s p_e_max=%.3f p_c_max=%.3f n_spans=%d "
            "thresholds=(e>=%.2f,c<%.2f)",
            db.question_id,
            decision,
            p_entail_max,
            p_contra_max,
            n_spans,
            VERIFY_ENTAIL_THRESHOLD,
            VERIFY_CONTRA_THRESHOLD_UPPER,
        )
        return verified, {
            "decision": decision,
            "p_entail_max": round(p_entail_max, 4),
            "p_contra_max": round(p_contra_max, 4),
            "n_spans": n_spans,
            "query": query[:200],
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("finalizer verify raised")
        return False, {
            "decision": "error",
            "p_entail_max": 0.0,
            "p_contra_max": 0.0,
            "n_spans": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


# answer_search_support
#
# Called by Paths 2 / 3a / 3b / 4 of the answer-submission pipeline. Given a
# candidate answer (produced by the LLM but not yet committable, e.g.
# post-hardgate rejection, post-educated-guess, post-prior-rollout),
# iteratively retrieve passages and either:
#
#   * auto-commit a ``verified_supporting`` nugget (strong NLI);
#   * queue a reflect-hint for the model to consider (moderate NLI);
#   * declare exhaustion after 3 reformulated queries.
#
# Shares three primitives with ``_verify_guess_with_retrieval``
# (_retrieve_for_support, _score_spans_with_nli, _decide_verification)
# so all paths get identical retrieval + NLI semantics.
#
# Budget: no shared cap. The recursive cascade runs as many ASS calls as
# the tier structure requires. Each invocation is internally capped at
# ANSWER_SEARCH_SUPPORT_MAX_ITERS reformulations.


def _reformulate_query(
    db: PipelineState,
    candidate_answer: str,
    iter_idx: int,
) -> str:
    """Produce a distinct query for each answer_search_support iteration.

    Iter 0: ``_build_finalizer_search_query`` — entities + filtered
      question tokens + answer (the verification query).
    Iter 1: entities + answer (drop the question tokens to surface
      answer-adjacent passages that the full query missed).
    Iter 2: answer verbatim (last resort — lets BM25 hit on the
      answer's proper-noun span alone).

    Returns an empty string when a reformulation would produce an
    empty query (caller should treat that as ``no_spans``).
    """
    cleaned = (candidate_answer or "").strip()
    if not cleaned:
        return ""
    if iter_idx <= 0:
        return _build_finalizer_search_query(db, cleaned)
    if iter_idx == 1:
        try:
            entities = extract_question_entities(db.question or "") or []
        except Exception:  # noqa: BLE001
            entities = []
        parts: list[str] = []
        seen: set[str] = set()
        for ent in entities:
            for tok in (ent or "").split():
                k = tok.lower()
                if k and k not in seen:
                    seen.add(k)
                    parts.append(tok)
        parts.append(cleaned)
        return " ".join(p for p in parts if p).strip()[:400]
    return cleaned[:400]


def _auto_commit_support_nugget(
    db: PipelineState,
    passage: dict[str, Any],
    claim: str,
    p_entail: float,
    p_contra: float,
    origin: str,
) -> str | None:
    """Commit / extend a ``verified_supporting`` Nugget sourced from
    answer_search_support retrieval.

    When a retrieved passage's span entails the same claim text as an
    existing verified_supporting nugget (case-insensitive substring
    match), the new pid is appended to that nugget's ``supporting_pids``
    instead of creating a duplicate row. A fresh nugget is minted only
    when the claim is new. This is how passages corroborate one another:
    `n_eff` (over pids) grows without the nugget list fragmenting.

    Creates / upgrades the matching Passage row so
    ``effective_supporting_pids`` can count the corroboration toward the
    hardgate.

    Returns the nugget_id the pid was attached to (new or existing), or
    ``None`` when the pid was already part of that nugget's
    ``supporting_pids`` (idempotent under retry) or on malformed input.
    """
    pid = passage.get("passage_id") if isinstance(passage, dict) else None
    text = (passage.get("text") if isinstance(passage, dict) else "") or ""
    title = passage.get("title") if isinstance(passage, dict) else None
    score = passage.get("score") if isinstance(passage, dict) else None
    if not pid or not isinstance(pid, str):
        return None

    # NLI gate — refuse to append a pid whose entailment score doesn't
    # clear the global verify threshold. The caller (answer_search_support)
    # already guards at AUTO_COMMIT_NLI_THRESHOLD which is > this floor,
    # but the defensive check makes this helper safe to wire into paths
    # that use looser NLI bands.
    if float(p_entail) < VERIFY_ENTAIL_THRESHOLD:
        return None

    claim_truncated = claim[:240]
    claim_norm = _normalize_nugget_text(claim_truncated)

    # Extension path: find an existing verified_supporting nugget whose
    # text matches (normalized equality or case-insensitive substring
    # containment). If it already cites this pid, do nothing. Otherwise
    # append the pid and record the NLI score.
    for existing in db.nuggets.values():
        if existing.status != "verified_supporting":
            continue
        existing_norm = _normalize_nugget_text(existing.text)
        text_match = existing_norm == claim_norm or (
            existing_norm
            and claim_norm
            and (existing_norm in claim_norm or claim_norm in existing_norm)
        )
        if not text_match:
            continue
        if pid in existing.supporting_pids:
            # Already corroborated — no-op.
            log.info(
                "pipeline ass.auto_commit.noop qid=%s pid=%s nid=%s "
                "(already in supporting_pids)",
                db.question_id,
                pid,
                existing.nugget_id,
            )
            return None
        # Seed / upgrade passage row so the pid is a first-class citizen
        # of the DB before the pid is referenced by the nugget.
        if pid not in db.passages:
            db.passages[pid] = Passage(
                passage_id=pid,
                text=text,
                title=title,
                score=float(score) if isinstance(score, (int, float)) else None,
                source_tool_call_id=f"answer_search_support_{origin}",
                source_msg_idx=db.current_turn,
                status="accepted",
                status_reason=f"auto_commit via answer_search_support ({origin})",
            )
        else:
            existing_p = db.passages[pid]
            if existing_p.status in {"unclassified", "rejected"}:
                existing_p.status = "accepted"
                existing_p.status_reason = (
                    f"auto_commit via answer_search_support ({origin})"
                )
        existing.supporting_pids.append(pid)
        existing.pid_nli_scores[pid] = float(p_entail)
        ev = TransitionEvent(
            step_id="answer_search_support.pid_append",
            iteration=db.current_turn,
            actor="system",
            from_status="verified_supporting",
            to_status="verified_supporting",
            reason=(
                f"origin={origin} append_pid={pid} "
                f"p_entail={p_entail:.3f} p_contra={p_contra:.3f}"
            ),
        )
        append_event(db, ev)
        existing.history.append(ev)
        log.info(
            "pipeline ass.auto_commit.extend qid=%s pid=%s nid=%s "
            "p_entail=%.3f p_contra=%.3f",
            db.question_id,
            pid,
            existing.nugget_id,
            p_entail,
            p_contra,
        )
        return existing.nugget_id

    # No textual match — mint a fresh verified_supporting nugget with
    # the pid pre-seeded into supporting_pids.
    if pid not in db.passages:
        db.passages[pid] = Passage(
            passage_id=pid,
            text=text,
            title=title,
            score=float(score) if isinstance(score, (int, float)) else None,
            source_tool_call_id=f"answer_search_support_{origin}",
            source_msg_idx=db.current_turn,
            status="accepted",
            status_reason=f"auto_commit via answer_search_support ({origin})",
        )
    else:
        existing_p = db.passages[pid]
        if existing_p.status in {"unclassified", "rejected"}:
            existing_p.status = "accepted"
            existing_p.status_reason = (
                f"auto_commit via answer_search_support ({origin})"
            )
    nid = f"n_{uuid.uuid4().hex[:8]}"
    nug = Nugget(
        nugget_id=nid,
        passage_id=pid,
        text=claim_truncated,
        proposed_stance="supporting",
        source_step=f"answer_search_support_{origin}",
        source_tool_call_id=f"answer_search_support_t{db.current_turn}",
        status="verified_supporting",
        status_reason=(
            f"auto_commit (p_entail={p_entail:.3f}, p_contra={p_contra:.3f})"
        ),
        nli_label="entails",
        nli_score=float(p_entail),
        supporting_pids=[pid],
        pid_nli_scores={pid: float(p_entail)},
    )
    db.nuggets[nid] = nug
    db.nuggets_recorded += 1
    ev = TransitionEvent(
        step_id="answer_search_support.auto_commit",
        iteration=db.current_turn,
        actor="system",
        from_status="unclassified",
        to_status="verified_supporting",
        reason=(f"origin={origin} p_entail={p_entail:.3f} p_contra={p_contra:.3f}"),
    )
    append_event(db, ev)
    nug.history.append(ev)
    return nid


def _build_soft_inject_reflect(
    passage: dict[str, Any],
    p_entail: float,
    origin: str,
) -> str:
    """Render a reflect string pointing the model at a moderate-NLI span.

    Short, action-oriented — the reflect is queued onto
    ``db.pending_reflects`` for the next loop turn to surface.
    """
    pid = (passage.get("passage_id") if isinstance(passage, dict) else "?") or "?"
    title = (passage.get("title") if isinstance(passage, dict) else "") or ""
    text = (passage.get("text") if isinstance(passage, dict) else "") or ""
    snippet = text.strip().replace("\n", " ")[:200]
    title_fragment = f" ({title[:80]})" if title else ""
    return (
        f"[{origin}] passage {pid}{title_fragment} partially supports your "
        f'answer (NLI entailment={p_entail:.2f}). Quote: "{snippet}". '
        "Read it with get_passage, then decide: record_evidence(stance="
        '"supporting") + submit_answer, or refuse if still unclear.'
    )


async def answer_search_support(
    db: PipelineState,
    candidate_answer: str,
    mcp_client: Any,
    *,
    origin: str = "path2",
) -> tuple[bool, dict[str, Any]]:
    """Active evidence-building loop around a candidate answer.

    Iterates up to ``ANSWER_SEARCH_SUPPORT_MAX_ITERS`` reformulated
    queries:

      * ``p_entail >= AUTO_COMMIT_NLI_THRESHOLD`` and
        ``p_contra < AUTO_COMMIT_CONTRA_MAX``:
          auto-commit a verified_supporting nugget and return
          ``(True, ...)`` immediately.
      * ``SOFT_INJECT_NLI_MIN <= p_entail < AUTO_COMMIT_NLI_THRESHOLD``:
          queue a reflect-hint on ``db.pending_reflects`` and continue
          the next iteration.
      * otherwise: fall through to the next reformulation.

    No shared budget cap — the counter
    ``db.answer_search_support_calls_used`` is telemetry only. Each
    invocation is internally capped at ``ANSWER_SEARCH_SUPPORT_MAX_ITERS``
    reformulations; the recursive cascade runs as many invocations as
    the tier structure requires.

    Never raises. Returns ``(verified, details)`` where ``details``
    carries ``iterations``, ``auto_nuggets``, ``soft_injects``,
    ``last_decision``, ``queries_tried``, ``calls_used``,
    ``p_entail_max``, ``p_contra_max``, ``origin``.
    """
    details: dict[str, Any] = {
        "origin": origin,
        "iterations": 0,
        "auto_nuggets": [],
        "soft_injects": [],
        "last_decision": "",
        "queries_tried": [],
        "calls_used": 0,
        "p_entail_max": 0.0,
        "p_contra_max": 0.0,
        "budget_exhausted": False,
    }
    cleaned = _strip_finalizer_prefixes(candidate_answer or "")
    if not cleaned or cleaned.upper() == "UNKNOWN":
        details["last_decision"] = "skip_empty_or_unknown_answer"
        log.info(
            "pipeline answer_search_support qid=%s origin=%s skip=%s",
            db.question_id,
            origin,
            details["last_decision"],
        )
        return False, details

    # Pipeline-level format gate — the auto-commit below writes the
    # cleaned string into a nugget; if it contains ``\n/\r/\t/;`` or a
    # sentinel fragment we'd pollute the AC file. Reject format-failing
    # candidates up-front, same contract as every other emission site.
    _ass_fmt = _validate_r2c2_answer(cleaned)
    if not _ass_fmt.ok:
        details["last_decision"] = f"skip_format_{_ass_fmt.reason}"
        log.warning(
            "pipeline answer_search_support qid=%s origin=%s skip=%s",
            db.question_id,
            origin,
            details["last_decision"],
        )
        return False, details
    cleaned = _ass_fmt.cleaned

    qtype = db.question_type or _question_shape_classifier(db.question or "")
    claim = _claim_template_for_verification(db.question or "", cleaned, qtype)
    p_entail_max_overall = 0.0
    p_contra_max_overall = 0.0
    queries_seen: set[str] = set()
    # Passages already NLI-scored in a prior iteration of this ASS
    # invocation are skipped. Re-scoring the same (premise, claim) pair
    # returns identical probabilities (the model is deterministic) and
    # costs a DeBERTa HTTP round-trip.
    scored_pids: set[str] = set()

    for iter_idx in range(ANSWER_SEARCH_SUPPORT_MAX_ITERS):
        query = _reformulate_query(db, cleaned, iter_idx)
        if not query:
            details["last_decision"] = "empty_query"
            continue
        if query in queries_seen:
            # Skip duplicate reformulations — answer-only can collide
            # with earlier entity+answer on short questions.
            continue
        queries_seen.add(query)
        details["queries_tried"].append(query[:200])

        # Budget is debited before the call so a crashed MCP doesn't let
        # the loop spin forever.
        db.answer_search_support_calls_used += 1
        details["calls_used"] = db.answer_search_support_calls_used
        details["iterations"] = iter_idx + 1

        passages, retrieval_meta = await _retrieve_for_support(db, query, mcp_client)
        if retrieval_meta.get("error"):
            details["last_decision"] = "retrieval_error"
            db.answer_search_support_decisions.append(
                f"iter={iter_idx} error={retrieval_meta['error']}"
            )
            log.info(
                "pipeline answer_search_support qid=%s iter=%d error=%s",
                db.question_id,
                iter_idx,
                retrieval_meta["error"],
            )
            continue
        if not passages:
            details["last_decision"] = "no_passages"
            db.answer_search_support_decisions.append(
                f"iter={iter_idx} decision=no_passages query={query[:80]}"
            )
            continue

        # Filter passages to only those not already NLI-scored in a prior
        # iteration.  Overlapping BM25 result sets across reformulations
        # are common; scoring the same span twice adds zero information.
        fresh_passages = [
            p
            for p in passages
            if isinstance(p, dict) and p.get("passage_id") not in scored_pids
        ]
        scored_pids.update(
            p["passage_id"]
            for p in fresh_passages
            if isinstance(p, dict) and p.get("passage_id")
        )
        if not fresh_passages:
            details["last_decision"] = "no_passages"
            db.answer_search_support_decisions.append(
                f"iter={iter_idx} decision=all_pids_already_scored"
            )
            continue

        nli_results, p_entail_max, p_contra_max, n_spans = await _score_spans_with_nli(
            db, fresh_passages, claim, qtype
        )
        p_entail_max_overall = max(p_entail_max_overall, p_entail_max)
        p_contra_max_overall = max(p_contra_max_overall, p_contra_max)
        details["p_entail_max"] = round(p_entail_max_overall, 4)
        details["p_contra_max"] = round(p_contra_max_overall, 4)

        # Scan the per-span results in order; strongest match wins.
        auto_committed = False
        for passage, probs in nli_results:
            pe = float(probs.get("entailment", 0.0))
            pc = float(probs.get("contradiction", 0.0))
            if pe >= AUTO_COMMIT_NLI_THRESHOLD and pc < AUTO_COMMIT_CONTRA_MAX:
                nid = _auto_commit_support_nugget(db, passage, claim, pe, pc, origin)
                if nid is not None:
                    details["auto_nuggets"].append(nid)
                    db.answer_search_support_auto_nuggets.append(nid)
                    db.answer_search_support_decisions.append(
                        f"iter={iter_idx} decision=auto_commit nid={nid} "
                        f"pid={passage.get('passage_id')} pe={pe:.3f} pc={pc:.3f}"
                    )
                    auto_committed = True
                    break
            # Soft-inject covers everything that passes the lower NLI floor
            # but did not auto-commit (failed p_contra check, or auto-commit
            # returned None for text mismatch). This includes a span at
            # pe=0.85, pc=0.25, which clears auto-commit's pe floor but
            # fails pc<0.25 and would otherwise be dropped silently.
            if pe >= SOFT_INJECT_NLI_MIN:
                _soft_pid = (
                    passage.get("passage_id") if isinstance(passage, dict) else None
                ) or ""
                # Skip if a reflect for this pid is already queued. Two ASS
                # invocations for the same answer can both soft-inject the same
                # passage, which would spend model context on a duplicate hint.
                _pid_already_reflected = _soft_pid and any(
                    f"passage {_soft_pid}" in r for r in db.pending_reflects
                )
                if _pid_already_reflected:
                    db.answer_search_support_decisions.append(
                        f"iter={iter_idx} decision=soft_inject_dedup_skip "
                        f"pid={_soft_pid}"
                    )
                else:
                    reflect = _build_soft_inject_reflect(passage, pe, origin)
                    db.pending_reflects.append(reflect)
                    details["soft_injects"].append(reflect[:120])
                    db.answer_search_support_decisions.append(
                        f"iter={iter_idx} decision=soft_inject "
                        f"pid={passage.get('passage_id')} pe={pe:.3f}"
                    )
        if auto_committed:
            details["last_decision"] = "auto_committed"
            log.info(
                "pipeline answer_search_support qid=%s origin=%s auto_committed "
                "iter=%d p_entail=%.3f",
                db.question_id,
                origin,
                iter_idx,
                details["p_entail_max"],
            )
            return True, details

        verified, decision = _decide_verification(p_entail_max, p_contra_max, n_spans)
        details["last_decision"] = decision
        if verified and details["last_decision"] != "auto_committed":
            # NLI says verified but no single span cleared the stricter
            # auto-commit band — treat as evidence-gathered but not
            # auto-committable. Caller (Path 2 retry-hardgate) must
            # decide whether to emit.
            db.answer_search_support_decisions.append(
                f"iter={iter_idx} decision=verified_weak "
                f"pe={p_entail_max:.3f} pc={p_contra_max:.3f}"
            )
            log.info(
                "pipeline answer_search_support qid=%s origin=%s verified_weak "
                "iter=%d (no span cleared auto-commit band)",
                db.question_id,
                origin,
                iter_idx,
            )
            return True, details

        db.answer_search_support_decisions.append(
            f"iter={iter_idx} decision={decision} "
            f"pe={p_entail_max:.3f} pc={p_contra_max:.3f}"
        )

    # Accumulate across invocations (cumulative sum) so iter_count
    # reports the total number of iterations across all ASS calls for
    # this qid, the same semantics as calls_used. max() would report 3
    # for two 3-iteration calls, not 6.
    db.answer_search_support_iter_count += details["iterations"]
    log.info(
        "pipeline answer_search_support qid=%s origin=%s finished "
        "iters=%d verified=False last=%s pe_max=%.3f pc_max=%.3f "
        "auto_nuggets=%d soft_injects=%d calls=%d",
        db.question_id,
        origin,
        details["iterations"],
        details["last_decision"],
        details["p_entail_max"],
        details["p_contra_max"],
        len(details["auto_nuggets"]),
        len(details["soft_injects"]),
        db.answer_search_support_calls_used,
    )
    return False, details


# NLI client (localhost server; 2-worker asyncio semaphore pattern)


def _nli_server_url() -> str:
    """Resolve NLI server URL from $NLI_PORT env or default 18500."""
    port = os.environ.get("NLI_PORT", "18500")
    return f"http://localhost:{port}/check_entailment"


async def _call_nli_one(
    client: httpx.AsyncClient,
    url: str,
    passage_text: str,
    claim_text: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """POST a single (passage, claim) to the NLI server; no read timeout.

    No per-call read timeout: the read phase waits indefinitely, and only
    a connect failure or genuine network error surfaces as a failure
    (fallback label ``bogus`` with reason ``nli_timeout``). Per-qid
    wall-clock is bounded at the orchestrator level, not here.

    Return ``{"label": str, "score": float}``. On any transport error,
    return ``{"label": "bogus", "score": 0.0, "reason": "nli_timeout"}``.
    """
    async with sem:
        try:
            resp = await client.post(
                url,
                json={"passage_text": passage_text, "claim_text": claim_text},
            )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("pipeline NLI call failed: %s", exc)
            return {"label": "bogus", "score": 0.0, "reason": "nli_timeout"}

    # Normalize / clamp.
    label = data.get("label") or "bogus"
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if not (score == score):  # NaN guard
        score = 0.0
    score = max(0.0, min(1.0, score))
    if label not in {"entails", "bogus"}:
        label = "bogus"
    return {"label": label, "score": score}


async def _batch_nli(
    pairs: list[tuple[str, str]],
    *,
    nli_url: str | None = None,
    http_post: Any | None = None,
) -> list[dict[str, Any]]:
    """Run up to ``NLI_EXECUTOR_WORKERS`` concurrent NLI calls.

    ``http_post`` is an optional test injection: a sync callable
    ``(url, body) -> {label, score}``. When provided, calls happen serially
    (tests don't need async concurrency).
    """
    if http_post is not None:
        out: list[dict[str, Any]] = []
        for passage_text, claim_text in pairs:
            try:
                data = http_post(
                    nli_url or _nli_server_url(),
                    {"passage_text": passage_text, "claim_text": claim_text},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("pipeline NLI test-injection failed: %s", exc)
                out.append({"label": "bogus", "score": 0.0, "reason": "nli_timeout"})
                continue
            label = data.get("label") or "bogus"
            try:
                score = float(data.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if not (score == score):
                score = 0.0
            score = max(0.0, min(1.0, score))
            if label not in {"entails", "bogus"}:
                label = "bogus"
            out.append({"label": label, "score": score})
        return out

    if not pairs:
        return []

    url = nli_url or _nli_server_url()
    sem = asyncio.Semaphore(NLI_EXECUTOR_WORKERS)
    # connect=30s (fail fast on dead server), read=None (wait for a
    # healthy but slow response), write=30s, pool=60s. Timeouts do not
    # cascade per call.
    nli_timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=60.0)
    async with httpx.AsyncClient(timeout=nli_timeout) as client:
        tasks = [
            _call_nli_one(client, url, p_text, c_text, sem) for p_text, c_text in pairs
        ]
        return await asyncio.gather(*tasks)


# Helpers for budget envelope + session status


def _budget_envelope(db: PipelineState) -> dict[str, Any]:
    """Budget dict, present in every tool response."""
    return {
        "calls_used": db.calls_used,
        "calls_remaining": max(0, db.calls_total - db.calls_used),
        "nuggets_recorded": db.nuggets_recorded,
        "recorded_pids_count": len(db.recorded_pids),
    }


def _session_status(db: PipelineState) -> dict[str, Any]:
    """Lightweight session summary for every response."""
    verified = sum(1 for n in db.nuggets.values() if n.status == "verified_supporting")
    bogus = sum(1 for n in db.nuggets.values() if n.status == "bogus")
    refusal_active = any(n.status == "refusal" for n in db.nuggets.values())
    return {
        "verified_count": verified,
        "bogus_count": bogus,
        "refusal_active": refusal_active,
    }


# handle_record_evidence


def handle_record_evidence(
    db: PipelineDB,
    args: dict[str, Any],
    *,
    _http_post: Any | None = None,
) -> dict[str, Any]:
    """Commit stance + ≤3 nuggets for a passage.

    Tool arguments:
        pid: str, required
        stance: Literal["supporting","counter_evidence","irrelevant","context"]
        rationale: str (≤400 chars)
        nuggets: optional list of ≤3 dicts with keys
            {claim, polarity, quoted_span}
        supersede_by: optional nugget_id of a prior nugget to replace

    Pipeline (server-side, pre-commit):
        1. passage existence check → unknown_pid early-return (no mutation)
        2. apply stance → Passage.status (+ cascade on demotion)
        3. per-nugget quoted_span validation
        4. batch NLI (2-worker asyncio)
        5. three-band decision per nugget: verify / reflect / reject
        6. diagnose_nli_failure() on rejections
        7. supersede_by wiring
        8. budget-cap enforcement

    Never raises — all errors surface as per-nugget reject records or
    top-level ``error`` fields. The PipelineDB is the single source of truth:
    callers may replay by re-running the event log.

    Args:
        db: Pipeline DB (must be PipelineState).
        args: Tool-call kwargs.
        _http_post: Optional test injection — sync callable
            ``(url, body) -> {label, score}``; bypasses async executor.

    Returns:
        Response dict (committed, nugget_ids, per_nugget, ...).
        The ``budget`` envelope is added by ``wrap_tool_response``; the
        returned dict carries the raw handler fields.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    # Step 0: arg validation
    pid = args.get("pid")
    stance = args.get("stance")
    rationale = args.get("rationale", "")
    nuggets_in = args.get("nuggets")
    if nuggets_in is None:
        nuggets_in = []
    supersede_by = args.get("supersede_by")

    # vLLM's Hermes tool-call parser delivers array-typed arguments as
    # JSON-encoded strings (`'[{"claim": "..."}]'`) rather than Python lists.
    # One round of json.loads before the list type-check keeps valid nugget
    # payloads from being rejected with a misleading "wrong shape" error.
    if isinstance(nuggets_in, str):
        try:
            nuggets_in = json.loads(nuggets_in)
        except (json.JSONDecodeError, TypeError):
            return {
                "committed": False,
                "error": "nuggets_malformed_json",
                "message": (
                    "nuggets arg arrived as a string but did not parse as "
                    "JSON — pass nuggets as a JSON array of ≤3 objects with "
                    "keys {claim, polarity, quoted_span}."
                ),
                "budget": _budget_envelope(pipeline_db),
                "session_status": _session_status(pipeline_db),
            }

    if not isinstance(pid, str) or not pid:
        return {
            "committed": False,
            "error": "invalid_pid",
            "message": "pid must be a non-empty string",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if stance not in _VALID_STANCES:
        return {
            "committed": False,
            "error": "invalid_stance",
            "message": f"stance must be one of {sorted(_VALID_STANCES)}",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if not isinstance(rationale, str) or len(rationale) > MAX_RATIONALE_CHARS:
        return {
            "committed": False,
            "error": "invalid_rationale",
            "message": (f"rationale must be ≤{MAX_RATIONALE_CHARS} chars (non-empty)"),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    # "Wrong shape" and "too many" get distinct error codes: a shared
    # ``too_many_nuggets`` code misleads the LLM into shrinking valid
    # payloads.
    if not isinstance(nuggets_in, list):
        return {
            "committed": False,
            "error": "nuggets_not_list",
            "message": (
                f"nuggets must be a JSON array of ≤{MAX_NUGGETS_PER_CALL} "
                f"objects with keys {{claim, polarity, quoted_span}}; got "
                f"{type(nuggets_in).__name__}."
            ),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if len(nuggets_in) > MAX_NUGGETS_PER_CALL:
        return {
            "committed": False,
            "error": "nuggets_over_cap",
            "message": (
                f"nuggets must be ≤{MAX_NUGGETS_PER_CALL} per call; got "
                f"{len(nuggets_in)}."
            ),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    # Handler entry trace: what the LLM attempted.
    log.info(
        "pipeline record_evidence.enter qid=%s pid=%s stance=%s "
        "nuggets_in=%d rationale=%r",
        pipeline_db.question_id,
        pid,
        stance,
        len(nuggets_in),
        rationale[:80],
    )
    _emit_event(
        "pipeline.record_evidence.enter",
        qid=pipeline_db.question_id,
        pid=pid,
        stance=stance,
        nuggets_in=len(nuggets_in),
        rationale_preview=rationale[:120],
    )

    # Step 1: passage existence check (no mutation)
    if pid not in pipeline_db.passages:
        return {
            "committed": False,
            "error": "unknown_pid",
            "message": f"pid {pid!r} not found; call get_passage first",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    # Step 2: stance application + cascade on demotion
    p = pipeline_db.passages[pid]
    target_status = STANCE_TO_PASSAGE_STATUS[stance]
    from_status = p.status
    cascaded_nuggets: list[str] = []
    if from_status != target_status:
        p.status = target_status  # type: ignore[assignment]
        p.status_reason = f"pipeline stance={stance}: {rationale[:100]}"
        ev = TransitionEvent(
            step_id="m11_record_evidence",
            iteration=pipeline_db.current_turn,
            actor="llm_tool",
            from_status=from_status,
            to_status=target_status,
            reason=f"stance={stance}",
        )
        append_event(pipeline_db, ev)
        p.history.append(ev)

        # Cascade on demotion: a passage going from accepted/counter to rejected
        # flips its downstream verified nuggets to bogus.
        if (
            from_status in {"accepted", "counter_evidence"}
            and target_status == "rejected"
        ):
            for nid, n in pipeline_db.nuggets.items():
                if n.passage_id == pid and n.status in {
                    "verified_supporting",
                    "verified_infirming",
                }:
                    n_from = n.status
                    n.status = "bogus"
                    n.status_reason = "cascade: passage demoted to rejected"
                    ev_c = TransitionEvent(
                        step_id="m11_record_evidence",
                        iteration=pipeline_db.current_turn,
                        actor="system",
                        from_status=n_from,
                        to_status="bogus",
                        reason="cascade on stance=irrelevant demotion",
                    )
                    append_event(pipeline_db, ev_c)
                    n.history.append(ev_c)
                    cascaded_nuggets.append(nid)

            # Cascade transition log: a visible state mutation other than
            # commit/reject.
            if cascaded_nuggets:
                log.info(
                    "pipeline cascade qid=%s pid=%s demoted %d "
                    "verified nuggets -> bogus",
                    pipeline_db.question_id,
                    pid,
                    len(cascaded_nuggets),
                )
                _emit_event(
                    "pipeline.cascade",
                    qid=pipeline_db.question_id,
                    pid=pid,
                    demoted=len(cascaded_nuggets),
                    nugget_ids=list(cascaded_nuggets),
                )

    # Step 3: per-nugget span validation
    span_valid_indices: list[int] = []
    per_nugget: list[dict[str, Any]] = []
    for i, nug in enumerate(nuggets_in):
        if not isinstance(nug, dict):
            per_nugget.append(
                {
                    "committed": False,
                    "index": i,
                    "nli_label": "bogus",
                    "reason": "invalid_nugget_format",
                }
            )
            continue
        claim = nug.get("claim", "")
        polarity = nug.get("polarity", "")
        quoted_span = nug.get("quoted_span", "")
        # Schema guards
        if not isinstance(claim, str) or not (
            MIN_CLAIM_CHARS <= len(claim) <= MAX_CLAIM_CHARS
        ):
            per_nugget.append(
                {
                    "committed": False,
                    "index": i,
                    "nli_label": "bogus",
                    "reason": "invalid_claim_length",
                }
            )
            continue
        if polarity not in _VALID_POLARITIES:
            per_nugget.append(
                {
                    "committed": False,
                    "index": i,
                    "nli_label": "bogus",
                    "reason": "invalid_polarity",
                }
            )
            continue
        if not isinstance(quoted_span, str) or not quoted_span:
            per_nugget.append(
                {
                    "committed": False,
                    "index": i,
                    "nli_label": "bogus",
                    "reason": "invalid_quoted_span",
                }
            )
            continue
        # Fuzzy check
        if not quoted_span_match(quoted_span, p.text):
            per_nugget.append(
                {
                    "committed": False,
                    "index": i,
                    "nli_label": "bogus",
                    "reason": "quoted_span_not_in_passage",
                    "diagnose": "span_missing",
                }
            )
            continue
        # Placeholder entry — we fill in NLI result at Step 5.
        per_nugget.append(
            {
                "committed": None,  # sentinel; replaced in Step 5
                "index": i,
                "claim": claim,
                "polarity": polarity,
                "quoted_span": quoted_span,
            }
        )
        span_valid_indices.append(i)

    # Step 4: batch NLI on span-valid nuggets.
    if span_valid_indices:
        pairs: list[tuple[str, str]] = [
            (
                per_nugget[i]["quoted_span"],  # type: ignore[index]
                per_nugget[i]["claim"],  # type: ignore[index]
            )
            for i in span_valid_indices
        ]
        import time as _time

        _t_nli = _time.monotonic()
        try:
            nli_results = asyncio.run(_batch_nli(pairs, http_post=_http_post))
        except RuntimeError:
            # Already inside an event loop (tests): use get_event_loop.
            loop = asyncio.new_event_loop()
            try:
                nli_results = loop.run_until_complete(
                    _batch_nli(pairs, http_post=_http_post)
                )
            finally:
                loop.close()
        _nli_dt = _time.monotonic() - _t_nli
        _scores_str = ",".join(f"{float(r.get('score', 0.0)):.2f}" for r in nli_results)
        log.info(
            "pipeline NLI.batch qid=%s pid=%s pairs=%d dt=%.2fs scores=[%s]",
            pipeline_db.question_id,
            pid,
            len(pairs),
            _nli_dt,
            _scores_str,
        )
        _emit_event(
            "pipeline.nli.batch",
            qid=pipeline_db.question_id,
            pid=pid,
            pairs=len(pairs),
            dt_s=round(_nli_dt, 3),
            scores=[round(float(r.get("score", 0.0)), 3) for r in nli_results],
        )
    else:
        nli_results = []

    # Step 5: three-band decision per nugget (with the hybrid cascade)
    # Each raw MiniCheck score flows through ``_hybrid_score``, which
    # applies the hard caps and the paraphrase-shy lift. The rule name +
    # sub-signals land on the Nugget so the trace audit can read the
    # breakdown.
    committed_nids: list[str] = []
    nli_scores: list[float] = []
    rejects_this_call = 0
    reject_reasons: list[str] = []
    for idx_in_pairs, i in enumerate(span_valid_indices):
        nli = nli_results[idx_in_pairs]
        raw_score = float(nli.get("score", 0.0))
        # Budget: hard-cap guard on nugget count.
        if pipeline_db.nuggets_recorded >= MAX_NUGGETS_HARD:
            per_nugget[i] = {
                "committed": False,
                "index": i,
                "nli_label": "bogus",
                "reason": "nugget_hard_cap_reached",
                "nli_score": raw_score,
            }
            continue

        claim = per_nugget[i]["claim"]  # type: ignore[index]
        polarity = per_nugget[i]["polarity"]  # type: ignore[index]
        quoted_span = per_nugget[i]["quoted_span"]  # type: ignore[index]

        # DeBERTa is the polarity detector in the scoring path. When
        # ``DEBERTA_NLI_URL`` is unset, the client returns ``None`` on
        # every call and the cascade degrades to the MiniCheck +
        # lexical-statistics branches, with polarity detection
        # disabled (acceptable for development without a DeBERTa
        # sidecar; production runs with it).
        score, hybrid_rule, hybrid_signals = _hybrid_score(
            raw_score,
            claim,
            quoted_span,
            polarity,
            deberta_probs_fn=_deberta_label_probs,
        )

        if score >= NLI_TRUST_THRESHOLD:
            nid = _commit_nugget(
                pipeline_db,
                pid=pid,
                claim=claim,
                polarity=polarity,
                quoted_span=quoted_span,
                status="verified_supporting"
                if polarity == "affirming"
                else "verified_infirming",
                nli_label="entails",
                nli_score=score,
                reason=f"pipeline record_evidence trust band ({hybrid_rule})",
                hybrid_rule=hybrid_rule,
                hybrid_signals=hybrid_signals,
            )
            committed_nids.append(nid)
            nli_scores.append(score)
            per_nugget[i] = {
                "committed": True,
                "index": i,
                "nugget_id": nid,
                "nli_score": score,
                "nli_raw_score": raw_score,
                "nli_label": "entails",
                "hybrid_rule": hybrid_rule,
                "hybrid_signals": hybrid_signals,
            }
        elif score >= NLI_REFLECT_THRESHOLD:
            nid = _commit_nugget(
                pipeline_db,
                pid=pid,
                claim=claim,
                polarity=polarity,
                quoted_span=quoted_span,
                status="verified_supporting"
                if polarity == "affirming"
                else "verified_infirming",
                nli_label="entails",  # store as entails; reflect on label
                nli_score=score,
                reason=f"pipeline record_evidence reflect band ({hybrid_rule})",
                hybrid_rule=hybrid_rule,
                hybrid_signals=hybrid_signals,
            )
            committed_nids.append(nid)
            nli_scores.append(score)
            per_nugget[i] = {
                "committed": True,
                "index": i,
                "nugget_id": nid,
                "nli_score": score,
                "nli_raw_score": raw_score,
                "nli_label": "uncertain",
                "hybrid_rule": hybrid_rule,
                "hybrid_signals": hybrid_signals,
            }
        else:
            # Reject — do not commit. The hybrid_rule is the headline
            # reason (polarity_flip / entity_swap / quantifier_drift /
            # minicheck_default); diagnose_nli_failure stays as the
            # existing sub-diagnosis used by the reflect-nudge path.
            reason_tag = (
                nli.get("reason") or hybrid_rule
                if hybrid_rule != "minicheck_default"
                else (nli.get("reason") or diagnose_nli_failure(quoted_span, claim))
            )
            rejects_this_call += 1
            reject_reasons.append(reason_tag)
            per_nugget[i] = {
                "committed": False,
                "index": i,
                "nli_label": "bogus",
                "nli_score": score,
                "nli_raw_score": raw_score,
                "reason": reason_tag,
                "hybrid_rule": hybrid_rule,
                "hybrid_signals": hybrid_signals,
            }

    # Step 7: supersede_by handling
    superseded_nid: str | None = None
    if supersede_by and isinstance(supersede_by, str):
        if supersede_by in pipeline_db.nuggets:
            old = pipeline_db.nuggets[supersede_by]
            n_from = old.status
            old.status = "bogus"
            old.status_reason = (
                f"superseded_by={committed_nids[0] if committed_nids else 'pending'}"  # noqa: E501
            )
            ev_s = TransitionEvent(
                step_id="m11_record_evidence",
                iteration=pipeline_db.current_turn,
                actor="system",
                from_status=n_from,
                to_status="bogus",
                reason=f"supersede_by={committed_nids[0] if committed_nids else 'none'}",  # noqa: E501
            )
            append_event(pipeline_db, ev_s)
            old.history.append(ev_s)
            superseded_nid = supersede_by
            # Supersede transition log.
            log.info(
                "pipeline supersede qid=%s old=%s new=%s",
                pipeline_db.question_id,
                supersede_by,
                committed_nids[0] if committed_nids else "pending",
            )
            _emit_event(
                "pipeline.supersede",
                qid=pipeline_db.question_id,
                old_nid=supersede_by,
                new_nid=committed_nids[0] if committed_nids else None,
            )
        else:
            log.warning(
                "pipeline record_evidence: supersede_by=%r references unknown nid; "
                "proceeding without supersede.",
                supersede_by,
            )

    # Step 8: rejection-streak + budget-cap reflect
    reflect: str | None = None
    # Structured label echoed alongside ``reflect`` in the tool response.
    # Downstream training loaders join on this to filter band-bypass /
    # budget-clamp / hard-cap cases without regex over the free-form text.
    reflect_source: str | None = None
    if committed_nids:
        pipeline_db.pid_reject_streak[pid] = 0
    else:
        if rejects_this_call > 0:
            pipeline_db.pid_reject_streak[pid] = pipeline_db.pid_reject_streak.get(pid, 0) + 1
            # Handler-local reflect with diagnose_nli_failure result.
            # Priority over wrap_tool_response generic nudges.
            primary_reason = reject_reasons[0]
            # Cascade-aware reflect for entity_swap rejections: surface the
            # offending entities and tell the model to commit an alias nugget
            # first, with a span containing both names. This is the pattern
            # the `<cascade_nuggets>` prompt block prescribes, turned from a
            # post-hoc diagnostic into a plan for the next turn.
            if primary_reason == "entity_swap":
                # Recompute details from the last rejected pair. Since all
                # rejected pairs here share the same pid (and thus the same
                # quoted_span universe), we use the first rejected entry's
                # (span, claim) for diagnosis.
                _first_rejected_span = ""
                _first_rejected_claim = ""
                for _pn in per_nugget:
                    if isinstance(_pn, dict) and not _pn.get("committed"):
                        # Look up the nugget in the ``nuggets`` input arg by
                        # index; fall back to the server-side recorded form.
                        _idx = _pn.get("index")
                        if isinstance(_idx, int) and 0 <= _idx < len(nuggets_in):
                            _n = nuggets_in[_idx]
                            if isinstance(_n, dict):
                                _first_rejected_span = str(_n.get("quoted_span", ""))
                                _first_rejected_claim = str(_n.get("claim", ""))
                                break
                offending = _entity_swap_details(
                    _first_rejected_span, _first_rejected_claim
                )
                if offending:
                    entity_list = ", ".join(f'"{e}"' for e in offending[:3])
                    reflect = (
                        f"NLI rejected your claim (entity_swap): "
                        f"{entity_list} appear in the claim but not in the "
                        "quoted_span. Before re-submitting, commit an ALIAS "
                        "nugget whose quoted_span contains BOTH the name in "
                        "your claim AND the name used by the passage — e.g. "
                        'span: "Star Wars: Episode I The Phantom Menace", '
                        'claim: "Episode I and The Phantom Menace refer to '
                        'the same film". Then re-submit the substantive '
                        "claim."
                    )
                else:
                    reflect = (
                        "NLI rejected your claim (entity_swap). "
                        "Try a tighter quoted_span or a more literal claim."
                    )
                reflect_source = "nli_entity_swap"
            else:
                reflect = (
                    f"NLI rejected your claim ({primary_reason}). "
                    "Try a tighter quoted_span or a more literal claim."
                )
                reflect_source = f"nli_reject:{primary_reason}"

    if pid not in pipeline_db.recorded_pids:
        pipeline_db.recorded_pids.append(pid)

    # Budget: "reflect: Budget exhausted" once at cap (evaluated in wrap too,
    # but echo here so caller can see it on the same response).
    if pipeline_db.calls_used + 1 >= pipeline_db.calls_total and reflect is None:
        reflect = "Budget exhausted — you must call submit_answer now."
        reflect_source = "budget_exhausted"
    if pipeline_db.nuggets_recorded >= MAX_NUGGETS_HARD and reflect is None:
        reflect = (
            f"Nugget hard cap ({MAX_NUGGETS_HARD}) reached — no further "
            "record_evidence accepted."
        )
        reflect_source = "nugget_hard_cap"

    out: dict[str, Any] = {
        "committed": bool(committed_nids),
        "nugget_ids": committed_nids,
        "nli_scores": nli_scores,
        "stance_accepted": stance,
        "per_nugget": per_nugget,
        "cascaded_nuggets_to_bogus": cascaded_nuggets,
    }
    if superseded_nid:
        out["superseded_nugget_id"] = superseded_nid
    if reflect:
        out["reflect"] = reflect
        if reflect_source:
            out["reflect_source"] = reflect_source
    return out


def _commit_nugget(
    db: PipelineState,
    *,
    pid: str,
    claim: str,
    polarity: str,
    quoted_span: str,
    status: str,
    nli_label: str,
    nli_score: float,
    reason: str,
    hybrid_rule: str | None = None,
    hybrid_signals: dict[str, object] | None = None,
) -> str:
    """Insert a Nugget row + TransitionEvent + bump counters.

    ``hybrid_rule`` and ``hybrid_signals`` record which branch of the
    hybrid cascade fired (e.g. ``lexical_overlap_override``) and the
    underlying signal values. Persisted on the Nugget so the trace
    audit can show the full breakdown per nugget.

    Returns the new nugget_id.
    """
    proposed_stance: Literal["supporting", "infirming"] = (
        "supporting" if polarity == "affirming" else "infirming"
    )
    # Dedup / extension guard. Two-level match against existing nuggets:
    #
    # (1) Same-pid near-duplicate: the LLM re-phrases a
    #     claim across multiple record_evidence calls on the same
    #     passage. Skip the insert and return the existing nugget_id.
    #
    # (2) Cross-pid same-claim (extension path): the LLM
    #     records the same claim on a different passage — this is the
    #     corroboration pattern. append the new pid to the existing
    #     verified_supporting nugget's supporting_pids rather than
    #     minting a duplicate row. Mirrors _auto_commit_support_nugget
    #     so the model-driven path and the ASS path produce the same
    #     DB shape. Only extends into verified_supporting rows with
    #     text match (normalized equality or case-insensitive substring
    #     containment); mismatched stance or different claim text falls
    #     through to the mint branch.
    norm_claim = _normalize_nugget_text(claim)
    same_stance = (
        "verified_supporting"
        if proposed_stance == "supporting"
        else "verified_infirming"
    )
    for existing_id, existing_nug in db.nuggets.items():
        if existing_nug.status in ("bogus",):
            continue
        existing_norm = _normalize_nugget_text(existing_nug.text)
        # (1) Same-pid near-duplicate.
        if existing_nug.passage_id == pid and existing_norm == norm_claim:
            log.info(
                "pipeline record_evidence.dedup qid=%s pid=%s "
                "skipping near-duplicate of %s",
                db.question_id,
                pid,
                existing_id,
            )
            return existing_id
        # (2) Cross-pid same-claim extension. Only extend when the
        # existing nugget is already verified with matching stance —
        # otherwise it may still be "pending" or classified differently
        # and shouldn't carry a second pid yet.
        if (
            existing_nug.status == same_stance
            and status == same_stance
            and existing_nug.passage_id != pid
        ):
            text_match = existing_norm == norm_claim or (
                existing_norm
                and norm_claim
                and (existing_norm in norm_claim or norm_claim in existing_norm)
            )
            if text_match and pid not in existing_nug.supporting_pids:
                existing_nug.supporting_pids.append(pid)
                existing_nug.pid_nli_scores[pid] = float(nli_score)
                log.info(
                    "pipeline record_evidence.extend qid=%s pid=%s "
                    "appended to nid=%s (cross-pid corroboration)",
                    db.question_id,
                    pid,
                    existing_id,
                )
                return existing_id
    nid = f"n_{uuid.uuid4().hex[:8]}"
    # Seed supporting_pids / pid_nli_scores for verified-supporting
    # rows so effective_supporting_pids counts this passage toward n_eff.
    # Infirming / bogus rows get empty lists — the strengthens-by-negation
    # tag is applied later in handle_submit_answer when the rationale is
    # written, and the Nugget validator tops them up on read if needed.
    seed_pids: list[str] = []
    seed_scores: dict[str, float] = {}
    if status == "verified_supporting" and pid:
        seed_pids = [pid]
        seed_scores = {pid: float(nli_score)}
    n = Nugget(
        nugget_id=nid,
        passage_id=pid,
        text=claim,
        proposed_stance=proposed_stance,
        source_step="m11_record_evidence",
        source_tool_call_id=f"m11_turn_{db.current_turn}",
        status=status,  # type: ignore[arg-type]
        status_reason=reason,
        nli_label="entails",  # type: ignore[arg-type]
        nli_score=nli_score,
        hybrid_rule=hybrid_rule,
        hybrid_signals=dict(hybrid_signals or {}),
        supporting_pids=seed_pids,
        pid_nli_scores=seed_scores,
    )
    # Persist quoted_span on the nugget via status_reason — Nugget model
    # doesn't have a dedicated field; the event log captures full context.
    db.nuggets[nid] = n
    ev = TransitionEvent(
        step_id="m11_record_evidence",
        iteration=db.current_turn,
        actor="llm_tool",
        from_status="unclassified",
        to_status=status,
        reason=f"{reason} | claim={claim[:60]} | span={quoted_span[:60]}",
    )
    append_event(db, ev)
    n.history.append(ev)
    _ = nli_label  # reserved for future telemetry
    db.nuggets_recorded += 1
    return nid


# Recursive submission cycle
#
# Every emission goes through the model's own ``submit_answer`` tool call.
# When ``handle_submit_answer`` rejects for thin/zero evidence, the server
# runs ``answer_search_support`` (ASS) to find a supporting passage. If ASS
# auto-commits a nugget, a ``record_evidence`` tool turn is synthesized (as
# if the model had called it) and the agentic loop is re-entered for one
# additional turn so the model can invoke ``submit_answer`` again with the
# new evidence in context. The recursion is bounded by
# ``TRY_SUBMIT_RECURSION_CAP`` and an ``(answer, pid)`` dedup set that
# prevents infinite cycles.

# Max cycles of (submit → reject → ASS → synthesize → re-enter) before
# giving up and routing to the ``force_deliver`` backstop.
TRY_SUBMIT_RECURSION_CAP = 3


def _ass_recursive_submit_enabled() -> bool:
    """Feature flag: default ON. Set ``ASS_RECURSIVE_SUBMIT=0`` to restore
    the legacy flat-band emission path (A/B comparison, emergency rollback)."""
    return os.getenv("ASS_RECURSIVE_SUBMIT", "1") not in ("0", "false", "False", "")


def _synthesize_record_evidence_turn(
    db: PipelineState, nugget: Nugget
) -> list[dict[str, Any]]:
    """Build an (assistant, tool) message pair that mimics the model
    having invoked ``record_evidence`` with ``nugget``'s fields.

    Returns a 2-element list suitable for ``messages.extend(...)``. The
    tool response JSON matches the shape returned by
    ``handle_record_evidence`` on a committed nugget: ``committed=True``,
    a single ``nugget_ids`` entry, matching ``nli_scores``,
    ``stance_accepted``, ``per_nugget``, and an empty
    ``cascaded_nuggets_to_bogus`` field. Chat templates that inspect the
    ``committed`` field and ``nugget_ids`` treat this as genuine.

    The nugget passed in must already be present in ``db.nuggets`` (the
    caller — ``try_submit`` + ASS — commits it before synthesizing).

    Args:
        db: Pipeline DB. Read-only for state (no mutations).
        nugget: A verified_supporting (or verified_infirming) nugget
            recently committed by ``_auto_commit_support_nugget``.

    Returns:
        ``[assistant_msg, tool_msg]`` — ready to append to a running
        conversation list.
    """
    # Unique ids for the synthetic turn so later re-synthesis doesn't
    # collide; same ``call_`` prefix vLLM/Hermes use.
    tool_call_id = f"call_synth_{uuid.uuid4().hex[:10]}"
    polarity = "affirming" if nugget.proposed_stance == "supporting" else "infirming"
    stance = (
        "supporting" if nugget.proposed_stance == "supporting" else "counter_evidence"
    )
    # Fabricate an empty quoted_span tag — the real ``handle_record_evidence``
    # response does not echo the span back, it echoes ``per_nugget`` entries
    # keyed by index. We reconstruct the same shape.
    per_nugget_entry: dict[str, Any] = {
        "committed": True,
        "index": 0,
        "nugget_id": nugget.nugget_id,
        "nli_score": float(nugget.nli_score or 0.0),
        "nli_raw_score": float(nugget.nli_score or 0.0),
        "nli_label": "entails",
        "hybrid_rule": "synthesized_from_ass",
        "hybrid_signals": {},
    }
    tool_response: dict[str, Any] = {
        "committed": True,
        "nugget_ids": [nugget.nugget_id],
        "nli_scores": [float(nugget.nli_score or 0.0)],
        "stance_accepted": stance,
        "per_nugget": [per_nugget_entry],
        "cascaded_nuggets_to_bogus": [],
        "budget": _budget_envelope(db),
        "session_status": _session_status(db),
    }
    # Arguments reflect what a real ``record_evidence`` call would carry.
    tool_call_args: dict[str, Any] = {
        "pid": nugget.passage_id,
        "stance": stance,
        "rationale": (
            "[synthesized] Corpus evidence discovered via answer_search_support."
        )[:MAX_RATIONALE_CHARS],
        "nuggets": [
            {
                "claim": nugget.text[:MAX_CLAIM_CHARS],
                "polarity": polarity,
                # quoted_span is required on real calls; use the nugget
                # text as a proxy so any downstream validator can round-
                # trip. The tool response above already carries committed=True
                # so no validator re-runs.
                "quoted_span": nugget.text[:240],
            }
        ],
    }
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "record_evidence",
                    "arguments": json.dumps(tool_call_args, ensure_ascii=False),
                },
                # Also mirror the flat shape so either consumer format is
                # satisfied.
                "name": "record_evidence",
                "arguments": tool_call_args,
            }
        ],
    }
    tool_msg: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "record_evidence",
        "content": json.dumps(tool_response, ensure_ascii=False),
    }
    return [assistant_msg, tool_msg]


# Prose → submit_answer fallback

# ``Answer: X`` style prefix extraction. The LLM often produces
# ``Answer: Luke Skywalker`` (or variants like ``The answer is Jane Austen``)
# in prose when tool_choice is not enforceable. These regexes surface the
# payload from the first matching prefix in a stable order.
_PROSE_ANSWER_LINE_PREFIXES: tuple[str, ...] = (
    "answer:",
    "final answer:",
    "best guess:",
    "my answer is",
    "the answer is",
    "guess:",
)
_PROSE_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\*+\s*)?"
    r"(?:answer|final answer|best guess|my answer|the answer|guess)"
    r"(?:\*+)?"
    r"(?:\s+is)?"
    r"\s*(?:\*+)?\s*[:\-]?\s*(?:\*+)?\s*(.+)$"
)


def _extract_prose_answer(content: str) -> str | None:
    """Extract a concise answer payload from free-form prose.

    Walks ``content`` line-by-line, matches ``_PROSE_ANSWER_LINE_RE``, and
    hands off to :func:`_strip_finalizer_prefixes` for quote / period /
    prefix cleanup (same helper the finalizer ladder uses). Returns the
    first non-empty match, or ``None`` if no match is found.

    Returns ``None`` for:
      * Empty / whitespace input
      * No ``Answer:``-style line found
      * Extracted payload is empty after stripping
      * Extracted payload upper-cases to ``UNKNOWN`` / contains the
        canonical-refusal markers (so the prose-fallback never promotes
        a refusal to a commit)
    """
    if not content or not isinstance(content, str):
        return None
    for m in _PROSE_ANSWER_LINE_RE.finditer(content):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        # Only look at the first line of the capture group — "Answer: X\n
        # rationale..." must not leak the rationale into the answer.
        first_line = raw.splitlines()[0].strip()
        if not first_line:
            continue
        cleaned = _strip_finalizer_prefixes(first_line)
        if not cleaned:
            continue
        if cleaned.upper() == "UNKNOWN":
            return None
        if _looks_like_canonical_refusal(cleaned):
            return None
        return cleaned
    return None


def _try_prose_submit_fallback(
    db: PipelineState,
    content: str,
) -> dict[str, Any] | None:
    """Programmatic ``submit_answer`` fallback.

    Called when the re-entry LLM turn produced prose with no tool call
    (either because ``tool_choice="required"`` wasn't forwarded, or
    because vLLM let a non-tool response slip through). Extracts the
    answer payload from the prose and invokes
    :func:`handle_submit_answer` directly — the hardgate remains the
    final admission authority.

    Returns:
        The ``handle_submit_answer`` envelope (may or may not be
        committed). ``None`` when no answer was extractable (the caller
        treats this as a genuine refusal).
    """
    if not content:
        return None
    extracted = _extract_prose_answer(content)
    if not extracted:
        return None
    # Respect refusal: the extraction already rejects UNKNOWN /
    # canonical-refusal markers, but also bail if the payload is
    # unreasonably long (>200 chars → probably a free-form rationale
    # that got through the line-matcher).
    if len(extracted) > 200:
        return None
    # Build a minimal synthesized rationale from the first ~240 chars of
    # the prose so the audit trail carries the model's reasoning. Clip
    # at the known MAX_RATIONALE_CHARS ceiling.
    rationale = f"[prose-fallback] {content.strip()[:MAX_RATIONALE_CHARS]}"[
        :MAX_RATIONALE_CHARS
    ]
    args: dict[str, Any] = {
        "answer": extracted,
        "refusal": False,
        "rationale": rationale,
    }
    try:
        return handle_submit_answer(db, args)
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "pipeline _try_prose_submit_fallback handle_submit_answer raised: %s",
            exc,
        )
        return None


async def _reenter_agentic_loop(
    db: PipelineState,
    messages: list[dict[str, Any]],
    max_additional_turns: int = 1,
) -> dict[str, Any]:
    """Re-invoke the agentic loop for ``max_additional_turns`` more turns.

    Uses the finalizer-registered LLM client (same one that feeds tier-4
    educated-guess) and MCP client. Each additional turn dispatches the
    model's chosen tool call via the in-module handlers, so a
    ``submit_answer`` commit produces the same envelope
    ``handle_submit_answer`` would.

    Returns a small dict describing the outcome:

        * ``tool_called``: name of the tool the model invoked (or None).
        * ``handler_result``: the wrapped tool response (when a tool fired).
        * ``turns``: number of model turns consumed.
        * ``skipped`` / ``error``: set when the re-entry couldn't run
          (no LLM client, LLM exception, etc.).

    Never raises. On any exception the DB is left unchanged and an
    ``error`` field is populated.
    """
    llm_client, mcp_client = _get_finalizer_clients()
    if llm_client is None:
        return {
            "skipped": True,
            "tool_called": None,
            "turns": 0,
            "error": "no_llm_client",
        }

    turns_consumed = 0
    last_result: dict[str, Any] = {"tool_called": None, "turns": 0}

    # Retrieve the stashed tool catalogue so we can forward it
    # alongside ``tool_choice="required"``. OpenAI/vLLM reject
    # ``tool_choice="required"`` without a ``tools`` list (the Hermes
    # parser needs a catalogue to choose from), so when the catalogue is
    # empty we intentionally skip the constraint rather than submitting a
    # payload that 400s.
    reentry_tools = _get_finalizer_tools()
    use_tool_choice_required = bool(reentry_tools)

    for _turn in range(max(1, int(max_additional_turns))):
        turns_consumed += 1
        try:
            # Primary path: pass the tool catalogue with
            # ``tool_choice="required"`` so vLLM forces a tool call instead
            # of prose. A well-behaved client (``OpenAILLMClient``) accepts
            # both kwargs; older stubs may not, so we degrade gracefully.
            if use_tool_choice_required:
                try:
                    write_response = await llm_client.chat(
                        messages=messages,
                        tools=list(reentry_tools),
                        tool_choice="required",
                        enable_thinking=True,
                    )
                except TypeError:
                    # Stub doesn't accept ``tool_choice`` — try without it
                    # but keep the tools list so the model still sees the
                    # catalogue. The prose fallback rescues prose
                    # outputs if the model refuses to call a tool.
                    try:
                        write_response = await llm_client.chat(
                            messages=messages,
                            tools=list(reentry_tools),
                            enable_thinking=True,
                        )
                    except TypeError:
                        write_response = await llm_client.chat(
                            messages=messages,
                            tools=list(reentry_tools),
                        )
            else:
                # No tool catalogue cached (e.g. admin_inject_state probes
                # that never call ``set_finalizer_tools``). Fall back to
                # the tools=None path; the prose fallback
                # catches any prose answer the model emits.
                try:
                    write_response = await llm_client.chat(
                        messages=messages,
                        tools=None,
                        enable_thinking=True,
                    )
                except TypeError:
                    write_response = await llm_client.chat(
                        messages=messages,
                        tools=None,
                    )
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline _reenter_agentic_loop LLM call raised")
            return {
                "skipped": False,
                "tool_called": None,
                "turns": turns_consumed,
                "error": f"llm_exception:{type(exc).__name__}:{exc!s}"[:240],
            }

        tool_calls = getattr(write_response, "tool_calls", None) or []
        if not tool_calls:
            # The model produced prose instead of a tool call
            # (either because tool_choice wasn't forwarded, or because
            # vLLM let the refusal slip through). Try to extract an
            # answer from the prose and synthesize a submit_answer call
            # programmatically. The hardgate in ``handle_submit_answer``
            # still decides whether to commit.
            content = getattr(write_response, "content", "") or ""
            fallback = _try_prose_submit_fallback(db, content)
            if fallback is not None:
                last_result = {
                    "skipped": False,
                    "tool_called": "submit_answer",
                    "turns": turns_consumed,
                    "handler_result": fallback,
                    "prose_fallback": True,
                }
                _emit_event(
                    "pipeline.reentry.prose_fallback",
                    qid=db.question_id if hasattr(db, "question_id") else None,
                    committed=bool(fallback.get("committed")),
                    content_preview=content[:120],
                )
                if bool(fallback.get("committed")):
                    return last_result
                # Non-committed fallback — still more informative than a
                # silent refusal. Break out so the outer try_submit loop
                # can decide whether to keep recursing.
                break
            # Genuine refusal / ambiguous prose — cycle complete with no
            # commit via tool call.
            _emit_event(
                "pipeline.reentry.model_refused",
                qid=db.question_id if hasattr(db, "question_id") else None,
                content_preview=content[:120],
            )
            return {
                "skipped": False,
                "tool_called": None,
                "turns": turns_consumed,
                "content_preview": content[:240],
            }

        # Dispatch the first tool call inline using the in-module handlers.
        tc = tool_calls[0]
        tool_name = getattr(tc, "name", None) or ""
        tool_args = getattr(tc, "arguments", {}) or {}
        handler_result: dict[str, Any] = {}
        try:
            if tool_name == "submit_answer":
                handler_result = handle_submit_answer(db, tool_args)
            elif tool_name == "record_evidence":
                handler_result = handle_record_evidence(db, tool_args)
            elif tool_name == "flag_nugget_bogus":
                handler_result = handle_flag_nugget_bogus(db, tool_args)
            elif tool_name == "list_nuggets":
                handler_result = handle_list_nuggets(db, tool_args)
            elif tool_name == "batch_record_stance":
                handler_result = handle_batch_record_stance(db, tool_args)
            elif mcp_client is not None and tool_name in (
                "search_passages",
                "search_entities",
                "get_passage",
                "extract_from_passage",
            ):
                handler_result = await mcp_client.call_tool(
                    name=tool_name, arguments=tool_args, timeout=30.0
                )
            else:
                handler_result = {
                    "error": "unsupported_reentry_tool",
                    "tool": tool_name,
                }
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "pipeline _reenter_agentic_loop handler raised for tool=%s",
                tool_name,
            )
            handler_result = {
                "error": "handler_exception",
                "message": str(exc)[:240],
            }

        last_result = {
            "skipped": False,
            "tool_called": tool_name or None,
            "turns": turns_consumed,
            "handler_result": handler_result,
        }
        # Terminate immediately on submit_answer commit — no more cycles.
        if tool_name == "submit_answer" and bool(handler_result.get("committed")):
            return last_result
        # For non-terminating tool calls we'd normally keep looping. With
        # ``max_additional_turns=1`` (the default), we exit after the
        # single turn regardless — the outer ``try_submit`` decides what
        # to do next.
        break

    return last_result


def try_submit(
    db: PipelineState,
    answer: str,
    evidence: Any | None = None,
    *,
    depth: int = 0,
) -> dict[str, Any]:
    """Recursive primitive for the submission cycle.

    Sequence:
      1. If ``evidence`` is a committed Nugget, synthesize the
         ``record_evidence`` turn and re-enter the loop for one more
         model turn. That turn's ``submit_answer`` call (if any) runs
         through ``handle_submit_answer`` — hardgate + compute_confidence
         — and commits at the evidence-derived confidence, not a tier band.
      2. If ``evidence`` is None, call ``answer_search_support`` (ASS)
         against the candidate answer. On auto-commit, recurse with the
         new nugget as evidence. On no finding, return a rejection
         envelope (not a flat-band commit).
      3. Bounded by ``TRY_SUBMIT_RECURSION_CAP``. After that many cycles
         the caller (``handle_submit_answer``) routes to ``force_deliver``
         as the narrow backstop.

    Dedup: ``db._try_submit_seen`` (created on first call) tracks
    ``(answer_norm, pid)`` tuples so ASS cannot re-find the same
    passage and cycle.

    Args:
        db: Pipeline DB (must be PipelineState).
        answer: The candidate answer the model (or ladder) proposed.
        evidence: Optional pre-found Nugget (skips ASS if provided).
        depth: Internal recursion counter. Callers pass 0.

    Returns:
        The envelope from the terminal action — either
        ``handle_submit_answer``'s committed envelope (normal path) or
        ``force_deliver``'s (backstop). Always dict-shaped.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    # Initialise the dedup set on first entry. Stored directly on the DB
    # as a plain attribute (Pydantic permits attribute set; we avoid
    # putting it on the schema so trace serialisation doesn't carry it).
    seen: set[tuple[str, str]] = getattr(pipeline_db, "_try_submit_seen", None) or set()
    if not hasattr(pipeline_db, "_try_submit_seen"):
        try:
            pipeline_db._try_submit_seen = seen  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            # Pydantic v2 permissive — assignment should succeed; the
            # try is defence-in-depth.
            pass

    answer_norm = _normalize_nugget_text(answer or "")

    # Recursion cap — bail to force_deliver backstop.
    if depth >= TRY_SUBMIT_RECURSION_CAP:
        log.info(
            "pipeline try_submit qid=%s recursion_cap_hit depth=%d",
            pipeline_db.question_id,
            depth,
        )
        _emit_event(
            "pipeline.try_submit.cap",
            qid=pipeline_db.question_id,
            depth=depth,
        )
        return force_deliver(
            pipeline_db,
            reason=f"try_submit_recursion_cap:depth={depth}",
            preferred_answer=answer or None,
        )

    if evidence is not None and isinstance(evidence, Nugget):
        # Step 1: synthesize record_evidence + re-enter.
        pid_key = evidence.passage_id or ""
        dedup_key = (answer_norm, pid_key)
        if dedup_key in seen:
            log.info(
                "pipeline try_submit qid=%s dedup_skip answer=%r pid=%s",
                pipeline_db.question_id,
                answer[:60],
                pid_key,
            )
            return force_deliver(
                pipeline_db,
                reason=f"try_submit_dedup:pid={pid_key}",
                preferred_answer=answer or None,
            )
        seen.add(dedup_key)

        session_messages = list(_get_finalizer_session_messages() or [])
        session_messages.extend(_synthesize_record_evidence_turn(pipeline_db, evidence))
        reentry = _run_coro_sync(
            _reenter_agentic_loop(pipeline_db, session_messages, max_additional_turns=1),
            timeout=FINALIZER_LLM_TIMEOUT_S + FINALIZER_SEARCH_TIMEOUT_S,
        )
        log.info(
            "pipeline try_submit qid=%s depth=%d reentry tool=%s",
            pipeline_db.question_id,
            depth,
            (reentry or {}).get("tool_called"),
        )
        _emit_event(
            "pipeline.try_submit.reentry",
            qid=pipeline_db.question_id,
            depth=depth,
            tool=(reentry or {}).get("tool_called"),
            committed=bool(
                ((reentry or {}).get("handler_result") or {}).get("committed")
            ),
        )
        if isinstance(reentry, dict):
            handler_result = reentry.get("handler_result") or {}
            if reentry.get("tool_called") == "submit_answer" and bool(
                handler_result.get("committed")
            ):
                return handler_result
            # Model refused despite evidence, or called a different tool
            # (record_evidence / search_passages). Fall through to a
            # narrow retry: try_submit one more cycle with no evidence,
            # which may trigger a fresh ASS search or hit the cap.
            return try_submit(pipeline_db, answer, evidence=None, depth=depth + 1)
        return force_deliver(
            pipeline_db,
            reason=f"try_submit_reentry_failed:depth={depth}",
            preferred_answer=answer or None,
        )

    # Step 2: no evidence yet — run ASS.
    _, mcp_client = _get_finalizer_clients()
    if mcp_client is None:
        log.info(
            "pipeline try_submit qid=%s no_mcp_client depth=%d",
            pipeline_db.question_id,
            depth,
        )
        return force_deliver(
            pipeline_db,
            reason=f"try_submit_no_mcp:depth={depth}",
            preferred_answer=answer or None,
        )
    try:
        verified, details = _run_coro_sync(
            answer_search_support(pipeline_db, answer, mcp_client, origin="try_submit"),
            timeout=_verify_outer_timeout_s() * ANSWER_SEARCH_SUPPORT_MAX_ITERS,
        )
    except Exception:  # noqa: BLE001
        log.exception("pipeline try_submit ASS raised qid=%s", pipeline_db.question_id)
        return force_deliver(
            pipeline_db,
            reason=f"try_submit_ass_exception:depth={depth}",
            preferred_answer=answer or None,
        )

    auto_ids = details.get("auto_nuggets") or []
    if not (verified and auto_ids):
        # ASS found nothing — do not emit flat-band. Force-deliver is the
        # narrow backstop; the canonical-refusal / prior-rollout branches
        # inside ``_force_deliver_impl`` are the only survivors.
        log.info(
            "pipeline try_submit qid=%s ass_no_find depth=%d last=%s",
            pipeline_db.question_id,
            depth,
            details.get("last_decision"),
        )
        return force_deliver(
            pipeline_db,
            reason=f"try_submit_ass_no_find:{details.get('last_decision', 'unknown')}",
            preferred_answer=answer or None,
        )

    # Find the most-recently committed ASS nugget and recurse with it as
    # evidence. (``auto_nuggets`` last entry is freshest.)
    new_nid = auto_ids[-1]
    new_nug = pipeline_db.nuggets.get(new_nid)
    if new_nug is None:
        return force_deliver(
            pipeline_db,
            reason=f"try_submit_ass_nugget_missing:{new_nid}",
            preferred_answer=answer or None,
        )
    return try_submit(pipeline_db, answer, evidence=new_nug, depth=depth + 1)


# handle_submit_answer


def handle_submit_answer(
    db: PipelineDB,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Emit final answer + rationale; terminate the loop on commit.

    Args:
        db: Pipeline DB (must be PipelineState).
        args: ``{answer, refusal, rationale, citation_pids?}``; no
            ``confidence`` — the server computes it.

    Behavior:
        refusal=True → canonical text + synth refusal nugget + conf=0.0 +
            exit_reason="low_confidence_refusal".
        refusal=False → derive citations if omitted, compute confidence,
            tiered hard-gate with 1 retry slot if any of:
              • n_eff_supporting == 0                   (zero-evidence block)
              • n_eff_supporting == 1 and conf < 0.70   (thin-evidence guard)
              • conf < 0.50                             (legacy low-conf guard)
            else commit + exit_reason="answer_submitted".
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    # arg validation
    answer_text = args.get("answer", "")
    refusal = bool(args.get("refusal", False))
    rationale = args.get("rationale", "")
    citation_pids = args.get("citation_pids")

    # Same JSON-string deserialization as handle_record_evidence: the
    # Hermes tool-call parser ships array-typed args as strings. A silent
    # accept here would let `citation_pids` be serialized back as a JSON
    # string and break the PR-ranking pipeline downstream.
    if isinstance(citation_pids, str):
        try:
            citation_pids = json.loads(citation_pids)
        except (json.JSONDecodeError, TypeError):
            return {
                "committed": False,
                "error": "citation_pids_malformed_json",
                "message": (
                    "citation_pids arg arrived as a string but did not parse "
                    "as JSON — pass citation_pids as a JSON array of pids, or "
                    "omit the arg to let the server derive it."
                ),
                "budget": _budget_envelope(pipeline_db),
                "session_status": _session_status(pipeline_db),
            }
    if citation_pids is not None and not isinstance(citation_pids, list):
        return {
            "committed": False,
            "error": "citation_pids_not_list",
            "message": (
                "citation_pids must be a JSON array of pid strings when "
                f"provided; got {type(citation_pids).__name__}."
            ),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    if (
        not isinstance(rationale, str)
        or not rationale
        or len(rationale) > MAX_RATIONALE_CHARS
    ):
        return {
            "committed": False,
            "error": "invalid_rationale",
            "message": f"rationale must be 1..{MAX_RATIONALE_CHARS} chars",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if not refusal and (not isinstance(answer_text, str) or not answer_text):
        return {
            "committed": False,
            "error": "invalid_answer",
            "message": "answer must be non-empty when refusal=False",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    # submit_history tracking
    # Append an entry for every submit attempt that passes arg validation.
    # Covers refusal commits, answer commits, and all rejection branches
    # below. Final-fallback path 4 (force_deliver) picks the max-confidence
    # entry when every other path has failed. Confidence is populated later
    # (answer branch) once computed; refusal entries carry 0.0.
    submit_history_entry: dict[str, Any] = {
        "turn": pipeline_db.current_turn,
        "answer": answer_text if not refusal else "",
        "refusal": refusal,
        "rationale": rationale[:MAX_RATIONALE_CHARS],
        "citation_pids": list(citation_pids) if citation_pids else [],
        "confidence": 0.0,
        "committed": False,
        "outcome": "",
    }
    pipeline_db.submit_history.append(submit_history_entry)

    # Refusal branch
    if refusal:
        # submit_answer.refusal entry log: surfaces the LLM's own rationale
        # for refusing, which is otherwise visible only in the trace file.
        log.info(
            "pipeline submit_answer.refusal qid=%s rationale=%r",
            pipeline_db.question_id,
            rationale[:160],
        )
        _emit_event(
            "pipeline.submit_answer.refusal",
            qid=pipeline_db.question_id,
            rationale_preview=rationale[:240],
        )

        # Exhaustion gate. Bypassed in Phase-2 (the evidence surface is
        # already frozen; further retrieval is pointless). Checks that
        # the agent has satisfied minimum per-tool usage thresholds (12
        # calls, 6 unique queries, 2 get_passage calls, 1 deep-read, no
        # byte-repeat query loops) before admitting a refusal. On failure:
        #   - attempts 1..MAX_REFUSAL_REJECTIONS: return committed=False
        #     with a reflect enumerating the specific missing actions
        #   - attempt MAX_REFUSAL_REJECTIONS+1 (safety valve):
        #     route to force_deliver(reason="exhaustion_safety_valve")
        #     so a zero-confidence refusal is never submitted on a qid
        #     where the agent hasn't actually exhausted search.
        if not pipeline_db.phase2_active:
            gate_passed, missing = _exhaustion_check(pipeline_db)
            if not gate_passed:
                if pipeline_db.refusal_exhaustion_rejects < MAX_REFUSAL_REJECTIONS:
                    pipeline_db.refusal_exhaustion_rejects += 1
                    reflect_msg = "Before refusing, complete: " + "; ".join(missing)
                    log.info(
                        "pipeline submit_answer.refusal.gate_rejected "
                        "qid=%s attempt=%d/%d missing=%s",
                        pipeline_db.question_id,
                        pipeline_db.refusal_exhaustion_rejects,
                        MAX_REFUSAL_REJECTIONS,
                        missing,
                    )
                    _emit_event(
                        "pipeline.submit_answer.refusal.gate_rejected",
                        qid=pipeline_db.question_id,
                        attempt=pipeline_db.refusal_exhaustion_rejects,
                        missing=missing[:5],
                    )
                    return {
                        "committed": False,
                        "reason": "exhaustion_gate_unmet",
                        "reflect": reflect_msg,
                        "confidence": 0.0,
                        "budget": _budget_envelope(pipeline_db),
                        "session_status": _session_status(pipeline_db),
                    }
                # Safety valve: gates still unmet after MAX_REFUSAL_REJECTIONS
                # bounces. Route to force_deliver instead of emitting a
                # zero-confidence refusal — calibrated low-conf guess is
                # strictly better than refusal under R2C2 scoring.
                log.warning(
                    "pipeline submit_answer.refusal.safety_valve qid=%s "
                    "still missing after %d rejects: %s — force-delivering",
                    pipeline_db.question_id,
                    pipeline_db.refusal_exhaustion_rejects,
                    missing,
                )
                _emit_event(
                    "pipeline.submit_answer.refusal.safety_valve",
                    qid=pipeline_db.question_id,
                    rejects_exhausted=pipeline_db.refusal_exhaustion_rejects,
                    missing=missing[:5],
                )
                submit_history_entry["committed"] = True
                submit_history_entry["outcome"] = "exhaustion_safety_valve"
                return force_deliver(
                    pipeline_db,
                    reason="exhaustion_safety_valve",
                    preferred_answer=answer_text or None,
                )

        # Every gate-passing model refusal is routed through force_deliver
        # so the emitted envelope picks up tier-4 (conf=0.10) or tier-5
        # (canonical refusal at conf=0.05) instead of a bare conf=0
        # commit. Under R2C2 calibration scoring, a 0.05-0.10 guess
        # strictly beats a zero-confidence refusal. Phase-2 benefits too:
        # its terminal refusal upgrades to a tier-routed guess when
        # any real passage was retrieved during Phase-1.
        #
        # Escape hatch: set ``R2C2_NO_BARE_REFUSAL=0`` for the bare
        # refusal (A/B comparison or emergency rollback without a code
        # change).
        _snapshot_ensemble_touched(pipeline_db)
        _refusal_reason = (
            "refusal branch (phase2 bypass)"
            if pipeline_db.phase2_active
            else "refusal branch"
        )

        if _refusal_routes_through_force_deliver():
            # Deliberately do not write rationale onto db.answer — a
            # refusal rationale is prose like "No supporting evidence..."
            # which would hijack force_deliver's tier-4 selector and ship
            # as the final answer. The LLM's rationale is still preserved
            # in the TransitionEvent reason below for replay audit.
            # _auto_reject_unclassified and validate_finish are skipped:
            # force_deliver owns the terminal path.
            if not pipeline_db.answer.reasoning_trace.strip():
                pipeline_db.answer.reasoning_trace = "refusal path: routed to force_deliver."
            ev = TransitionEvent(
                step_id="m11_submit_answer",
                iteration=pipeline_db.current_turn,
                actor="llm_tool",
                from_status=None,
                to_status="exit=model_requested_refusal_routed_to_force_deliver",
                reason=(_refusal_reason + " | rationale=" + rationale[:200]),
            )
            append_event(pipeline_db, ev)
            submit_history_entry["committed"] = True
            submit_history_entry["outcome"] = "refusal_routed_to_force_deliver"
            # Refused-with-nuggets rescue: when the
            # agent has already recorded real verified_supporting nuggets
            # and has a non-empty candidate answer floating in state
            # (db.answer.draft_answer or an earlier non-refusal
            # submit_history entry), promote that candidate to
            # ``preferred_answer`` so ``_best_non_empty_answer`` re-tries
            # the T1→T2 NLI gate. If T1/T2 verifies, force_deliver emits
            # at tier 1/2. If verification fails, the ladder falls
            # through to canonical_refusal (preserved backstop). The
            # model's stated refusal becomes advisory, not binding,
            # whenever the corpus has already endorsed an answer.
            rescue_candidate = _refusal_rescue_candidate(pipeline_db)
            if rescue_candidate is not None:
                log.info(
                    "pipeline submit_answer.refusal.rescue_attempt qid=%s "
                    "n_real_nuggets=%d candidate=%r",
                    pipeline_db.question_id,
                    sum(
                        1
                        for n in pipeline_db.nuggets.values()
                        if n.status == "verified_supporting"
                        and not getattr(n, "is_synthetic", False)
                    ),
                    rescue_candidate[:60],
                )
                _emit_event(
                    "pipeline.submit_answer.refusal.rescue_attempt",
                    qid=pipeline_db.question_id,
                    candidate_preview=rescue_candidate[:80],
                )
            return force_deliver(
                pipeline_db,
                reason="model_requested_refusal",
                preferred_answer=rescue_candidate,
            )

        # Bare-refusal path (R2C2_NO_BARE_REFUSAL=0): canonical refusal at
        # conf=0, for A/B comparison only. The invariant rescue in
        # loop.py:_finalize_trace_from_db still clamps to the
        # confidence floor of 5, but the answer stays the canonical
        # refusal string.
        _auto_reject_unclassified(pipeline_db)
        synthesize_refusal(pipeline_db)
        pipeline_db.answer.draft_answer = CANONICAL_REFUSAL_ANSWER
        pipeline_db.answer.confidence = 0.0
        pipeline_db.answer.rationale = rationale
        pipeline_db.answer.exit_reason = "low_confidence_refusal"
        if not pipeline_db.answer.reasoning_trace.strip():
            pipeline_db.answer.reasoning_trace = "refusal path: no supporting evidence."
        pipeline_db.terminated = True
        ev = TransitionEvent(
            step_id="m11_submit_answer",
            iteration=pipeline_db.current_turn,
            actor="llm_tool",
            from_status=None,
            to_status="exit=low_confidence_refusal",
            reason=_refusal_reason,
        )
        append_event(pipeline_db, ev)
        submit_history_entry["committed"] = True
        submit_history_entry["outcome"] = "refusal_committed"
        return {
            "committed": True,
            "answer": CANONICAL_REFUSAL_ANSWER,
            "confidence": 0.0,
            "exit_reason": "low_confidence_refusal",
            "rationale": rationale,
        }

    # Normal branch
    # Ensemble feed-forward: snapshot the agent-judged pids before the
    # auto-reject sweep runs.
    _snapshot_ensemble_touched(pipeline_db)
    # Auto-reject any still-unclassified real passages: the validator
    # requires zero unclassified at finish time. The unified loop accepts or
    # ignores passages explicitly via record_evidence; unseen hits (e.g.
    # search results the LLM didn't promote) default to rejected.
    _auto_reject_unclassified(pipeline_db)
    # n_eff counts distinct supporting passages, not nugget rows.
    # ``effective_supporting_ids`` still drives code
    # paths that need nugget identity (citation picker, rationale tag
    # binding) — this is strictly the hardgate counter.
    eff_pids = effective_supporting_pids(pipeline_db)
    n_eff = len(eff_pids)

    # Pipeline-level R2C2 format gate, symmetric with the tier-2+ ladder.
    # The model's ``answer_text`` went through the Hermes tool-call schema
    # so it is non-empty, but nothing has yet checked for ``\n/\r/\t/;``
    # or sentinel leaks. ``validate_r2c2_answer`` is the single source of
    # truth for every emission site; on failure the canonical refusal is
    # substituted at floor confidence (same contract as tier-2+). The
    # `format_rejected` outcome is recorded on ``submit_history``.
    qtype = pipeline_db.question_type or _question_shape_classifier(pipeline_db.question or "")
    _fmt = _validate_r2c2_answer(answer_text, qtype)
    if not _fmt.ok:
        log.warning(
            "pipeline submit_answer.format_reject qid=%s reason=%s cleaned_preview=%r",
            pipeline_db.question_id,
            _fmt.reason,
            (_fmt.cleaned or "")[:120],
        )
        _emit_event(
            "pipeline.submit_answer.format_reject",
            qid=pipeline_db.question_id,
            reason=_fmt.reason,
            preview=(_fmt.cleaned or "")[:200],
        )
        submit_history_entry["committed"] = True
        submit_history_entry["outcome"] = f"format_rejected:{_fmt.reason}"
        return force_deliver(
            pipeline_db,
            reason=f"format_rejected:{_fmt.reason}",
            preferred_answer=None,
        )
    answer_text = _fmt.cleaned

    # Clear a stale ``low_confidence_refusal`` exit_reason before calling
    # ``compute_confidence``, which returns 0.0 whenever ``exit_reason ==
    # "low_confidence_refusal"``. That guard is correct when the refusal
    # was just committed. But when the main loop terminates with
    # low_confidence_refusal and ``force_deliver`` then runs the finalizer
    # ladder (tier-4 → ``try_submit`` → synthesized record_evidence →
    # re-entry → the model calls ``submit_answer`` with a fresh
    # non-refusal answer), the stale flag would zero every subsequent
    # ``compute_confidence`` call, giving conf=0.00 with n_eff=1 at
    # pe=0.98. The draft answer is about to be overwritten with a new
    # non-refusal submission, so the prior refusal is no longer the
    # current state. Only the value that would short-circuit the score is
    # cleared; ``high_confidence``, ``error`` and ``force_deliver`` are
    # unaffected. On reject no fresh exit_reason is set here; the caller
    # (``try_submit`` or ``force_deliver``) writes the final value.
    if pipeline_db.answer.exit_reason == "low_confidence_refusal":
        pipeline_db.answer.exit_reason = None  # type: ignore[assignment]

    # Preset answer state for compute_confidence — it reads from db.answer.
    pipeline_db.answer.draft_answer = answer_text
    pipeline_db.answer.rationale = rationale
    # Recompute the auto-derived supportability from the current
    # n_effective_supporting count on every submit attempt. Deriving it
    # only when unset would pin the label on the first attempt: n_eff=1
    # gives "insufficient" (base 0.55) and the hardgate blocks; a second
    # attempt with n_eff=2 would keep "insufficient" and land at conf 0.61
    # instead of 0.91.
    #
    # Only the two server-derived values {sufficient, insufficient} are
    # overwritten. Externally-set {contradicted, unresolvable} labels
    # reflect richer state (verified_infirming evidence, explicit refusal
    # reasoning) and are preserved.
    if pipeline_db.answer.supportability in {
        "",
        "sufficient",
        "insufficient",
        None,
    }:
        pipeline_db.answer.supportability = "sufficient" if n_eff >= 2 else "insufficient"  # type: ignore[assignment]
    conf = compute_confidence(pipeline_db)
    pipeline_db.answer.confidence = conf

    # Hard gate (tiered): zero evidence is always blocked; thin evidence
    # (1 supporter) passes only through the thin-evidence escape; otherwise
    # conf < 0.5 is blocked. A plain `conf<0.5 and n_eff<2` conjunction
    # would let through conf=0.55 + n_eff=0 answers, because the default
    # supportability "insufficient" pins compute_confidence to exactly 0.55
    # (R2C2_DECOUPLED_SCORING=0), one tick above the 0.5 threshold.
    #
    # Path 2: retries 1..SUBMIT_RETRY_CAP-1 → reflect-reject so the model
    # can self-correct; on the SUBMIT_RETRY_CAP'th rejection the server
    # escalates to ``answer_search_support`` to fetch evidence around the
    # model's answer. If that surfaces a verified_supporting nugget and
    # the hardgate clears, fall through to the normal commit. Otherwise
    # route to ``force_deliver``; nothing is committed silently.
    #
    # Thin-evidence escape: n_eff=1 passes if the single supporting
    # passage entails the claim with NLI >= 0.90 and the computed conf is
    # at or above the default base 0.55. It is the only admission
    # criterion for n_eff==1: supportability is "insufficient" whenever
    # n_eff<2, which caps compute_confidence at 0.58 under
    # R2C2_DECOUPLED_SCORING=0 (≈0.77 for high NLI under =1), and no other
    # code path can pin supportability="sufficient" with n_eff==1. A
    # `conf < 0.70` test on that branch would therefore be vacuous, and the
    # gate is a Boolean predicate with no arithmetic on the conf value.
    thin_ok = _thin_evidence_clears(pipeline_db, n_eff, conf)
    if n_eff == 0 or (n_eff == 1 and not thin_ok) or conf < 0.50:
        if pipeline_db.answer_submit_retries < SUBMIT_RETRY_CAP:
            pipeline_db.answer_submit_retries += 1
            retry_num = pipeline_db.answer_submit_retries
            log.info(
                "pipeline submit_answer.hardgate qid=%s conf=%.2f n_eff=%d retry=%d/%d",
                pipeline_db.question_id,
                conf,
                n_eff,
                retry_num,
                SUBMIT_RETRY_CAP,
            )
            _emit_event(
                "pipeline.submit_answer.hardgate",
                qid=pipeline_db.question_id,
                confidence=round(conf, 3),
                n_effective_supporting=n_eff,
                retry=retry_num,
                retry_cap=SUBMIT_RETRY_CAP,
            )
            submit_history_entry["confidence"] = float(conf)
            submit_history_entry["outcome"] = f"hardgate_rejected_{retry_num}"
            return {
                "committed": False,
                "reason": "low_confidence_or_insufficient_evidence",
                "reflect": (
                    f"Submit #{retry_num}/{SUBMIT_RETRY_CAP} failed: "
                    f"conf={conf:.2f} with n_eff={n_eff}. Fetch one more "
                    "supporting passage via search_passages + record_evidence, "
                    "or call submit_answer(refusal=true)."
                ),
                "confidence": conf,
                "n_effective_supporting": n_eff,
            }

        # SUBMIT_RETRY_CAP reached.
        #
        # Under ASS_RECURSIVE_SUBMIT=1 (default), the hardgate-reject
        # branch routes through ``try_submit``, the recursive primitive
        # that runs ASS, synthesizes a record_evidence turn when a
        # nugget is auto-committed, re-enters the agentic loop, and lets
        # the model invoke submit_answer again: a full tool-mediated
        # commit. The flat-band Path-2 block below runs only when the
        # flag is off.
        if _ass_recursive_submit_enabled():
            submit_history_entry["confidence"] = float(conf)
            submit_history_entry["outcome"] = "try_submit_entry"
            log.info(
                "pipeline submit_answer.try_submit qid=%s conf=%.2f n_eff=%d — "
                "routing reject branch through recursive primitive",
                pipeline_db.question_id,
                conf,
                n_eff,
            )
            _emit_event(
                "pipeline.submit_answer.try_submit_entry",
                qid=pipeline_db.question_id,
                confidence=round(conf, 3),
                n_effective_supporting=n_eff,
            )
            # ``try_submit`` runs ASS → on auto-commit synthesizes a
            # record_evidence turn + re-enters the agentic loop →
            # model's ``submit_answer`` on the re-entry turn reaches
            # this handler again with evidence in play and clears the
            # hardgate. On no-find / model-refusal / cap-hit,
            # try_submit routes to force_deliver (backstop).
            return try_submit(pipeline_db, answer_text, evidence=None)

        # Flag-off path: Path 2 answer_search_support + flat
        # rescue, behind ``ASS_RECURSIVE_SUBMIT=0`` for
        # A/B comparison and emergency rollback.
        path2_rescued = False
        if _route_to_search_support_enabled():
            _llm_client, mcp_client = _get_finalizer_clients()
            if mcp_client is not None:
                # Covers the MCP retry budget inside _retrieve_for_support
                # × the 3 reformulation iterations.
                verify_timeout = (
                    _verify_outer_timeout_s() * ANSWER_SEARCH_SUPPORT_MAX_ITERS
                )
                try:
                    verified, details = _run_coro_sync(
                        answer_search_support(
                            pipeline_db, answer_text, mcp_client, origin="path2"
                        ),
                        timeout=verify_timeout,
                    )
                except Exception:  # noqa: BLE001
                    log.exception(
                        "pipeline submit_answer.path2 qid=%s raised",
                        pipeline_db.question_id,
                    )
                    verified, details = (
                        False,
                        {
                            "last_decision": "runner_error",
                            "iterations": 0,
                            "auto_nuggets": [],
                            "soft_injects": [],
                        },
                    )
                log.info(
                    "pipeline submit_answer.path2 qid=%s verified=%s "
                    "last=%s auto_nuggets=%d soft_injects=%d iters=%d",
                    pipeline_db.question_id,
                    verified,
                    details.get("last_decision"),
                    len(details.get("auto_nuggets", [])),
                    len(details.get("soft_injects", [])),
                    details.get("iterations", 0),
                )
                _emit_event(
                    "pipeline.submit_answer.path2",
                    qid=pipeline_db.question_id,
                    verified=bool(verified),
                    last_decision=details.get("last_decision"),
                    auto_nuggets=len(details.get("auto_nuggets", [])),
                    soft_injects=len(details.get("soft_injects", [])),
                    iterations=details.get("iterations", 0),
                )
                if verified and details.get("auto_nuggets"):
                    # Re-evaluate the hardgate with the freshly committed
                    # verified_supporting nugget (or pid append) in play,
                    # counting distinct supporting pids, not rows.
                    new_eff_pids = effective_supporting_pids(pipeline_db)
                    new_n_eff = len(new_eff_pids)
                    if pipeline_db.answer.supportability in {
                        "",
                        "sufficient",
                        "insufficient",
                        None,
                    }:
                        pipeline_db.answer.supportability = (  # type: ignore[assignment]
                            "sufficient" if new_n_eff >= 2 else "insufficient"
                        )
                    new_conf = compute_confidence(pipeline_db)
                    pipeline_db.answer.confidence = new_conf
                    # Same thin-evidence escape as the primary hardgate.
                    new_thin_ok = _thin_evidence_clears(pipeline_db, new_n_eff, new_conf)
                    hardgate_still_fails = (
                        new_n_eff == 0
                        or (new_n_eff == 1 and not new_thin_ok)
                        or new_conf < 0.50
                    )
                    if not hardgate_still_fails:
                        conf = new_conf
                        n_eff = new_n_eff
                        path2_rescued = True
                        submit_history_entry["outcome"] = "path2_rescued"
                        log.info(
                            "pipeline submit_answer.path2 qid=%s RESCUED "
                            "new_conf=%.2f new_n_eff=%d",
                            pipeline_db.question_id,
                            new_conf,
                            new_n_eff,
                        )

        if not path2_rescued:
            submit_history_entry["confidence"] = float(conf)
            # ``committed`` has a single meaning: committed at the tier-1
            # hardgate clear or the tier-2 Path-2 rescue. An attempt that
            # ends in force_deliver keeps committed=False and carries
            # force_delivered=True as its outcome marker.
            submit_history_entry["force_delivered"] = True
            submit_history_entry["outcome"] = "submit_retries_exhausted_path2_fail"
            return force_deliver(
                pipeline_db,
                f"submit_retries_exhausted:conf={conf:.2f},n_eff={n_eff}",
                preferred_answer=answer_text,
            )
        # path2_rescued: fall through to the normal commit flow.

    # Derive citations if omitted
    if citation_pids is None:
        citation_pids = [
            n.passage_id
            for nid, n in pipeline_db.nuggets.items()
            if n.status == "verified_supporting"
        ]
        # Dedup preserving order
        seen: set[str] = set()
        citation_pids = [p for p in citation_pids if not (p in seen or seen.add(p))]

    # Ensemble feed-forward: union the citation pids into the
    # touched-set snapshot even if their Passage.status stayed
    # unclassified. Mirrors the force_deliver rule.
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID_SUBMIT  # noqa: N811

    for _cp in citation_pids or []:
        if _cp and _cp != _REFUSAL_PID_SUBMIT:
            pipeline_db.ensemble_touched_snapshot.add(_cp)

    pipeline_db.answer.exit_reason = "high_confidence"
    # Reasoning trace guard (validate_finish requires non-empty in high_conf).
    if not pipeline_db.answer.reasoning_trace.strip():
        pipeline_db.answer.reasoning_trace = f"pipeline loop: {rationale[:200]}"

    # Thin-evidence reconciliation: ``_thin_evidence_clears`` is the
    # admission path for n_eff=1 (single supporter at NLI ≥ 0.90). When
    # that escape fires and conf is above the high_confidence validator
    # threshold, the validator's ``supportability=sufficient`` invariant
    # must agree with the admission. Otherwise the hardgate admits the
    # thin-evidence n_eff=1 answer and validate_finish then rejects it on
    # the same DB state. Supportability is upgraded to "sufficient" only
    # when the thin-evidence escape admits the answer.
    if (
        thin_ok
        and conf >= 0.8
        and pipeline_db.answer.supportability in {"insufficient", "", None}
    ):
        pipeline_db.answer.supportability = "sufficient"  # type: ignore[assignment]

    # The validator's "high_confidence" exit requires conf>=0.8. The
    # exit_reason here is "answer_submitted", not "high_confidence", so
    # both are used: validate as "high_confidence" for invariants, expose
    # "answer_submitted" to the caller.
    err: str | None = None
    if conf >= 0.8:
        err = validate_finish(pipeline_db, "high_confidence")
    else:
        # Below high-conf threshold — skip validator's high-conf gates. We
        # still require universal invariants (no unclassified passages etc.)
        err = _validate_universal_invariants(pipeline_db)

    if err is not None:
        submit_history_entry["confidence"] = float(conf)
        submit_history_entry["outcome"] = "validation_failed"
        return {
            "committed": False,
            "reason": "validation_failed",
            "errors": [err],
            "reflect": f"Validator rejected submit: {err}",
            "confidence": conf,
        }

    pipeline_db.terminated = True
    ev = TransitionEvent(
        step_id="m11_submit_answer",
        iteration=pipeline_db.current_turn,
        actor="llm_tool",
        from_status=None,
        to_status="exit=answer_submitted",
        reason=f"conf={conf:.2f} n_eff={n_eff}",
    )
    append_event(pipeline_db, ev)
    submit_history_entry["citation_pids"] = list(citation_pids)
    submit_history_entry["confidence"] = float(conf)
    submit_history_entry["committed"] = True
    submit_history_entry["outcome"] = "answer_committed"
    return {
        "committed": True,
        "answer": answer_text,
        "confidence": conf,
        "exit_reason": "answer_submitted",
        "citation_pids": citation_pids,
        "rationale": rationale,
        "n_effective_supporting": n_eff,
    }


def _snapshot_ensemble_touched(db: PipelineState) -> None:
    """Capture the set of pids the agent has explicitly judged so far.

    Writes the snapshot onto ``db.ensemble_touched_snapshot``. Union —
    never overwrites a previous snapshot — because ``handle_submit_answer``
    may be retried across multiple submit attempts and each attempt
    contributes the same or an expanded set.

    Called from the top of the refusal + normal branches of
    ``handle_submit_answer`` before ``_auto_reject_unclassified`` runs,
    and from ``force_deliver`` after the citation pid is chosen. See
    ``touched_pids()`` above for what consumers do with it.
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _RID

    current = getattr(db, "ensemble_touched_snapshot", None)
    if current is None:
        db.ensemble_touched_snapshot = set()
        current = db.ensemble_touched_snapshot
    for pid, p in db.passages.items():
        if pid == _RID:
            continue
        if p.status in {"accepted", "rejected", "counter_evidence"}:
            current.add(pid)


def _auto_reject_unclassified(db: PipelineState) -> None:
    """Flip any leftover unclassified real passages to 'rejected'.

    The unified-agentic loop does not mandate the LLM classify every
    retrieved passage explicitly — but the validate_finish() invariants
    require it. On submit_answer we bridge that by demoting unreviewed pids
    to rejected (they're not supporting evidence anyway).
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _RID

    for pid, p in db.passages.items():
        if pid == _RID:
            continue
        if p.status == "unclassified":
            from_status = p.status
            p.status = "rejected"
            p.status_reason = "auto-reject at submit_answer (not reviewed)"
            ev = TransitionEvent(
                step_id="m11_submit_answer",
                iteration=db.current_turn,
                actor="system",
                from_status=from_status,
                to_status="rejected",
                reason="auto-reject unreviewed at submit",
            )
            append_event(db, ev)
            p.history.append(ev)


def _validate_universal_invariants(db: PipelineState) -> str | None:
    """Subset of validate_finish that only checks invariants that don't
    depend on a particular exit_reason's threshold (no conf≥0.8 gate)."""
    # Budget caps
    if len(db.passages) > db.max_passages + 1:
        return f"passages={len(db.passages)} exceeds cap {db.max_passages}"
    real_nuggets = sum(1 for n in db.nuggets.values() if n.status != "refusal")
    if real_nuggets > db.max_nuggets_global:
        return f"nuggets exceed cap {db.max_nuggets_global}"
    # Verified nuggets must anchor to accepted/counter_evidence pids.
    for n in db.nuggets.values():
        if n.status.startswith("verified_"):
            if n.passage_id not in db.passages:
                return f"nugget {n.nugget_id} has unknown pid"
            if db.passages[n.passage_id].status not in {
                "accepted",
                "counter_evidence",
            }:
                return f"nugget {n.nugget_id} anchored to non-accepted pid"
    return None


# wrap_tool_response — budget envelope + already_recorded + reflect nudges


# Tools that attach an already_recorded flag to returned passages.
_PASSAGE_RETURNING_TOOLS = {"search_passages", "get_passage"}


def wrap_tool_response(
    db: PipelineDB,
    tool_name: str,
    raw_response: dict[str, Any] | str,
) -> dict[str, Any]:
    """Decorate every tool response with budget envelope + reflect nudges.

    Called by the dispatcher after each tool handler runs. Responsibilities:

    1. Increment ``db.calls_used`` (the server-side counter).
    2. Parse raw_response (MCP tools return JSON strings; orchestrator
       handlers return dicts directly).
    3. Inject the ``budget`` envelope.
    4. Inject ``already_recorded`` flags on returned passages for
       ``search_passages`` / ``get_passage`` (within-session watermark).
    5. Inject ``session_status``.
    6. Compute a ``reflect`` nudge if any of the 5 triggers fire, with
       priority order + 2-turn cooldown.

    Any handler-set ``reflect`` field survives — generic nudges are never
    overwritten onto an existing reflect string (handler context beats the
    deterministic trigger).
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    # Parse raw response — MCP tools JSON-encode, orchestrator handlers don't.
    if isinstance(raw_response, str):
        try:
            merged: dict[str, Any] = json.loads(raw_response)
        except json.JSONDecodeError:
            merged = {"error": "invalid_json", "raw": raw_response}
    elif isinstance(raw_response, dict):
        merged = dict(raw_response)
    else:
        merged = {"error": "invalid_response_type", "raw": repr(raw_response)}

    # Increment before decorating so budget reflects post-call state.
    pipeline_db.calls_used += 1
    pipeline_db.current_turn += 1
    if tool_name == "record_evidence":
        pipeline_db.calls_used_on_record_evidence += 1
    elif tool_name == "search_passages":
        pipeline_db.calls_used_on_search_passages += 1
        # Log the query for the no_answer_keyword_search nudge.
        # search_passages schema: queries: str or list — normalize.
        q = None
        if isinstance(raw_response, str):
            try:
                raw_dict = json.loads(raw_response)
                q = raw_dict.get("main_query") or raw_dict.get("queries")
            except json.JSONDecodeError:
                q = None
        if q:
            if isinstance(q, list):
                pipeline_db.search_queries.extend(str(x) for x in q)
            else:
                pipeline_db.search_queries.append(str(q))
        # untouched_top_chunk nudge: track rank-1 pids seen in search results.
        # merged is always a dict at this point (JSON parse above); grab
        # the top-ranked passage_id if present.
        passages_list = merged.get("passages") if isinstance(merged, dict) else None
        if isinstance(passages_list, list) and passages_list:
            first = passages_list[0]
            if isinstance(first, dict):
                first_pid = first.get("passage_id") or first.get("pid")
                if isinstance(first_pid, str) and first_pid:
                    pipeline_db.seen_top_pids.add(first_pid)
    elif tool_name == "get_passage":
        pipeline_db.calls_used_on_get_passage += 1
        # untouched_top_chunk nudge: record which pids the agent actually opened.
        # Parse the tool's returned passages — each has a ``passage_id``
        # field. Only center chunks (is_center=True) when window>0; also
        # records the requested pid unconditionally.
        p_list = merged.get("passages") if isinstance(merged, dict) else None
        if isinstance(p_list, list):
            for p_row in p_list:
                if isinstance(p_row, dict):
                    pid_val = p_row.get("passage_id") or p_row.get("pid")
                    if isinstance(pid_val, str) and pid_val:
                        pipeline_db.top_chunk_read_pids.add(pid_val)
    elif tool_name == "extract_from_passage":
        pipeline_db.calls_used_on_extract_from_passage += 1
    elif tool_name == "search_entities":
        pipeline_db.calls_used_on_search_entities += 1

    # already_recorded flag on passage-returning tools.
    if tool_name in _PASSAGE_RETURNING_TOOLS:
        if "passages" in merged and isinstance(merged["passages"], list):
            for p_row in merged["passages"]:
                if isinstance(p_row, dict):
                    pid = p_row.get("passage_id") or p_row.get("pid")
                    if pid:
                        p_row["already_recorded"] = _is_recorded(pipeline_db, pid)
        elif "passage_id" in merged and isinstance(merged["passage_id"], str):
            merged["already_recorded"] = _is_recorded(pipeline_db, merged["passage_id"])

    # Reflect nudge, only if the handler didn't already set one. A log
    # line records each nudge that fires, since a nudge changes the
    # conversation.
    handler_reflect = merged.get("reflect")
    if not handler_reflect:
        nudge = fire_reflect_nudge(pipeline_db, tool_name)
        if nudge is not None:
            merged["reflect"] = nudge
            pipeline_db.last_reflect_turn = pipeline_db.current_turn
            log.info(
                "pipeline reflect qid=%s source=nudge tool=%s turn=%d text=%r",
                pipeline_db.question_id,
                tool_name,
                pipeline_db.current_turn,
                nudge[:160],
            )
            _emit_event(
                "pipeline.reflect",
                qid=pipeline_db.question_id,
                source="nudge",
                tool=tool_name,
                turn=pipeline_db.current_turn,
                text=nudge[:240],
            )

    # Budget envelope + session_status.
    merged["budget"] = _budget_envelope(pipeline_db)
    merged["session_status"] = _session_status(pipeline_db)
    return merged


def _is_recorded(db: PipelineState, pid: str) -> bool:
    """Within-session watermark: pid was recorded on this qid."""
    return pid in db.passages and db.passages[pid].status != "unclassified"


def touched_pids(db: PipelineState) -> set[str]:
    """Ensemble feed-forward: every pid the agent has seen this iteration.

    This is the cross-iteration uniqueness invariant: any passage id
    that surfaced to the agent (via ``search_passages``, ``get_passage``,
    or ``extract_from_passage``), regardless of what the agent did with
    it, must be excluded from the next iteration so the same pid cannot
    re-surface. This includes:

    - Every pid in ``db.passages`` (entered via retrieval / read seeding
      in ``loop._seed_passages_from_mcp``), independent of
      ``Passage.status``. Restricting to
      ``status ∈ {accepted, rejected, counter_evidence}`` would let
      unjudged-but-retrieved pids leak across iterations.
    - Every pid captured in ``db.ensemble_touched_snapshot`` by
      ``handle_submit_answer`` and ``force_deliver`` (defence-in-depth
      for force-delivered / successful-submit citation pids).
    - Every ``citation_pids`` entry on ``db.force_deliver_envelope``
      (legacy trace replay path).

    Excludes the synthetic refusal pid in all branches.

    Returns:
        Set of pids to subtract from the virtual retrieval index when
        the next ensemble iteration runs.
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID

    # Snapshot every retrieved/read pid, not only judged ones.
    out: set[str] = {pid for pid in db.passages.keys() if pid and pid != _REFUSAL_PID}
    # Union in the pre-submit snapshot (captures force_deliver.chosen_pid
    # and successful-submit citation_pids even when their Passage.status
    # stayed unclassified, and any pid added externally — e.g. tracer
    # tooling — that did not flow through db.passages).
    out |= {
        pid
        for pid in getattr(db, "ensemble_touched_snapshot", set())
        if pid and pid != _REFUSAL_PID
    }
    # Defense-in-depth: if a force_deliver envelope is on db but the
    # snapshot was never updated (e.g. legacy trace replay), still honour
    # its citation_pids list.
    env = getattr(db, "force_deliver_envelope", None) or {}
    for _cp in env.get("citation_pids") or []:
        if _cp and _cp != _REFUSAL_PID:
            out.add(_cp)
    return out


def extract_question_entities(question: str) -> list[str]:
    """Extract capitalized entities from the question (regex-only).

    Used by the no_answer_keyword_search nudge: if the LLM has made ≥5
    tool calls and no search_passages query mentions any of these entities,
    nudge it to search directly.
    """
    if not question:
        return []
    return [m for m in _ENTITY_RE.findall(question)]


def _exhaustion_check(db: PipelineState) -> tuple[bool, list[str]]:
    """Exhaustion gate: has the agent demonstrated enough evidence-gathering?

    Called from the refusal branch of ``handle_submit_answer``. Returns
    ``(all_passed, missing_action_strings)``. Each failed gate appends a
    concrete, actionable string naming the specific shortfall (e.g.
    pid that was never opened, query that was byte-repeated). These
    strings are surfaced to the LLM as a reflect nudge when the refusal
    is rejected, so the agent knows exactly what to do next.

    Gates (all must pass to admit a refusal):
      1. ``calls_used >= MIN_CALLS_BEFORE_REFUSAL``
      2. ``>= MIN_UNIQUE_QUERIES`` distinct normalized search queries
      3. ``>= MIN_GET_PASSAGE_CALLS`` get_passage invocations
      4. ``>= MIN_EXTRACT_OR_TOPCHUNK_READ`` either extract_from_passage
         calls or the number of top-ranked pids the agent actually
         opened via get_passage (whichever is larger)
      5. No ``MAX_JACCARD_CONSECUTIVE_QUERIES`` threshold exceeded on
         two or more consecutive query pairs (one high-Jaccard pair is
         a legitimate rephrase; two in a row is a loop).
    """
    missing: list[str] = []

    # Gate 1 — minimum call budget consumed.
    if db.calls_used < MIN_CALLS_BEFORE_REFUSAL:
        short = MIN_CALLS_BEFORE_REFUSAL - db.calls_used
        missing.append(
            f"make {short} more tool call(s) "
            f"(have {db.calls_used}/{MIN_CALLS_BEFORE_REFUSAL})"
        )

    # Gate 2 — query diversity.
    normalized_queries = {_normalize_nugget_text(q) for q in db.search_queries if q}
    n_unique = len(normalized_queries)
    if n_unique < MIN_UNIQUE_QUERIES:
        missing.append(
            f"try {MIN_UNIQUE_QUERIES - n_unique} more DIFFERENT search query "
            f"(have {n_unique}/{MIN_UNIQUE_QUERIES} unique)"
        )

    # Gate 3 — get_passage depth.
    if db.calls_used_on_get_passage < MIN_GET_PASSAGE_CALLS:
        # Prefer specific action: name a top-ranked pid the agent never
        # opened. Fall back to generic "call get_passage" if none.
        unopened_top = sorted(db.seen_top_pids - db.top_chunk_read_pids)
        if unopened_top:
            missing.append(
                f"call get_passage on your top-ranked hit(s), e.g. "
                f"get_passage('{unopened_top[0]}', window=2) — "
                f"{db.calls_used_on_get_passage}/{MIN_GET_PASSAGE_CALLS}"
            )
        else:
            missing.append(
                f"call get_passage at least {MIN_GET_PASSAGE_CALLS} times "
                f"(have {db.calls_used_on_get_passage})"
            )

    # Gate 4 — deep read on at least one passage (extract or top-chunk).
    deep_read_count = db.calls_used_on_extract_from_passage + len(
        db.top_chunk_read_pids
    )
    if deep_read_count < MIN_EXTRACT_OR_TOPCHUNK_READ:
        missing.append(
            "deep-read at least one passage — call "
            "extract_from_passage(pid, predicate) or get_passage(pid) to "
            "pull its full text"
        )

    # Gate 5 — no back-to-back byte-similar query pairs. Uses the
    # normalized form to match the dedup nudge at fire time. We flip
    # fail only when >=2 consecutive pairs exceed the threshold, so a
    # single legitimate rephrase doesn't trip the gate.
    raw = [q for q in db.search_queries if q]
    high_jaccard_streak = 0
    for i in range(1, len(raw)):
        if _jaccard_unigrams(raw[i - 1], raw[i]) >= MAX_JACCARD_CONSECUTIVE_QUERIES:
            high_jaccard_streak += 1
            if high_jaccard_streak >= 2:
                missing.append(
                    "vary your query tokens — you have 2+ consecutive "
                    f"near-identical queries (last: {raw[i][:80]!r})"
                )
                break
        else:
            high_jaccard_streak = 0

    return (len(missing) == 0, missing)


def _count_verified_supporting(db: PipelineState) -> int:
    """Number of nuggets with ``status == "verified_supporting"``.

    Shared helper used by the reminder-nudge logic and the
    ``answer_search_support`` procedure to decide whether
    there is enough evidence to recommend ``submit_answer``.
    """
    return sum(1 for n in db.nuggets.values() if n.status == "verified_supporting")


def _periodic_reminder_if_due(db: PipelineState) -> str | None:
    """Return a submit-answer reminder text iff ``calls_used`` just
    crossed a ``REMINDER_NUDGE_EVERY_N_CALLS`` boundary and the model
    holds ≥ 1 verified supporting nugget.

    The conditions together capture: (a) model has burned ≥ 10, 20, 30
    ... calls, (b) model has actionable evidence on hand, (c) it still
    hasn't committed. The nudge is a periodic guard against drift: it
    reminds the model that ``submit_answer`` exists.
    """
    if db.calls_used <= 0:
        return None
    if db.calls_used % REMINDER_NUDGE_EVERY_N_CALLS != 0:
        return None
    n_verified = _count_verified_supporting(db)
    if n_verified < 1:
        return None
    remaining = max(0, db.calls_total - db.calls_used)
    return (
        f"You've used {db.calls_used}/{db.calls_total} calls and have "
        f"{n_verified} verified supporting nugget(s). "
        f"{remaining} call(s) remaining. If the answer is clear, call "
        "submit_answer(answer='...', citation_pids=[...]) now."
    )


def _record_evidence_success_nudge(
    db: PipelineState,
    tool_name: str,
    last_nli_score: float | None,
) -> str | None:
    """Return a post-record_evidence reminder iff the just-completed
    tool call was ``record_evidence`` and it produced a new
    ``verified_supporting`` nugget.

    Fires at a natural decision point — the model just committed
    evidence; this nudge reminds it that calling ``submit_answer`` is
    the next logical step once the evidence is strong.
    """
    if tool_name != "record_evidence":
        return None
    n_verified = _count_verified_supporting(db)
    if n_verified < 1:
        return None
    nli_tag = (
        f"NLI={last_nli_score:.2f}" if last_nli_score is not None else "NLI=committed"
    )
    return (
        f"Nugget recorded ({nli_tag}). You now have {n_verified} verified "
        "supporting nugget(s). Consider submit_answer."
    )


def fire_reflect_nudge(db: PipelineState, tool_name: str) -> str | None:
    """Priority-ordered nudge firing with 2-turn cooldown.

    Returns the nudge string or None. Priorities:
        0.  record_evidence_success (just committed a verified nugget)
        0b. periodic_reminder (calls_used % 10 == 0 and n_verified ≥ 1)
        1.  nuggets_exhausted (recorded_nuggets ≥ 9)
        2.  budget_low (calls_remaining ≤ 4)
        3.  zero_verified_after_3 (≥3 record_evidence, 0 verified)
        4.  no_answer_keyword_search (≥5 turns, no query hits Q entities)
        5.  nli_repeat_fail (same pid failed 2× in a row)

    Priorities 0 and 0b are additional sources of nudges — the 2-turn
    cooldown still applies, so adding them cannot cause back-to-back
    spam. They push the model toward submit_answer at decision moments
    (priority 0: evidence just committed) and at steady-state drift
    boundaries (priority 0b: every 10 calls).
    """
    # Cooldown guard.
    if db.current_turn < db.last_reflect_turn + REFLECT_COOLDOWN_TURNS:
        return None

    # 0. record_evidence_success — fires at the natural commit moment.
    evidence_nudge = _record_evidence_success_nudge(db, tool_name, None)
    if evidence_nudge is not None:
        return evidence_nudge

    # 0b. periodic_reminder — fires every N calls if the model has usable evidence.
    periodic = _periodic_reminder_if_due(db)
    if periodic is not None:
        return periodic

    # 1. nuggets_exhausted
    if db.nuggets_recorded >= 9:
        return "One nugget slot left — save it for your strongest claim."

    # 2. budget_low
    remaining = max(0, db.calls_total - db.calls_used)
    if remaining <= 4:
        return "Budget low — consider submit_answer now."

    # 3. zero_verified_after_3
    verified_count = sum(
        1 for n in db.nuggets.values() if n.status == "verified_supporting"
    )
    if db.calls_used_on_record_evidence >= 3 and verified_count == 0:
        return (
            "Three record_evidence attempts, zero verified — re-read the "
            "passage or try a different claim shape."
        )

    # 4. no_answer_keyword_search
    if db.calls_used >= 5:
        entities = extract_question_entities(db.question)
        entity_lower = {e.lower() for e in entities}
        if entity_lower:
            matched = any(
                any(e in q.lower() for e in entity_lower) for q in db.search_queries
            )
            if not matched:
                return "Have you searched for the answer entity directly?"

    # 5. nli_repeat_fail
    for pid, streak in db.pid_reject_streak.items():
        if streak >= 2:
            return (
                f"Two consecutive NLI rejections on pid {pid} — consider "
                "discarding it and retrieving new evidence."
            )

    # 6. duplicate_query — fires on byte-identical (after normalize)
    # consecutive search_passages queries. Signals the agent is looping
    # instead of exploring.
    if len(db.search_queries) >= 2:
        if _normalize_nugget_text(db.search_queries[-1]) == _normalize_nugget_text(
            db.search_queries[-2]
        ):
            return (
                "Duplicate query — your last two search_passages calls used "
                "the same query tokens. Try synonyms, a different proper noun, "
                "or a different query shape (e.g. add quoted phrase)."
            )

    # 7. uncovered_keywords — once a handful of searches have run,
    # surface the question-keywords that have never appeared in any
    # retrieved passage text. This tells the agent which specific terms
    # need a fresh search.
    if db.calls_used >= 6:
        try:
            _, missed = _keyword_coverage(db.question, db.passages)
        except Exception:  # noqa: BLE001
            missed = set()
        if missed:
            sample = sorted(missed)[:5]
            return (
                f"Question keywords not yet covered by any retrieved passage: "
                f"{', '.join(sample)}. Search for them directly."
            )

    # 8. untouched_top_chunk — agent retrieved top-ranked passages
    # but never fetched their full text via get_passage. Top-1 hits are
    # usually the answer-bearing chunk; rejecting without opening them
    # is premature.
    if db.calls_used >= 8:
        untouched = db.seen_top_pids - db.top_chunk_read_pids
        if untouched:
            pick = sorted(untouched)[0]
            return (
                f"You've retrieved top-ranked passages but never pulled the "
                f"full text of {pick}. Call get_passage('{pick}', window=2) "
                "before concluding evidence is insufficient."
            )

    _ = tool_name  # reserved for per-tool throttling
    return None


# Regex nugget fallback


def _normalize_nugget_text(text: str) -> str:
    """Normalize for dedup: lowercase + collapse internal whitespace."""
    return " ".join(text.lower().strip().split())


def parse_regex_nugget_fallback(
    db: PipelineDB,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Scrape free-text NUGGET[pid]: claim entries from assistant messages.

    Runs only on non-refusal terminations. Dedupes
    against existing ``db.nuggets`` by ``(pid, normalize(text))`` so it
    does not duplicate nuggets already committed via ``record_evidence``.

    Each scrape becomes a ``Nugget`` with ``source_step='regex_fallback'``,
    ``status='verified_supporting'`` (or ``verified_infirming`` if the
    optional ``[counter]`` suffix was present), and ``nli_label/nli_score =
    None`` — the regex path is intentionally an NLI bypass because its
    purpose is to recover nuggets the LLM typed in prose instead of the
    tool call.

    Args:
        db: PipelineDB (reads ``passages``, writes ``nuggets``).
        messages: conversation message list (role/content dicts).

    Returns:
        Stats dict:
            ``scraped`` — new nuggets committed
            ``deduped`` — matches skipped because content was already recorded
            ``unknown_pid`` — matches skipped because pid not in db.passages
            ``scraped_nugget_ids`` — list of committed nugget_ids
            ``skipped_reason`` — set to "refusal_path" on a refusal termination
    """
    stats: dict[str, Any] = {
        "scraped": 0,
        "deduped": 0,
        "unknown_pid": 0,
        "scraped_nugget_ids": [],
    }

    # Skip on the refusal path. The refusal nugget owns the slot; regex
    # would introduce races and possibly contradict the canonical refusal.
    if db.answer.exit_reason == "low_confidence_refusal":
        stats["skipped_reason"] = "refusal_path"
        return stats

    # Build dedup set from nuggets already committed (excluding bogus /
    # superseded since those aren't "occupying" the slot anymore).
    existing: set[tuple[str, str]] = set()
    for n in db.nuggets.values():
        if n.status in {"bogus", "superseded"}:
            continue
        existing.add((n.passage_id, _normalize_nugget_text(n.text)))

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str) or not content:
            continue
        for match in REGEX_NUGGET_PATTERN.finditer(content):
            pid = match.group("pid")
            text = match.group("text").strip()
            stance_raw = match.group("stance") or "supporting"
            if not pid or not text:
                continue

            # Filter: pid must exist in the session. Drops false positives
            # like NUGGET[p_example]: ... that the LLM may fabricate.
            if pid not in db.passages:
                stats["unknown_pid"] += 1
                continue

            key = (pid, _normalize_nugget_text(text))
            if key in existing:
                stats["deduped"] += 1
                continue
            existing.add(key)

            # Map stance → status + proposed_stance (the Nugget Stance Literal
            # is only "supporting"|"infirming"). Missing stance → supporting.
            if stance_raw == "counter":
                nugget_status = "verified_infirming"
                proposed_stance: Literal["supporting", "infirming"] = "infirming"
            else:
                nugget_status = "verified_supporting"
                proposed_stance = "supporting"

            nugget_id = f"n_{uuid.uuid4().hex[:8]}"
            nug = Nugget(
                nugget_id=nugget_id,
                passage_id=pid,
                text=text,
                proposed_stance=proposed_stance,
                source_step="regex_fallback",
                source_tool_call_id="regex_fallback",
                status=nugget_status,  # type: ignore[arg-type]
                status_reason="scraped from assistant text (Strangler Fig)",
                nli_label=None,
                nli_score=None,
            )
            db.nuggets[nugget_id] = nug

            # Keep the PipelineState counters in sync so downstream nudges / budget
            # reporting reflect regex-scraped nuggets too.
            if isinstance(db, PipelineState):
                db.nuggets_recorded += 1
                if pid not in db.recorded_pids:
                    db.recorded_pids.append(pid)

            stats["scraped"] += 1
            stats["scraped_nugget_ids"].append(nugget_id)

    return stats


# Internal utilities


def _ensure_pipeline_db(db: PipelineDB) -> None:
    """Raise if ``db`` is not an ``PipelineState`` — handlers need the extra
    fields (budget, termination flags). Callers should seed via
    ``seed_db_empty()`` which returns ``PipelineState`` typed as
    ``PipelineDB``.
    """
    if not isinstance(db, PipelineState):
        raise TypeError(
            "pipeline handlers require an PipelineState; seed with "
            "seed_db_empty(qid, question, qtype) — not PipelineDB()."
        )


# force-deliver floor
# The "never return an empty quadruple" terminal safety net. Called from:
#   - answer_question_pipeline terminal branches (no_tool_call / max_iter /
#     token_budget / proactive-near-budget)
#   - handle_submit_answer refusal branch on validate_finish reject
#   - handle_submit_answer normal branch on answer_submit_retries >= 3


_PR_MAX_PASSAGES = 20  # R2C2 submission hard cap


def _prune_passages_to_cap(db: PipelineState, keep: set[str] | None = None) -> None:
    """Ensure len(db.passages) <= _PR_MAX_PASSAGES + 1 (virtual refusal pid).

    Keeps the virtual REFUSAL_PID always, plus any pid in ``keep``, plus
    accepted/counter_evidence passages (highest score first). Drops
    unclassified passages first, then rejected. Deterministic.
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID

    keep_set: set[str] = {_REFUSAL_PID}
    if keep:
        keep_set |= keep

    for pid, p in db.passages.items():
        if p.status in ("accepted", "counter_evidence"):
            keep_set.add(pid)

    if len(db.passages) <= _PR_MAX_PASSAGES + 1:
        return

    # Candidates to drop, in priority order (weakest first):
    droppable_unclassified = [
        pid
        for pid, p in db.passages.items()
        if pid not in keep_set and p.status == "unclassified"
    ]
    droppable_rejected = [
        pid
        for pid, p in db.passages.items()
        if pid not in keep_set and p.status == "rejected"
    ]
    droppable_rest = [pid for pid in db.passages if pid not in keep_set]

    target_drop = len(db.passages) - (_PR_MAX_PASSAGES + 1)
    for ordered in (droppable_unclassified, droppable_rejected, droppable_rest):
        while target_drop > 0 and ordered:
            pid = ordered.pop()
            if pid in db.passages and pid not in keep_set:
                del db.passages[pid]
                target_drop -= 1
        if target_drop <= 0:
            return


def _db_has_corpus_evidence(db: PipelineState) -> bool:
    """True when the DB carries at least one accepted passage or verified
    supporting nugget.

    Used as a cheap presence-check before running the per-answer
    verification step. The presence-check alone is not sufficient — an
    accepted passage proves the agent judged something, not that the
    specific answer is entailed by the corpus.
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID

    for p in db.passages.values():
        if p.passage_id == _REFUSAL_PID:
            continue
        if p.status in ("accepted", "counter_evidence"):
            return True
    for n in db.nuggets.values():
        if n.status == "verified_supporting":
            return True
    return False


def _db_nugget_entails_answer(db: PipelineState, answer: str) -> bool:
    """True if the DB already has a verified_supporting nugget whose text
    has meaningful overlap with ``answer``.

    Fast-path gate before running full retrieval+NLI verification: if the
    agent already produced a nugget whose NLI-checked claim covers the
    answer text, we do not need to re-verify. Token-Jaccard overlap
    ≥ 0.3 (after stopword removal) is the threshold — chosen small
    enough to admit "police officer" against a nugget saying "Min-Woo
    works as a police officer" but large enough to reject incidental
    same-topic nuggets.
    """
    if not answer:
        return False
    ans_tokens = {t for t in _tokenize_lower(answer) if t not in _KEYWORD_STOPWORDS}
    if not ans_tokens:
        return False
    for n in db.nuggets.values():
        if n.status != "verified_supporting":
            continue
        n_tokens = {t for t in _tokenize_lower(n.text) if t not in _KEYWORD_STOPWORDS}
        if not n_tokens:
            continue
        jac = len(ans_tokens & n_tokens) / len(ans_tokens | n_tokens)
        if jac >= 0.3:
            return True
    return False


def _verify_candidate_synchronously(
    db: PipelineState,
    candidate: str,
) -> tuple[bool, dict[str, Any]]:
    """Synchronous wrapper around ``_verify_guess_with_retrieval``.

    The verification coroutine is async; this wrapper lets the sync
    callers in ``_best_non_empty_answer`` and ``force_deliver`` use the
    same check, so every emission path runs one NLI pipeline.

    Returns ``(verified, stats)`` where ``stats`` mirrors
    ``_verify_guess_with_retrieval``'s schema. Never raises.
    """
    _, mcp_client = _get_finalizer_clients()
    if mcp_client is None:
        return False, {
            "decision": "skip",
            "error": "no_mcp_client",
            "p_entail_max": 0.0,
            "p_contra_max": 0.0,
            "n_spans": 0,
        }
    timeout = _verify_outer_timeout_s()
    try:
        verified, stats = _run_coro_sync(
            _verify_guess_with_retrieval(db, candidate, mcp_client),
            timeout=timeout,
        )
    except TimeoutError:
        return False, {
            "decision": "error",
            "error": "verify_timeout",
            "p_entail_max": 0.0,
            "p_contra_max": 0.0,
            "n_spans": 0,
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("verify_candidate_synchronously raised")
        return False, {
            "decision": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "p_entail_max": 0.0,
            "p_contra_max": 0.0,
            "n_spans": 0,
        }
    return verified, stats


def _best_non_empty_answer(
    db: PipelineState,
    preferred_answer: str | None = None,
) -> tuple[str | None, int]:
    """Sequential T1 → T2 fallback with universal corpus-entailment check.

    Builds the candidate list in strict tier order:

      * Tier 1 — ``preferred_answer`` (from submit_history / caller)
      * Tier 2 — ``draft_answer`` (from the model's last submit attempt)

    Each candidate is tested independently with the same NLI gate the
    educated-guess ladder uses (format → nugget fastpath → NLI). The
    first tier that passes wins; if T1 fails format or NLI, T2 is
    tested on its own merits, not skipped. If both fail, returns
    ``(None, 0)`` so T4 runs.

    One NLI gate for every emission path regardless of tier origin.

    Returns:
      * ``(answer, 1)`` — preferred_answer verified.
      * ``(answer, 2)`` — draft_answer verified (T1 absent or failed).
      * ``(None, 0)`` — both failed; T4 ladder must handle it.
    """
    candidates: list[tuple[str, int]] = []
    if preferred_answer and preferred_answer.strip():
        candidates.append((preferred_answer.strip()[:2000], 1))
    draft = (db.answer.draft_answer or "").strip() if db.answer else ""
    if draft and draft != CANONICAL_REFUSAL_ANSWER:
        candidates.append((draft[:2000], 2))

    if not candidates:
        return None, 0

    qtype = db.question_type or _question_shape_classifier(db.question or "")

    for raw_candidate, source_tier in candidates:
        # Pipeline-level R2C2 format gate — a candidate that fails
        # the AC-file rules (\n/\r/\t/;/sentinel/>200 chars) is rejected
        # before the NLI-verify spend. On fail try the next tier.
        fmt_res = _validate_r2c2_answer(raw_candidate, qtype)
        if not fmt_res.ok:
            log.warning(
                "pipeline best_answer qid=%s tier=%d format_fail reason=%s "
                "cleaned_preview=%r — trying next tier",
                db.question_id,
                source_tier,
                fmt_res.reason,
                (fmt_res.cleaned or "")[:120],
            )
            continue
        cleaned = fmt_res.cleaned

        if _db_nugget_entails_answer(db, cleaned):
            log.info(
                "pipeline best_answer qid=%s tier=%d path=nugget_fastpath",
                db.question_id,
                source_tier,
            )
            return cleaned, source_tier

        verified, stats = _verify_candidate_synchronously(db, cleaned)
        log.info(
            "pipeline best_answer qid=%s tier=%d verify=%s decision=%s "
            "p_e=%.3f p_c=%.3f n_spans=%d",
            db.question_id,
            source_tier,
            verified,
            stats.get("decision"),
            float(stats.get("p_entail_max", 0.0)),
            float(stats.get("p_contra_max", 0.0)),
            int(stats.get("n_spans", 0)),
        )
        if verified:
            return cleaned, source_tier

    return None, 0


def _pick_citation_pid(db: PipelineState) -> str:
    """First accepted pid by score desc → any non-rejected real pid →
    any real pid at all (top-scored) → REFUSAL_PID.

    The final "any real pid" tier ensures that when the agent retrieved
    passages but never committed evidence, the submission still cites
    something retrievable rather than the synthetic refusal sentinel.
    """
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID

    accepted = [
        p for p in db.passages.values() if p.status in ("accepted", "counter_evidence")
    ]
    if accepted:
        accepted.sort(key=lambda p: p.score or 0.0, reverse=True)
        return accepted[0].passage_id
    real_unrejected = [
        p
        for p in db.passages.values()
        if p.passage_id != _REFUSAL_PID and p.status != "rejected"
    ]
    if real_unrejected:
        real_unrejected.sort(key=lambda p: p.score or 0.0, reverse=True)
        return real_unrejected[0].passage_id
    any_real = [p for p in db.passages.values() if p.passage_id != _REFUSAL_PID]
    if any_real:
        any_real.sort(key=lambda p: p.score or 0.0, reverse=True)
        return any_real[0].passage_id
    return _REFUSAL_PID


def _try_answer_search_support_for_guess(
    pipeline_db: PipelineState,
    guess: str,
    mcp_client: Any,
    origin: str,
) -> tuple[bool, dict[str, Any]]:
    """Secondary verification attempt for a tier-4 educated guess.

    Called after ``_verify_guess_with_retrieval`` returns not-verified.
    Runs ``answer_search_support`` against the guess — that procedure
    reformulates the query up to 3 times and, on a strong NLI hit,
    auto-commits a verified_supporting nugget on the retrieved passage.

    Returns ``(verified, details)`` where ``verified`` is True iff
    ``answer_search_support`` produced at least one auto-committed
    nugget or a "verified_weak" decision (both imply the corpus
    endorses the guess).

    Never raises — errors become ``verified=False`` with a diagnostic
    ``details['error']``.
    """
    aux: dict[str, Any] = {
        "attempted": True,
        "origin": origin,
        "verified": False,
        "auto_nuggets": 0,
        "soft_injects": 0,
        "iterations": 0,
        "last_decision": "",
        "p_entail_max": 0.0,
        "p_contra_max": 0.0,
        "error": None,
    }
    if mcp_client is None:
        aux["error"] = "no_mcp_client"
        return False, aux

    verify_timeout = _verify_outer_timeout_s() * ANSWER_SEARCH_SUPPORT_MAX_ITERS
    try:
        verified, details = _run_coro_sync(
            answer_search_support(pipeline_db, guess, mcp_client, origin=origin),
            timeout=verify_timeout,
        )
    except TimeoutError:
        aux["error"] = "path3_timeout"
        return False, aux
    except Exception as exc:  # noqa: BLE001
        aux["error"] = f"path3_exception:{type(exc).__name__}"
        log.exception("path3 answer_search_support raised")
        return False, aux

    aux["verified"] = bool(verified)
    aux["auto_nuggets"] = len(details.get("auto_nuggets", []))
    aux["soft_injects"] = len(details.get("soft_injects", []))
    aux["iterations"] = int(details.get("iterations", 0))
    aux["last_decision"] = str(details.get("last_decision", ""))
    aux["p_entail_max"] = float(details.get("p_entail_max", 0.0))
    aux["p_contra_max"] = float(details.get("p_contra_max", 0.0))
    # ``answer_search_support`` returns verified=True even on the
    # "verified_weak" branch (moderate NLI, no span ≥ 0.85, so no
    # auto-commit). Treating that as verified for tier emission would let
    # the Path-3 rescue emit without a verified_supporting nugget backing
    # the citation pid. As in the Path-2 re-check, accept only when an
    # auto-commit landed a nugget on the DB.
    verified_strong = bool(verified) and bool(details.get("auto_nuggets"))
    log.info(
        "pipeline %s qid=%s path3 verified=%s auto_nuggets=%d "
        "soft_injects=%d iters=%d last=%s",
        origin,
        pipeline_db.question_id,
        verified_strong,
        aux["auto_nuggets"],
        aux["soft_injects"],
        aux["iterations"],
        aux["last_decision"],
    )
    return verified_strong, aux


def _path4_reduced_trust_ok(
    pipeline_db: PipelineState,
    p4_aux: dict[str, Any],
) -> bool:
    """Whether the prior answer is a reduced-trust emit candidate.

    True iff: prior_rollout_answer is set, prior_rollout_confidence
    already cleared PRIOR_ROLLOUT_MIN_CONFIDENCE (70), and Path 4 ASS
    failed for infrastructure reasons (timeout / budget / exception),
    not because the corpus refuted it.

    path4_exception:* errors (raised when the ASS coroutine fails
    mid-flight) are infrastructure failures like the others: the prior
    rollout's vetted answer is still emittable at the reduced-trust band
    instead of dropping to canonical refusal.
    """
    if not pipeline_db.prior_rollout_answer:
        return False
    if pipeline_db.prior_rollout_confidence < PRIOR_ROLLOUT_MIN_CONFIDENCE:
        return False
    err = p4_aux.get("error") or ""
    return err in {"path4_timeout", "no_path4_budget"} or err.startswith(
        "path4_exception:"
    )


def _try_path4_prior_rollout_seed(
    pipeline_db: PipelineState,
    mcp_client: Any,
) -> tuple[bool, dict[str, Any]]:
    """Path 4: verify a prior-rollout answer through answer_search_support.

    Called from the ``force_deliver`` ladder between tier-4 and tier-5.
    ``db.prior_rollout_answer`` is seeded by the per-iter worker when
    the ensemble's earlier rollouts have already committed an answer
    for this qid at conf >= PRIOR_ROLLOUT_MIN_CONFIDENCE.

    Returns ``(verified, aux)`` where verified is True iff
    ``answer_search_support`` produced at least one auto-committed
    nugget (the prior answer now has corpus backing in this rollout's
    evidence surface too).

    Never raises. Short-circuits on missing prior answer, missing MCP
    client, or exhausted budget.
    """
    aux: dict[str, Any] = {
        "attempted": False,
        "seed_answer": (pipeline_db.prior_rollout_answer or "")[:120],
        "seed_confidence": pipeline_db.prior_rollout_confidence,
        "source_iter": pipeline_db.prior_rollout_source_iter,
        "verified": False,
        "error": None,
    }
    if not pipeline_db.prior_rollout_answer:
        aux["error"] = "no_prior_answer"
        log.info(
            "pipeline path4 qid=%s aborted reason=no_prior_answer",
            pipeline_db.question_id,
        )
        return False, aux
    aux["attempted"] = True
    if mcp_client is None:
        aux["error"] = "no_mcp_client"
        log.info(
            "pipeline path4 qid=%s aborted reason=no_mcp_client",
            pipeline_db.question_id,
        )
        return False, aux

    verify_timeout = _verify_outer_timeout_s() * ANSWER_SEARCH_SUPPORT_MAX_ITERS
    try:
        verified, details = _run_coro_sync(
            answer_search_support(
                pipeline_db,
                pipeline_db.prior_rollout_answer,
                mcp_client,
                origin="path4",
            ),
            timeout=verify_timeout,
        )
    except TimeoutError:
        aux["error"] = "path4_timeout"
        log.info(
            "pipeline path4 qid=%s aborted reason=path4_timeout seed_conf=%d iters=%d",
            pipeline_db.question_id,
            pipeline_db.prior_rollout_confidence,
            ANSWER_SEARCH_SUPPORT_MAX_ITERS,
        )
        return False, aux
    except Exception as exc:  # noqa: BLE001
        aux["error"] = f"path4_exception:{type(exc).__name__}"
        log.exception("path4 answer_search_support raised")
        log.info(
            "pipeline path4 qid=%s aborted reason=%s",
            pipeline_db.question_id,
            aux["error"],
        )
        return False, aux

    aux["verified"] = bool(verified)
    aux["auto_nuggets"] = len(details.get("auto_nuggets", []))
    aux["iterations"] = int(details.get("iterations", 0))
    aux["last_decision"] = str(details.get("last_decision", ""))
    # Require auto_nuggets > 0 for a rescue — the prior answer is
    # already someone else's best guess; we only emit when the corpus
    # explicitly backs it in this rollout's retrieval.
    rescued = bool(verified) and aux["auto_nuggets"] > 0
    log.info(
        "pipeline path4 qid=%s rescued=%s auto_nuggets=%d iters=%d last=%s",
        pipeline_db.question_id,
        rescued,
        aux["auto_nuggets"],
        aux["iterations"],
        aux["last_decision"],
    )
    return rescued, aux


def _refusal_rescue_candidate(pipeline_db: PipelineState) -> str | None:
    """Pick a candidate answer for the refused-with-nuggets rescue path.

    When the model calls ``submit_answer(refusal=true)`` after recording
    real verified_supporting nuggets,
    ``force_deliver(reason="model_requested_refusal")`` on its own would
    drop straight to canonical_refusal without consulting those nuggets.
    This helper extracts the best candidate answer from
    ``db.answer.draft_answer`` or earlier non-refusal
    ``submit_history`` entries so the caller can pass it as
    ``preferred_answer`` to ``force_deliver``. The downstream
    ``_best_non_empty_answer`` then runs the T1→T2 NLI gate over the
    candidate; if verification fails, the ladder still falls through
    to canonical_refusal.

    Rescue is attempted unconditionally; the downstream
    ``_best_non_empty_answer`` NLI gate is the sole safety check.

    Returns:
        Candidate answer string (already stripped, ≤2000 chars), or
        ``None`` when no candidate is available. Never returns
        ``CANONICAL_REFUSAL_ANSWER`` (that would be self-defeating —
        the rescue must propose a real answer, not the refusal
        itself).
    """
    # Source 1: db.answer.draft_answer, only if non-empty and not the
    # canonical refusal sentinel (refusal-shaped drafts must not be
    # promoted as a rescue candidate — they would just round-trip
    # through the refusal verifier).
    draft = ""
    if pipeline_db.answer is not None:
        draft = (pipeline_db.answer.draft_answer or "").strip()
    if (
        draft
        and draft != CANONICAL_REFUSAL_ANSWER
        and not _looks_like_canonical_refusal(draft)
    ):
        return draft[:2000]
    # Source 2: most-recent non-refusal submit_history entry whose
    # answer field is non-empty. Walk in reverse so the latest
    # candidate (closest to the refusal) wins.
    history = getattr(pipeline_db, "submit_history", None) or []
    for entry in reversed(history):
        if entry.get("refusal"):
            continue
        cand = str(entry.get("answer") or "").strip()
        if not cand:
            continue
        if cand == CANONICAL_REFUSAL_ANSWER or _looks_like_canonical_refusal(cand):
            continue
        return cand[:2000]
    return None


def _pick_best_submit_history_answer(
    pipeline_db: PipelineState,
) -> tuple[str | None, float, str]:
    """Final fallback: pick the highest-confidence entry
    from ``db.submit_history`` when every other path has failed.

    Filters to non-refusal answer commits / hardgate-rejected attempts
    (any entry with a non-empty ``answer`` field). Refusal entries
    never win — those are already the "give up" signal.

    Returns ``(answer, confidence, source_tag)`` or
    ``(None, 0.0, "empty_history")`` when nothing is eligible. The
    source_tag encodes the outcome tag for telemetry.
    """
    if not pipeline_db.submit_history:
        return None, 0.0, "empty_history"
    best: tuple[float, int, dict[str, Any]] | None = None
    for idx, entry in enumerate(pipeline_db.submit_history):
        if entry.get("refusal"):
            continue
        ans = str(entry.get("answer") or "").strip()
        if not ans:
            continue
        conf = float(entry.get("confidence") or 0.0)
        # Format-rejected entries default to ``confidence=0.0`` and would
        # emit at ``min(0.0, 0.15) = 0.0``, strictly worse than
        # canonical_refusal's 0.05 floor. Skip them so the caller falls
        # through to ``canonical_refusal`` instead.
        if conf <= 0.0:
            continue
        # Tie-break on first entry wins (earliest is more independent).
        if best is None or conf > best[0]:
            best = (conf, idx, entry)
    if best is None:
        return None, 0.0, "no_eligible_entry"
    conf, idx, entry = best
    raw = str(entry.get("answer") or "")
    qtype = pipeline_db.question_type or _question_shape_classifier(pipeline_db.question or "")
    emit, fmt_reason = _coerce_answer_for_emit(
        raw, qtype, site="submit_history_fallback"
    )
    tag = f"submit_history[{idx}]:{entry.get('outcome') or '?'}"
    if fmt_reason:
        tag = f"{tag}:format_{fmt_reason}"
    return emit, conf, tag


def _try_tier4_fresh_session_guess(
    pipeline_db: PipelineState,
    llm_client: Any,
    mcp_client: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Attempt the tier-4 closed-book educated guess and its NLI
    verification. Safety contract: never raises;
    every failure path writes telemetry and returns ``(None, details)``.

    Tier-4 does not require retrieved passages: the
    prompt is closed-book. NLI verification still requires an
    MCP client and a non-empty search_passages response; when either
    is missing, tier-4 cannot verify and falls through.
    """
    details: dict[str, Any] = {
        "attempted": False,
        "raw": "",
        "shape_pass": False,
        "verified": False,
        "p_entail": 0.0,
        "p_contra": 0.0,
        "n_spans": 0,
        "error": None,
    }
    if llm_client is None:
        details["error"] = "no_llm_client"
        log.info("pipeline tier4 qid=%s skip reason=no_llm_client", pipeline_db.question_id)
        return None, details
    if not _finalizer_budget_allows(pipeline_db, needs_llm=True):
        details["error"] = "no_budget_llm"
        log.info("pipeline tier4 qid=%s skip reason=no_budget_llm", pipeline_db.question_id)
        return None, details

    details["attempted"] = True
    pipeline_db.finalizer_tiers_attempted.append(4)
    pipeline_db.finalizer_llm_calls_used += 1
    log.info("pipeline tier4 qid=%s attempt fresh-session LLM call", pipeline_db.question_id)

    try:
        _raw, stats = _run_coro_sync(
            _make_educated_guess(pipeline_db, llm_client),
            timeout=FINALIZER_LLM_TIMEOUT_S,
        )
    except TimeoutError:
        details["error"] = "llm_timeout"
        return None, details
    except Exception as exc:  # noqa: BLE001
        details["error"] = f"llm_exception:{type(exc).__name__}"
        log.exception("tier-4 LLM call raised")
        return None, details

    inline_err = stats.get("error") if isinstance(stats, dict) else None
    raw = _raw or ""
    pipeline_db.tier4_guess_raw = raw[:200]
    cleaned = _strip_finalizer_prefixes(raw)
    details["raw"] = cleaned[:80]
    if inline_err:
        details["error"] = f"llm_inline:{inline_err}"
        log.info(
            "pipeline tier4 qid=%s abort reason=llm_inline:%s",
            pipeline_db.question_id,
            inline_err,
        )
        return None, details
    if not cleaned or cleaned.upper() == "UNKNOWN":
        details["error"] = "empty_or_unknown"
        log.info(
            "pipeline tier4 qid=%s abort reason=empty_or_unknown raw=%r",
            pipeline_db.question_id,
            raw[:80],
        )
        return None, details
    if _looks_like_canonical_refusal(cleaned):
        details["error"] = "refusal_like_guess"
        log.info(
            "pipeline tier4 qid=%s abort reason=refusal_like_guess raw=%r",
            pipeline_db.question_id,
            raw[:80],
        )
        return None, details

    qtype = pipeline_db.question_type or _question_shape_classifier(pipeline_db.question or "")
    try:
        shape_ok = _answer_shape_check(cleaned, qtype, pipeline_db.question or "")
    except Exception:  # noqa: BLE001
        shape_ok = False
    pipeline_db.tier4_guess_shape_pass = bool(shape_ok)
    details["shape_pass"] = bool(shape_ok)
    log.info(
        "pipeline tier4 qid=%s shape_check qtype=%s guess=%r pass=%s",
        pipeline_db.question_id,
        qtype,
        cleaned[:60],
        bool(shape_ok),
    )
    if not shape_ok:
        details["error"] = "shape_fail"
        return None, details

    if mcp_client is None:
        details["error"] = "no_mcp_client"
        log.info("pipeline tier4 qid=%s abort reason=no_mcp_client", pipeline_db.question_id)
        return None, details
    if not _finalizer_budget_allows(pipeline_db, needs_search=True):
        details["error"] = "no_budget_search"
        log.info(
            "pipeline tier4 qid=%s abort reason=no_budget_search",
            pipeline_db.question_id,
        )
        return None, details

    pipeline_db.finalizer_search_calls_used += 1
    log.info(
        "pipeline tier4 qid=%s enter verification for guess=%r",
        pipeline_db.question_id,
        cleaned[:60],
    )
    verify_timeout = _verify_outer_timeout_s()
    details["guess"] = cleaned
    try:
        verified, vstats = _run_coro_sync(
            _verify_guess_with_retrieval(pipeline_db, cleaned, mcp_client),
            timeout=verify_timeout,
        )
    except TimeoutError:
        details["error"] = "verify_timeout"
        return None, details
    except Exception as exc:  # noqa: BLE001
        details["error"] = f"verify_exception:{type(exc).__name__}"
        log.exception("tier-4 verify raised")
        return None, details

    pipeline_db.tier4_support_p_entail = float(vstats.get("p_entail_max", 0.0))
    pipeline_db.tier4_support_p_contra = float(vstats.get("p_contra_max", 0.0))
    pipeline_db.tier4_verified = bool(verified)
    details["p_entail"] = pipeline_db.tier4_support_p_entail
    details["p_contra"] = pipeline_db.tier4_support_p_contra
    details["n_spans"] = int(vstats.get("n_spans", 0))
    details["verified"] = bool(verified)
    details["guess"] = cleaned
    details["decision"] = vstats.get("decision") if isinstance(vstats, dict) else None
    log.info(
        "pipeline tier4 qid=%s exit verified=%s guess=%r p_e=%.2f p_c=%.2f",
        pipeline_db.question_id,
        bool(verified),
        cleaned[:60],
        pipeline_db.tier4_support_p_entail,
        pipeline_db.tier4_support_p_contra,
    )
    if verified:
        # Emit the verified guess directly at the flat tier band; there is
        # no second-stage recursive commit.
        emit, fmt_reason = _coerce_answer_for_emit(
            cleaned[:200], qtype, site="tier4_verified"
        )
        if fmt_reason:
            details["format_reason"] = fmt_reason
        return emit, details

    # Path 3: primary verify failed, so try answer_search_support.
    # Skip only on "refuted" (strong NLI contradiction). Other primary
    # errors (retrieval_error, no_spans, transport_flake, uncertain,
    # not_verified) allow the ASS reformulations to retry.
    primary_decision = str(vstats.get("decision") or "")
    if primary_decision == "refuted":
        log.info(
            "pipeline tier4 qid=%s skip path3 reason=primary_refuted",
            pipeline_db.question_id,
        )
        details["error"] = primary_decision
        return None, details
    path3_verified, path3_aux = _try_answer_search_support_for_guess(
        pipeline_db, cleaned, mcp_client, origin="path3_tier4"
    )
    details["path3"] = path3_aux
    if path3_verified:
        pipeline_db.tier4_verified = True
        details["verified"] = True
        details["error"] = None
        log.info(
            "pipeline tier4 qid=%s PATH3-RESCUED guess=%r auto_nuggets=%d",
            pipeline_db.question_id,
            cleaned[:60],
            path3_aux["auto_nuggets"],
        )
        # Emit the Path-3 rescue answer directly at the flat tier band
        # (ASS_RECURSIVE_SUBMIT=0 semantics).
        emit, fmt_reason = _coerce_answer_for_emit(
            cleaned[:200], qtype, site="tier4_path3_rescue"
        )
        if fmt_reason:
            details["format_reason"] = fmt_reason
        return emit, details

    details["error"] = vstats.get("decision") or "not_verified"
    return None, details


def force_deliver(
    db: PipelineDB,
    reason: str,
    *,
    preferred_answer: str | None = None,
) -> dict[str, Any]:
    """Terminal safety net — guarantee a valid (answer, nugget, pid, conf) tuple.

    Idempotent: re-calling on an already-force-delivered DB returns the
    memoised envelope without mutating state.

    Tier map:
      1. ``preferred_answer`` arg (driver-scraped last assistant prose)
      2. ``db.answer.draft_answer`` (model's ``submit_answer.answer``)
      4. fresh-session LLM educated guess (closed-book), verified by
         ``_verify_guess_with_retrieval`` (NLI against answer-seeded
         retrieval) at confidence ``TIER4_FRESH_SESSION_VERIFIED``
      5. ``CANONICAL_REFUSAL_ANSWER`` at ``FORCE_DELIVER_TIER5_CONFIDENCE``
    There is no tier 3. Passage titles and rationale prose are not
    emitted as answers.

    Args:
        db: Pipeline DB (must be PipelineState).
        reason: Short tag identifying why force-deliver fired.
        preferred_answer: Driver-scraped assistant-message content (tier 1).

    Returns:
        ``{committed, answer, confidence, exit_reason="force_deliver",
            force_deliver_reason, chosen_answer_tier}``.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    try:
        return _force_deliver_impl(pipeline_db, reason, preferred_answer)
    except BaseException as exc:  # noqa: BLE001 — terminal safety net
        log.exception(
            "force_deliver catastrophic failure; emitting inline tier-5 envelope"
        )
        from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _RPID_CATA

        # Preserve the ladder path if the impl had progressed far enough
        # to stamp it (e.g. PATH4-REDUCED-TRUST log fired, then a
        # downstream exception). Otherwise the envelope loses the tag.
        _cata_ladder_path = (
            getattr(pipeline_db, "finalizer_ladder_path", None) or "cata_exception"
        )
        envelope = {
            "committed": True,
            "answer": CANONICAL_REFUSAL_ANSWER,
            "confidence": round(FORCE_DELIVER_TIER5_CONFIDENCE, 3),
            "exit_reason": "force_deliver",
            "force_deliver_reason": (f"{reason}:catastrophic:{type(exc).__name__}")[
                :400
            ],
            "ladder_path": _cata_ladder_path,
            "chosen_answer_tier": 5,
            "citation_pid": _RPID_CATA,
            "citation_pids": [_RPID_CATA],
            "budget": {},
            "session_status": {},
        }
        # Best-effort state flips — guarded because the DB itself may be
        # corrupt at this point.
        try:
            pipeline_db.terminated = True
        except Exception:  # noqa: BLE001
            pass
        try:
            pipeline_db.finalizer_tier_emitted = 5
        except Exception:  # noqa: BLE001
            pass
        return envelope


def _force_deliver_impl(
    pipeline_db: PipelineState,
    reason: str,
    preferred_answer: str | None,
) -> dict[str, Any]:
    """Inner force_deliver body — wrapped by ``force_deliver`` in an
    outer ``try / except BaseException`` so any unexpected failure
    still emits a tier-5 envelope."""
    # Idempotent re-entry. deepcopy so a caller who mutates the returned
    # envelope (e.g. rewrites the nested ``budget``/``session_status``
    # dicts downstream) cannot leak into the memoised store and poison
    # the 2nd call.
    import copy as _copy

    # db.terminated short-circuit. If the DB is already terminated (from
    # ``handle_submit_answer`` committing at the tier-1 hardgate clear,
    # exit_reason == "answer_submitted"), a subsequent ``force_deliver``
    # call must not re-run the ladder. A memo check keyed on
    # exit_reason == "force_deliver" alone would let an answer_submitted
    # DB fall through and run the ladder again. Return the memoised
    # force_deliver envelope when present, or a minimal echo of the
    # committed answer otherwise (Phase-1 commit).
    if pipeline_db.terminated:
        if pipeline_db.force_deliver_envelope:
            return _copy.deepcopy(pipeline_db.force_deliver_envelope)
        if pipeline_db.answer and (pipeline_db.answer.draft_answer or "").strip():
            return {
                "committed": True,
                "answer": pipeline_db.answer.draft_answer,
                "confidence": float(pipeline_db.answer.confidence or 0.0),
                "exit_reason": pipeline_db.answer.exit_reason or "answer_submitted",
                "citation_pid": None,
                "citation_pids": [],
                "ladder_path": None,
                "chosen_answer_tier": 0,
                "force_deliver_reason": f"echo_terminated:{reason}"[:400],
                "budget": _budget_envelope(pipeline_db),
                "session_status": _session_status(pipeline_db),
            }

    if (
        pipeline_db.answer
        and pipeline_db.answer.exit_reason == "force_deliver"
        and pipeline_db.force_deliver_envelope
    ):
        return _copy.deepcopy(pipeline_db.force_deliver_envelope)

    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID

    # Tier 1 / tier 2: driver-scraped prose or the model's own answer.
    picked_answer, picked_tier = _best_non_empty_answer(pipeline_db, preferred_answer)

    # Educated-guess ladder when tiers 1/2 didn't hit.
    tier4_details: dict[str, Any] = {}
    ladder_path: str | None = None  # PATH1..PATH5 / CANONICAL_REFUSAL — envelope tag
    # The final-fallback confidence lives at function scope so the
    # confidence map (below) can thread it into the envelope.
    # None when the fallback branch didn't fire.
    final_fallback_conf: float | None = None
    if picked_tier == 0:
        llm_client, mcp_client = _get_finalizer_clients()
        log.info(
            "pipeline ladder qid=%s entry llm_client=%s mcp_client=%s "
            "submit_history=%d prior_rollout_answer=%s",
            pipeline_db.question_id,
            "set" if llm_client is not None else "None",
            "set" if mcp_client is not None else "None",
            len(getattr(pipeline_db, "submit_history", []) or []),
            "set" if pipeline_db.prior_rollout_answer else "None",
        )
        # The ladder goes directly to the tier-4 fresh-session guess. A
        # same-session guess tier fired 0/236 times in a 65-question,
        # 4-rollout production audit, so the ladder has none.
        if True:
            tier4_answer, tier4_details = _try_tier4_fresh_session_guess(
                pipeline_db, llm_client, mcp_client
            )
            # Same recursive short-circuit for tier-4 Path-3.
            _t4_env = (
                tier4_details.get("recursive_envelope")
                if isinstance(tier4_details, dict)
                else None
            )
            if isinstance(_t4_env, dict) and _t4_env.get("committed"):
                log.info(
                    "pipeline ladder qid=%s tier4_recursive_commit conf=%.2f",
                    pipeline_db.question_id,
                    float(_t4_env.get("confidence") or 0.0),
                )
                return _t4_env
            if tier4_answer:
                picked_answer = tier4_answer
                picked_tier = 4
                ladder_path = "path3b_tier4_verified"
            else:
                # Path 4: prior-rollout seed. If a previous
                # rollout for this qid committed a high-confidence answer
                # (threaded onto ``db.prior_rollout_answer``), verify it
                # through answer_search_support before falling to tier-5.
                p4_verified, p4_aux = _try_path4_prior_rollout_seed(pipeline_db, mcp_client)
                if p4_verified and pipeline_db.prior_rollout_answer:
                    picked_answer = pipeline_db.prior_rollout_answer
                    picked_tier = 4  # shares tier-4 confidence band
                    ladder_path = "path4_prior_rollout_seed"
                    log.info(
                        "pipeline ladder qid=%s PATH4-SEEDED answer=%r "
                        "from iter=%s conf=%d",
                        pipeline_db.question_id,
                        picked_answer[:60],
                        pipeline_db.prior_rollout_source_iter,
                        pipeline_db.prior_rollout_confidence,
                    )
                elif _path4_reduced_trust_ok(pipeline_db, p4_aux):
                    # Path 4 ASS failed for infrastructure
                    # reasons (timeout / budget) — not because the corpus
                    # refuted the answer. The prior rollout already cleared
                    # the conf>=70 bar under its own retrieval; emit the
                    # seeded answer at a "reduced-trust" band (prior_conf
                    # * 0.33 → ~conf=0.30 for a conf=91 prior). Strictly
                    # dominates canonical refusal and Path 5 floor when
                    # the prior answer is correct.
                    picked_answer = pipeline_db.prior_rollout_answer
                    picked_tier = 5  # conf computed below, not via tier map
                    ladder_path = "path4_prior_reduced_trust"
                    # Stamp immediately so an exception between here and
                    # the envelope write still preserves the tag.
                    pipeline_db.finalizer_ladder_path = ladder_path
                    log.info(
                        "pipeline ladder qid=%s PATH4-REDUCED-TRUST "
                        "answer=%r from iter=%s orig_conf=%d reason=%s",
                        pipeline_db.question_id,
                        picked_answer[:60],
                        pipeline_db.prior_rollout_source_iter,
                        pipeline_db.prior_rollout_confidence,
                        p4_aux.get("error"),
                    )
                else:
                    # Path 5: infrastructure-failed
                    # tier-4 rescue. When tier-4 produced a shape-passing
                    # guess but verify failed for infrastructure reasons
                    # (verify_timeout / verify_exception / transport flake)
                    # — not refuted and not no_spans — the guess is
                    # plausible but unverified. Emit it at the same conf
                    # band as tier-5 canonical refusal (0.05, Brier-safe
                    # floor). A plausible guess at 0.05 strictly dominates
                    # canonical refusal at 0.05 when the correct answer
                    # actually exists in the corpus.
                    t4_guess = (
                        (tier4_details.get("guess") or "").strip()
                        if tier4_details
                        else ""
                    )
                    t4_err = (
                        str(tier4_details.get("error") or "") if tier4_details else ""
                    )
                    t4_shape = (
                        bool(tier4_details.get("shape_pass"))
                        if tier4_details
                        else False
                    )
                    # Infrastructure/retrieval failures — not semantic
                    # rejections. Path 5 rescues these because the guess
                    # is plausible but we couldn't get corpus evidence.
                    # Excludes "refuted" (NLI said no), "uncertain",
                    # "not_verified", "verified" (kept as-is upstream).
                    infra_err = (
                        t4_err == "verify_timeout"
                        or t4_err.startswith("verify_exception:")
                        or t4_err == "retrieval_error"
                        or t4_err == "no_spans"
                    )
                    if t4_guess and t4_shape and infra_err:
                        # Before falling to floor conf,
                        # try answer_search_support on the guess. If ASS
                        # succeeds (auto-commits a nugget), we can promote
                        # to tier-4 verified (0.15) instead of tier-5
                        # floor (0.05). Pareto-superior: infra-rescue
                        # guess + corpus endorsement strictly dominates
                        # infra-rescue guess alone.
                        p5_ass_verified = False
                        # ASS runs even when tier-4's verify failed with
                        # retrieval_error or no_spans: it verifies a closed-book
                        # guess through up to 3 different query reformulations,
                        # and a different query may succeed where the single
                        # verify query failed. ASS has no budget cap; the
                        # recursive cascade runs as many ASS calls as the tier
                        # structure requires.
                        if mcp_client is None:
                            # Tag the skip so later analysis can tell whether ASS
                            # was even attempted.
                            pipeline_db.answer_search_support_decisions.append(
                                "path5_ass_skip_no_mcp"
                            )
                        if mcp_client is not None:
                            try:
                                p5_ass_verified, _p5_aux = _run_coro_sync(
                                    answer_search_support(
                                        pipeline_db,
                                        t4_guess,
                                        mcp_client,
                                        origin="path5_tier4_rescue",
                                    ),
                                    timeout=_verify_outer_timeout_s() * 2,
                                )
                            except TimeoutError:
                                p5_ass_verified = False
                            except Exception:  # noqa: BLE001
                                log.exception("Path 5 answer_search_support raised")
                                p5_ass_verified = False
                        if p5_ass_verified:
                            # Recursive submission cycle: with the flag
                            # on, the Path 5 ASS rescue routes through the
                            # model's tool call instead of emitting the
                            # guess flat-band. On commit, short-circuit
                            # the ladder and return the envelope.
                            if _ass_recursive_submit_enabled():
                                if (
                                    isinstance(p5_env, dict)
                                    and p5_env.get("committed")
                                    and p5_env.get("exit_reason")
                                    in ("answer_submitted", "high_confidence")
                                ):
                                    log.info(
                                        "pipeline ladder qid=%s "
                                        "path5_recursive_commit conf=%.2f",
                                        pipeline_db.question_id,
                                        float(p5_env.get("confidence") or 0.0),
                                    )
                                    return p5_env
                                # Re-entry refused → fall through to
                                # canonical_refusal (not flat-band emit).
                                picked_answer = CANONICAL_REFUSAL_ANSWER
                                picked_tier = 5
                                ladder_path = "canonical_refusal"
                            else:
                                picked_answer = t4_guess[:200]
                                picked_tier = 4
                                ladder_path = "path5_tier4_ass_rescued"
                                log.info(
                                    "pipeline ladder qid=%s PATH5-TIER4-ASS-RESCUED "
                                    "answer=%r (ASS endorsed after infra_err=%s)",
                                    pipeline_db.question_id,
                                    picked_answer[:60],
                                    t4_err,
                                )
                        else:
                            picked_answer = t4_guess[:200]
                            picked_tier = 5  # floor confidence, non-refusal text
                            ladder_path = "path5_tier4_infra_rescue"
                            log.info(
                                "pipeline ladder qid=%s PATH5-TIER4-INFRA-RESCUE "
                                "answer=%r t4_err=%s",
                                pipeline_db.question_id,
                                picked_answer[:60],
                                t4_err,
                            )
                    else:
                        # Both flag modes fall through to the
                        # max-confidence ``submit_history`` walker, so
                        # that a prior tool-submitted answer is not lost
                        # to an early refusal. The walker filters out
                        # ``conf <= 0.0`` entries, so a format-rejected
                        # zero-conf attempt cannot emit below
                        # canonical_refusal's 0.05 floor. If
                        # ``submit_history`` has no eligible entry the
                        # final ``else`` branch falls to
                        # ``canonical_refusal``.
                        # Final fallback: max-confidence submit_history entry.
                        fallback_answer, fallback_conf, fallback_meta = (
                            _pick_best_submit_history_answer(pipeline_db)
                        )
                        if fallback_answer:
                            picked_answer = fallback_answer
                            picked_tier = 4  # non-refusal final fallback
                            ladder_path = "final_fallback_submit_history"
                            # Thread fallback_conf up to the
                            # confidence map so the emitted confidence
                            # reflects the entry's recorded conf,
                            # not the flat TIER4_FRESH_SESSION_VERIFIED.
                            final_fallback_conf = fallback_conf
                            log.info(
                                "pipeline ladder qid=%s FINAL-FALLBACK answer=%r "
                                "conf=%.2f from %s",
                                pipeline_db.question_id,
                                picked_answer[:60],
                                fallback_conf,
                                fallback_meta,
                            )
                        else:
                            picked_answer = CANONICAL_REFUSAL_ANSWER
                            picked_tier = 5
                            ladder_path = "canonical_refusal"
        log.info(
            "pipeline ladder qid=%s chose tier=%d t3_err=%s t4_err=%s",
            pipeline_db.question_id,
            picked_tier,
            tier4_details.get("error") if tier4_details else "na",
        )

    # picked_answer is always a non-empty string at this point (a tier
    # returned text, or tier-5 wrote CANONICAL_REFUSAL_ANSWER).
    # Pipeline-level terminal format gate. If any upstream tier leaked a
    # string containing ``\n/\r/\t/;`` or a sentinel, substitute the
    # canonical refusal at tier-5 so the envelope that ships into the AC
    # file emitter is always format-clean. The tier-4 and ASS sites
    # validate their own emits; this is the terminal net.
    _qtype_fd = pipeline_db.question_type or _question_shape_classifier(
        pipeline_db.question or ""
    )
    _fd_emit, _fd_reason = _coerce_answer_for_emit(
        picked_answer,
        _qtype_fd,
        site=f"force_deliver:{ladder_path or f'tier{picked_tier}'}",
    )
    if _fd_reason:
        # Before downgrading to tier-5 canonical refusal, try to rescue a
        # short, well-formed candidate from the malformed
        # ``picked_answer``. The format gate fires most often on
        # ``too_long`` strings like "6: Memento (2000), ...": the model
        # answered correctly but added an explanatory tail that breaks
        # the 200-char R2C2 cap. ``_extract_first_short_token`` peels
        # off the leading short token (e.g. "6"); we then re-run
        # ``_best_non_empty_answer`` with that token as
        # ``preferred_answer`` so the universal NLI gate gets a chance
        # to verify it. If verification succeeds, emit at the tier
        # returned (1 or 2). Otherwise fall through to canonical
        # refusal.
        rescue_emit: str | None = None
        rescue_tier: int = 0
        rescue_token = _extract_first_short_token(picked_answer)
        if rescue_token and rescue_token != picked_answer:
            rescue_fmt = _validate_r2c2_answer(rescue_token, _qtype_fd)
            if rescue_fmt.ok:
                _r_ans, _r_tier = _best_non_empty_answer(pipeline_db, rescue_token)
                if _r_ans and _r_tier in (1, 2):
                    rescue_emit = _r_ans
                    rescue_tier = _r_tier
                    log.info(
                        "pipeline force_deliver qid=%s format_gate_rescue "
                        "reason=%s rescue_tier=%d rescue_answer=%r "
                        "(was: %r)",
                        pipeline_db.question_id,
                        _fd_reason,
                        rescue_tier,
                        rescue_emit[:60],
                        picked_answer[:80],
                    )
                    _emit_event(
                        "pipeline.force_deliver.format_gate_rescue",
                        qid=pipeline_db.question_id,
                        reason=_fd_reason,
                        rescue_tier=rescue_tier,
                        rescue_preview=rescue_emit[:80],
                    )
        if rescue_emit is not None:
            picked_answer = rescue_emit
            picked_tier = rescue_tier
            ladder_path = f"format_gate_rescue_tier{rescue_tier}"
        else:
            # Substituted — downgrade to tier-5 so the confidence map picks
            # FORCE_DELIVER_TIER5_CONFIDENCE below.
            picked_answer = _fd_emit
            picked_tier = 5
            ladder_path = f"canonical_refusal_format_{_fd_reason}"
    else:
        picked_answer = _fd_emit
    best_answer = picked_answer  # name used by the rest of this function
    chosen_tier = picked_tier
    pipeline_db.finalizer_tier_emitted = chosen_tier

    chosen_pid = _pick_citation_pid(pipeline_db)

    # Ensemble feed-forward: union agent-judged pids and the
    # force-delivered citation pid into the touched-set snapshot before
    # _prune_passages_to_cap drops the pid from db.passages.
    _snapshot_ensemble_touched(pipeline_db)
    if chosen_pid and chosen_pid != _REFUSAL_PID:
        pipeline_db.ensemble_touched_snapshot.add(chosen_pid)

    # Synthesise a nugget if none exist (status=verified_supporting on a
    # real pid, status=refusal on the virtual pid). ``is_synthetic=True``
    # marks it as a bookkeeping placeholder so the output adapter
    # excludes it from the emitted ``reconciled_nuggets`` list: the stub
    # serves replay/audit only and must not inflate the user-facing
    # nugget count for refused rollouts.
    has_real_nugget = any(
        n.status == "verified_supporting" and not n.is_synthetic
        for n in pipeline_db.nuggets.values()
    )
    if not has_real_nugget:
        synth_status = (
            "verified_supporting" if chosen_pid != _REFUSAL_PID else "refusal"
        )
        synth = Nugget(
            passage_id=chosen_pid,
            text=f"[auto-synthesised] {best_answer[:200]}",
            proposed_stance="supporting",
            source_step="force_deliver",
            source_tool_call_id="force_deliver",
            status=synth_status,  # type: ignore[arg-type]
            nli_label="entails",
            nli_score=0.5,
            is_synthetic=True,
        )
        pipeline_db.nuggets[synth.nugget_id] = synth
        pipeline_db.nuggets_recorded += 1
        append_event(
            pipeline_db,
            TransitionEvent(
                step_id="force_deliver_synth_nugget",
                iteration=pipeline_db.current_turn,
                actor="system",
                to_status=synth_status,
                reason=f"force-deliver synthesised nugget on pid={chosen_pid}",
            ),
        )

    # Pre-ladder tier-1/2 label, set before the confidence map so the
    # map's explicit branches can key on ``pre_ladder_tier{1,2}``.
    # Without those branches these paths would fall through to
    # ``has_real_nugget → compute_confidence`` and emit at 0.70-0.80
    # despite termination=force_deliver. The model's submit path already
    # ran and did not produce a clean ``answer_submitted`` commit;
    # force_deliver is a backstop and emits at a flat band calibrated for
    # that.
    if ladder_path is None and chosen_tier in (1, 2):
        ladder_path = f"pre_ladder_tier{chosen_tier}"

    # Confidence map:
    #   * ladder_path == path4_prior_reduced_trust → TIER5_REDUCED_TRUST (0.10)
    #     regardless of has_real_nugget (ASS rescue auto-commits a
    #     nugget but the answer is the ladder guess, not the model's
    #     own; emit at the tier band, not at compute_confidence which
    #     would overweight the local supporter).
    #   * ladder_path == path3b_tier4_verified → TIER4_FRESH_SESSION_VERIFIED
    #     (same reasoning).
    #   * ladder_path == path4_prior_rollout_seed → TIER4_FRESH_SESSION_VERIFIED
    #     (the ASS auto-committed nugget corroborates the prior
    #     rollout's answer, but the emit is still a force_deliver backstop:
    #     the model did not clear the hardgate via submit_answer this rollout.
    #     The flat band matches path3b, fresh-session corroboration.)
    #   * ladder_path == pre_ladder_tier1 → TIER4_FRESH_SESSION_VERIFIED
    #     (``_best_non_empty_answer`` NLI-verified the caller's
    #     preferred_answer, but force_deliver only runs when upstream submit
    #     failed, so it is capped at the fresh-session verified band.)
    #   * ladder_path == pre_ladder_tier2 → TIER4_FRESH_SESSION_VERIFIED
    #     (NLI-verified draft_answer, lower preference than
    #     tier-1; match the fresh-session verified band.)
    #   * real supporter present → compute_confidence
    #   * tier 4 (verified fresh-session guess) → TIER4_FRESH_SESSION_VERIFIED
    #   * tier 5 or citation is the refusal sentinel → TIER5_CONFIDENCE
    #   * tier 1/2 with no real supporter → TIER4_CONFIDENCE (the model
    #     produced a plausible string but nothing backs it up)
    # Absolute floor enforced afterward.
    if ladder_path == "path4_prior_reduced_trust":
        # Reduced-trust emit: the prior rollout's answer passed the hardgate
        # previously, but in this rollout the ASS verification hit an infra
        # failure (timeout / budget / exception) so we couldn't refresh the
        # corpus evidence. Emit flat TIER5_REDUCED_TRUST_CONFIDENCE (0.10),
        # deliberately below seed-rescue (0.15) because seed-rescue has
        # fresh corroboration and above canonical refusal (0.05) because we
        # still trust the historical prior. No formula over
        # prior_rollout_confidence: compute_confidence forces the admitted
        # domain into the 3-point set {91, 94, 97}, so any continuous
        # function collapses to a constant under any reasonable cap. Keep
        # local nuggets out of this emit — they're evidence for the model's
        # failed answer, not for the seeded prior_rollout_answer.
        confidence = TIER5_REDUCED_TRUST_CONFIDENCE
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif ladder_path == "path3b_tier4_verified":
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif ladder_path == "path4_prior_rollout_seed":
        # Path-4's ASS auto-committed a nugget on the prior rollout's
        # answer, making ``has_real_nugget=True``. Without this explicit
        # branch, the fallthrough to ``compute_confidence`` would emit at
        # 0.70-0.80 despite being a force_deliver backstop.
        # path4_prior_rollout_seed parallels path3b_tier4_verified (fresh
        # corroboration of a non-local answer), so it uses the same flat
        # band.
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif ladder_path == "pre_ladder_tier1":
        # ``_best_non_empty_answer`` NLI-verified the
        # caller-supplied ``preferred_answer``, but force_deliver runs
        # only because upstream submit_answer/try_submit did not produce
        # a clean ``answer_submitted`` commit. The verification is real
        # but the emission is a backstop — cap at the fresh-session
        # verified band (0.25), not at compute_confidence (~0.76).
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif ladder_path == "pre_ladder_tier2":
        # NLI-verified ``draft_answer`` with lower
        # preference than tier-1. Match the fresh-session verified band
        # (0.15) — same reasoning as pre_ladder_tier1, one band lower.
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif (
        ladder_path == "final_fallback_submit_history"
        and final_fallback_conf is not None
    ):
        # The final fallback emits at the submit_history entry's own
        # recorded confidence, capped at TIER4_FRESH_SESSION_VERIFIED.
        # Without the cap, a hardgate-rejected attempt with raw
        # compute_confidence = 0.58 (n_eff=1, supportability="insufficient";
        # ≈0.77 under R2C2_DECOUPLED_SCORING=1) would emit at 0.58 and
        # dominate every verified tier-4 answer under Brier scoring. A
        # failed attempt must never out-rank a corpus-backed tier emit.
        confidence = min(float(final_fallback_conf), TIER4_FRESH_SESSION_VERIFIED)
    elif ladder_path == "canonical_refusal" or (ladder_path or "").startswith(
        "canonical_refusal_format_"
    ):
        # Canonical refusal always emits at FORCE_DELIVER_TIER5_CONFIDENCE,
        # whatever real nuggets happen to be on the DB. An unrelated
        # verified_supporting nugget (e.g. from a prior record_evidence that
        # didn't survive the hardgate) would otherwise fall through to the
        # ``has_real_nugget`` branch and call ``compute_confidence``, emitting
        # the refusal string at ~0.13 instead of the 0.05 floor and breaking
        # Brier calibration on the refusal equivalence class. This clause
        # also covers the format-substituted variant
        # ``canonical_refusal_format_<reason>`` written by the terminal
        # _coerce_answer_for_emit gate.
        confidence = FORCE_DELIVER_TIER5_CONFIDENCE
    elif has_real_nugget:
        try:
            confidence = float(compute_confidence(pipeline_db))
        except Exception:  # noqa: BLE001
            confidence = 0.3
    elif chosen_tier == 3:
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif chosen_tier == 4:
        confidence = TIER4_FRESH_SESSION_VERIFIED
    elif chosen_tier == 5 or chosen_pid == _REFUSAL_PID:
        confidence = FORCE_DELIVER_TIER5_CONFIDENCE
    else:
        confidence = FORCE_DELIVER_TIER4_CONFIDENCE
    confidence = max(confidence, CONFIDENCE_FLOOR)

    # Write through db.answer.
    if pipeline_db.answer is not None:
        pipeline_db.answer.draft_answer = best_answer
        pipeline_db.answer.confidence = confidence
        existing_rationale = pipeline_db.answer.rationale or ""
        pipeline_db.answer.rationale = (
            f"[force-delivered: {reason}] " + existing_rationale
        )[:400]
        pipeline_db.answer.exit_reason = "force_deliver"  # type: ignore[assignment]

    # If we're force-delivering into the refusal sentinel, synthesise the
    # canonical refusal nugget the way the normal refusal branch does.
    if chosen_pid == _REFUSAL_PID and not any(
        n.status == "refusal" for n in pipeline_db.nuggets.values()
    ):
        try:
            synthesize_refusal(pipeline_db)
        except Exception:  # noqa: BLE001
            pass

    # Prune over-cap passages keeping the chosen pid + accepted ones.
    try:
        _prune_passages_to_cap(pipeline_db, keep={chosen_pid})
    except Exception:  # noqa: BLE001
        log.exception("force_deliver: passage pruning failed (non-fatal)")

    pipeline_db.terminated = True
    append_event(
        pipeline_db,
        TransitionEvent(
            step_id="force_deliver",
            iteration=pipeline_db.current_turn,
            actor="system",
            to_status="exit=force_deliver",
            reason=reason[:400],
        ),
    )

    # The pre_ladder_tier1/2 default is set above the confidence map
    # so those branches can key on it. This is a safety net for the
    # case where ``chosen_tier`` is tier 1/2 with no ladder_path: the
    # format gate sets ladder_path to ``canonical_refusal_format_*``
    # explicitly, so the branch is reached only if some other code
    # path picks tier 1/2 without a ladder_path, which should still
    # be labelled consistently.
    if ladder_path is None and chosen_tier in (1, 2):
        ladder_path = f"pre_ladder_tier{chosen_tier}"
    # Mirror on db so the catastrophic handler and any post-emit
    # inspection see the authoritative final value.
    pipeline_db.finalizer_ladder_path = ladder_path
    envelope: dict[str, Any] = {
        "committed": True,
        "answer": best_answer,
        "confidence": round(confidence, 3),
        "exit_reason": "force_deliver",
        "force_deliver_reason": reason,
        "ladder_path": ladder_path,
        "chosen_answer_tier": chosen_tier,
        "citation_pid": chosen_pid,
        # Plural list alongside the singular form, for consumers that
        # iterate citations. force_deliver writes one pid, so the
        # list length is 1.
        "citation_pids": [chosen_pid] if chosen_pid else [],
        "budget": _budget_envelope(pipeline_db),
        "session_status": _session_status(pipeline_db),
    }
    # deepcopy on write too — pair with the deepcopy on read above so the
    # stored envelope is independent of the caller's local mutations.
    pipeline_db.force_deliver_envelope = _copy.deepcopy(envelope)
    _emit_event(
        "pipeline.force_deliver",
        qid=pipeline_db.question_id,
        reason=reason,
        chosen_answer_tier=chosen_tier,
        citation_pid=chosen_pid,
        confidence=round(confidence, 3),
        had_real_nugget=has_real_nugget,
    )
    log.info(
        "pipeline force_deliver qid=%s tier=%d pid=%s conf=%.2f reason=%s",
        pipeline_db.question_id,
        chosen_tier,
        chosen_pid,
        confidence,
        reason[:120],
    )
    return envelope


# read + retract tools: list_nuggets and flag_nugget_bogus

_STATUS_PRIORITY: dict[str, int] = {
    "verified_supporting": 0,
    "verified_infirming": 1,
    "reflected": 2,
    "bogus": 3,
    "refusal": 4,
    "unclassified": 5,
}
_LIST_NUGGETS_MAX = 20


def handle_list_nuggets(
    db: PipelineDB,
    args: dict[str, Any],
) -> dict[str, Any]:
    """List nuggets with their reference passage. Read-only.

    Args:
        db: Pipeline DB (must be PipelineState).
        args: ``{status_in?, passage_id?, polarity?, created_this_session?}``.
            - ``status_in``: list of statuses to keep (default: all non-refusal).
            - ``passage_id``: filter to one pid.
            - ``polarity``: ``"affirming"`` or ``"infirming"``.
            - ``created_this_session``: Phase-2 only; no-op in Phase 1
              (returns a `phase_2_only` warning tag).

    Returns:
        ``{nuggets, total_matching, truncated_to, phase_2_only_warnings,
            budget, session_status}``. Ranking: ``(status_priority,
        -created_at_turn)``; truncated to 20.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    status_in = args.get("status_in")
    passage_id_filter = args.get("passage_id")
    polarity_filter = args.get("polarity")
    created_this_session = bool(args.get("created_this_session", False))

    if status_in is None:
        status_set: set[str] | None = None
    elif isinstance(status_in, list):
        status_set = {str(s).strip() for s in status_in if s}
    else:
        return {
            "error": "invalid_status_in",
            "message": "status_in must be a list of status strings or omitted",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    warnings: list[str] = []
    if created_this_session and not getattr(pipeline_db, "phase2_active", False):
        warnings.append(
            "created_this_session is Phase-2-only; Phase 1 treats it as a no-op"
        )
        created_this_session = False

    session_start_turn = int(getattr(pipeline_db, "phase2_session_start_turn", 0) or 0)

    matches: list[Nugget] = []
    for n in pipeline_db.nuggets.values():
        if status_set is None:
            if n.status == "refusal":
                continue
        elif n.status not in status_set:
            continue
        if passage_id_filter and n.passage_id != passage_id_filter:
            continue
        if polarity_filter:
            stored = getattr(n, "proposed_stance", "")
            pol = "affirming" if stored == "supporting" else "infirming"
            if pol != polarity_filter:
                continue
        if created_this_session:
            created_iter = int(getattr(n, "created_iter", 0) or 0)
            if created_iter < session_start_turn:
                continue
        matches.append(n)

    def _key(n: Nugget) -> tuple[int, int]:
        prio = _STATUS_PRIORITY.get(n.status, 99)
        turn = -int(getattr(n, "created_iter", 0) or 0)
        return (prio, turn)

    matches.sort(key=_key)
    total = len(matches)
    truncated = matches[:_LIST_NUGGETS_MAX]

    rows: list[dict[str, Any]] = []
    for n in truncated:
        passage = pipeline_db.passages.get(n.passage_id)
        title = (passage.title if passage and passage.title else "")[:80]
        stored_stance = getattr(n, "proposed_stance", "")
        polarity = "affirming" if stored_stance == "supporting" else "infirming"
        rows.append(
            {
                "nugget_id": n.nugget_id,
                "claim": n.text[:400],
                "polarity": polarity,
                "status": n.status,
                "passage_id": n.passage_id,
                "passage_title": title,
                "nli_label": getattr(n, "nli_label", None),
                "nli_score": round(float(getattr(n, "nli_score", 0.0) or 0.0), 3),
                "created_at_turn": int(getattr(n, "created_iter", 0) or 0),
                "status_reason": (getattr(n, "status_reason", "") or "")[:200],
            }
        )

    return {
        "nuggets": rows,
        "total_matching": total,
        "truncated_to": min(total, _LIST_NUGGETS_MAX),
        "phase_2_only_warnings": warnings,
        "budget": _budget_envelope(pipeline_db),
        "session_status": _session_status(pipeline_db),
    }


def handle_flag_nugget_bogus(
    db: PipelineDB,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Flip a nugget's status to ``bogus`` with an audit-trail event.

    Args:
        db: Pipeline DB (must be PipelineState).
        args: ``{nugget_id, reason}`` — ``reason`` ≤200 chars.

    Behavior:
        - Nugget stays in ``db.nuggets`` (not deleted) — audit trail preserved.
        - Emits a ``TransitionEvent(actor="llm_tool", ...)``.
        - Idempotent: re-flagging a bogus nugget returns ``error=already_bogus``.
        - Rejects: refusal sentinels (``error=refusal_immutable``) and unknown
          ids (``error=unknown_nugget_id``).
        - NLI-authoritative guard: if the nugget was committed with
          ``nli_score<0.55`` and is already status=bogus (NLI-rejected),
          return ``error=nli_authoritative`` on attempts to un-flag. This
          guard is inert here (we only flip into bogus) but kept so the
          response shape is stable across both flip directions for tests.

    Returns:
        Success: ``{committed:True, nugget_id, from_status, to_status:"bogus",
            budget, session_status}``.
        Error: ``{committed:False, error, message, budget, session_status}``.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    nugget_id = args.get("nugget_id")
    reason = args.get("reason", "")

    if not isinstance(nugget_id, str) or not nugget_id:
        return {
            "committed": False,
            "error": "invalid_nugget_id",
            "message": "nugget_id must be a non-empty string",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if not isinstance(reason, str) or not reason.strip():
        return {
            "committed": False,
            "error": "invalid_reason",
            "message": "reason must be a non-empty string (≤200 chars)",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if len(reason) > 200:
        return {
            "committed": False,
            "error": "reason_too_long",
            "message": f"reason must be ≤200 chars (got {len(reason)})",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    n = pipeline_db.nuggets.get(nugget_id)
    if n is None:
        return {
            "committed": False,
            "error": "unknown_nugget_id",
            "message": f"no nugget with id={nugget_id!r}",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if n.status == "refusal":
        return {
            "committed": False,
            "error": "refusal_immutable",
            "message": "refusal sentinel nuggets cannot be flagged bogus",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if n.status == "bogus":
        return {
            "committed": False,
            "error": "already_bogus",
            "message": f"nugget {nugget_id} is already bogus (idempotent no-op)",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    from_status = n.status
    n.status = "bogus"
    # The stored reason is truncated at the same 200-char cap the
    # validator accepts, so a valid reason is stored whole.
    _flag_reason = f"flagged_by_model: {reason[:200]}"
    n.status_reason = _flag_reason
    ev = TransitionEvent(
        step_id="m11_flag_nugget_bogus",
        iteration=pipeline_db.current_turn,
        actor="llm_tool",
        from_status=from_status,
        to_status="bogus",
        reason=_flag_reason,
    )
    append_event(pipeline_db, ev)
    n.history.append(ev)

    return {
        "committed": True,
        "nugget_id": nugget_id,
        "from_status": from_status,
        "to_status": "bogus",
        "budget": _budget_envelope(pipeline_db),
        "session_status": _session_status(pipeline_db),
    }


# batch_record_stance (Phase 2 only)
# Bulk-demote up to BATCH_RECORD_STANCE_MAX pids in a single tool call. Only
# available in Phase 2 — Phase 1 must use record_evidence per-pid (forces
# engagement with each passage's evidence). Only stance="irrelevant" is
# supported; bulk-promoting to supporting/context is unsafe because those
# stances carry nugget implications that can't be meaningfully batched.

BATCH_RECORD_STANCE_MAX: int = 20
_BATCH_RECORD_STANCE_ALLOWED_STANCES: frozenset[str] = frozenset({"irrelevant"})


def handle_batch_record_stance(
    db: PipelineDB,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Bulk-demote pids to ``stance="irrelevant"`` — Phase 2 only.

    Args:
        db: Pipeline DB (must be PipelineState).
        args: ``{pids: list[str], stance: "irrelevant", reason: str}``.
            ``pids`` holds 1..BATCH_RECORD_STANCE_MAX pids; each must
            already be in ``db.passages``. ``reason`` is ≤200 chars.

    Returns:
        Success:
            ``{committed: True, n_demoted, n_skipped,
               skipped: [{pid, reason}], budget, session_status}``.
        Error:
            ``{committed: False, error, message, budget, session_status}``.
            Errors: ``phase_2_only``, ``invalid_stance``,
            ``invalid_pids``, ``invalid_reason``, ``too_many_pids``.

    Behaviour:
        - Sets ``Passage.status = "rejected"`` for each pid.
        - Any nuggets previously attached to a demoted pid are cascaded
          to ``status="bogus"`` with ``status_reason="cascade: passage
          demoted by batch_record_stance"`` (matches the single-pid
          record_evidence cascade).
        - Emits one ``TransitionEvent`` per pid plus one batch-level
          event carrying the full pid list.
        - Idempotent per-pid: a pid already ``rejected`` is reported in
          ``skipped`` but does not abort the batch.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    if not pipeline_db.phase2_active:
        return {
            "committed": False,
            "error": "phase_2_only",
            "message": (
                "batch_record_stance is only available during a Phase 2 "
                "synthesis session. Use record_evidence per-pid in Phase 1."
            ),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    stance = args.get("stance", "irrelevant")
    if stance not in _BATCH_RECORD_STANCE_ALLOWED_STANCES:
        return {
            "committed": False,
            "error": "invalid_stance",
            "message": (
                "batch_record_stance only accepts stance='irrelevant'; "
                f"got {stance!r}. Use record_evidence for other stances."
            ),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    pids = args.get("pids")
    if not isinstance(pids, list) or not pids:
        return {
            "committed": False,
            "error": "invalid_pids",
            "message": "pids must be a non-empty list of passage_id strings",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if len(pids) > BATCH_RECORD_STANCE_MAX:
        return {
            "committed": False,
            "error": "too_many_pids",
            "message": (
                f"batch_record_stance accepts at most "
                f"{BATCH_RECORD_STANCE_MAX} pids; got {len(pids)}"
            ),
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    if not all(isinstance(p, str) and p for p in pids):
        return {
            "committed": False,
            "error": "invalid_pids",
            "message": "every pid must be a non-empty string",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }

    reason = args.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        return {
            "committed": False,
            "error": "invalid_reason",
            "message": "reason must be a non-empty string (≤200 chars)",
            "budget": _budget_envelope(pipeline_db),
            "session_status": _session_status(pipeline_db),
        }
    reason = reason[:200]

    # Deduplicate while preserving order — the model may emit the same pid
    # twice, which would otherwise inflate the skipped counter.
    seen: set[str] = set()
    unique_pids: list[str] = []
    for p in pids:
        if p in seen:
            continue
        seen.add(p)
        unique_pids.append(p)

    demoted: list[str] = []
    skipped: list[dict[str, str]] = []
    cascaded_nuggets: list[str] = []

    for pid in unique_pids:
        passage = pipeline_db.passages.get(pid)
        if passage is None:
            skipped.append({"pid": pid, "reason": "unknown_pid"})
            continue
        if passage.status == "rejected":
            skipped.append({"pid": pid, "reason": "already_rejected"})
            continue

        prev_status = passage.status
        passage.status = "rejected"  # type: ignore[assignment]
        demoted.append(pid)

        # Cascade attached nuggets to bogus — matches single-pid behaviour.
        for n in pipeline_db.nuggets.values():
            if n.passage_id != pid or n.status == "bogus":
                continue
            prev_nugget_status = n.status
            n.status = "bogus"
            n.status_reason = "cascade: passage demoted by batch_record_stance"
            cev = TransitionEvent(
                step_id="m11_batch_record_stance_cascade",
                iteration=pipeline_db.current_turn,
                actor="system",
                from_status=prev_nugget_status,
                to_status="bogus",
                reason=f"cascade from pid={pid} demotion",
            )
            append_event(pipeline_db, cev)
            n.history.append(cev)
            cascaded_nuggets.append(n.nugget_id)

        append_event(
            pipeline_db,
            TransitionEvent(
                step_id="m11_batch_record_stance",
                iteration=pipeline_db.current_turn,
                actor="llm_tool",
                from_status=prev_status,
                to_status="rejected",
                reason=f"batch_demote: {reason}",
            ),
        )

    # Batch-level audit event carrying the full demoted pid list.
    append_event(
        pipeline_db,
        TransitionEvent(
            step_id="m11_batch_record_stance_batch",
            iteration=pipeline_db.current_turn,
            actor="llm_tool",
            to_status=f"batch_demoted_{len(demoted)}",
            reason=f"batch: n_pids={len(unique_pids)} n_demoted={len(demoted)}",
        ),
    )
    _emit_event(
        "pipeline.batch_record_stance",
        qid=pipeline_db.question_id,
        n_pids=len(unique_pids),
        n_demoted=len(demoted),
        n_skipped=len(skipped),
        n_cascaded_nuggets=len(cascaded_nuggets),
        reason=reason[:120],
    )

    return {
        "committed": True,
        "stance": "irrelevant",
        "n_demoted": len(demoted),
        "n_skipped": len(skipped),
        "demoted_pids": demoted,
        "skipped": skipped,
        "cascaded_nuggets": cascaded_nuggets,
        "budget": _budget_envelope(pipeline_db),
        "session_status": _session_status(pipeline_db),
    }


# Phase 2 synthesis-session DB serialisation
# The Phase-2 recovery loop re-serialises the live DB into a single user
# message each session (fresh 131k budget, thinking=ON). Format chosen to
# keep the system prompt prefix-cacheable and the handoff payload human-
# auditable.

PHASE2_MAX_PASSAGES_INLINED = 200
PHASE2_SNIPPET_MAX_CHARS = 180
PHASE2_TITLE_MAX_CHARS = 80

_PHASE2_PASSAGE_ORDER: dict[str, int] = {
    "supporting": 0,
    "counter_evidence": 1,
    "accepted": 0,  # legacy alias
    "context": 2,
    "unclassified": 3,
    "irrelevant": 4,
    "rejected": 5,
}


def _phase2_snippet(text: str) -> str:
    """First sentence or first ``PHASE2_SNIPPET_MAX_CHARS`` chars, whichever
    shorter. Trailing ellipsis when truncated; none otherwise."""
    if not text:
        return ""
    clean = " ".join(text.split())
    # Try to cut at the first sentence boundary within the cap.
    for marker in (". ", "? ", "! "):
        idx = clean.find(marker)
        if 0 < idx < PHASE2_SNIPPET_MAX_CHARS:
            return clean[: idx + 1].rstrip()
    if len(clean) <= PHASE2_SNIPPET_MAX_CHARS:
        return clean
    return clean[:PHASE2_SNIPPET_MAX_CHARS].rstrip() + "…"


def _phase2_title(title: str | None) -> str:
    if not title:
        return ""
    return title[:PHASE2_TITLE_MAX_CHARS]


def _phase2_yaml_escape(s: str) -> str:
    """Minimal YAML-safe scalar: wrap in double quotes and escape backslashes +
    quotes. Also rewrites ``{``/``}``/backticks to non-structural equivalents
    so the LLM reading the handoff cannot confuse them with the format
    placeholders that the `<cascade_nuggets>` worked example renders."""
    if s is None:
        return '""'
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ")
    # Neutralise structural chars that would otherwise confuse the LLM:
    #   { / } : risk of being parsed as a format placeholder
    #   ` : often wraps code spans in markdown contexts
    s = s.replace("{", "(").replace("}", ")").replace("`", "'")
    return f'"{s}"'


def build_phase2_db_yaml(
    db: PipelineDB,
    *,
    session_idx: int,
    max_sessions: int,
    calls_total_phase2: int,
    tokens_remaining: int,
    prior_delta: dict[str, Any] | None = None,
) -> str:
    """Serialise the live DB into the Phase 2 handoff payload.

    The payload is a single string wrapped in a ``<phase2_state>`` XML tag
    (YAML body inside). Passages are ordered by
    ``(status_priority, -score)`` and truncated to
    ``PHASE2_MAX_PASSAGES_INLINED``; any overflow is reported in
    ``stats.n_passages_over_soft_cap``. Recorded nuggets are inlined under
    their passage when status ∈ {verified_supporting, verified_infirming,
    reflected}; rejected / bogus are surfaced in aggregate counts only.

    Args:
        db: Live pipeline DB (must be PipelineState).
        session_idx: 1-based session counter.
        max_sessions: Budget for Phase 2 sessions before force_deliver.
        calls_total_phase2: Per-session tool-call budget.
        tokens_remaining: Fresh context tokens available for this session.
        prior_delta: Optional dict with keys ``pids_demoted``,
            ``pids_promoted``, ``nuggets_added``, ``nuggets_superseded``,
            ``submit_answer_attempts``, ``submit_answer_rejection_reasons``,
            ``tokens_consumed``, ``tool_calls_used`` — rendered as a delta
            block above the main payload. ``None`` on session 1.

    Returns:
        The full ``<phase2_state>…</phase2_state>`` string, suitable for
        inclusion as a user message in the synthesis turn.
    """
    _ensure_pipeline_db(db)
    pipeline_db = db  # type: ignore[assignment]

    lines: list[str] = ["<phase2_state>"]
    lines.append(f"session: {session_idx}")
    lines.append(f"max_sessions: {max_sessions}")
    lines.append(f"calls_used_phase1: {int(pipeline_db.calls_used)}")
    lines.append(f"calls_total_phase2: {int(calls_total_phase2)}")
    lines.append(f"tokens_remaining: {int(tokens_remaining)}")
    lines.append("")

    q_text = (pipeline_db.question or "").strip()
    lines.append("question: |")
    for qline in q_text.splitlines() or [""]:
        lines.append(f"  {qline}")
    lines.append(f"question_type: {pipeline_db.question_type or 'simple'}")
    entities = extract_question_entities(q_text)
    if entities:
        lines.append(
            "question_entities: ["
            + ", ".join(_phase2_yaml_escape(e) for e in entities)
            + "]"
        )
    else:
        lines.append("question_entities: []")
    lines.append("")

    answer = pipeline_db.answer
    draft_text = (answer.draft_answer or "") if answer else ""
    draft_rationale = (answer.rationale or "") if answer else ""
    draft_reasoning = (answer.reasoning_trace or "") if answer else ""
    draft_conf = float(answer.confidence or 0.0) if answer else 0.0
    lines.append("answer_draft:")
    lines.append(f"  text: {_phase2_yaml_escape(draft_text[:800])}")
    lines.append(f"  confidence: {draft_conf:.3f}")
    lines.append(f"  rationale: {_phase2_yaml_escape(draft_rationale[:400])}")
    lines.append(f"  reasoning_trace: {_phase2_yaml_escape(draft_reasoning[:400])}")
    lines.append("")

    total_rejects = int(pipeline_db.answer_submit_retries + pipeline_db.refusal_validator_rejects)
    lines.append(f"consecutive_submit_rejects: {total_rejects}")
    lines.append("")

    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID as _REFUSAL_PID

    real_passages = [
        p for p in pipeline_db.passages.values() if p.passage_id != _REFUSAL_PID
    ]
    n_total = len(real_passages)
    n_supporting = sum(
        1 for n in pipeline_db.nuggets.values() if n.status == "verified_supporting"
    )
    n_infirming = sum(
        1 for n in pipeline_db.nuggets.values() if n.status == "verified_infirming"
    )
    n_rejected = sum(1 for n in pipeline_db.nuggets.values() if n.status in ("bogus",))
    n_over_soft = n_total > PHASE2_MAX_PASSAGES_INLINED
    lines.append("stats:")
    lines.append(f"  n_passages: {n_total}")
    lines.append(f"  n_passages_over_soft_cap: {'true' if n_over_soft else 'false'}")
    lines.append(f"  n_nuggets_verified_supporting: {n_supporting}")
    lines.append(f"  n_nuggets_verified_infirming: {n_infirming}")
    lines.append(f"  n_nuggets_rejected: {n_rejected}")
    lines.append("")

    # Prior-session delta (session 2+).
    if prior_delta:
        lines.append(f"# --- delta from session {session_idx - 1} → {session_idx} ")
        lines.append(f"session_{session_idx - 1}_actions:")
        for key in (
            "pids_demoted",
            "pids_promoted",
            "nuggets_added",
            "nuggets_superseded",
            "submit_answer_rejection_reasons",
        ):
            vals = prior_delta.get(key) or []
            if isinstance(vals, list):
                rendered = ", ".join(_phase2_yaml_escape(str(v)) for v in vals)
                lines.append(f"  {key}: [{rendered}]")
        for key in ("submit_answer_attempts", "tokens_consumed", "tool_calls_used"):
            val = prior_delta.get(key, 0)
            lines.append(f"  {key}: {int(val)}")
        lines.append("")

    # Ordered passages.
    def _rank(p: Any) -> tuple[int, float]:
        prio = _PHASE2_PASSAGE_ORDER.get(getattr(p, "status", "unclassified"), 99)
        return (prio, -float(getattr(p, "score", 0.0) or 0.0))

    ordered = sorted(real_passages, key=_rank)
    inlined = ordered[:PHASE2_MAX_PASSAGES_INLINED]

    lines.append("passages:")
    if not inlined:
        lines.append("  []")
    for p in inlined:
        pid = p.passage_id
        title = _phase2_title(getattr(p, "title", "") or "")
        stance = getattr(p, "status", "unclassified")
        score = float(getattr(p, "score", 0.0) or 0.0)
        snippet = _phase2_snippet(getattr(p, "text", "") or "")
        lines.append(f"  - pid: {_phase2_yaml_escape(pid)}")
        lines.append(f"    title: {_phase2_yaml_escape(title)}")
        lines.append(f"    stance: {stance}")
        lines.append(f"    score: {score:.3f}")
        lines.append(f"    snippet: {_phase2_yaml_escape(snippet)}")
        attached = [
            n
            for n in pipeline_db.nuggets.values()
            if n.passage_id == pid
            and n.status in ("verified_supporting", "verified_infirming", "reflected")
        ]
        if not attached:
            lines.append("    nuggets: []")
        else:
            lines.append("    nuggets:")
            for n in attached:
                polarity = (
                    "affirming"
                    if getattr(n, "proposed_stance", "") == "supporting"
                    else "infirming"
                )
                label = getattr(n, "nli_label", "") or ""
                score_val = float(getattr(n, "nli_score", 0.0) or 0.0)
                lines.append(f"      - nid: {_phase2_yaml_escape(n.nugget_id)}")
                lines.append(f"        claim: {_phase2_yaml_escape(n.text[:400])}")
                lines.append(f"        polarity: {polarity}")
                lines.append(f"        status: {n.status}")
                lines.append(
                    f"        nli: {{label: {label or 'none'}, score: {score_val:.3f}}}"
                )
    lines.append("")

    # Cross-ref index for verified_supporting nuggets only.
    lines.append("nuggets_verified_supporting:")
    verified = [n for n in pipeline_db.nuggets.values() if n.status == "verified_supporting"]
    if not verified:
        lines.append("  {}")
    else:
        for n in verified:
            nid = _phase2_yaml_escape(n.nugget_id)
            pid = _phase2_yaml_escape(n.passage_id)
            lines.append(f"  {nid}: {pid}")
    lines.append(f"nuggets_bogus_count: {n_rejected}")
    lines.append("</phase2_state>")
    return "\n".join(lines)
