"""Phase 2 synthesis-session recovery loop for the unified pipeline.

Fires after Phase 1 (``answer_question_pipeline``) exhausts its budget without
successfully committing an answer. Each Phase 2 session starts with a
fresh 131 k-token window, thinking=ON, a filter-only tool subset (no
``search_passages``/``search_entities``), and the live DB serialised into
a single USER message via :func:`build_phase2_db_yaml`.

Up to ``PHASE2_MAX_SESSIONS`` (default 3) sessions may run; the loop
exits early via either:

* ``submit_answer`` success (``db.terminated`` flips): normal commit.
* stuck-detection circuit breaker (no DB delta on the session): route
  to :func:`force_deliver`.
* wall-clock cap (``PHASE2_MAX_TOTAL_SECONDS``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from r2c2_agent.agent.config import LoopConfig
from r2c2_agent.agent.interfaces.llm import LLMClientProtocol
from r2c2_agent.agent.interfaces.mcp import MCPClientProtocol
from r2c2_agent.agent.interfaces.types import Question, QuestionTrace
from r2c2_agent.agent.prompt import (
    build_synthesis_system_prompt,
    build_synthesis_user_message,
)

__all__ = [
    "PHASE2_MAX_SESSIONS",
    "PHASE2_MAX_TOTAL_SECONDS",
    "PHASE2_TOKEN_BUDGET_WATERMARK",
    "PHASE2_TOKEN_BUDGET_CEILING",
    "PHASE2_HARD_ITER_CEILING",
    "PHASE2_LOOP_CONFIG",
    "at_token_budget_watermark",
    "PHASE2_TOOL_NAMES",
    "PhaseSnapshot",
    "SessionResult",
    "db_adequate_for_phase2",
    "decide_terminal_route",
    "filter_tools_for_phase2",
    "run_phase2_loop",
    "run_synthesis_session",
]

log = logging.getLogger(__name__)

# Constants

PHASE2_MAX_SESSIONS: int = 3
PHASE2_MAX_TOTAL_SECONDS: float = 900.0  # 15 min hard cap across all sessions

# Token-budget watermark: routes a qid to Phase 2 once the agent has
# spent this fraction of its call budget without committing. It adds to
# the retry-based triggers, so a qid can reach Phase 2 through either
# the retry path or this watermark.
#
# PHASE2_TOKEN_BUDGET_WATERMARK is the enter threshold (0.85 = 85% of
# MAX_CALLS_TOTAL used without a committed answer). PHASE2_TOKEN_BUDGET_
# CEILING is the inner-loop exit (0.95 of the LoopConfig.token_budget).
# PHASE2_HARD_ITER_CEILING is a second, independent cap on iterations.
PHASE2_TOKEN_BUDGET_WATERMARK: float = 0.85
PHASE2_TOKEN_BUDGET_CEILING: float = 0.95
PHASE2_HARD_ITER_CEILING: int = 500

# Tool subset exposed in Phase 2. Drops search_passages / search_entities
# (retrieval-free phase); keeps read + revise + commit surfaces.
PHASE2_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_passage",
        "extract_from_passage",
        "record_evidence",
        "list_nuggets",
        "flag_nugget_bogus",
        "batch_record_stance",
        "submit_answer",
    }
)

PHASE2_LOOP_CONFIG: LoopConfig = LoopConfig(
    max_iterations=20,
    context_budget=118_000,
    token_budget=125_000,
    tool_timeout=900.0,
    synthesis_timeout=60.0,
    concurrency=1,
)

# DB-adequacy thresholds. Below these we route directly to
# force_deliver rather than burn a 131 k budget on an empty DB.
_DB_ADEQUATE_MIN_REAL_PASSAGES: int = 5
_DB_ADEQUATE_MIN_SEARCHES: int = 1


# Stuck detection


@dataclass(frozen=True)
class PhaseSnapshot:
    """Immutable DB fingerprint used to detect session progress.

    Captured before and after each synthesis session; a session that
    produces an identical snapshot did literally nothing (no stance flips,
    no nuggets, no submit attempts) and trips the circuit breaker.
    """

    nuggets_recorded: int
    pid_statuses: tuple[tuple[str, str], ...]
    answer_submit_attempts: int

    @classmethod
    def from_db(cls, db: Any) -> PhaseSnapshot:
        return cls(
            nuggets_recorded=int(getattr(db, "nuggets_recorded", 0) or 0),
            pid_statuses=tuple(
                sorted(
                    (p.passage_id, getattr(p, "status", "unclassified"))
                    for p in db.passages.values()
                )
            ),
            answer_submit_attempts=int(
                getattr(db, "answer_submit_retries", 0)
                + getattr(db, "refusal_validator_rejects", 0)
            ),
        )


def session_made_progress(before: PhaseSnapshot, after: PhaseSnapshot) -> bool:
    """Return ``True`` iff the session produced any DB delta.

    A session that attempted (and failed) ``submit_answer`` still
    counts: the retry counter moved, so the next session sees richer
    state. Only a session with no side-effectful tool call at all is
    flagged as stuck.
    """
    return (
        after.nuggets_recorded != before.nuggets_recorded
        or after.pid_statuses != before.pid_statuses
        or after.answer_submit_attempts > before.answer_submit_attempts
    )


# Trigger gate


def db_adequate_for_phase2(db: Any) -> bool:
    """True iff the DB holds enough real state for a filter-only session."""
    from r2c2_agent.orchestrator.nugget_reconciler import REFUSAL_PID
    from r2c2_agent.orchestrator.pipeline_handlers import (  # local import avoids cycles
        PipelineState,  # noqa: F401
    )

    real_passages = [p for p in db.passages.values() if p.passage_id != REFUSAL_PID]
    n_searches = len(getattr(db, "search_queries", []) or [])
    return (
        len(real_passages) >= _DB_ADEQUATE_MIN_REAL_PASSAGES
        and n_searches >= _DB_ADEQUATE_MIN_SEARCHES
    )


def at_token_budget_watermark(
    db: Any, *, watermark: float = PHASE2_TOKEN_BUDGET_WATERMARK
) -> bool:
    """True iff the watermark should fire: agent has consumed ≥
    ``watermark`` of its call budget and the cheap retry path is also
    exhausted (``answer_submit_retries >= SUBMIT_RETRY_CAP``).

    The retry-exhaustion gate keeps the cheap path first: retries run
    until the model's self-correction budget is spent before a fresh
    131K-context Phase 2 session is started. If the model never submits,
    retries stay at 0 and the watermark never fires; the global
    ``calls_total`` hard cap is the backstop in that case.
    """
    calls_total = int(getattr(db, "calls_total", 0) or 0)
    if calls_total <= 0:
        return False
    calls_used = int(getattr(db, "calls_used", 0) or 0)
    terminated = bool(getattr(db, "terminated", False))
    if terminated or calls_used < watermark * calls_total:
        return False
    # Gate on retry exhaustion. Local import avoids a circular import
    # between loop_phase2 and pipeline_handlers at module-load time.
    from r2c2_agent.orchestrator.pipeline_handlers import SUBMIT_RETRY_CAP

    retries = int(getattr(db, "answer_submit_retries", 0) or 0)
    return retries >= SUBMIT_RETRY_CAP


def decide_terminal_route(db: Any, reason: str) -> str:
    """Decide whether a terminal Phase-1 branch routes to Phase 2 or force_deliver.

    Primary triggers (``answer_submit_retries>=3``,
    ``refusal_validator_rejected``) are unconditional: the model has
    already tried to commit, so by definition the DB is adequate.
    Conditional triggers must pass :func:`db_adequate_for_phase2`.

    The ``token_budget_watermark`` reason is also unconditional: the
    loop fires Phase 2 when the agent has consumed the watermark
    fraction of its call budget without committing.

    Args:
        db: The live PipelineState.
        reason: Short tag identifying the terminal branch that fired.

    Returns:
        ``"phase2"`` or ``"force_deliver"``.
    """
    reason_lc = (reason or "").lower()
    if (
        "refusal_validator_rejected" in reason_lc
        or "retries>=3" in reason_lc
        or "answer_submit_retries" in reason_lc
        or "token_budget_watermark" in reason_lc
    ):
        return "phase2"
    if db_adequate_for_phase2(db):
        return "phase2"
    return "force_deliver"


def filter_tools_for_phase2(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the tools allowed in Phase 2 (no retrieval tools)."""
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        func = tool.get("function", tool)
        name = func.get("name", "")
        if name in PHASE2_TOOL_NAMES:
            out.append(tool)
    return out


# Session result


@dataclass
class SessionResult:
    """Outcome of one Phase 2 synthesis session."""

    committed: bool
    made_progress: bool
    iterations: int
    wall_seconds: float
    terminated_by: (
        str  # "commit" | "no_tool_call" | "max_iter" | "token_budget" | "error"
    )
    session_messages: list[dict[str, Any]] = field(default_factory=list)


# Per-session inner loop


def _count_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap char/3.5 estimator, the same heuristic as loop.count_tokens."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        reasoning = msg.get("reasoning", "")
        if isinstance(reasoning, str):
            total_chars += len(reasoning)
    return int(total_chars / 3.5)


def _build_assistant_message(response: Any) -> dict[str, Any]:
    """Minimal reconstruction of an assistant message from LLMResponse."""
    msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    reasoning = getattr(response, "reasoning", "") or ""
    if reasoning:
        msg["reasoning"] = reasoning
    if response.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments or {}, ensure_ascii=False),
                },
            }
            for tc in response.tool_calls
        ]
    return msg


async def run_synthesis_session(
    db: Any,
    question: Question,
    *,
    session_idx: int,
    max_sessions: int,
    llm_client: LLMClientProtocol,
    mcp_client: MCPClientProtocol,
    tools: list[dict[str, Any]],
    config: LoopConfig,
    trace: QuestionTrace,
) -> SessionResult:
    """Run one Phase 2 synthesis session end-to-end.

    Mirrors the Phase 1 ReAct inner loop but with a filter-only tool
    catalogue, a fresh ``[system, user]`` message pair, and relaxed
    termination: the inner loop never calls ``force_deliver`` itself,
    the outer :func:`run_phase2_loop` decides whether to run another
    session or escalate.
    """
    # Lazy imports keep the module importable without the orchestrator
    # on sys.path (worker-side wiring lives next to this file).
    from r2c2_agent.orchestrator.pipeline_handlers import build_phase2_db_yaml

    from r2c2_agent.agent.loop import _dispatch_pipeline_tool
    from r2c2_agent.agent.obs import emit_event

    qid = db.question_id
    db.phase2_active = True
    db.phase2_session_start_turn = int(db.current_turn)

    # Serialise the DB into the handoff payload. tokens_remaining is the
    # session's *fresh* budget, not Phase 1 residual.
    yaml_payload = build_phase2_db_yaml(
        db,
        session_idx=session_idx,
        max_sessions=max_sessions,
        calls_total_phase2=int(config.max_iterations),
        tokens_remaining=int(config.token_budget),
    )
    system_prompt = build_synthesis_system_prompt(
        tools, session_index=session_idx, max_sessions=max_sessions
    )
    user_msg = build_synthesis_user_message(
        yaml_payload,
        question.text,
        question.question_type if question.question_type != "default" else None,
        session_index=session_idx,
        max_sessions=max_sessions,
        calls_remaining=int(config.max_iterations),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    iteration = 0
    terminated_by = "max_iter"
    t_start = time.monotonic()

    emit_event(
        "pipeline.phase2.session.turn0",
        qid=qid,
        session_idx=session_idx,
        tokens=_count_tokens(messages),
    )

    while not db.terminated and iteration < config.max_iterations:
        token_count = _count_tokens(messages)
        if token_count > int(config.token_budget * 0.95):
            terminated_by = "token_budget"
            break

        try:
            try:
                response = await llm_client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    enable_thinking=True,
                    user=f"qid={qid};phase2_s{session_idx};iter={iteration}",
                )
            except TypeError:
                response = await llm_client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    enable_thinking=True,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "phase2 LLM call failed qid=%s s=%d: %s", qid, session_idx, exc
            )
            emit_event(
                "pipeline.phase2.llm_error",
                qid=qid,
                session_idx=session_idx,
                iter=iteration,
                error=str(exc)[:200],
            )
            terminated_by = "error"
            break

        assistant_msg = _build_assistant_message(response)
        messages.append(assistant_msg)
        trace.append_turn(assistant_msg)

        if not response.tool_calls:
            # No tool call ends the session. The outer loop decides
            # whether to run another session or force-deliver, so
            # force_deliver is not called from here.
            terminated_by = "no_tool_call"
            emit_event(
                "pipeline.phase2.no_tool_call",
                qid=qid,
                session_idx=session_idx,
                iter=iteration,
                content_preview=(response.content or "")[:240],
            )
            break

        for tc in response.tool_calls:
            if tc.name not in PHASE2_TOOL_NAMES:
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": json.dumps(
                        {
                            "error": "tool_not_available_in_phase2",
                            "message": (
                                f"Tool {tc.name!r} is disabled in Phase 2. Allowed: "
                                + ", ".join(sorted(PHASE2_TOOL_NAMES))
                            ),
                        },
                        ensure_ascii=False,
                    ),
                }
                messages.append(tool_msg)
                trace.append_tool_result(tc, {"error": "tool_not_available_in_phase2"})
                continue

            wrapped = await _dispatch_pipeline_tool(
                db, tc, mcp_client, config, iteration=iteration
            )
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(wrapped, ensure_ascii=False),
            }
            messages.append(tool_msg)
            trace.append_tool_result(tc, wrapped)

            if db.terminated:
                terminated_by = "commit"
                break

        iteration += 1

    wall = time.monotonic() - t_start
    # Clear the Phase-2 flag before returning so downstream handlers (e.g.
    # post-session metrics) see clean state.
    db.phase2_active = False

    result = SessionResult(
        committed=bool(db.terminated),
        made_progress=True,  # outer loop computes the authoritative value
        iterations=iteration,
        wall_seconds=wall,
        terminated_by=terminated_by,
        session_messages=messages,
    )
    return result


# Outer control loop


async def run_phase2_loop(
    db: Any,
    question: Question,
    *,
    llm_client: LLMClientProtocol,
    mcp_client: MCPClientProtocol,
    tools: list[dict[str, Any]],
    trace: QuestionTrace,
    parent_reason: str,
) -> None:
    """Outer Phase 2 controller: runs up to ``PHASE2_MAX_SESSIONS`` sessions.

    Applies session-level circuit breakers: wall-clock timeout, stuck
    detection (no DB delta), and per-session error-tolerance. On any
    non-commit exit the loop calls :func:`force_deliver` with a tagged
    reason so the final trace distinguishes Phase 2 outcomes from direct
    force-delivers.

    Side effects:
      * Flips ``db.terminated`` (via ``submit_answer`` or ``force_deliver``).
      * Sets the ``m11_phase2_*`` fields on ``trace``.
    """
    from r2c2_agent.orchestrator.pipeline_handlers import (
        force_deliver,
        set_finalizer_clients,
        set_finalizer_tools,
    )

    from r2c2_agent.agent.obs import emit_event

    qid = db.question_id
    tools_filtered = filter_tools_for_phase2(tools)
    t_phase_start = time.monotonic()

    # Idempotent: ``loop.py`` sets these at the top of
    # ``answer_question_pipeline``; this covers the path where Phase-2
    # is invoked standalone (e.g. in tests).
    set_finalizer_clients(llm_client, mcp_client)
    # ``_reenter_agentic_loop`` uses this catalogue with
    # ``tool_choice="required"`` when a submission is retried. Phase-2
    # forbids retrieval tools, so pass the filtered catalogue (not the
    # full one) to keep the re-entry tool surface consistent with the
    # enclosing phase.
    set_finalizer_tools(tools_filtered)

    emit_event(
        "pipeline.phase2.enter",
        qid=qid,
        reason=parent_reason,
        nuggets=int(db.nuggets_recorded),
        passages=len(db.passages),
    )
    setattr(trace, "m11_phase2_entered", True)
    setattr(trace, "m11_phase2_sessions", 0)
    setattr(trace, "m11_phase2_outcome", "in_progress")

    for session_idx in range(1, PHASE2_MAX_SESSIONS + 1):
        elapsed = time.monotonic() - t_phase_start
        if elapsed >= PHASE2_MAX_TOTAL_SECONDS:
            emit_event(
                "pipeline.phase2.timeout",
                qid=qid,
                elapsed=round(elapsed, 2),
                cap=PHASE2_MAX_TOTAL_SECONDS,
            )
            setattr(trace, "m11_phase2_outcome", "timeout")
            setattr(trace, "m11_phase2_wall_seconds", round(elapsed, 2))
            try:
                force_deliver(db, f"phase2_timeout:{PHASE2_MAX_TOTAL_SECONDS:.0f}s")
            except Exception:  # noqa: BLE001
                log.exception("phase2 timeout: force_deliver raised")
            return

        before = PhaseSnapshot.from_db(db)
        emit_event(
            "pipeline.phase2.session.start",
            qid=qid,
            session_idx=session_idx,
            max_sessions=PHASE2_MAX_SESSIONS,
            nuggets_before=before.nuggets_recorded,
            passages=len(db.passages),
        )

        try:
            result = await run_synthesis_session(
                db,
                question,
                session_idx=session_idx,
                max_sessions=PHASE2_MAX_SESSIONS,
                llm_client=llm_client,
                mcp_client=mcp_client,
                tools=tools_filtered,
                config=PHASE2_LOOP_CONFIG,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("phase2 session %d raised qid=%s", session_idx, qid)
            emit_event(
                "pipeline.phase2.session.error",
                qid=qid,
                session_idx=session_idx,
                error=str(exc)[:200],
            )
            setattr(trace, "m11_phase2_outcome", "error")
            setattr(trace, "m11_phase2_sessions", session_idx)
            setattr(
                trace,
                "m11_phase2_wall_seconds",
                round(time.monotonic() - t_phase_start, 2),
            )
            try:
                force_deliver(
                    db, f"phase2_error:session={session_idx}:{str(exc)[:160]}"
                )
            except Exception:  # noqa: BLE001
                log.exception("phase2 error-path: force_deliver raised")
            return

        after = PhaseSnapshot.from_db(db)
        made_progress = session_made_progress(before, after)
        result.made_progress = made_progress

        emit_event(
            "pipeline.phase2.session.end",
            qid=qid,
            session_idx=session_idx,
            committed=result.committed,
            made_progress=made_progress,
            iters=result.iterations,
            wall_s=round(result.wall_seconds, 2),
            terminated_by=result.terminated_by,
            nuggets_after=after.nuggets_recorded,
        )

        if db.terminated:
            emit_event(
                "pipeline.phase2.commit",
                qid=qid,
                session_idx=session_idx,
            )
            setattr(trace, "m11_phase2_committed_session", session_idx)
            setattr(trace, "m11_phase2_outcome", "commit")
            setattr(trace, "m11_phase2_sessions", session_idx)
            setattr(
                trace,
                "m11_phase2_wall_seconds",
                round(time.monotonic() - t_phase_start, 2),
            )
            return

        if not made_progress:
            emit_event(
                "pipeline.phase2.stuck",
                qid=qid,
                session_idx=session_idx,
            )
            setattr(trace, "m11_phase2_outcome", "stuck")
            setattr(trace, "m11_phase2_sessions", session_idx)
            setattr(
                trace,
                "m11_phase2_wall_seconds",
                round(time.monotonic() - t_phase_start, 2),
            )
            try:
                force_deliver(db, f"phase2_stuck:session={session_idx}")
            except Exception:  # noqa: BLE001
                log.exception("phase2 stuck: force_deliver raised")
            return

    # Exhausted max_sessions without commit.
    emit_event(
        "pipeline.phase2.exhausted",
        qid=qid,
        sessions=PHASE2_MAX_SESSIONS,
    )
    setattr(trace, "m11_phase2_outcome", "exhausted")
    setattr(trace, "m11_phase2_sessions", PHASE2_MAX_SESSIONS)
    setattr(
        trace,
        "m11_phase2_wall_seconds",
        round(time.monotonic() - t_phase_start, 2),
    )
    try:
        force_deliver(db, f"phase2_exhausted:sessions={PHASE2_MAX_SESSIONS}")
    except Exception:  # noqa: BLE001
        log.exception("phase2 exhausted: force_deliver raised")
