"""Hook dataclasses for the agentic loop.

The agentic loop (``r2c2_agent.agent.loop.answer_question``) supports optional
post-draft callbacks that a caller can use to implement check-points (e.g.
CP2 verifier-driven self-correction).

Why this module lives outside ``r2c2_agent.agent/interfaces/``:
    These dataclasses are plain in-process control structures, not
    versioned protocols, so they live at the package root alongside
    ``config.py``. The loop imports them directly, with no factory wiring.

Why the callback exists:
    The callback needs three things that are otherwise not reachable from
    inside the loop:

    1. The draft answer the agent just produced.
    2. The gold / expected answer, which must not be put on
       :class:`r2c2_agent.agent.interfaces.types.Question` (which the LLM sees)
       because it would leak the answer into the prompt.
    3. A decision channel back into the loop: inject a reflection turn,
       continue silently, or abort.

    The worker (``src/r2c2_agent/orchestrator/worker.py``) constructs a closure
    that captures ``expected_answer`` from the bundle and passes the
    closure in :attr:`r2c2_agent.agent.config.LoopConfig.on_draft`. The loop
    never sees the gold answer directly.

All dataclasses in this module are in-process control structures only.
Nothing here is serialized into trace JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DraftContext", "DraftDecision"]


@dataclass
class DraftContext:
    """Inputs to the on_draft CP2 callback.

    Attributes:
        question: The original user question text (never the injected
            reflection text — this is stable across multiple callback
            invocations for a single trace).
        expected_answer: The gold reference answer, derived from the
            bundle by the caller's closure. Empty string if the bundle
            has no computable gold (the caller should short-circuit the
            callback in that case rather than pass through).
        draft_answer: The draft answer the agent just produced in its
            most recent assistant turn. May be empty if the agent
            emitted only tool calls.
        retrieved_passages: Up to the last 3 tool-result contents
            (string-serialized JSON), already windowed to the recent
            context. The callback is expected to further trim these
            before sending to the verifier.
        injections_so_far: How many reflection turns have already been
            injected into this trace. The CP2 cap is 2; a callback that
            sees ``injections_so_far >= 2`` and still wants to fire
            must set :attr:`DraftDecision.abort` instead of
            :attr:`DraftDecision.inject`.
    """

    question: str
    expected_answer: str
    draft_answer: str
    retrieved_passages: list[str]
    injections_so_far: int


@dataclass
class DraftDecision:
    """Return value of the on_draft CP2 callback.

    Attributes:
        inject: When True, the loop appends ``reflection_text`` as a
            user-role turn with a ``_injected_reflection=True`` marker
            and continues. Ignored if :attr:`abort` is also True.
        reflection_text: The message content to append. Must be
            non-empty when ``inject`` is True.
        abort: When True, the loop terminates with
            ``trace.termination = "cp2_abort"`` on the next opportunity.
            Used to enforce the 2-injection cap (3rd trigger = abort).
        verdict_info: Arbitrary audit-trail dict appended to
            ``trace.trace_metadata["verifier_verdicts"]``. Not used by
            the loop itself — purely for downstream analysis.
    """

    inject: bool
    reflection_text: str
    abort: bool
    verdict_info: dict[str, Any] = field(default_factory=dict)
