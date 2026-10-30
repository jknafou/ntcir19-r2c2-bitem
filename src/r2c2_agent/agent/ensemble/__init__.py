"""Ensemble-mode helpers for per-iteration dispatch.

Public API: :func:`load_touched_set` reconstructs the cumulative
``touched_pids`` snapshot for ensemble iteration K from the on-disk
records of iterations 0..K-1. Moving this out of the worker's
in-memory loop makes each iteration node-independent.
"""

from r2c2_agent.agent.ensemble.prior_state import (
    MissingPriorIterError,
    PriorAnswerResult,
    PriorIterResult,
    load_prior_rollout_answer,
    load_touched_set,
)

__all__ = [
    "MissingPriorIterError",
    "PriorAnswerResult",
    "PriorIterResult",
    "load_prior_rollout_answer",
    "load_touched_set",
]
