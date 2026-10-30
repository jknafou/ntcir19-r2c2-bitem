"""Structured-event logging helpers for the agentic pipeline.

Provides a single dedicated logger (``pipeline.events``) that emits one JSON line
per semantic event. The plain-text module loggers (``r2c2_agent.agent.loop``,
``pipeline_handlers``) stay untouched so operator-visible terminal output remains
human-readable. Post-hoc analysis uses the JSONL stream via ``jq`` or pandas.

Verbosity is gated by the ``PIPELINE_LOG_VERBOSITY`` env var:

- ``quiet``  — FINAL + adapter events only (~3 lines / qid).
- ``normal`` — full instrumentation (~50-100 lines / qid). Default.
- ``verbose`` — ``normal`` plus any verbose-tagged emissions.

Callers pass an optional ``level`` kwarg per event. ``level="quiet"`` events
always fire; ``level="normal"`` fires when verbosity is ``normal`` or above;
``level="verbose"`` fires only when verbosity is ``verbose``.

Worker-side setup (attaching a JSONL file handler) lives in
``src/r2c2_agent/orchestrator/worker.py::main`` — this module stays pure so it can
be imported from test code without side effects.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

__all__ = [
    "emit_event",
    "get_events_logger",
    "verbosity",
]


_LOG_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2}


def verbosity() -> str:
    """Return the current ``PIPELINE_LOG_VERBOSITY`` value, normalised.

    Falls back to ``"normal"`` for any unrecognised value.
    """
    raw = os.environ.get("PIPELINE_LOG_VERBOSITY", "normal").lower().strip()
    return raw if raw in _LOG_LEVELS else "normal"


def _should_emit(event_level: str) -> bool:
    """Return True if an event tagged ``event_level`` should fire now."""
    current = _LOG_LEVELS.get(verbosity(), 1)
    required = _LOG_LEVELS.get(event_level, 1)
    return current >= required


def get_events_logger() -> logging.Logger:
    """Return the shared ``pipeline.events`` logger (no side effects)."""
    return logging.getLogger("pipeline.events")


def emit_event(event: str, level: str = "normal", **fields: Any) -> None:
    """Emit a structured event to the ``pipeline.events`` logger as JSON.

    Args:
        event: Event type name (e.g. ``"pipeline.record_evidence.outcome"``).
        level: Verbosity gate — ``"quiet"``, ``"normal"``, or ``"verbose"``.
        **fields: Arbitrary key/value pairs to serialise in the JSON line.
            ``event`` and ``ts`` are injected automatically.

    The emission never raises — any serialisation failure is caught and
    logged to the same logger via ``exception`` so the plain-text log still
    carries a trace of the lost event.
    """
    if not _should_emit(level):
        return
    logger = get_events_logger()
    payload: dict[str, Any] = {"ts": time.time(), "event": event}
    payload.update(fields)
    try:
        logger.info(json.dumps(payload, default=str, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        logger.exception("pipeline.events emit_event failed for %s", event)
