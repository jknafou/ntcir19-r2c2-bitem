"""r2c2_agent — agentic-RAG inference subpackage.

The published code path is T1 → T2 → T4 → T5 — a four-tier
educated-guess ladder followed by a canonical-refusal floor.

Top-level CLI::

    python -m r2c2_agent submit       --questions Q.jsonl --output OUT/
    python -m r2c2_agent reaggregate-ac --pr-dir DIR --traces-dir DIR --output OUT/

Sub-package layout:

* ``orchestrator/`` — CPU dispatch, GPU worker pool, ReAct driver, force-deliver
  ladder, ensemble aggregator, AC re-aggregator, NLI verifier.
* ``mcp_server/``   — FastMCP server exposing the 4 deployed retrieval tools
  (``search_passages``, ``get_passage``, ``extract_from_passage``,
  ``search_entities``).
* ``validation/``   — 7-check R2C2-spec submission validator (the
  aggregator runs the same checks plus a source-coverage check inline).
* ``shared/``       — answer-extraction + answer-normalization helpers.
* ``slurm/``        — SLURM templates for orchestrator + workers.
"""

from __future__ import annotations

__all__: list[str] = ["__version__"]

__version__ = "0.1.0"
