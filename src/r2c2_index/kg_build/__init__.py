"""Stage 3: build the entity co-occurrence Knowledge Graph (NPMI-scored).

Single sub-tool:
  - ``build_kg``  Stream the enriched JSONL, accumulate entity + pair counts,
                  compute NPMI, prune (NPMI ≥ 0.1, count ≥ 3), write
                  ``bitem-kg.json`` (~80-120 MB).
"""

from __future__ import annotations

__all__ = ["build_kg"]
