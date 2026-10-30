"""Stage 1: parse R2C2 wikitext shards into ~200-word passages with offsets.

The deployed entrypoint is :mod:`chunk_structured` (see
``src/r2c2_index/slurm/run_pipeline.sbatch``). It uses :mod:`chunker` for
chunking primitives and the collection-specific parsers under
:mod:`parsers`.
"""

from __future__ import annotations

__all__ = [
    "chunker",
    "chunk_structured",
]
