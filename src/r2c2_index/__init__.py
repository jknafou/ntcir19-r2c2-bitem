"""r2c2_index — corpus indexation pipeline for the R2C2 agentic-RAG system.

Five stages map 1:1 to subpackages, each runnable as ``python -m r2c2_index.<stage>``:

================  =======================================================
Stage             Purpose
================  =======================================================
corpus_parse      Wikitext → structured documents → ~200-word passages
                  (with raw_char offsets) sharded across the corpus.
ner               LLM-driven named entity extraction + within-doc and
                  cross-doc refinement; merges entities into enriched
                  passages JSONL with NER+offset fields.
kg_build          Entity co-occurrence graph (NPMI scored) from the
                  enriched JSONL.
es_index          Bulk-load three Elasticsearch indexes
                  (bitem-passages, bitem-passages-ner, bitem-documents).
faiss_encode      Sentence-transformer dense encoding into two FAISS
                  IndexFlatIP indexes (plain + entity-enriched).
================  =======================================================

SLURM templates live in :mod:`r2c2_index.slurm` (one .sbatch per stage,
following the shard → load-to-RAM → multiprocess → cat-merge pattern).

See ``src/r2c2_index/README.md`` for the canonical run order and reference
artefact paths.
"""

from __future__ import annotations

__all__ = [
    "corpus_parse",
    "ner",
    "kg_build",
    "es_index",
    "faiss_encode",
]
