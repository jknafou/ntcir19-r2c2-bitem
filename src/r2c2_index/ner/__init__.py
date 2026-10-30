"""Stage 2: LLM-driven named entity extraction + within/cross-doc refinement.

Sub-tools (each with its own ``main()``; dispatched via
``python -m r2c2_index.ner <tool>``):

  - ``extract_entities_llm``  Document-level (or per-passage) LLM extraction;
                              writes ``passages_enriched.jsonl`` with `entities` +
                              `entity_types` per passage.
  - ``extract_entities``      Dictionary-based fallback / pre-LLM baseline.
  - ``classify_entities``     Type classification of dictionary-extracted entities.
  - ``filter_entities``       Frequency / blocklist filtering.
  - ``refine_entities``       Within-doc occurrence-based FP/FN refinement
                              (one LLM call per detected entity).
  - ``fast_crossdoc``         Fast cross-document co-occurrence filter
                              (multiprocess, NPMI-style scoring).
  - ``parallel_crossdoc``     Sharded variant of fast_crossdoc.
  - ``judge_annotate``        LLM-judge that builds the silver gold standard.
  - ``convert_ner_to_enriched``    Convert raw NER output JSONL → enriched format.
  - ``merge_offsets_to_enriched``  Merge ``raw_char_offset`` from passages_v3.jsonl
                                   into the enriched JSONL by passage_id.

Helper modules (private) live in :mod:`r2c2_index.ner._lib`:
``llm_utils``, ``entity_utils``, ``bio_utils``, ``span_utils``,
``build_document_prompt``, ``document_reconstruction``.
"""

from __future__ import annotations

__all__ = [
    "extract_entities_llm",
    "extract_entities",
    "classify_entities",
    "filter_entities",
    "refine_entities",
    "fast_crossdoc",
    "parallel_crossdoc",
    "judge_annotate",
    "convert_ner_to_enriched",
    "merge_offsets_to_enriched",
]
