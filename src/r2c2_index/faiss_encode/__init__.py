"""Stage 5: dense encode + FAISS IndexFlatIP build (2 indexes).

Sub-tools:
  - ``build_dense_index``  Sentence-transformer encode (BAAI/bge-base-en-v1.5
                           or nomic-embed-text-v1.5), L2-normalize, write
                           ``faiss.index`` + ``id_map.npy`` + ``build_meta.json``.
                           ``--index-type {plain,enriched}`` selects the text
                           construction (plain = title|section|contents;
                           enriched = entities. + plain).
  - ``merge_faiss_shards`` Merge per-shard FAISS outputs into a single
                           ``IndexFlatIP`` (fast cat-style concatenation).
"""

from __future__ import annotations

__all__ = ["build_dense_index", "merge_faiss_shards"]
