#!/usr/bin/env python3
"""Bulk-index passages.jsonl into Elasticsearch.

Creates an ES index with custom BM25 similarity (k1=0.82, b=0.68) and
bulk-indexes all passages. Supports two modes:
  - base: core text fields + offset metadata (no entity fields)
  - ner:  base fields + entity fields (entities, entity_types, entity_text, etc.)

Usage:
    # Base index (from passages.jsonl)
    python3 index_to_elasticsearch.py \
        --input data/chunks-v3/passages.jsonl \
        --es-url http://localhost:9200 \
        --index-name bitem-passages \
        --mode base --recreate

    # NER index (from passages_enriched_v3.jsonl)
    python3 index_to_elasticsearch.py \
        --input data/chunks-v3/passages_enriched_v3.jsonl \
        --es-url http://localhost:9200 \
        --index-name bitem-passages-ner \
        --mode ner --recreate

Requires: elasticsearch>=8.12,<9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

__all__ = ["main", "passage_actions"]

DEFAULT_INDEX_NAME = "bitem-passages"
BATCH_SIZE = 5000

# BM25 parameters tuned on MS MARCO.
BM25_K1 = 0.82
BM25_B = 0.68

# Base mapping: core text fields + offset metadata
_BASE_PROPERTIES = {
    "contents": {"type": "text", "similarity": "custom_bm25"},
    "title": {
        "type": "text",
        "similarity": "custom_bm25",
        "fields": {"keyword": {"type": "keyword"}},
    },
    "section": {
        "type": "text",
        "similarity": "custom_bm25",
        "fields": {"keyword": {"type": "keyword"}},
    },
    "source": {"type": "keyword"},
    "doc_id": {"type": "keyword"},
    # Offset metadata (stored, not searchable except passage_index)
    "raw_char_offset": {"type": "long", "index": False},
    "raw_char_length": {"type": "integer", "index": False},
    "passage_index": {"type": "integer"},
    "word_count": {"type": "integer", "index": False},
}

# NER fields added in ner mode
_NER_PROPERTIES = {
    "entities": {"type": "keyword"},
    "entity_types": {"type": "keyword"},
    "entity_text": {"type": "text", "similarity": "custom_bm25"},
    "doc_entities": {"type": "keyword"},
    "text_entities": {"type": "keyword"},
}

# Base fields to extract from each JSONL record
_BASE_FIELDS = (
    "contents",
    "title",
    "section",
    "source",
    "doc_id",
    "raw_char_offset",
    "raw_char_length",
    "word_count",
)

# NER fields to extract (only in ner mode)
_NER_FIELDS = (
    "entities",
    "entity_types",
    "entity_text",
    "doc_entities",
    "text_entities",
)


_DOCUMENT_PROPERTIES = {
    "title": {
        "type": "text",
        "fields": {"keyword": {"type": "keyword"}},
    },
    "source": {"type": "keyword"},
    "sections": {"type": "keyword"},
    "passage_count": {"type": "integer"},
    "raw_path": {"type": "keyword", "index": False},
    "norm_doc_text": {"type": "text", "index": False},
}


def _build_index_settings(mode: str) -> dict:
    """Build ES index settings+mappings for the given mode.

    Args:
        mode: 'base', 'ner', or 'documents'.

    Returns:
        Complete index settings dict.
    """
    if mode == "documents":
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {"properties": dict(_DOCUMENT_PROPERTIES)},
        }

    properties = dict(_BASE_PROPERTIES)
    if mode == "ner":
        properties.update(_NER_PROPERTIES)

    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": BM25_K1,
                    "b": BM25_B,
                }
            },
        },
        "mappings": {"properties": properties},
    }


def passage_actions(
    input_path: str,
    index_name: str,
    mode: str = "base",
) -> Iterator[dict]:
    """Yield bulk-index actions from a passages.jsonl file.

    Args:
        input_path: Path to JSONL file.
        index_name: Target ES index name.
        mode: 'base' or 'ner'. Controls which fields are indexed.

    Yields:
        Bulk action dicts for elasticsearch.helpers.streaming_bulk.
    """
    fields = _BASE_FIELDS
    if mode == "ner":
        fields = _BASE_FIELDS + _NER_FIELDS

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARN: skipping line {line_num}: {exc}", file=sys.stderr)
                continue

            doc_id = doc.get("id", f"line_{line_num}")

            source: dict = {}
            for field in fields:
                if field in doc:
                    source[field] = doc[field]

            # A passage ID has the form "docid__N"; N is its position in the document.
            try:
                source["passage_index"] = int(doc_id.rsplit("__", 1)[1])
            except (IndexError, ValueError):
                source["passage_index"] = 0

            yield {
                "_index": index_name,
                "_id": doc_id,
                "_source": source,
            }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk-index passages into Elasticsearch",
    )
    parser.add_argument("--input", required=True, help="Path to passages.jsonl")
    parser.add_argument(
        "--es-url", default="http://localhost:9200", help="Elasticsearch URL"
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"Elasticsearch index name (default: {DEFAULT_INDEX_NAME})",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "ner"],
        default="base",
        help="Index mode: base (no entity fields) or ner (with entities). Default: base.",
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate index"
    )
    parser.add_argument(
        "--shard-mode",
        action="store_true",
        help="Shard mode: skip index creation/deletion and force-merge. "
        "Index must already exist. Used by parallel job arrays.",
    )
    args = parser.parse_args()

    index_name = args.index_name
    mode = args.mode
    shard_mode = args.shard_mode
    es = Elasticsearch(args.es_url, request_timeout=120)

    # Fail early if the server is unreachable.
    info = es.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")
    print(f"Mode: {mode}, Shard mode: {shard_mode}")

    if shard_mode:
        # Parallel shard jobs share one index, so none of them may create or delete it.
        if not es.indices.exists(index=index_name):
            print(
                f"ERROR: Index '{index_name}' does not exist (required in --shard-mode)"
            )
            sys.exit(1)
        print(f"Shard mode: appending to existing index '{index_name}'")
    else:
        if es.indices.exists(index=index_name):
            if args.recreate:
                print(f"Deleting existing index '{index_name}'...")
                es.indices.delete(index=index_name)
            else:
                print(
                    f"Index '{index_name}' already exists. Use --recreate to rebuild."
                )
                count = es.count(index=index_name)["count"]
                print(f"Current document count: {count:,}")
                return

        index_settings = _build_index_settings(mode)
        print(
            f"Creating index '{index_name}' (BM25 k1={BM25_K1}, b={BM25_B}, mode={mode})..."
        )
        es.indices.create(index=index_name, body=index_settings)

    print(f"Indexing from {args.input}...")
    t0 = time.time()
    indexed = 0
    errors = 0

    for ok, result in streaming_bulk(
        es,
        passage_actions(args.input, index_name, mode=mode),
        chunk_size=BATCH_SIZE,
        raise_on_error=False,
    ):
        if ok:
            indexed += 1
        else:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: {result}", file=sys.stderr)

        if indexed % 100_000 == 0 and indexed > 0:
            elapsed = time.time() - t0
            rate = indexed / elapsed
            print(f"  {indexed:>10,} docs indexed ({rate:,.0f} docs/s)")

    elapsed = time.time() - t0
    rate = indexed / elapsed
    print(
        f"\nIndexing complete: {indexed:,} docs in {elapsed:.1f}s ({rate:,.0f} docs/s)"
    )
    if errors:
        print(f"  Errors: {errors:,}")

    if shard_mode:
        # The finalize job refreshes and force-merges once all shards are in.
        print("\nShard indexing complete. Skipping refresh/force-merge (shard mode).")
    else:
        print("Refreshing index...")
        es.indices.refresh(index=index_name)

        print("Force-merging to 1 segment (this may take a few minutes)...")
        es.indices.forcemerge(index=index_name, max_num_segments=1, request_timeout=600)

        count = es.count(index=index_name)["count"]
        print(f"\nFinal document count: {count:,}")


if __name__ == "__main__":
    main()
