#!/usr/bin/env python3
"""Bulk-index documents.jsonl into Elasticsearch.

Creates the `bitem-documents` index with document metadata + norm_doc_text.
The norm_doc_text is stored but not searchable (used by get_raw_document tool).

Supports shard mode for parallel bulk insert via SLURM job arrays.

Usage:
    # Full index (single process)
    python3 index_documents_to_es.py \
        --input data/chunks-v3/documents.jsonl \
        --es-url http://localhost:9200 \
        --index-name bitem-documents \
        --recreate

    # Shard mode (parallel job array, skip create/delete)
    python3 index_documents_to_es.py \
        --input data/chunks-v3/documents_shard_0003.jsonl \
        --es-url http://localhost:9200 \
        --index-name bitem-documents \
        --shard-mode

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

__all__ = ["main", "document_actions"]

DEFAULT_INDEX_NAME = "bitem-documents"
BATCH_SIZE = 2000

# Document index mapping — metadata + stored norm_doc_text
_DOCUMENT_PROPERTIES = {
    "title": {
        "type": "text",
        "fields": {"keyword": {"type": "keyword"}},
    },
    "source": {"type": "keyword"},
    "sections": {"type": "keyword"},
    "passage_count": {"type": "integer"},
    "raw_path": {"type": "keyword", "index": False},
    "norm_doc_text": {
        "type": "text",
        "index": False,
    },  # stored only, for get_raw_document
}

_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {"properties": _DOCUMENT_PROPERTIES},
}

# Fields to extract from each JSONL record
_FIELDS = (
    "title",
    "source",
    "sections",
    "passage_count",
    "raw_path",
    "norm_doc_text",
)


def document_actions(
    input_path: str,
    index_name: str,
) -> Iterator[dict]:
    """Yield bulk-index actions from a documents.jsonl file.

    Args:
        input_path: Path to JSONL file.
        index_name: Target ES index name.

    Yields:
        Bulk action dicts for elasticsearch.helpers.streaming_bulk.
    """
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] Skipping malformed JSON at line {line_num}",
                    file=sys.stderr,
                )
                continue

            doc_id = record.get("doc_id", "")
            if not doc_id:
                print(
                    f"[WARN] Skipping record without doc_id at line {line_num}",
                    file=sys.stderr,
                )
                continue

            source = {}
            for field in _FIELDS:
                if field in record:
                    source[field] = record[field]

            yield {
                "_index": index_name,
                "_id": doc_id,
                "_source": source,
            }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Bulk-index documents.jsonl into Elasticsearch.",
    )
    ap.add_argument(
        "--input", required=True, help="Path to documents.jsonl (or shard)."
    )
    ap.add_argument("--es-url", required=True, help="Elasticsearch URL.")
    ap.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="ES index name.")
    ap.add_argument(
        "--recreate", action="store_true", help="Delete and recreate index."
    )
    ap.add_argument(
        "--shard-mode",
        action="store_true",
        help="Shard mode: skip index creation/deletion (index must already exist).",
    )

    args = ap.parse_args()

    es = Elasticsearch(args.es_url, request_timeout=120)

    if not es.ping():
        print(f"ERROR: Cannot connect to ES at {args.es_url}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to ES at {args.es_url}")

    # In shard mode the index is created once, before the job array.
    if not args.shard_mode:
        if args.recreate and es.indices.exists(index=args.index_name):
            print(f"Deleting existing index: {args.index_name}")
            es.indices.delete(index=args.index_name)

        if not es.indices.exists(index=args.index_name):
            print(f"Creating index: {args.index_name}")
            es.indices.create(index=args.index_name, body=_INDEX_SETTINGS)

    print(f"Indexing from {args.input} → {args.index_name}")
    t0 = time.time()
    success_count = 0
    error_count = 0

    for ok, result in streaming_bulk(
        es,
        document_actions(args.input, args.index_name),
        chunk_size=BATCH_SIZE,
        raise_on_error=False,
    ):
        if ok:
            success_count += 1
        else:
            error_count += 1
            if error_count <= 5:
                print(f"[ERROR] {result}", file=sys.stderr)

        if success_count % 50000 == 0 and success_count > 0:
            elapsed = time.time() - t0
            rate = success_count / elapsed
            print(f"  Indexed {success_count} documents ({rate:.0f} docs/sec)")

    elapsed = time.time() - t0
    print(f"\nDone. {success_count} indexed, {error_count} errors in {elapsed:.1f}s")

    # Force merge (skip in shard mode — done once after all shards)
    if not args.shard_mode:
        print("Force-merging index...")
        es.indices.forcemerge(index=args.index_name, max_num_segments=1)
        print("Force-merge complete")

    count = es.count(index=args.index_name)["count"]
    print(f"Index {args.index_name}: {count} documents")


if __name__ == "__main__":
    main()
