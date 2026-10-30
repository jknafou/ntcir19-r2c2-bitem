#!/usr/bin/env python3
"""Create or finalize an Elasticsearch index (no data ingestion).

Used as pre/post steps around parallel bulk-indexing job arrays:
  - create: Create the index with proper settings/mappings
  - finalize: Refresh + force-merge to 1 segment + report count

Usage:
    # Before job array: create the index
    python3 es_manage_index.py create \
        --es-url http://localhost:9200 \
        --index-name bitem-passages \
        --mode base --recreate

    # After job array: finalize
    python3 es_manage_index.py finalize \
        --es-url http://localhost:9200 \
        --index-name bitem-passages

Requires: elasticsearch>=8.12,<9
"""

from __future__ import annotations

import argparse
import sys

from elasticsearch import Elasticsearch

__all__ = ["main"]

# BM25 parameters tuned on MS MARCO
BM25_K1 = 0.82
BM25_B = 0.68

# Import mappings from the indexing script to stay DRY
from .index_to_elasticsearch import _build_index_settings


def cmd_create(args: argparse.Namespace) -> None:
    """Create an ES index with custom BM25 settings."""
    es = Elasticsearch(args.es_url)
    info = es.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")

    index_name = args.index_name
    mode = args.mode

    if es.indices.exists(index=index_name):
        if args.recreate:
            print(f"Deleting existing index '{index_name}'...")
            es.indices.delete(index=index_name)
        else:
            print(f"Index '{index_name}' already exists. Use --recreate to rebuild.")
            count = es.count(index=index_name)["count"]
            print(f"Current document count: {count:,}")
            return

    settings = _build_index_settings(mode)
    print(
        f"Creating index '{index_name}' (BM25 k1={BM25_K1}, b={BM25_B}, mode={mode})..."
    )
    es.indices.create(index=index_name, body=settings)
    print(f"Index '{index_name}' created.")


def cmd_finalize(args: argparse.Namespace) -> None:
    """Refresh and force-merge an ES index after parallel bulk indexing."""
    es = Elasticsearch(args.es_url, request_timeout=300)
    info = es.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")

    index_name = args.index_name

    if not es.indices.exists(index=index_name):
        print(f"ERROR: Index '{index_name}' does not exist.")
        sys.exit(1)

    print("Refreshing index...")
    es.indices.refresh(index=index_name)

    count = es.count(index=index_name)["count"]
    print(f"Document count after refresh: {count:,}")

    print("Force-merging to 1 segment (this may take a few minutes)...")
    es.options(request_timeout=600).indices.forcemerge(
        index=index_name, max_num_segments=1
    )

    count = es.count(index=index_name)["count"]
    print(f"\nFinal document count: {count:,}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage ES index lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create index with settings/mappings")
    p_create.add_argument("--es-url", default="http://localhost:9200")
    p_create.add_argument("--index-name", required=True)
    p_create.add_argument(
        "--mode", choices=["base", "ner", "documents"], default="base"
    )
    p_create.add_argument("--recreate", action="store_true")

    p_finalize = sub.add_parser("finalize", help="Refresh + force-merge index")
    p_finalize.add_argument("--es-url", default="http://localhost:9200")
    p_finalize.add_argument("--index-name", required=True)

    args = parser.parse_args()

    if args.command == "create":
        cmd_create(args)
    elif args.command == "finalize":
        cmd_finalize(args)


if __name__ == "__main__":
    main()
