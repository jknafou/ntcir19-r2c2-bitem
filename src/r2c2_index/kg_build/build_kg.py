#!/usr/bin/env python3
"""Build knowledge graph from enriched passages using entity co-occurrence.

Streams passages_enriched_v3.jsonl, accumulates entity pair counts and
individual entity counts, then computes NPMI for all pairs.

Modes:
  - Normal: full pipeline (count + NPMI) on entire input
  - Shard (map): accumulate counts on a shard, write partial counts JSON
  - Reduce: merge partial counts from multiple shards, compute NPMI

Usage:
    # Normal (single process)
    python build_kg.py \
        --input data/passages_enriched_v3.jsonl \
        --output data/bitem-kg.json

    # Map phase (parallel job array)
    python build_kg.py map \
        --input shards/enriched_0003.jsonl \
        --output shards/kg_counts_0003.json

    # Reduce phase (after all map tasks complete)
    python build_kg.py reduce \
        --input-dir shards/ \
        --input-pattern 'kg_counts_*.json' \
        --output data/bitem-kg.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

__all__ = ["main"]

# Minimum co-occurrence count to include a pair in the KG
MIN_PAIR_COUNT = 3
# Minimum NPMI to include a pair
MIN_NPMI = 0.1


def _count_from_jsonl(input_path: Path) -> tuple[Counter, Counter, int]:
    """Stream enriched JSONL and accumulate entity counts.

    Args:
        input_path: Path to enriched JSONL (full or shard).

    Returns:
        Tuple of (entity_count, pair_count, total_passages).
    """
    entity_count: Counter = Counter()
    pair_count: Counter = Counter()
    total_passages = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            entities = rec.get("entities", [])
            if not entities:
                continue

            total_passages += 1

            # An entity counts once per passage, however often it is mentioned.
            unique_entities = sorted(set(entities))
            for e in unique_entities:
                entity_count[e] += 1

            # Entities are sorted, so each pair has one canonical key.
            for e1, e2 in combinations(unique_entities, 2):
                pair_count[(e1, e2)] += 1

            if total_passages % 500_000 == 0:
                print(
                    f"  {total_passages:>10,} passages processed, "
                    f"{len(entity_count):,} entities, {len(pair_count):,} pairs"
                )

    return entity_count, pair_count, total_passages


def _compute_npmi(
    entity_count: Counter,
    pair_count: Counter,
    total_passages: int,
) -> list[dict]:
    """Compute NPMI for all entity pairs.

    Args:
        entity_count: Entity -> count of passages containing it.
        pair_count: (entity1, entity2) -> co-occurrence count.
        total_passages: Total number of passages with entities.

    Returns:
        List of edge dicts sorted by NPMI descending.
    """
    print(f"Computing NPMI for {len(pair_count):,} pairs...")
    edges = []
    n = total_passages

    for (e1, e2), count in pair_count.items():
        if count < MIN_PAIR_COUNT:
            continue

        p_e1 = entity_count[e1] / n
        p_e2 = entity_count[e2] / n
        p_pair = count / n

        # PMI = log(p(e1,e2) / (p(e1) * p(e2)))
        # NPMI = PMI / -log(p(e1,e2))  (normalized to [-1, 1])
        if p_pair <= 0 or p_e1 <= 0 or p_e2 <= 0:
            continue

        pmi = math.log(p_pair / (p_e1 * p_e2))
        npmi = pmi / (-math.log(p_pair))

        if npmi < MIN_NPMI:
            continue

        edges.append(
            {
                "e1": e1,
                "e2": e2,
                "npmi": round(npmi, 4),
                "count": count,
                "count_e1": entity_count[e1],
                "count_e2": entity_count[e2],
            }
        )

    edges.sort(key=lambda x: x["npmi"], reverse=True)
    return edges


def cmd_normal(args: argparse.Namespace) -> None:
    """Full pipeline: count + NPMI on entire input."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Counting entities from {input_path}...")
    entity_count, pair_count, total_passages = _count_from_jsonl(input_path)
    print(
        f"  {total_passages:,} passages, {len(entity_count):,} entities, "
        f"{len(pair_count):,} pairs"
    )

    edges = _compute_npmi(entity_count, pair_count, total_passages)

    kg = {
        "total_passages": total_passages,
        "total_entities": len(entity_count),
        "total_pairs_raw": len(pair_count),
        "total_edges": len(edges),
        "min_pair_count": MIN_PAIR_COUNT,
        "min_npmi": MIN_NPMI,
        "edges": edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False)
    print(f"\nKG written: {len(edges):,} edges -> {output_path}")


def cmd_map(args: argparse.Namespace) -> None:
    """Map phase: accumulate partial counts from a shard."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Map: counting entities from {input_path}...")
    entity_count, pair_count, total_passages = _count_from_jsonl(input_path)
    print(
        f"  {total_passages:,} passages, {len(entity_count):,} entities, "
        f"{len(pair_count):,} pairs"
    )

    # JSON keys must be strings, so a pair is written as "e1\te2".
    partial = {
        "total_passages": total_passages,
        "entity_count": dict(entity_count),
        "pair_count": {f"{e1}\t{e2}": c for (e1, e2), c in pair_count.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Partial counts written: {size_mb:.1f} MB -> {output_path}")


def cmd_reduce(args: argparse.Namespace) -> None:
    """Reduce phase: merge partial counts, compute NPMI."""
    input_dir = Path(args.input_dir)
    pattern = args.input_pattern
    output_path = Path(args.output)

    files = sorted(glob.glob(str(input_dir / pattern)))
    if not files:
        print(f"ERROR: No files matching {input_dir / pattern}")
        sys.exit(1)

    print(f"Reduce: merging {len(files)} partial count files...")

    total_passages = 0
    entity_count: Counter = Counter()
    pair_count: Counter = Counter()

    for i, path in enumerate(files):
        with open(path) as f:
            partial = json.load(f)
        total_passages += partial["total_passages"]
        for entity, count in partial["entity_count"].items():
            entity_count[entity] += count
        for pair_key, count in partial["pair_count"].items():
            e1, e2 = pair_key.split("\t", maxsplit=1)
            pair_count[(e1, e2)] += count
        print(
            f"  [{i + 1}/{len(files)}] {Path(path).name}: "
            f"+{partial['total_passages']:,} passages"
        )

    print(
        f"\nMerged totals: {total_passages:,} passages, "
        f"{len(entity_count):,} entities, {len(pair_count):,} pairs"
    )

    edges = _compute_npmi(entity_count, pair_count, total_passages)

    kg = {
        "total_passages": total_passages,
        "total_entities": len(entity_count),
        "total_pairs_raw": len(pair_count),
        "total_edges": len(edges),
        "min_pair_count": MIN_PAIR_COUNT,
        "min_npmi": MIN_NPMI,
        "edges": edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False)
    print(f"\nKG written: {len(edges):,} edges -> {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from entity co-occurrence in enriched passages."
    )
    sub = parser.add_subparsers(dest="command")

    # Without a subcommand the parser runs in normal mode.
    parser.add_argument("--input", help="Input enriched JSONL (normal mode)")
    parser.add_argument("--output", help="Output KG JSON (normal mode)")

    p_map = sub.add_parser(
        "map", help="Map phase: accumulate partial counts from shard"
    )
    p_map.add_argument("--input", required=True, help="Input shard JSONL")
    p_map.add_argument("--output", required=True, help="Output partial counts JSON")

    p_reduce = sub.add_parser(
        "reduce", help="Reduce phase: merge partial counts + compute NPMI"
    )
    p_reduce.add_argument(
        "--input-dir", required=True, help="Directory with partial count files"
    )
    p_reduce.add_argument(
        "--input-pattern",
        default="kg_counts_*.json",
        help="Glob pattern for partial count files",
    )
    p_reduce.add_argument("--output", required=True, help="Output KG JSON")

    args = parser.parse_args()

    if args.command == "map":
        cmd_map(args)
    elif args.command == "reduce":
        cmd_reduce(args)
    else:
        if not args.input or not args.output:
            parser.error("--input and --output required for normal mode")
        cmd_normal(args)


if __name__ == "__main__":
    main()
