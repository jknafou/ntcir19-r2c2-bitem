#!/usr/bin/env python3
"""Merge offset fields from passages_v3.jsonl into passages_enriched.jsonl.

Also fixes entity_types format: splits "Name::TYPE" composites into
separate entity_types (flat type list) and entities (flat name list).
Produces entity_type_map.json sidecar for FAISS enriched index building.

Supports two execution modes:
  - Normal: process entire enriched file, produce final output + entity map
  - Shard:  process a shard of enriched file, produce output shard + partial
            entity_type_counts JSON (for later reduction)

Usage:
    # Normal (single process)
    python merge_offsets_to_enriched.py \
        --passages-v3 chunks-v2/passages.jsonl \
        --enriched chunks-v2/passages_enriched.jsonl \
        --output chunks-v2/passages_enriched_v3.jsonl \
        --entity-map chunks-v2/entity_type_map.json

    # Shard mode (parallel job array)
    python merge_offsets_to_enriched.py \
        --passages-v3 chunks-v3/passages.jsonl \
        --enriched shards/enriched_0003.jsonl \
        --output shards/output_0003.jsonl \
        --entity-map shards/entity_counts_0003.json \
        --shard-mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["main"]

# Fields to remove from enriched JSONL (replaced by fixed entity_types)
_TYPED_ENTITY_FIELDS = {
    "entities_person",
    "entities_character",
    "entities_location",
    "entities_organization",
    "entities_event",
    "entities_work",
    "entities_date",
}


def _process_enriched(
    enriched_path: Path,
    output_path: Path,
    offset_data: dict[str, dict],
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Process enriched JSONL: merge offsets, fix entity_types.

    Args:
        enriched_path: Input enriched JSONL (full file or shard).
        output_path: Output merged JSONL.
        offset_data: Mapping of passage ID to offset fields.

    Returns:
        Tuple of (merged_count, missing_count, entity_type_counts).
    """
    entity_type_counts: dict[str, dict[str, int]] = {}
    merged_count = 0
    missing_count = 0

    with (
        open(enriched_path, encoding="utf-8") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            rec = json.loads(line)
            pid = rec["id"]

            # Add offset fields
            offsets = offset_data.get(pid)
            if offsets:
                rec.update(offsets)
                merged_count += 1
            else:
                missing_count += 1

            # Fix entity_types: split "Name::TYPE" composites
            raw_et = rec.get("entity_types", [])
            fixed_types: list[str] = []
            for composite in raw_et:
                if "::" in composite:
                    name, etype = composite.rsplit("::", 1)
                    fixed_types.append(etype)
                    # Track for entity_type_map
                    if name not in entity_type_counts:
                        entity_type_counts[name] = {}
                    entity_type_counts[name][etype] = (
                        entity_type_counts[name].get(etype, 0) + 1
                    )

            rec["entity_types"] = sorted(set(fixed_types))

            # Remove old typed entity fields
            for field in _TYPED_ENTITY_FIELDS:
                rec.pop(field, None)

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return merged_count, missing_count, entity_type_counts


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Merge offset fields into enriched JSONL + fix entity_types.",
    )
    ap.add_argument(
        "--passages-v3",
        required=True,
        help="Path to passages.jsonl with offset fields (from chunker.py v3 run).",
    )
    ap.add_argument(
        "--enriched",
        required=True,
        help="Path to existing passages_enriched.jsonl (or a shard of it).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for merged passages_enriched_v3.jsonl (or output shard).",
    )
    ap.add_argument(
        "--entity-map",
        required=True,
        help="Output path for entity_type_map.json (normal) or partial counts (shard).",
    )
    ap.add_argument(
        "--shard-mode",
        action="store_true",
        help="Shard mode: write raw entity_type_counts instead of resolved entity_type_map. "
        "The merge step will reduce partial counts into the final map.",
    )

    args = ap.parse_args()

    passages_v3_path = Path(args.passages_v3)
    enriched_path = Path(args.enriched)
    output_path = Path(args.output)
    entity_map_path = Path(args.entity_map)

    # Pass 1: load offset fields from passages_v3.jsonl.
    # Normal and shard mode both load the full offset dict.
    # ~500 MB for 2.09M passages, fits comfortably in 4G RAM.
    print(f"Pass 1: Loading offsets from {passages_v3_path}...")
    offset_data: dict[str, dict] = {}
    v3_count = 0
    with open(passages_v3_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["id"]
            offset_data[pid] = {
                "raw_char_offset": rec["raw_char_offset"],
                "raw_char_length": rec["raw_char_length"],
                "word_count": rec["word_count"],
            }
            v3_count += 1
    print(f"  Loaded {v3_count} passages with offsets")

    # Pass 2: stream enriched, merge offsets, fix entity_types.
    print(f"Pass 2: Merging into {output_path}...")
    merged_count, missing_count, entity_type_counts = _process_enriched(
        enriched_path, output_path, offset_data
    )

    if args.shard_mode:
        # Shard mode: write raw counts for later reduction
        with open(entity_map_path, "w", encoding="utf-8") as f:
            json.dump(entity_type_counts, f, ensure_ascii=False)
        print(
            f"  Shard entity counts: {len(entity_type_counts)} entities -> {entity_map_path}"
        )
    else:
        # Normal mode: resolve to final entity_type_map (most frequent type wins)
        entity_type_map: dict[str, str] = {}
        for entity, type_counts in entity_type_counts.items():
            best_type = max(type_counts, key=lambda t: type_counts[t])
            entity_type_map[entity] = best_type

        with open(entity_map_path, "w", encoding="utf-8") as f:
            json.dump(entity_type_map, f, ensure_ascii=False)
        print(
            f"  Entity type map: {len(entity_type_map)} entities -> {entity_map_path}"
        )

    print(f"  Merged: {merged_count}, Missing offsets: {missing_count}")
    print()

    if missing_count > 0:
        pct = missing_count / (merged_count + missing_count) * 100
        print(
            f"WARNING: {missing_count} enriched passages ({pct:.1f}%) had no matching v3 passage"
        )


if __name__ == "__main__":
    main()
