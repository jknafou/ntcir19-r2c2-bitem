#!/usr/bin/env python3
"""Convert NER pipeline output to enriched passages format for indexing.

Transforms typed entity dicts (list[dict]) from refined_crossdoc.jsonl into
the flat string lists (list[str]) that the indexing stack expects, plus
per-type entity fields for type-aware search.

Usage:
    python convert_ner_to_enriched.py \
        --input releases/v1.0-9B-full/refined_crossdoc.jsonl \
        --output data/chunks-v2/passages_enriched.jsonl \
        --stats
"""

import argparse
import json
import sys
import time

ENTITY_TYPES = [
    "PERSON",
    "CHARACTER",
    "LOCATION",
    "ORGANIZATION",
    "EVENT",
    "WORK",
    "DATE",
]


def extract_names(entity_list: list[dict]) -> list[str]:
    """Extract unique sorted entity names from a list of entity dicts."""
    return sorted(set(e["text"] for e in entity_list if "text" in e))


def convert_passage(passage: dict, include_spans: bool = True) -> dict:
    """Convert NER pipeline output to indexable format."""
    all_entities = passage.get("entities", [])

    # The Elasticsearch mapping and the search tools expect plain lists of names.
    entities_flat = extract_names(all_entities)
    text_entities_flat = extract_names(passage.get("text_entities", []))
    doc_entities_flat = extract_names(passage.get("doc_entities", []))

    out = {
        # Pass through unchanged
        "id": passage["id"],
        "contents": passage["contents"],
        "title": passage["title"],
        "section": passage.get("section", ""),
        "source": passage["source"],
        "doc_id": passage["doc_id"],
        "entities": entities_flat,
        "text_entities": text_entities_flat,
        "doc_entities": doc_entities_flat,
        # Typed entity composite field (Name::TYPE)
        "entity_types": sorted(
            set(
                f"{e['text']}::{e['type']}"
                for e in all_entities
                if "text" in e and "type" in e
            )
        ),
        # Entity names as one text field, so that BM25 can match them
        "entity_text": " ".join(entities_flat),
    }

    for etype in ENTITY_TYPES:
        field_name = f"entities_{etype.lower()}"
        out[field_name] = sorted(
            set(
                e["text"]
                for e in all_entities
                if e.get("type") == etype and "text" in e
            )
        )

    # Spans are optional; they allow highlighting entities in the text.
    if include_spans:
        for span_field in ("content_spans", "title_spans", "section_spans"):
            if passage.get(span_field):
                out[span_field] = passage[span_field]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert NER pipeline output to enriched passages format for indexing."
    )
    parser.add_argument("--input", required=True, help="Path to refined_crossdoc.jsonl")
    parser.add_argument(
        "--output", required=True, help="Path to write passages_enriched.jsonl"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only first N passages (0=all)"
    )
    parser.add_argument(
        "--no-spans",
        action="store_true",
        help="Omit span data (reduces file size ~30%%)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate required fields; error on missing",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print per-type entity statistics"
    )
    args = parser.parse_args()

    t0 = time.time()
    include_spans = not args.no_spans

    total = 0
    total_entities = 0
    type_counts = {t: 0 for t in ENTITY_TYPES}
    passages_with_entities = 0
    validation_errors = 0

    required_fields = ["id", "contents", "title", "source", "doc_id", "entities"]

    with (
        open(args.input, encoding="utf-8") as fin,
        open(args.output, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            passage = json.loads(line)
            total += 1

            if args.validate:
                missing = [f for f in required_fields if f not in passage]
                if missing:
                    print(
                        f"ERROR: passage {passage.get('id', '?')} missing fields: {missing}",
                        file=sys.stderr,
                    )
                    validation_errors += 1
                    continue

            out = convert_passage(passage, include_spans=include_spans)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            n_ents = len(out["entities"])
            total_entities += n_ents
            if n_ents > 0:
                passages_with_entities += 1
            for etype in ENTITY_TYPES:
                type_counts[etype] += len(out[f"entities_{etype.lower()}"])

            if total % 100000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  Converted {total:,} passages ({rate:.0f}/s)", flush=True)

            if args.limit and total >= args.limit:
                break

    elapsed = time.time() - t0
    print(f"\nDone: {total:,} passages in {elapsed:.1f}s ({total / elapsed:.0f}/s)")
    print(f"  Output: {args.output}")

    if validation_errors:
        print(f"  WARNING: {validation_errors} validation errors", file=sys.stderr)

    if args.stats:
        print("\n=== Entity Statistics ===")
        print(f"  Passages:          {total:,}")
        print(
            f"  With entities:     {passages_with_entities:,} ({passages_with_entities / max(1, total) * 100:.1f}%)"
        )
        print(f"  Total entities:    {total_entities:,}")
        print(f"  Avg per passage:   {total_entities / max(1, total):.1f}")
        print("\n  Per-type counts (unique per passage, summed):")
        for etype in ENTITY_TYPES:
            print(f"    {etype:15s}  {type_counts[etype]:>10,}")

    sys.exit(1 if validation_errors > 0 else 0)


if __name__ == "__main__":
    main()
