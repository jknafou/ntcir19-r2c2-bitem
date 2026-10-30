#!/usr/bin/env python3
"""Build FAISS dense vector index from chunked passages.

Encodes all passages using nomic-embed-text-v1.5 (768-dim) and builds a flat
FAISS index for cosine similarity search. Designed to run on GPU (A100) for
fast encoding.

Supports two index types:
  - plain:    "{title} | {section} | {contents}" with search_document: prefix
  - enriched: entity-prefixed text with type-sorted entities

Modes:
  1. Full build (default): encodes passages → builds FAISS index in one shot.
  2. Shard encode (--shard-index + --total-shards): encodes a subset of passages
     and saves raw embeddings + ids as .npy files. Use merge_faiss_shards.py to
     combine shards into the final index.

Full build saves:
  1. faiss.index    -- FAISS index (flat inner product on L2-normalized vectors)
  2. id_map.npy     -- NumPy array mapping FAISS integer IDs to passage string IDs
  3. build_meta.json -- Build metadata (model, dim, count, timing, prefixes)

Shard encode saves:
  1. embeddings_XXXX.npy -- float32 embeddings for this shard
  2. ids_XXXX.npy        -- string passage IDs for this shard
  3. shard_meta_XXXX.json -- Shard metadata (count, timing, shard info)

Passage content is NOT stored here -- the dense MCP server looks up full passage
data from Elasticsearch.

Usage:
    # Plain index (full build):
    python build_dense_index.py \
        --passages data/chunks-v3/passages.jsonl \
        --output-dir data/indexes/bitem-faiss-plain \
        --model nomic-ai/nomic-embed-text-v1.5 \
        --index-type plain \
        --batch-size 1024 --device cuda

    # Shard encode (1 of 4 shards):
    python build_dense_index.py \
        --passages data/chunks-v3/passages.jsonl \
        --output-dir data/indexes/bitem-faiss-plain/shards \
        --model nomic-ai/nomic-embed-text-v1.5 \
        --index-type plain \
        --shard-index 0 --total-shards 4 \
        --batch-size 1024 --device cuda

    # Enriched index:
    python build_dense_index.py \
        --passages data/chunks-v3/passages_enriched_v3.jsonl \
        --output-dir data/indexes/bitem-faiss-enriched \
        --model nomic-ai/nomic-embed-text-v1.5 \
        --index-type enriched \
        --entity-type-map data/chunks-v3/entity_type_map.json \
        --batch-size 1024 --device cuda

Dependencies:
    pip install sentence-transformers faiss-cpu numpy
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

__all__ = ["main"]

# Nomic model prefixes (required for nomic-embed-text-v1.5)
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Entity type priority for enriched text construction
TYPE_PRIORITY = {
    "CHARACTER": 0,
    "LOCATION": 1,
    "EVENT": 2,
    "ORGANIZATION": 3,
    "PERSON": 4,
    "WORK": 5,
    "DATE": 6,
}


def _count_lines(input_path: Path) -> int:
    """Count total lines in a file efficiently.

    Args:
        input_path: Path to file.

    Returns:
        Number of lines.
    """
    count = 0
    with open(input_path, "rb") as f:
        for _ in f:
            count += 1
    return count


def _shard_range(
    total_lines: int,
    shard_index: int,
    total_shards: int,
) -> tuple[int, int]:
    """Compute (start_line, end_line) for a shard (0-indexed, exclusive end).

    Distributes remainder lines to first shards so all shards differ by at most 1.

    Args:
        total_lines: Total number of lines in the file.
        shard_index: 0-based shard index.
        total_shards: Total number of shards.

    Returns:
        (start_line, end_line) tuple, 0-indexed, end exclusive.
    """
    base = total_lines // total_shards
    remainder = total_lines % total_shards
    start = base * shard_index + min(shard_index, remainder)
    end = start + base + (1 if shard_index < remainder else 0)
    return start, end


def _stream_passages_plain(
    input_path: Path,
    limit: int = 0,
) -> Iterator[tuple[str, str]]:
    """Stream (passage_id, text) pairs for plain index.

    Text format: "search_document: {title} | {section} | {contents}"

    Args:
        input_path: Path to passages.jsonl.
        limit: Max passages (0=all).

    Yields:
        (passage_id, prefixed_text) tuples.
    """
    count = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title", "")
            section = doc.get("section", "")
            contents = doc.get("contents", "")
            text = f"{DOCUMENT_PREFIX}{title} | {section} | {contents}"
            yield doc["id"], text
            count += 1
            if limit and count >= limit:
                return


def _stream_passages_enriched(
    input_path: Path,
    entity_type_map: dict[str, str],
    limit: int = 0,
) -> Iterator[tuple[str, str]]:
    """Stream (passage_id, text) pairs for enriched index.

    Text format: "search_document: {sorted_entities}. {title} | {section} | {contents}"
    Entities sorted by type priority.

    Args:
        input_path: Path to passages_enriched_v3.jsonl.
        entity_type_map: Mapping entity_name -> primary_type.
        limit: Max passages (0=all).

    Yields:
        (passage_id, prefixed_text) tuples.
    """
    count = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title", "")
            section = doc.get("section", "")
            contents = doc.get("contents", "")

            # Only the first 10 entities are kept, ordered by type priority.
            text_entities = doc.get("text_entities", [])[:10]
            sorted_entities = sorted(
                text_entities,
                key=lambda e: TYPE_PRIORITY.get(entity_type_map.get(e, ""), 99),
            )
            prefix = ". ".join(sorted_entities) + ". " if sorted_entities else ""
            text = f"{DOCUMENT_PREFIX}{prefix}{title} | {section} | {contents}"
            yield doc["id"], text
            count += 1
            if limit and count >= limit:
                return


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Build FAISS dense index from passages.",
    )
    ap.add_argument(
        "--passages",
        required=True,
        help="Path to passages.jsonl (plain or enriched).",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for FAISS index and metadata.",
    )
    ap.add_argument(
        "--model",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Sentence-transformers model for encoding (default: nomic-embed-text-v1.5).",
    )
    ap.add_argument(
        "--index-type",
        choices=["plain", "enriched"],
        default="plain",
        help="Index type: plain (title|section|contents) or enriched (entity-prefixed).",
    )
    ap.add_argument(
        "--entity-type-map",
        default=None,
        help="Path to entity_type_map.json (required for enriched index type).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Encoding batch size (default: 1024).",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for encoding (default: cuda).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of passages (0=all, for testing).",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index (0-based) for multi-GPU parallel encoding.",
    )
    ap.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of shards for multi-GPU parallel encoding.",
    )

    args = ap.parse_args()

    if (args.shard_index is None) != (args.total_shards is None):
        ap.error("--shard-index and --total-shards must be used together")
    if args.shard_index is not None:
        if args.shard_index < 0 or args.shard_index >= args.total_shards:
            ap.error(
                f"--shard-index must be in [0, {args.total_shards - 1}], "
                f"got {args.shard_index}"
            )

    if args.index_type == "enriched" and args.entity_type_map is None:
        ap.error("--entity-type-map is required for enriched index type")

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.passages)

    entity_type_map: dict[str, str] = {}
    if args.index_type == "enriched":
        print(f"Loading entity type map from {args.entity_type_map}...")
        with open(args.entity_type_map, encoding="utf-8") as f:
            entity_type_map = json.load(f)
        print(f"  {len(entity_type_map)} entity->type mappings")

    print(f"Loading model: {args.model} (device={args.device})...")
    model = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
    )
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded: dim={dim}")

    # A shard covers a contiguous line range of the input; a full build covers all of it.
    is_sharded = args.shard_index is not None
    shard_label = ""
    start_line, end_line = 0, 0

    if is_sharded:
        total_lines = _count_lines(input_path)
        start_line, end_line = _shard_range(
            total_lines, args.shard_index, args.total_shards
        )
        shard_label = f"{args.shard_index:04d}"
        print(
            f"Shard mode: {args.shard_index}/{args.total_shards} "
            f"(lines {start_line:,}-{end_line:,} of {total_lines:,})"
        )

    print(
        f"Encoding passages (index_type={args.index_type}, batch_size={args.batch_size})..."
    )

    if args.index_type == "enriched":
        full_stream = _stream_passages_enriched(
            input_path,
            entity_type_map,
            limit=args.limit,
        )
    else:
        full_stream = _stream_passages_plain(input_path, limit=args.limit)

    # Keep only the lines that belong to this shard.
    if is_sharded:
        passage_stream = (
            item for i, item in enumerate(full_stream) if start_line <= i < end_line
        )
    else:
        passage_stream = full_stream

    # Embeddings are held in memory: both modes write them out in one piece.
    all_embeddings: list[np.ndarray] = []
    all_ids: list[str] = []

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    t0 = time.time()
    total_encoded = 0

    for pid, text in passage_stream:
        batch_ids.append(pid)
        batch_texts.append(text)

        if len(batch_texts) >= args.batch_size:
            embeddings = model.encode(
                batch_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings_np = np.array(embeddings, dtype=np.float32)
            all_embeddings.append(embeddings_np)
            all_ids.extend(batch_ids)
            total_encoded += len(batch_texts)

            if total_encoded % (args.batch_size * 10) == 0:
                elapsed = time.time() - t0
                rate = total_encoded / elapsed if elapsed > 0 else 0
                print(f"  {total_encoded:>10,} passages encoded ({rate:,.0f}/s)")

            batch_ids = []
            batch_texts = []

    # Encode the last, partial batch.
    if batch_texts:
        embeddings = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings_np = np.array(embeddings, dtype=np.float32)
        all_embeddings.append(embeddings_np)
        all_ids.extend(batch_ids)
        total_encoded += len(batch_texts)

    encode_time = time.time() - t0
    rate = total_encoded / encode_time if encode_time > 0 else 0
    print(f"  Encoding complete: {encode_time:.1f}s ({rate:.0f} passages/sec)")

    full_embeddings = (
        np.vstack(all_embeddings)
        if all_embeddings
        else np.empty((0, dim), dtype=np.float32)
    )

    if is_sharded:
        # A shard saves raw embeddings and IDs; merge_faiss_shards.py builds the index.
        emb_path = out_dir / f"embeddings_{shard_label}.npy"
        ids_path = out_dir / f"ids_{shard_label}.npy"
        meta_path = out_dir / f"shard_meta_{shard_label}.json"

        np.save(emb_path, full_embeddings)
        emb_size_mb = emb_path.stat().st_size / (1024 * 1024)
        print(f"  Embeddings saved: {emb_path} ({emb_size_mb:.0f} MB)")

        id_map_np = np.array(all_ids, dtype="U25")
        np.save(ids_path, id_map_np)
        print(f"  IDs saved: {ids_path} ({len(all_ids):,} entries)")

        meta = {
            "model": args.model,
            "dim": dim,
            "num_passages": total_encoded,
            "shard_index": args.shard_index,
            "total_shards": args.total_shards,
            "start_line": start_line,
            "end_line": end_line,
            "index_type": args.index_type,
            "encode_time_sec": round(encode_time, 1),
            "passages_per_sec": round(rate) if encode_time > 0 else 0,
            "device": args.device,
            "batch_size": args.batch_size,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Shard metadata: {meta_path}")

        print()
        print(f"Shard {args.shard_index}/{args.total_shards} complete:")
        print(f"  Passages: {total_encoded:,}")
        print(f"  Time:     {encode_time:.1f}s on {args.device}")
        print(f"  Output:   {out_dir}")
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(full_embeddings)

        index_path = out_dir / "faiss.index"
        faiss.write_index(index, str(index_path))
        index_size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"  Index saved: {index_path} ({index_size_mb:.0f} MB)")

        id_map_np = np.array(all_ids, dtype="U25")
        id_map_path = out_dir / "id_map.npy"
        np.save(id_map_path, id_map_np)
        print(f"  ID map saved: {id_map_path} ({len(all_ids):,} entries, dtype=U25)")

        # A leftover id_map.json would disagree with id_map.npy, the only ID map.
        old_json = out_dir / "id_map.json"
        if old_json.exists():
            old_json.unlink()
            print(f"  Removed old {old_json}")

        meta = {
            "model": args.model,
            "dim": dim,
            "num_passages": total_encoded,
            "index_type_faiss": "IndexFlatIP",
            "index_type": args.index_type,
            "normalization": "L2 (cosine via inner product)",
            "encode_time_sec": round(encode_time, 1),
            "passages_per_sec": round(rate) if encode_time > 0 else 0,
            "device": args.device,
            "batch_size": args.batch_size,
            "document_prefix": DOCUMENT_PREFIX,
            "query_prefix": QUERY_PREFIX,
            "metric": "cosine (via IP on L2-normalized)",
            "dtype": "float32",
        }
        meta_path = out_dir / "build_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Build metadata: {meta_path}")

        print()
        print("Dense index built successfully:")
        print(f"  Passages:   {total_encoded:,}")
        print(f"  Dimension:  {dim}")
        print(f"  Index:      {index_path} ({index_size_mb:.0f} MB)")
        print(f"  Index type: {args.index_type}")
        print(f"  Encoding:   {encode_time:.1f}s on {args.device}")


if __name__ == "__main__":
    main()
