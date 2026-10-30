#!/usr/bin/env python3
"""Merge sharded FAISS embeddings into a single index.

Loads all embeddings_XXXX.npy and ids_XXXX.npy from a shard directory,
stacks them, builds a FAISS IndexFlatIP, and saves the final index.

This is the reduce step after parallel shard encoding via build_dense_index.py
with --shard-index/--total-shards flags.

Usage:
    python merge_faiss_shards.py \
        --shard-dir data/indexes/bitem-faiss-plain/shards \
        --output-dir data/indexes/bitem-faiss-plain \
        --index-type plain

Dependencies:
    pip install faiss-cpu numpy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

__all__ = ["main"]

# Nomic model prefixes, recorded in the build metadata as build_dense_index.py does.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Merge sharded FAISS embeddings into a single index.",
    )
    ap.add_argument(
        "--shard-dir",
        required=True,
        help="Directory containing embeddings_XXXX.npy and ids_XXXX.npy files.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for final FAISS index.",
    )
    ap.add_argument(
        "--index-type",
        choices=["plain", "enriched"],
        required=True,
        help="Index type (for metadata).",
    )
    ap.add_argument(
        "--model",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Model name (for metadata, default: nomic-embed-text-v1.5).",
    )

    args = ap.parse_args()

    import faiss
    import numpy as np

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sorted file names give shard order, which fixes the row order of the index.
    emb_files = sorted(shard_dir.glob("embeddings_*.npy"))
    ids_files = sorted(shard_dir.glob("ids_*.npy"))
    meta_files = sorted(shard_dir.glob("shard_meta_*.npy"))

    if not emb_files:
        raise FileNotFoundError(f"No embeddings_*.npy files found in {shard_dir}")
    if len(emb_files) != len(ids_files):
        raise ValueError(
            f"Mismatch: {len(emb_files)} embedding files vs {len(ids_files)} id files"
        )

    print("=== Merge FAISS Shards ===")
    print(f"Shard dir: {shard_dir}")
    print(f"Found {len(emb_files)} shards")

    t0 = time.time()
    all_embeddings = []
    all_ids = []
    total_shard_time = 0.0

    for emb_path, ids_path in zip(emb_files, ids_files):
        print(f"  Loading {emb_path.name}...", end=" ", flush=True)
        emb = np.load(emb_path)
        ids = np.load(ids_path)
        all_embeddings.append(emb)
        all_ids.append(ids)
        print(f"{emb.shape[0]:,} passages, {emb.shape}")

        # Shards encode in parallel, so the slowest one gives the encode time.
        shard_idx = emb_path.stem.split("_")[-1]
        shard_meta_path = shard_dir / f"shard_meta_{shard_idx}.json"
        if shard_meta_path.exists():
            with open(shard_meta_path) as f:
                sm = json.load(f)
                total_shard_time = max(total_shard_time, sm.get("encode_time_sec", 0))

    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    print("Stacking embeddings...", end=" ", flush=True)
    t1 = time.time()
    full_embeddings = np.vstack(all_embeddings)
    full_ids = np.concatenate(all_ids)
    stack_time = time.time() - t1
    dim = full_embeddings.shape[1]
    total_passages = full_embeddings.shape[0]
    print(
        f"{total_passages:,} passages x {dim} dim "
        f"({full_embeddings.nbytes / (1024**3):.2f} GB), {stack_time:.1f}s"
    )

    print("Building FAISS IndexFlatIP...", end=" ", flush=True)
    t2 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(full_embeddings)
    build_time = time.time() - t2
    print(f"{build_time:.1f}s")

    index_path = out_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"  Index saved: {index_path} ({index_size_mb:.0f} MB)")

    id_map_path = out_dir / "id_map.npy"
    np.save(id_map_path, full_ids)
    print(f"  ID map saved: {id_map_path} ({len(full_ids):,} entries)")

    # A leftover id_map.json would disagree with id_map.npy, the only ID map.
    old_json = out_dir / "id_map.json"
    if old_json.exists():
        old_json.unlink()
        print(f"  Removed old {old_json}")

    total_time = time.time() - t0
    meta = {
        "model": args.model,
        "dim": dim,
        "num_passages": int(total_passages),
        "index_type_faiss": "IndexFlatIP",
        "index_type": args.index_type,
        "normalization": "L2 (cosine via inner product)",
        "encode_time_sec": round(total_shard_time, 1),
        "merge_time_sec": round(total_time, 1),
        "num_shards": len(emb_files),
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
    print("Merge complete:")
    print(f"  Total passages: {total_passages:,}")
    print(f"  Dimension:      {dim}")
    print(f"  Index size:     {index_size_mb:.0f} MB")
    print(f"  Index type:     {args.index_type}")
    print(f"  Shards merged:  {len(emb_files)}")
    print(f"  Max encode time (parallel): {total_shard_time:.1f}s")
    print(f"  Merge wall time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
