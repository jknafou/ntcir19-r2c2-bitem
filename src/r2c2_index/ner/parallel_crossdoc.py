#!/usr/bin/env python3
"""Parallel cross-document refinement preprocessing.

Parallelizes the slow parts of cross-doc refinement:
1. aggregate_document_stats_v2() — JSON parsing + regex occurrence counting
2. cross_document_candidates() — sparse matrix co-occurrence

Uses multiprocessing to process shard files in parallel, then merges results.
The output is a candidates JSON that can be fed to the LLM review stage.

Usage:
    python parallel_crossdoc.py \
        --shards-dir .workqueue/output/shards \
        --documents documents.jsonl \
        --output-candidates candidates.json \
        --output-stats doc_stats.pkl \
        --workers 64 \
        --iqr-k 1.5
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Helper modules live in src/r2c2_index/ner/_lib/ and are imported by bare name.
_LIB_DIR = str(Path(__file__).resolve().parent / "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from build_document_prompt import load_documents_index  # noqa: E402

# Per-shard worker: aggregate_document_stats_v2 for one shard

# Word boundary cache (per-process)
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern] = {}


def _entity_key(e: dict) -> tuple:
    """Normalize entity to (text_lower, type) key."""
    return (e.get("text", "").lower(), e.get("type", ""))


def _expand_entities(entities: list[dict]) -> list[dict]:
    """Expand multi-label entities."""
    result = []
    for e in entities:
        types = e.get("types", [e.get("type", "")])
        if len(types) <= 1:
            result.append(e)
        else:
            for t in types:
                result.append({**e, "type": t, "types": [t]})
    return result


def process_shard(args_tuple):
    """Process one shard file → dict of doc_stats (same format as aggregate_document_stats_v2)."""
    shard_path, doc_meta, stage = args_tuple
    t0 = time.time()

    docs = {}
    total_passages = 0

    with open(shard_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            passage = json.loads(line)
            total_passages += 1

            doc_id = passage.get("doc_id", "")
            pid = passage.get("id", "")

            if doc_id not in docs:
                docs[doc_id] = {
                    "title": passage.get("title", ""),
                    "total_passages": 0,
                    "passage_ids": [],
                    "entities": {},
                    "text_parts": [],
                }

            doc = docs[doc_id]
            doc["total_passages"] += 1
            doc["passage_ids"].append(pid)

            contents = passage.get("contents", "")
            if contents:
                doc["text_parts"].append(contents)
            section = passage.get("section", "")
            if section:
                doc["text_parts"].append(section)

            for e in _expand_entities(passage.get("text_entities", [])):
                ekey = _entity_key(e)
                if ekey not in doc["entities"]:
                    doc["entities"][ekey] = {
                        "form": e,
                        "detected_count": 0,
                        "text_occurrences": 0,
                        "passage_ids": set(),
                    }
                doc["entities"][ekey]["detected_count"] += 1
                doc["entities"][ekey]["passage_ids"].add(pid)

    # Build full_text with metadata
    for doc_id, doc in docs.items():
        title = doc["title"]
        if title:
            doc["text_parts"].insert(0, title)

        meta = doc_meta.get(doc_id, {})
        infobox = meta.get("infobox")
        if infobox and infobox.get("params"):
            for key, value in infobox["params"].items():
                if isinstance(value, str) and value.strip():
                    doc["text_parts"].append(f"{key}: {value}")
        categories = meta.get("categories", [])
        if categories:
            doc["text_parts"].extend(categories)
        appearances = meta.get("appearances", [])
        if appearances:
            doc["text_parts"].extend(appearances)

        doc["full_text"] = "\n".join(doc["text_parts"])
        del doc["text_parts"]

    # Count text occurrences (CPU-heavy regex matching)
    for doc_id, doc in docs.items():
        full_text = doc["full_text"]

        for ekey, info in doc["entities"].items():
            entity_text = info["form"]["text"]
            if len(entity_text) >= 3:
                key = entity_text.lower()
                if key not in _WORD_BOUNDARY_CACHE:
                    _WORD_BOUNDARY_CACHE[key] = re.compile(
                        r"\b" + re.escape(entity_text) + r"\b", re.IGNORECASE
                    )
                info["text_occurrences"] = len(
                    _WORD_BOUNDARY_CACHE[key].findall(full_text)
                )
            else:
                info["text_occurrences"] = full_text.lower().count(entity_text.lower())

        # Subtract occurrences covered by longer entities
        entity_texts_sorted = sorted(
            doc["entities"].keys(),
            key=lambda ek: len(ek[0]),
            reverse=True,
        )
        for i, longer_key in enumerate(entity_texts_sorted):
            longer_text = longer_key[0]
            if len(longer_text) < 3:
                continue
            for shorter_key in entity_texts_sorted[i + 1 :]:
                shorter_text = shorter_key[0]
                if shorter_text in longer_text:
                    overlap = doc["entities"][longer_key]["text_occurrences"]
                    doc["entities"][shorter_key]["text_occurrences"] = max(
                        0, doc["entities"][shorter_key]["text_occurrences"] - overlap
                    )

    elapsed = time.time() - t0
    shard_name = os.path.basename(shard_path)
    return shard_name, docs, total_passages, elapsed


def merge_doc_stats(all_shard_results):
    """Merge doc_stats from all shards (docs are unique per shard due to splitting)."""
    merged = {}
    total_passages = 0
    for shard_name, docs, n_passages, elapsed in all_shard_results:
        total_passages += n_passages
        # Documents should be unique across shards (split by doc_id)
        for doc_id, doc in docs.items():
            if doc_id in merged:
                # Shouldn't happen with proper splitting, but handle gracefully
                existing = merged[doc_id]
                existing["passage_ids"].extend(doc["passage_ids"])
                existing["total_passages"] += doc["total_passages"]
                for ekey, info in doc["entities"].items():
                    if ekey not in existing["entities"]:
                        existing["entities"][ekey] = info
                    else:
                        existing["entities"][ekey]["detected_count"] += info[
                            "detected_count"
                        ]
                        existing["entities"][ekey]["passage_ids"].update(
                            info["passage_ids"]
                        )
                        existing["entities"][ekey]["text_occurrences"] += info[
                            "text_occurrences"
                        ]
            else:
                merged[doc_id] = doc
    return merged, total_passages


# Cross-document candidates (sparse, same as refine_entities.py)


def cross_document_candidates_sparse(
    doc_stats,
    min_freq=2,
    iqr_k=1.5,
    min_cond_prob=0.05,
    top_percentile=95,
    max_per_doc=50,
    min_cooccur=2,
    target_doc_ids=None,
):
    """Compute cross-doc candidates using sparse matrices.

    Returns the same format as cross_document_candidates in refine_entities.py.
    """
    import numpy as np
    from scipy import sparse as sp

    t0 = time.time()

    # Build entity -> docs index
    entity_to_docs = {}
    entity_form = {}
    for doc_id, doc in doc_stats.items():
        for ekey, info in doc["entities"].items():
            if ekey not in entity_to_docs:
                entity_to_docs[ekey] = set()
                entity_form[ekey] = info["form"]
            entity_to_docs[ekey].add(doc_id)

    total_entities = len(entity_to_docs)
    print(f"  {total_entities:,} unique entities across {len(doc_stats):,} documents")

    # Filter by min_freq
    entity_to_docs = {e: d for e, d in entity_to_docs.items() if len(d) >= min_freq}
    filtered = total_entities - len(entity_to_docs)
    print(f"  {filtered:,} entities removed (doc freq < {min_freq})")

    if not entity_to_docs:
        print("  No entities remain after frequency filter.")
        return {}

    # IQR outlier removal
    doc_counts = sorted(len(d) for d in entity_to_docs.values())
    n = len(doc_counts)
    q1 = doc_counts[n // 4]
    q3 = doc_counts[3 * n // 4]
    iqr = q3 - q1
    upper_fence = q3 + iqr_k * iqr

    outlier_entities = {e for e, d in entity_to_docs.items() if len(d) > upper_fence}
    print(f"  IQR stats: Q1={q1}, Q3={q3}, IQR={iqr}, upper fence={upper_fence:.1f}")
    print(
        f"  {len(outlier_entities):,} outlier entities removed (doc freq > {upper_fence:.1f})"
    )

    if outlier_entities:
        outlier_sorted = sorted(
            outlier_entities, key=lambda e: len(entity_to_docs[e]), reverse=True
        )
        print("  Top outliers:")
        for ekey in outlier_sorted[:20]:
            text, etype = ekey
            print(
                f"    {text} [{etype}]{' ' * max(0, 50 - len(text) - len(etype) - 3)}docs={len(entity_to_docs[ekey]):,}"
            )

    valid_entities = set(entity_to_docs.keys()) - outlier_entities

    # Sparse matrix computation
    print("  Computing conditional probabilities (sparse)...")
    t1 = time.time()

    entity_list = sorted(valid_entities)
    entity_idx = {e: i for i, e in enumerate(entity_list)}
    doc_list = list(doc_stats.keys())
    doc_idx = {d: i for i, d in enumerate(doc_list)}
    n_ents = len(entity_list)
    n_docs = len(doc_list)

    rows, cols = [], []
    for doc_id, doc in doc_stats.items():
        d = doc_idx[doc_id]
        for ekey in doc["entities"]:
            if ekey in entity_idx:
                rows.append(d)
                cols.append(entity_idx[ekey])

    data = np.ones(len(rows), dtype=np.float32)
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, n_ents))
    nnz = M.nnz
    mem_sparse_mb = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1e6
    mem_dense_mb = n_docs * n_ents * 4 / 1e6
    print(
        f"  Matrix: {n_docs:,} × {n_ents:,} (sparse: {mem_sparse_mb:.0f} MB, {nnz:,} nnz; dense would be {mem_dense_mb:,.0f} MB)"
    )

    co_occur = (M.T @ M).tocsr()
    doc_count = np.asarray(M.sum(axis=0)).ravel()
    safe_count = np.maximum(doc_count, 1.0)
    inv_count = sp.diags(1.0 / safe_count)
    P = (inv_count @ co_occur).tocsr()

    mem_co = (
        co_occur.data.nbytes + co_occur.indices.nbytes + co_occur.indptr.nbytes
    ) / 1e6
    mem_P = (P.data.nbytes + P.indices.nbytes + P.indptr.nbytes) / 1e6
    print(
        f"  Co-occur: {co_occur.nnz:,} nnz ({mem_co:.0f} MB), P: {P.nnz:,} nnz ({mem_P:.0f} MB), {time.time() - t1:.1f}s"
    )

    # Memory tracking
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"  Process RSS: {rss:,.0f} MB")
    except ImportError:
        pass

    # Per-doc candidates
    candidate_doc_ids = (
        [d for d in doc_list if d in target_doc_ids] if target_doc_ids else doc_list
    )
    if target_doc_ids:
        print(f"  Target docs: {len(candidate_doc_ids):,} (of {len(doc_list):,})")

    doc_candidates = {}
    all_max_probs = []

    for doc_id in candidate_doc_ids:
        d = doc_idx[doc_id]
        d_ent_indices = M[d].indices
        if len(d_ent_indices) == 0:
            continue

        P_sub = P[d_ent_indices, :]
        max_probs = np.asarray(P_sub.max(axis=0).todense()).ravel()
        max_probs[d_ent_indices] = 0.0

        if min_cooccur > 1:
            co_sub = co_occur[d_ent_indices, :]
            max_cooccur = np.asarray(co_sub.max(axis=0).todense()).ravel()
            max_probs[max_cooccur < min_cooccur] = 0.0

        nonzero = np.nonzero(max_probs)[0]
        if len(nonzero) > 0:
            doc_candidates[doc_id] = {
                entity_list[i]: float(max_probs[i]) for i in nonzero
            }
            all_max_probs.extend(max_probs[nonzero].tolist())

    print(
        f"  Pass 1: {len(all_max_probs):,} (doc, candidate) pairs in {time.time() - t1:.1f}s"
    )

    if not all_max_probs:
        print("  No cross-document candidates found.")
        return {}

    # Adaptive threshold
    all_max_probs.sort()
    pct_idx = min(
        int(len(all_max_probs) * top_percentile / 100.0), len(all_max_probs) - 1
    )
    adaptive_threshold = all_max_probs[pct_idx]
    threshold = max(adaptive_threshold, min_cond_prob)
    print(
        f"  Threshold: P{top_percentile:.0f}={adaptive_threshold:.4f}, floor={min_cond_prob:.4f}, final={threshold:.4f}"
    )

    result = {}
    total_candidates = 0
    for doc_id, cands in doc_candidates.items():
        above = [(c, p) for c, p in cands.items() if p >= threshold]
        if not above:
            continue
        above.sort(key=lambda x: x[1], reverse=True)
        above = above[:max_per_doc]

        fn_entities = {}
        for c_key, _ in above:
            fn_entities[c_key] = entity_form.get(
                c_key, {"text": c_key[0], "type": c_key[1]}
            )

        fn_passages = {}
        for pid in doc_stats[doc_id]["passage_ids"]:
            fn_passages[pid] = list(fn_entities.keys())

        result[doc_id] = {
            "fp_entities": {},
            "fn_entities": fn_entities,
            "fp_passages": {},
            "fn_passages": fn_passages,
        }
        total_candidates += len(fn_entities)

    elapsed = time.time() - t0
    print(
        f"  {len(result):,} docs with candidates, {total_candidates:,} total in {elapsed:.1f}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Parallel cross-doc preprocessing")
    parser.add_argument(
        "--shards-dir", required=True, help="Directory with shard extraction outputs"
    )
    parser.add_argument("--documents", required=True, help="Path to documents.jsonl")
    parser.add_argument(
        "--output-candidates", required=True, help="Output candidates JSON"
    )
    parser.add_argument(
        "--output-stats", default=None, help="Output doc_stats pickle (optional)"
    )
    parser.add_argument(
        "--workers", type=int, default=64, help="Number of parallel workers"
    )
    parser.add_argument(
        "--iqr-k", type=float, default=1.5, help="IQR multiplier for outlier removal"
    )
    parser.add_argument(
        "--min-freq", type=int, default=2, help="Min document frequency"
    )
    parser.add_argument(
        "--min-cooccur", type=int, default=2, help="Min co-occurrence count"
    )
    parser.add_argument(
        "--top-pct", type=float, default=95, help="Top percentile for threshold"
    )
    parser.add_argument(
        "--min-prob", type=float, default=5, help="Min conditional probability (%)"
    )
    parser.add_argument(
        "--max-per-doc", type=int, default=50, help="Max candidates per doc"
    )
    parser.add_argument(
        "--stage",
        default="extraction",
        help="Which stage output to read (extraction or refined)",
    )
    parser.add_argument(
        "--target-input", default=None, help="Input JSONL to restrict target doc_ids"
    )
    args = parser.parse_args()

    t0 = time.time()

    # Find shard extraction files
    pattern = os.path.join(args.shards_dir, f"shard_*/{args.stage}.jsonl")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        print(f"ERROR: No shard files found: {pattern}")
        sys.exit(1)
    print(f"Found {len(shard_files)} shard files")

    # Load document metadata (shared across workers via fork)
    print("Loading document metadata...")
    doc_meta = load_documents_index(args.documents)
    print(f"  {len(doc_meta):,} documents loaded")

    # Parallel aggregation
    print(f"\nStep 1: Parallel document stats ({args.workers} workers)...")
    t1 = time.time()
    worker_args = [(f, doc_meta, args.stage) for f in shard_files]

    with Pool(processes=min(args.workers, len(shard_files))) as pool:
        results = pool.map(process_shard, worker_args)

    # Report per-shard timing
    for shard_name, _, n_passages, elapsed in results:
        print(f"  {shard_name}: {n_passages:,} passages in {elapsed:.1f}s")

    print("\nMerging shard results...")
    doc_stats, total_passages = merge_doc_stats(results)
    t1_elapsed = time.time() - t1
    print(
        f"  {len(doc_stats):,} documents, {total_passages:,} passages in {t1_elapsed:.1f}s"
    )

    # Memory tracking
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"  Process RSS after merge: {rss:,.0f} MB")
    except ImportError:
        pass

    # Save doc_stats if requested
    if args.output_stats:
        print(f"\nSaving doc_stats to {args.output_stats}...")
        with open(args.output_stats, "wb") as f:
            pickle.dump(doc_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        stats_size = os.path.getsize(args.output_stats) / 1e6
        print(f"  {stats_size:.0f} MB")

    # Determine target doc_ids
    target_doc_ids = None
    if args.target_input:
        print(f"\nLoading target doc_ids from {args.target_input}...")
        target_doc_ids = set()
        with open(args.target_input) as f:
            for line in f:
                target_doc_ids.add(json.loads(line)["doc_id"])
        print(f"  {len(target_doc_ids):,} target documents")

    print("\nStep 2: Cross-document candidate detection (sparse)...")
    candidates = cross_document_candidates_sparse(
        doc_stats,
        min_freq=args.min_freq,
        iqr_k=args.iqr_k,
        min_cond_prob=args.min_prob / 100.0,
        top_percentile=args.top_pct,
        max_per_doc=args.max_per_doc,
        min_cooccur=args.min_cooccur,
        target_doc_ids=target_doc_ids,
    )

    # Save candidates (JSON-serializable format)
    print(f"\nSaving candidates to {args.output_candidates}...")
    # Convert tuple keys to strings for JSON
    serializable = {}
    for doc_id, cand in candidates.items():
        serializable[doc_id] = {
            "fn_entities": {
                f"{k[0]}||{k[1]}": v for k, v in cand["fn_entities"].items()
            },
            "fn_passages": {
                pid: [f"{k[0]}||{k[1]}" for k in keys]
                for pid, keys in cand["fn_passages"].items()
            },
        }
    with open(args.output_candidates, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - t0
    print(f"\nTotal: {total_elapsed:.1f}s")
    print(f"  Documents with candidates: {len(candidates):,}")
    total_cands = sum(len(c["fn_entities"]) for c in candidates.values())
    print(f"  Total candidate entities: {total_cands:,}")

    # Final memory
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"  Final RSS: {rss:,.0f} MB")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
