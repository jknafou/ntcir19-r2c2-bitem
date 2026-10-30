#!/usr/bin/env python3
"""Fast parallel cross-document candidate computation.

Steps:
  1. Copy inputs to /tmp (SSD) for fast I/O
  2. Split corpus into N chunks using line count
  3. Fork N workers, each processes its chunk (JSON parse + regex)
  4. Merge doc_stats from all workers
  5. Sparse matrix co-occurrence → candidates
  6. Write candidates JSON

Designed for 100-192 cores on HPC bigmem/high-CPU nodes.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Helper modules live in src/r2c2_index/ner/_lib/ and are imported by bare name.
_LIB_DIR = str(Path(__file__).resolve().parent / "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from build_document_prompt import load_documents_index  # noqa: E402

# Worker function: process one chunk file
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern] = {}
_DOC_META: dict = {}  # Loaded in parent before fork → CoW shared with all workers

MAX_ENTITIES_PER_DOC = 500  # Cap to bound O(E²) overlap subtraction


def _entity_key(e: dict) -> tuple:
    return (e.get("text", "").lower(), e.get("type", ""))


def _expand_entities(entities: list[dict]) -> list[dict]:
    result = []
    for e in entities:
        types = e.get("types", [e.get("type", "")])
        if len(types) <= 1:
            result.append(e)
        else:
            for t in types:
                result.append({**e, "type": t, "types": [t]})
    return result


def process_chunk(chunk_path: str) -> tuple[dict, int, float]:
    """Process one chunk → (doc_stats_dict, passage_count, elapsed)."""
    t0 = time.time()
    docs: dict[str, dict] = {}
    total_passages = 0

    with open(chunk_path, encoding="utf-8") as f:
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

        meta = _DOC_META.get(doc_id, {})
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

    # Regex occurrence counting + overlap subtraction
    for doc_id, doc in docs.items():
        # Cap entities per doc to bound O(E²) overlap loop
        if len(doc["entities"]) > MAX_ENTITIES_PER_DOC:
            top_keys = sorted(
                doc["entities"].keys(),
                key=lambda k: doc["entities"][k]["detected_count"],
                reverse=True,
            )[:MAX_ENTITIES_PER_DOC]
            doc["entities"] = {k: doc["entities"][k] for k in top_keys}

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

        # Subtract longer-entity overlaps (O(E²) but capped at MAX_ENTITIES_PER_DOC)
        sorted_keys = sorted(
            doc["entities"].keys(), key=lambda ek: len(ek[0]), reverse=True
        )
        for i, longer_key in enumerate(sorted_keys):
            longer_text = longer_key[0]
            if len(longer_text) < 3:
                continue
            for shorter_key in sorted_keys[i + 1 :]:
                shorter_text = shorter_key[0]
                if shorter_text in longer_text:
                    overlap = doc["entities"][longer_key]["text_occurrences"]
                    doc["entities"][shorter_key]["text_occurrences"] = max(
                        0, doc["entities"][shorter_key]["text_occurrences"] - overlap
                    )

    return docs, total_passages, time.time() - t0


# Module-level globals for candidate pool workers (set before fork in main)
_P_MATRIX = None
_CO_MATRIX = None
_M_MATRIX = None
_ENTITY_LIST: list = []
_DOC_IDX: dict = {}
_MIN_COOCCUR: int = 2


def _process_doc_batch(doc_ids_batch):
    """Process a batch of doc_ids → list of (doc_id, {entity: prob}) pairs."""
    import numpy as _np

    results_batch = []
    for doc_id in doc_ids_batch:
        d = _DOC_IDX[doc_id]
        d_ent_indices = _M_MATRIX[d].indices
        if len(d_ent_indices) == 0:
            continue

        max_probs = _np.asarray(
            _P_MATRIX[d_ent_indices, :].max(axis=0).todense()
        ).ravel()
        max_probs[d_ent_indices] = 0.0

        if _MIN_COOCCUR > 1:
            max_cooccur = _np.asarray(
                _CO_MATRIX[d_ent_indices, :].max(axis=0).todense()
            ).ravel()
            max_probs[max_cooccur < _MIN_COOCCUR] = 0.0

        nonzero = _np.nonzero(max_probs)[0]
        if len(nonzero) > 0:
            cands = {_ENTITY_LIST[i]: float(max_probs[i]) for i in nonzero}
            results_batch.append((doc_id, cands))
    return results_batch


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel cross-doc preprocessing"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Extraction corpus JSONL (or dir with shard subdirs)",
    )
    parser.add_argument("--documents", required=True, help="Path to documents.jsonl")
    parser.add_argument("--output", required=True, help="Output candidates JSON")
    parser.add_argument("--workers", type=int, default=100, help="Parallel workers")
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--min-cooccur", type=int, default=2)
    parser.add_argument("--top-pct", type=float, default=95)
    parser.add_argument("--min-prob", type=float, default=5)
    parser.add_argument("--max-per-doc", type=int, default=50)
    parser.add_argument(
        "--target-input", default=None, help="Restrict candidates to these doc_ids"
    )
    parser.add_argument(
        "--tmpdir", default="/tmp/fast_crossdoc", help="Local SSD staging dir"
    )
    args = parser.parse_args()

    t_total = time.time()
    tmpdir = args.tmpdir
    os.makedirs(tmpdir, exist_ok=True)

    print("=== Step 0: Copy to local SSD ===")
    t0 = time.time()

    # Determine corpus file(s)
    if os.path.isdir(args.corpus):
        # Shards dir: cat all shard extraction files into one
        import glob

        shard_files = sorted(
            glob.glob(os.path.join(args.corpus, "shard_*/extraction.jsonl"))
        )
        if not shard_files:
            shard_files = sorted(
                glob.glob(os.path.join(args.corpus, "*/extraction.jsonl"))
            )
        corpus_local = os.path.join(tmpdir, "corpus.jsonl")
        print(f"  Concatenating {len(shard_files)} shard files to {corpus_local}...")
        with open(corpus_local, "w") as out:
            for sf in shard_files:
                with open(sf) as inp:
                    shutil.copyfileobj(inp, out)
    else:
        corpus_local = os.path.join(tmpdir, "corpus.jsonl")
        print(f"  Copying {args.corpus} to {corpus_local}...")
        shutil.copy2(args.corpus, corpus_local)

    corpus_lines = int(subprocess.check_output(["wc", "-l", corpus_local]).split()[0])
    corpus_size = os.path.getsize(corpus_local) / 1e6
    print(f"  Corpus: {corpus_lines:,} lines, {corpus_size:.0f} MB")

    # Load doc_meta into parent memory — workers inherit via fork CoW (zero I/O per worker)
    global _DOC_META
    print("  Loading documents metadata into parent memory...")
    _DOC_META = load_documents_index(args.documents)
    print(f"  {len(_DOC_META):,} documents loaded (shared via CoW after fork)")
    print(f"  SSD copy + metadata: {time.time() - t0:.1f}s")

    print(f"\n=== Step 1: Split into {args.workers} chunks (streaming) ===")
    t1 = time.time()

    chunk_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # Pass 1: find line numbers where new docs start
    doc_boundaries: list[int] = [0]
    current_doc = None
    t_scan = time.time()
    with open(corpus_local) as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                print(
                    f"  Scanning doc boundaries: {i:,}/{corpus_lines:,} lines...",
                    flush=True,
                )
            doc_id = json.loads(line).get("doc_id", "")
            if doc_id != current_doc:
                if current_doc is not None:
                    doc_boundaries.append(i)
                current_doc = doc_id
    doc_boundaries.append(corpus_lines)
    print(
        f"  Doc boundary scan: {len(doc_boundaries) - 1:,} docs in {time.time() - t_scan:.1f}s"
    )

    n_docs = len(doc_boundaries) - 1
    docs_per_chunk = max(1, n_docs // args.workers)

    chunk_ranges = []
    for ci in range(0, n_docs, docs_per_chunk):
        start_line = doc_boundaries[ci]
        end_doc = min(ci + docs_per_chunk, n_docs)
        end_line = doc_boundaries[end_doc]
        chunk_ranges.append((start_line, end_line))

    # Pass 2: stream-write chunks (constant memory, no readlines)
    chunk_files = []
    with open(corpus_local) as f:
        line_no = 0
        for chunk_idx, (start_line, end_line) in enumerate(chunk_ranges):
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.jsonl")
            chunk_files.append(chunk_path)
            with open(chunk_path, "w") as out:
                while line_no < end_line:
                    line = f.readline()
                    if line_no >= start_line:
                        out.write(line)
                    line_no += 1

    actual_workers = len(chunk_files)
    print(
        f"  {actual_workers} chunks, ~{docs_per_chunk} docs each, {time.time() - t1:.1f}s"
    )

    # Forked workers share parent memory; imap_unordered lets progress print as chunks finish.
    print(f"\n=== Step 2: Parallel doc stats ({actual_workers} workers) ===")
    t2 = time.time()

    results = []
    completed = 0

    with Pool(processes=min(actual_workers, args.workers)) as pool:
        for docs, n_pass, elapsed in pool.imap_unordered(process_chunk, chunk_files):
            completed += 1
            elapsed_total = time.time() - t2
            rate = completed / elapsed_total if elapsed_total > 0 else 0
            eta = (len(chunk_files) - completed) / rate if rate > 0 else 0
            print(
                f"  [{completed:3d}/{len(chunk_files)}] "
                f"{len(docs):,} docs, {n_pass:,} passages in {elapsed:.1f}s "
                f"| {elapsed_total:.0f}s elapsed, ETA {eta:.0f}s",
                flush=True,
            )
            results.append((docs, n_pass, elapsed))

    total_passages = sum(n for _, n, _ in results)
    print(f"  Total: {total_passages:,} passages in {time.time() - t2:.1f}s")

    print("\n=== Step 3: Merge ===")
    t3 = time.time()
    merged: dict[str, dict] = {}
    for docs, _, _ in results:
        merged.update(docs)  # Docs are unique per chunk (split by doc boundary)
    del results
    print(f"  {len(merged):,} documents in {time.time() - t3:.1f}s")

    # Memory check
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"  RSS: {rss:,.0f} MB")
    except ImportError:
        pass

    print("\n=== Step 4: Cross-doc candidates (sparse) ===")
    import numpy as np
    from scipy import sparse as sp

    t4 = time.time()

    # Entity → docs index
    entity_to_docs: dict[tuple, set] = {}
    entity_form: dict[tuple, dict] = {}
    for doc_id, doc in merged.items():
        for ekey, info in doc["entities"].items():
            if ekey not in entity_to_docs:
                entity_to_docs[ekey] = set()
                entity_form[ekey] = info["form"]
            entity_to_docs[ekey].add(doc_id)

    total_entities = len(entity_to_docs)
    print(f"  {total_entities:,} unique entities across {len(merged):,} documents")

    # Frequency filter
    entity_to_docs = {
        e: d for e, d in entity_to_docs.items() if len(d) >= args.min_freq
    }
    print(
        f"  {total_entities - len(entity_to_docs):,} removed (freq < {args.min_freq})"
    )

    if not entity_to_docs:
        print("  No entities remain. Writing empty candidates.")
        with open(args.output, "w") as f:
            json.dump({}, f)
        return

    # IQR outlier removal
    doc_counts = sorted(len(d) for d in entity_to_docs.values())
    n = len(doc_counts)
    q1 = doc_counts[n // 4]
    q3 = doc_counts[3 * n // 4]
    iqr = q3 - q1
    upper_fence = q3 + args.iqr_k * iqr

    outliers = {e for e, d in entity_to_docs.items() if len(d) > upper_fence}
    print(f"  IQR: Q1={q1}, Q3={q3}, IQR={iqr}, fence={upper_fence:.1f}")
    print(f"  {len(outliers):,} outliers removed (freq > {upper_fence:.1f})")

    if outliers:
        for ekey in sorted(
            outliers, key=lambda e: len(entity_to_docs[e]), reverse=True
        )[:10]:
            print(f"    {ekey[0]} [{ekey[1]}]  docs={len(entity_to_docs[ekey]):,}")

    valid = set(entity_to_docs.keys()) - outliers

    print("  Building sparse matrix...")
    entity_list = sorted(valid)
    entity_idx = {e: i for i, e in enumerate(entity_list)}
    doc_list = list(merged.keys())
    doc_idx = {d: i for i, d in enumerate(doc_list)}
    n_ents = len(entity_list)
    n_docs = len(doc_list)

    rows, cols = [], []
    for doc_id, doc in merged.items():
        d = doc_idx[doc_id]
        for ekey in doc["entities"]:
            if ekey in entity_idx:
                rows.append(d)
                cols.append(entity_idx[ekey])

    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_docs, n_ents)
    )
    del rows, cols

    mem_M = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1e6
    print(
        f"  M: {n_docs:,} × {n_ents:,}, {M.nnz:,} nnz, {mem_M:.0f} MB (dense would be {n_docs * n_ents * 4 / 1e9:.1f} GB)"
    )

    co_occur = (M.T @ M).tocsr()
    doc_count = np.asarray(M.sum(axis=0)).ravel()
    P = (sp.diags(1.0 / np.maximum(doc_count, 1.0)) @ co_occur).tocsr()

    mem_co = (
        co_occur.data.nbytes + co_occur.indices.nbytes + co_occur.indptr.nbytes
    ) / 1e6
    print(
        f"  Co-occur: {co_occur.nnz:,} nnz ({mem_co:.0f} MB), {time.time() - t4:.1f}s"
    )

    try:
        import psutil

        print(f"  RSS: {psutil.Process(os.getpid()).memory_info().rss / 1e6:,.0f} MB")
    except ImportError:
        pass

    # Target doc_ids
    target_doc_ids = None
    if args.target_input:
        target_doc_ids = set()
        with open(args.target_input) as f:
            for line in f:
                target_doc_ids.add(json.loads(line)["doc_id"])
        print(f"  Target docs: {len(target_doc_ids):,}")

    candidate_doc_ids = (
        [d for d in doc_list if d in target_doc_ids] if target_doc_ids else doc_list
    )

    # Set module globals for candidate pool workers (inherited via CoW fork)
    global _P_MATRIX, _CO_MATRIX, _M_MATRIX, _ENTITY_LIST, _DOC_IDX, _MIN_COOCCUR
    _P_MATRIX = P
    _CO_MATRIX = co_occur
    _M_MATRIX = M
    _ENTITY_LIST = entity_list
    _DOC_IDX = doc_idx
    _MIN_COOCCUR = args.min_cooccur

    # Split candidate_doc_ids into batches (1000 docs per batch)
    n_candidate_docs = len(candidate_doc_ids)
    batch_size = 1000
    batches = [
        candidate_doc_ids[i : i + batch_size]
        for i in range(0, n_candidate_docs, batch_size)
    ]
    n_batches = len(batches)
    n_cand_workers = min(args.workers, n_batches)

    print(
        f"  Candidate search: {n_candidate_docs:,} docs, "
        f"{n_batches} batches, {n_cand_workers} workers",
        flush=True,
    )

    doc_candidates: dict[str, dict] = {}
    all_max_probs: list[float] = []
    t_cand = time.time()
    completed_batches = 0

    with Pool(processes=n_cand_workers) as cand_pool:
        for batch_results in cand_pool.imap_unordered(_process_doc_batch, batches):
            completed_batches += 1
            for doc_id, cands in batch_results:
                doc_candidates[doc_id] = cands
                all_max_probs.extend(cands.values())

            if completed_batches % 50 == 0 or completed_batches == n_batches:
                elapsed_c = time.time() - t_cand
                rate_c = completed_batches / elapsed_c if elapsed_c > 0 else 0
                eta_c = (n_batches - completed_batches) / rate_c if rate_c > 0 else 0
                docs_done = completed_batches * batch_size
                print(
                    f"  Candidates: [{docs_done:,}/{n_candidate_docs:,}] "
                    f"{elapsed_c:.0f}s elapsed, ETA {eta_c:.0f}s, "
                    f"{len(doc_candidates):,} docs with candidates",
                    flush=True,
                )

    print(f"  {len(all_max_probs):,} (doc, candidate) pairs in {time.time() - t4:.1f}s")

    if not all_max_probs:
        print("  No candidates. Writing empty.")
        with open(args.output, "w") as f:
            json.dump({}, f)
        return

    # Threshold
    all_max_probs.sort()
    pct_idx = min(
        int(len(all_max_probs) * args.top_pct / 100.0), len(all_max_probs) - 1
    )
    threshold = max(all_max_probs[pct_idx], args.min_prob / 100.0)
    print(
        f"  Threshold: P{args.top_pct:.0f}={all_max_probs[pct_idx]:.4f}, final={threshold:.4f}"
    )

    result = {}
    total_cands = 0
    for doc_id, cands in doc_candidates.items():
        above = sorted(
            [(c, p) for c, p in cands.items() if p >= threshold], key=lambda x: -x[1]
        )
        above = above[: args.max_per_doc]
        if not above:
            continue

        fn_entities = {}
        for c_key, _ in above:
            fn_entities[c_key] = entity_form.get(
                c_key, {"text": c_key[0], "type": c_key[1]}
            )

        fn_passages = {
            pid: list(fn_entities.keys()) for pid in merged[doc_id]["passage_ids"]
        }
        result[doc_id] = {"fn_entities": fn_entities, "fn_passages": fn_passages}
        total_cands += len(fn_entities)

    print(f"  {len(result):,} docs with candidates, {total_cands:,} entities")

    print("\n=== Step 5: Save ===")
    serializable = {}
    for doc_id, cand in result.items():
        serializable[doc_id] = {
            "fn_entities": {
                f"{k[0]}||{k[1]}": v for k, v in cand["fn_entities"].items()
            },
            "fn_passages": {
                pid: [f"{k[0]}||{k[1]}" for k in keys]
                for pid, keys in cand["fn_passages"].items()
            },
        }
    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  {args.output}: {os.path.getsize(args.output) / 1e6:.1f} MB")

    total = time.time() - t_total
    print(f"\n=== DONE in {total:.1f}s ({total / 60:.1f} min) ===")
    print(f"  Docs: {len(merged):,}, Candidates: {total_cands:,}")
    try:
        import psutil

        print(
            f"  Peak RSS: {psutil.Process(os.getpid()).memory_info().rss / 1e6:,.0f} MB"
        )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
