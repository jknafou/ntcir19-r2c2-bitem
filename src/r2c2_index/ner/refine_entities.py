#!/usr/bin/env python3
"""Refine entity annotations using document-level statistics.

Two refinement modes:

**Occurrence-based (default):**
  Step A  — Aggregate per-document entity statistics with zone awareness.
            Zones include passage content, section headers, title, infobox,
            categories, and appearances from documents.jsonl.
  Step B  — Identify FP/FN candidates by occurrence counting:
            FP: entity text appears undetected in more zones than detected.
            FN: every zone where entity text appears but was not detected.
  Step B2 — (optional) Cross-document FN candidates via conditional probability.
  Step C  — LLM review grouped by entity type per document (type-specific prompts).

**Legacy threshold-based (--legacy-thresholds):**
  Step A  — Aggregate per-document entity statistics (passage-only).
  Step B  — Identify FP/FN candidates by document-ratio thresholds.
  Step C  — LLM review per passage.

Cross-document mode (--cross-document): finds entities that are likely missing
from a document by analyzing entity co-occurrence patterns across all documents.
Uses IQR-based outlier removal and adaptive conditional probability thresholds.

Entities are typed dicts: {"text": str, "type": str} where type is one of
PERSON, CHARACTER, LOCATION, ORGANIZATION, EVENT, WORK, DATE.

Requirements:
    pip install openai

Usage:
    # Stats only (no LLM, ~2 min):
    python3 src/r2c2_index/ner/refine_entities.py \
        --input passages_enriched.jsonl \
        --stats-only

    # Cross-document stats:
    python3 src/r2c2_index/ner/refine_entities.py \
        --input passages_enriched.jsonl \
        --cross-document --stats-only

    # Full refinement with cross-document (3xA100 on HPC):
    python3 src/r2c2_index/ner/refine_entities.py \
        --input passages_enriched.jsonl \
        --vllm-urls http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1 \
        --cross-document

    # Resume after interruption:
    python3 src/r2c2_index/ner/refine_entities.py \
        --input passages_enriched.jsonl \
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# Entity keys are (text_lower, entity_type).

EntityKey = tuple[str, str]


VLLM_MODEL = "Qwen/Qwen3-32B-FP8"

# Module-level counters for the diagnostic summary

_error_counts: dict[str, int] = {}
_total_requests = 0
_total_retries = 0


def _record_error(exc_type: str) -> None:
    """Record a final (non-retryable) error by exception type."""
    _error_counts[exc_type] = _error_counts.get(exc_type, 0) + 1


def _record_retry() -> None:
    """Record a retry attempt."""
    global _total_retries
    _total_retries += 1


def _record_request() -> None:
    """Record an LLM request attempt."""
    global _total_requests
    _total_requests += 1


def _print_error_summary() -> None:
    """Print error diagnostics if any errors occurred."""
    total_errors = sum(_error_counts.values())
    if total_errors == 0 and _total_retries == 0:
        return
    print("\n--- LLM Request Diagnostics ---")
    print(f"  Total requests: {_total_requests:,}")
    print(f"  Retries: {_total_retries:,}")
    print(f"  Final errors (all retries exhausted): {total_errors:,}")
    if _total_requests > 0:
        error_rate = total_errors / _total_requests * 100
        print(f"  Error rate: {error_rate:.1f}%")
    if _error_counts:
        print("  Errors by type:")
        for exc_type, count in sorted(
            _error_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {exc_type}: {count:,}")
    print()


VALID_ENTITY_TYPES = {
    "PERSON",
    "CHARACTER",
    "LOCATION",
    "ORGANIZATION",
    "EVENT",
    "WORK",
    "DATE",
}

# Regex for typed entity output: "Entity Text [TYPE]"
_TYPED_ENTITY_RE = re.compile(r"^(.+?)\s*\[([^\]]+)\]\s*$")

FP_SYSTEM_PROMPT = (
    "You verify entity annotations. For each entity listed, decide if "
    "it is a genuine named entity based on the document context.\n\n"
    "Output one line per entity: entity name | verdict\n"
    "Example: Luke Skywalker | ALL\n\n"
    "Verdicts: ALL (genuine), NONE (not genuine), KEEP (as-is)\n\n"
    "Output one line per entity, nothing else."
)

FN_SYSTEM_PROMPT = (
    "You check for missed entity annotations. For each entity listed, "
    "decide if all occurrences should be detected.\n\n"
    "Output one line per entity: entity name | verdict\n"
    "Example: Luke Skywalker | ALL\n\n"
    "Verdicts: ALL (add missing), NONE (remove all), KEEP (as-is)\n\n"
    "Output one line per entity, nothing else."
)

COMBINED_SYSTEM_PROMPT = (
    "You review named entity annotations for a passage. "
    "Each entity is shown as 'Entity [TYPE]'. "
    "You have two tasks:\n"
    "1. VERIFY: For each entity in the 'Verify' list, decide if it is a "
    "genuine named entity (a specific person, character, place, organization, "
    "title of a work, specific event, or species name) — not a generic "
    "descriptor, title/rank, common noun, or category term. "
    "Output: Entity [TYPE] | KEEP or Entity [TYPE] | REMOVE\n"
    "2. CHECK MISSING: For each entity in the 'Check missing' list, decide if "
    "it appears in the passage. Output: Entity [TYPE] | FOUND or Entity [TYPE] | ABSENT\n"
    "Preserve the [TYPE] tag in your output. "
    "Output one line per entity, nothing else."
)


def _load_refine_prompts(prompt_dir: str) -> dict[str, str]:
    """Load refinement prompts from a directory.

    Args:
        prompt_dir: Directory containing refine_fp.txt, refine_fn.txt,
            refine_combined.txt.

    Returns:
        Dict mapping prompt name to content.
    """
    prompts: dict[str, str] = {}
    p = Path(prompt_dir)
    for name in ("refine_fp", "refine_fn", "refine_combined"):
        f = p / f"{name}.txt"
        if f.exists():
            prompts[name] = f.read_text(encoding="utf-8").strip()
    return prompts


def _load_type_specific_prompt(
    prompt_dir: str, direction: str, entity_type: str
) -> str:
    """Load a type-specific refinement prompt, falling back to generic.

    Looks for ``refine_{direction}_{TYPE}.txt`` first, then
    ``refine_{direction}.txt`` (with ``{TYPE}`` placeholder replaced).

    Args:
        prompt_dir: Directory containing prompt files.
        direction: "verify" or "check".
        entity_type: Entity type, e.g. "LOCATION".

    Returns:
        Prompt text string.
    """
    p = Path(prompt_dir)
    specific = p / f"refine_{direction}_{entity_type}.txt"
    if specific.exists():
        return specific.read_text(encoding="utf-8").strip()
    fallback = p / f"refine_{direction}.txt"
    if fallback.exists():
        text = fallback.read_text(encoding="utf-8").strip()
        return text.replace("{TYPE}", entity_type)
    # Ultimate fallback: hardcoded prompts
    if direction == "verify":
        return FP_SYSTEM_PROMPT
    return FN_SYSTEM_PROMPT


# Pre-compiled word-boundary regex cache
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _text_appears_in_zone(entity_text: str, zone_text: str) -> bool:
    """Check if entity text appears in zone text using word-boundary matching.

    Uses word boundaries for entities >= 3 chars to prevent partial matches
    (e.g. "Han" matching "Chandra"). Short entities use exact substring match.

    Args:
        entity_text: The entity surface form to search for.
        zone_text: The zone text to search in.

    Returns:
        True if the entity text appears in the zone text.
    """
    if not entity_text or not zone_text:
        return False
    if len(entity_text) >= 3:
        key = entity_text.lower()
        if key not in _WORD_BOUNDARY_CACHE:
            _WORD_BOUNDARY_CACHE[key] = re.compile(
                r"\b" + re.escape(entity_text) + r"\b", re.IGNORECASE
            )
        return _WORD_BOUNDARY_CACHE[key].search(zone_text) is not None
    # Short entities: case-insensitive substring
    return entity_text.lower() in zone_text.lower()


# Typed entity helpers


def _normalize_entity(e: str | dict) -> dict:
    """Normalize an entity to canonical {"text", "type", "types"} format.

    Thin wrapper around entity_utils.normalize_entity for backward compat.
    """
    return normalize_entity(e)


def expand_entities(entities: list[dict]) -> list[dict]:
    """Expand multi-label entities into per-type entries.

    Thin wrapper around entity_utils.expand_entities.
    Each expanded entry has "text", "type", and "types" fields.
    """
    from entity_utils import expand_entities as _expand

    return _expand(entities)


def _entity_key(e: dict) -> EntityKey:
    """Return the dedup/comparison key for a typed entity dict.

    Thin wrapper around entity_utils.entity_key for backward compat.
    """
    return entity_key(e)


def _format_entity_for_prompt(e: dict[str, str]) -> str:
    """Format a typed entity dict for display in LLM prompts."""
    return f"{e['text']} [{e['type']}]"


# The helper modules live in _lib/ and are imported by bare name, so _lib/
# goes on sys.path before the imports below.
_LIB_DIR = str(Path(__file__).resolve().parent / "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
from entity_utils import ENTITY_TYPES, entity_key, normalize_entity  # noqa: E402,F401
from llm_utils import EndpointPool  # noqa: E402

# Step A: Aggregation


def aggregate_document_stats(input_path: str, min_doc_passages: int) -> dict[str, dict]:
    """Stream input and build per-document entity statistics.

    Entity keys are (text_lower, type) tuples so that e.g.
    "Brooklyn [LOCATION]" and "Brooklyn [PERSON]" are tracked separately.

    Returns:
        {doc_id: {
            title: str,
            total_passages: int,
            passage_ids: [str, ...],
            entities: {(text_lower, type): {form: dict, count: int, passage_ids: set}}
        }}
    """
    print("Step A: Aggregating document-level entity statistics...")
    t0 = time.time()

    docs: dict[str, dict] = {}
    total_passages = 0

    with open(input_path, encoding="utf-8") as f:
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
                }

            doc = docs[doc_id]
            doc["total_passages"] += 1
            doc["passage_ids"].append(pid)

            for raw_e in passage.get("text_entities", []):
                e = _normalize_entity(raw_e)
                ekey = _entity_key(e)
                if ekey not in doc["entities"]:
                    doc["entities"][ekey] = {
                        "form": e,
                        "count": 0,
                        "passage_ids": set(),
                    }
                doc["entities"][ekey]["count"] += 1
                doc["entities"][ekey]["passage_ids"].add(pid)

    filtered = {
        did: d for did, d in docs.items() if d["total_passages"] >= min_doc_passages
    }

    elapsed = time.time() - t0
    print(f"  {total_passages:,} passages, {len(docs):,} documents in {elapsed:.1f}s")
    print(
        f"  {len(filtered):,} documents with >= {min_doc_passages} passages "
        f"(skipped {len(docs) - len(filtered):,})"
    )

    return filtered


def aggregate_document_stats_v2(
    input_path: str,
    documents_path: str | None = None,
) -> dict[str, dict]:
    """Build per-document entity statistics with occurrence counting.

    For each detected entity, counts:
    - detected_count: number of passages where the entity was detected
    - text_occurrences: total times the entity text appears in the full
      document (all passages + metadata concatenated)

    No min_doc_passages filter — all documents participate.

    Args:
        input_path: Path to extraction JSONL (passages with text_entities).
        documents_path: Path to documents.jsonl for metadata (infobox,
            categories, appearances). Included in full document text.

    Returns:
        {doc_id: {
            title: str,
            total_passages: int,
            passage_ids: [str, ...],
            entities: {(text_lower, type): {
                form: dict,
                detected_count: int,
                text_occurrences: int,
                passage_ids: set[str],
            }},
            full_text: str,
        }}
    """
    print(
        "Step A: Aggregating document-level entity statistics (v2, occurrence-based)..."
    )
    t0 = time.time()

    doc_meta: dict[str, dict] = {}
    if documents_path:
        # _lib/ is on sys.path from the module-level setup.
        from build_document_prompt import load_documents_index  # noqa: E402

        doc_meta = load_documents_index(documents_path)
        print(f"  Loaded {len(doc_meta):,} document metadata records")

    docs: dict[str, dict] = {}
    total_passages = 0

    with open(input_path, encoding="utf-8") as f:
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
                    "text_parts": [],  # temporary, joined later
                }

            doc = docs[doc_id]
            doc["total_passages"] += 1
            doc["passage_ids"].append(pid)

            # Collect text parts for full document text
            contents = passage.get("contents", "")
            if contents:
                doc["text_parts"].append(contents)
            section = passage.get("section", "")
            if section:
                doc["text_parts"].append(section)

            # Count detected entities per passage (expand multi-label)
            for e in expand_entities(passage.get("text_entities", [])):
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

    # Build full document text including metadata
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

    # Count text occurrences for each entity in the full document text.
    # Then subtract occurrences covered by a longer detected entity to avoid
    # counting "McCartney" inside "Paul McCartney" as undetected.
    for doc_id, doc in docs.items():
        full_text = doc["full_text"]

        # First pass: raw occurrence counts
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

        # Second pass: subtract occurrences covered by longer detected entities.
        # If "McCartney" appears 12 times but 10 are inside "Paul McCartney"
        # (also detected), then effective occurrences for "McCartney" = 2.
        entity_texts_sorted = sorted(
            doc["entities"].keys(),
            key=lambda ek: len(ek[0]),
            reverse=True,
        )
        for i, short_ek in enumerate(entity_texts_sorted):
            short_text = short_ek[0]  # text_lower
            for long_ek in entity_texts_sorted[:i]:
                long_text = long_ek[0]
                if short_text in long_text and long_ek in doc["entities"]:
                    # short is a substring of long — subtract long's occurrences
                    long_count = doc["entities"][long_ek]["text_occurrences"]
                    doc["entities"][short_ek]["text_occurrences"] -= long_count
            # Floor at detected_count (can't have fewer occurrences than detections)
            info = doc["entities"][short_ek]
            if info["text_occurrences"] < info["detected_count"]:
                info["text_occurrences"] = info["detected_count"]

    elapsed = time.time() - t0
    docs_with_meta = sum(1 for d_id in docs if d_id in doc_meta)
    print(f"  {total_passages:,} passages, {len(docs):,} documents in {elapsed:.1f}s")
    print(f"  {docs_with_meta:,} docs with metadata")

    return docs


# Step B: Identify FP/FN candidates


def identify_candidates(
    doc_stats: dict[str, dict],
    fp_threshold: float,
    fn_threshold: float,
) -> dict[str, dict]:
    """Identify FP and FN candidate entities per document.

    Returns:
        {doc_id: {
            fp_entities: {(text_lower, type): entity_dict},
            fn_entities: {(text_lower, type): entity_dict},
            fp_passages: {pid: [(text_lower, type), ...]},
            fn_passages: {pid: [(text_lower, type), ...]},
        }}
    """
    print(
        f"\nStep B: Identifying candidates "
        f"(FP < {fp_threshold:.0%}, FN > {fn_threshold:.0%})..."
    )
    t0 = time.time()

    candidates: dict[str, dict] = {}
    total_fp_passages = 0
    total_fn_passages = 0
    total_fp_entities = 0
    total_fn_entities = 0

    for doc_id, doc in doc_stats.items():
        total = doc["total_passages"]
        fp_entities: dict[EntityKey, dict[str, str]] = {}
        fn_entities: dict[EntityKey, dict[str, str]] = {}

        for ekey, info in doc["entities"].items():
            ratio = info["count"] / total
            if ratio < fp_threshold:
                fp_entities[ekey] = info["form"]
            elif ratio > fn_threshold:
                fn_entities[ekey] = info["form"]

        if not fp_entities and not fn_entities:
            continue

        fp_passages: dict[str, list[EntityKey]] = {}
        fn_passages: dict[str, list[EntityKey]] = {}

        # FP: passages that have these low-ratio entities
        for ekey, info in doc["entities"].items():
            if ekey in fp_entities:
                for pid in info["passage_ids"]:
                    fp_passages.setdefault(pid, []).append(ekey)

        # FN: passages that are missing these high-ratio entities
        all_pids = set(doc["passage_ids"])
        for ekey, info in doc["entities"].items():
            if ekey in fn_entities:
                missing_pids = all_pids - info["passage_ids"]
                for pid in missing_pids:
                    fn_passages.setdefault(pid, []).append(ekey)

        candidates[doc_id] = {
            "fp_entities": fp_entities,
            "fn_entities": fn_entities,
            "fp_passages": fp_passages,
            "fn_passages": fn_passages,
        }
        total_fp_passages += len(fp_passages)
        total_fn_passages += len(fn_passages)
        total_fp_entities += len(fp_entities)
        total_fn_entities += len(fn_entities)

    elapsed = time.time() - t0
    print(f"  {len(candidates):,} documents with candidates in {elapsed:.1f}s")
    print(
        f"  FP: {total_fp_entities:,} unique entities across "
        f"{total_fp_passages:,} passages"
    )
    print(
        f"  FN: {total_fn_entities:,} unique entities across "
        f"{total_fn_passages:,} passages"
    )

    return candidates


def identify_candidates_occurrence(
    doc_stats: dict[str, dict],
) -> dict[str, dict]:
    """Identify FP, FN, and cross-type conflict candidates.

    For each entity E detected in document D:
      detected   = number of passages where E was detected
      occurrences = total times E's text appears in full document text
      undetected = occurrences - detected

      If undetected <= 0: nothing to review (detected everywhere).
      If detected < undetected: FP candidate — text appears mostly without
        detection, so maybe the few detections were wrong.
      If detected >= undetected: FN candidate — text appears mostly with
        detection, so maybe we missed some occurrences.

    Cross-type conflicts: if the same text (case-insensitive) is detected
    with different types across passages (e.g. "Empire" as LOCATION and
    ORGANIZATION), all involved EntityKeys are flagged for type review.

    Args:
        doc_stats: Per-document stats from aggregate_document_stats_v2().

    Returns:
        {doc_id: {
            fp_entities: {EntityKey: entity_dict},
            fn_entities: {EntityKey: entity_dict},
            fp_passages: {pid: [EntityKey, ...]},
            fn_passages: {pid: [EntityKey, ...]},
            type_conflicts: {text_lower: [EntityKey, ...]},
        }}
    """
    print("\nStep B: Identifying candidates (occurrence-based)...")
    t0 = time.time()

    candidates: dict[str, dict] = {}
    total_fp_entities = 0
    total_fn_entities = 0
    total_fp_passages = 0
    total_fn_passages = 0
    total_type_conflicts = 0

    for doc_id, doc in doc_stats.items():
        fp_entities: dict[EntityKey, dict[str, str]] = {}
        fn_entities: dict[EntityKey, dict[str, str]] = {}
        fp_passages: dict[str, list[EntityKey]] = {}
        fn_passages: dict[str, list[EntityKey]] = {}

        all_pids = set(doc["passage_ids"])

        # Detect cross-type conflicts: same text, different types
        text_to_types: dict[str, list[EntityKey]] = {}
        for ekey in doc["entities"]:
            text_lower = ekey[0]
            text_to_types.setdefault(text_lower, []).append(ekey)
        type_conflicts: dict[str, list[EntityKey]] = {
            text: ekeys for text, ekeys in text_to_types.items() if len(ekeys) > 1
        }

        for ekey, info in doc["entities"].items():
            detected = info["detected_count"]
            occurrences = info["text_occurrences"]
            undetected = occurrences - detected

            if undetected <= 0:
                continue

            if detected < undetected:
                fp_entities[ekey] = info["form"]
                for pid in info["passage_ids"]:
                    fp_passages.setdefault(pid, []).append(ekey)
            else:
                fn_entities[ekey] = info["form"]
                missing_pids = all_pids - info["passage_ids"]
                for pid in missing_pids:
                    fn_passages.setdefault(pid, []).append(ekey)

        if not fp_entities and not fn_entities and not type_conflicts:
            continue

        candidates[doc_id] = {
            "fp_entities": fp_entities,
            "fn_entities": fn_entities,
            "fp_passages": fp_passages,
            "fn_passages": fn_passages,
            "type_conflicts": type_conflicts,
        }
        total_fp_entities += len(fp_entities)
        total_fn_entities += len(fn_entities)
        total_fp_passages += len(fp_passages)
        total_fn_passages += len(fn_passages)
        total_type_conflicts += len(type_conflicts)

    elapsed = time.time() - t0
    print(f"  {len(candidates):,} documents with candidates in {elapsed:.1f}s")
    print(
        f"  FP: {total_fp_entities:,} unique entities across "
        f"{total_fp_passages:,} passages"
    )
    print(
        f"  FN: {total_fn_entities:,} unique entities across "
        f"{total_fn_passages:,} passages"
    )
    if total_type_conflicts:
        print(f"  Type conflicts: {total_type_conflicts:,} texts with multiple types")

    return candidates


# Step B2: Cross-document FN candidates


def cross_document_candidates(
    doc_stats: dict[str, dict],
    min_freq: int = 2,
    iqr_k: float = 1.5,
    min_cond_prob: float = 0.05,
    top_percentile: float = 95.0,
    max_per_doc: int = 30,
    min_cooccur: int = 1,
    stats_only: bool = False,
    target_doc_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Cross-document FN candidate detection using conditional probability.

    Args:
        doc_stats: Per-document entity stats from aggregate_document_stats().
            When using an external corpus, this should be the full corpus
            stats (e.g. 10k docs) so the probability matrix is meaningful.
        min_freq: Skip entities in fewer than this many docs (too rare).
        iqr_k: Tukey's fence multiplier for IQR outlier removal.
        min_cond_prob: Floor for adaptive threshold (prevents degenerate cases).
        top_percentile: Percentile of P(C|E) distribution for adaptive cutoff.
        max_per_doc: Cap candidates per document to limit LLM cost.
        stats_only: If True, print detailed stats and return empty dict.
        target_doc_ids: If provided, only generate candidates for these
            documents (but use all docs for probability computation).
            This allows building the matrix from a large corpus while
            only applying refinement to a smaller subset.

    Returns:
        Same structure as identify_candidates():
        {doc_id: {fp_entities, fn_entities, fp_passages, fn_passages}}
    """
    print("\nStep B2: Cross-document FN candidate detection...")
    t0 = time.time()

    # B2a: Build entity -> docs index (keyed on EntityKey tuples)
    entity_to_docs: dict[EntityKey, set[str]] = {}
    entity_form: dict[EntityKey, dict[str, str]] = {}  # ekey -> best surface form dict
    for doc_id, doc in doc_stats.items():
        for ekey, info in doc["entities"].items():
            if ekey not in entity_to_docs:
                entity_to_docs[ekey] = set()
                entity_form[ekey] = info["form"]
            entity_to_docs[ekey].add(doc_id)

    total_entities = len(entity_to_docs)
    print(f"  {total_entities:,} unique entities across {len(doc_stats):,} documents")

    entity_to_docs = {
        e: docs for e, docs in entity_to_docs.items() if len(docs) >= min_freq
    }
    filtered_by_freq = total_entities - len(entity_to_docs)
    print(f"  {filtered_by_freq:,} entities removed (doc freq < {min_freq})")

    if not entity_to_docs:
        print("  No entities remain after frequency filter.")
        return {}

    # B2b: IQR-based outlier removal
    doc_counts = sorted(len(docs) for docs in entity_to_docs.values())
    n = len(doc_counts)
    q1 = doc_counts[n // 4]
    q3 = doc_counts[3 * n // 4]
    iqr = q3 - q1
    upper_fence = q3 + iqr_k * iqr

    outlier_entities: set[EntityKey] = {
        e for e, docs in entity_to_docs.items() if len(docs) > upper_fence
    }
    print(f"  IQR stats: Q1={q1}, Q3={q3}, IQR={iqr}, upper fence={upper_fence:.1f}")
    print(
        f"  {len(outlier_entities):,} outlier entities removed (doc freq > {upper_fence:.1f})"
    )

    if outlier_entities:
        outlier_sorted = sorted(
            outlier_entities,
            key=lambda e: len(entity_to_docs[e]),
            reverse=True,
        )
        print("  Top outliers:")
        for ekey in outlier_sorted[:20]:
            form = entity_form.get(ekey, {"text": ekey[0], "type": ekey[1]})
            label = _format_entity_for_prompt(form)
            print(f"    {label:50s}  docs={len(entity_to_docs[ekey]):,}")

    # Working set: non-outlier entities with sufficient frequency
    valid_entities = set(entity_to_docs.keys()) - outlier_entities

    # B2c: Conditional probability via numpy matrix multiply
    #
    # Co-occurrence matrix: C = M.T @ M where M[doc, entity] is binary.
    # C[i,j] = number of docs containing both entity i and entity j.
    # P(j|i) = C[i,j] / doc_count[i].
    # For each doc D, max P(candidate|E) across all E in D.

    print("  Computing conditional probabilities (sparse)...")
    t1 = time.time()

    import numpy as np
    from scipy import sparse as sp

    entity_list = sorted(valid_entities)
    entity_idx: dict[EntityKey, int] = {e: i for i, e in enumerate(entity_list)}
    doc_list = list(doc_stats.keys())
    doc_idx: dict[str, int] = {d: i for i, d in enumerate(doc_list)}
    n_ents = len(entity_list)
    n_docs = len(doc_list)

    # Build sparse binary doc-entity matrix (CSR for fast row slicing)
    rows: list[int] = []
    cols: list[int] = []
    for doc_id, doc in doc_stats.items():
        d = doc_idx[doc_id]
        for ekey in doc["entities"]:
            if ekey in entity_idx:
                rows.append(d)
                cols.append(entity_idx[ekey])
    data = np.ones(len(rows), dtype=np.float32)
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, n_ents))
    nnz = M.nnz
    density = nnz / (n_docs * n_ents) * 100 if n_docs * n_ents > 0 else 0
    mem_sparse_mb = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1e6
    mem_dense_mb = n_docs * n_ents * 4 / 1e6
    print(
        f"  Matrix: {n_docs:,} docs × {n_ents:,} entities "
        f"(sparse: {mem_sparse_mb:.0f} MB, {nnz:,} nnz, {density:.4f}% dense; "
        f"would be {mem_dense_mb:,.0f} MB dense)"
    )

    # Co-occurrence matrix: C = M.T @ M (sparse × sparse → sparse)
    co_occur = (M.T @ M).tocsr()
    doc_count = np.asarray(M.sum(axis=0)).ravel()  # (n_ents,)
    safe_count = np.maximum(doc_count, 1.0)

    # Conditional probability: P[i,j] = C[i,j] / doc_count[i]
    # Scale rows of co_occur by 1/safe_count to get P
    inv_count = sp.diags(1.0 / safe_count)
    P = (inv_count @ co_occur).tocsr()

    mem_cooccur_mb = (
        co_occur.data.nbytes + co_occur.indices.nbytes + co_occur.indptr.nbytes
    ) / 1e6
    mem_P_mb = (P.data.nbytes + P.indices.nbytes + P.indptr.nbytes) / 1e6
    t_matmul = time.time() - t1
    print(
        f"  Co-occur: {co_occur.nnz:,} nnz ({mem_cooccur_mb:.0f} MB), "
        f"P: {P.nnz:,} nnz ({mem_P_mb:.0f} MB), "
        f"computed in {t_matmul:.1f}s"
    )

    # Memory summary
    import os

    try:
        import psutil

        proc_mem = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"  Process RSS: {proc_mem:,.0f} MB")
    except ImportError:
        pass

    # Per-doc candidate tracking
    # When target_doc_ids is set, only collect candidates for those docs
    # (but the probability matrix uses all docs for meaningful statistics)
    candidate_doc_ids = (
        [d for d in doc_list if d in target_doc_ids] if target_doc_ids else doc_list
    )
    if target_doc_ids:
        print(
            f"  Target docs: {len(candidate_doc_ids):,} "
            f"(of {len(doc_list):,} in corpus)"
        )

    doc_candidates: dict[str, dict[EntityKey, float]] = {}
    all_max_probs: list[float] = []

    for doc_id in candidate_doc_ids:
        d = doc_idx[doc_id]
        # Get entity indices present in this doc (sparse row)
        d_ent_indices = M[d].indices

        if len(d_ent_indices) == 0:
            continue

        # max P(candidate | E) across all E in this doc
        # P[d_ent_indices, :] is a sparse submatrix; take column-wise max
        P_sub = P[d_ent_indices, :]  # sparse (|D_ents| × n_ents)
        max_probs = np.asarray(P_sub.max(axis=0).todense()).ravel()  # (n_ents,)

        # Zero out entities already in doc (not candidates)
        max_probs[d_ent_indices] = 0.0

        # Apply min_cooccur filter: require at least N absolute co-occurrences
        if min_cooccur > 1:
            co_sub = co_occur[d_ent_indices, :]
            max_cooccur = np.asarray(co_sub.max(axis=0).todense()).ravel()
            max_probs[max_cooccur < min_cooccur] = 0.0

        # Collect non-zero candidates
        nonzero = np.nonzero(max_probs)[0]
        if len(nonzero) > 0:
            candidate_probs = {entity_list[i]: float(max_probs[i]) for i in nonzero}
            doc_candidates[doc_id] = candidate_probs
            all_max_probs.extend(max_probs[nonzero].tolist())

    elapsed_pass1 = time.time() - t1
    print(
        f"  Pass 1 done in {elapsed_pass1:.1f}s: {len(all_max_probs):,} (doc, candidate) pairs"
    )

    if not all_max_probs:
        print("  No cross-document candidates found.")
        return {}

    # Adaptive threshold: top_percentile of all conditional probabilities
    all_max_probs.sort()
    pct_idx = int(len(all_max_probs) * top_percentile / 100.0)
    pct_idx = min(pct_idx, len(all_max_probs) - 1)
    adaptive_threshold = all_max_probs[pct_idx]
    threshold = max(adaptive_threshold, min_cond_prob)

    print(
        f"  Adaptive threshold: P{top_percentile:.0f}={adaptive_threshold:.4f}, "
        f"floor={min_cond_prob:.4f}, final={threshold:.4f}"
    )

    if stats_only:
        print("\n--- Cross-Document Stats ---")
        print(f"  Total unique entities (after freq filter): {len(entity_to_docs):,}")
        print(f"  Valid entities (non-outlier): {len(valid_entities):,}")
        print(f"  Documents with cross-doc candidates: {len(doc_candidates):,}")

        # Count candidates per doc (above threshold)
        doc_cand_counts: list[tuple[str, int, list[tuple[EntityKey, float]]]] = []
        for doc_id, cands in doc_candidates.items():
            above = [(c, p) for c, p in cands.items() if p >= threshold]
            if above:
                above.sort(key=lambda x: x[1], reverse=True)
                doc_cand_counts.append((doc_id, len(above), above))

        doc_cand_counts.sort(key=lambda x: x[1], reverse=True)
        print(f"  Documents with candidates above threshold: {len(doc_cand_counts):,}")

        print("\n  Top 20 documents by cross-doc candidate count:")
        for doc_id, count, cands in doc_cand_counts[:20]:
            title = doc_stats[doc_id]["title"][:50]
            print(f"    {title:50s}  candidates={count:4d}")
            for c_key, prob in cands[:5]:
                form = entity_form.get(c_key, {"text": c_key[0], "type": c_key[1]})
                label = _format_entity_for_prompt(form)
                print(f"      {label:50s}  P={prob:.4f}")

        return {}

    # Second pass: filter and build candidate structure
    result: dict[str, dict] = {}
    total_candidates = 0

    for doc_id, cands in doc_candidates.items():
        above = [(c, p) for c, p in cands.items() if p >= threshold]
        if not above:
            continue

        # Sort by probability descending, cap at max_per_doc
        above.sort(key=lambda x: x[1], reverse=True)
        above = above[:max_per_doc]

        fn_entities: dict[EntityKey, dict[str, str]] = {}
        for c_key, _ in above:
            fn_entities[c_key] = entity_form.get(
                c_key, {"text": c_key[0], "type": c_key[1]}
            )

        # Add to fn_passages for all passages of this doc
        fn_passages: dict[str, list[EntityKey]] = {}
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
        f"  {len(result):,} documents with cross-doc candidates, "
        f"{total_candidates:,} total candidate entities in {elapsed:.1f}s"
    )

    return result


def load_gold_entities(path: str) -> dict[str, list[dict[str, str]]]:
    """Load gold standard JSONL and return {passage_id: [entity_dicts]}.

    Args:
        path: Path to gold standard JSONL (each line has 'id' and 'entities').

    Returns:
        Mapping from passage ID to list of entity dicts with 'text' and 'type'.
    """
    result: dict[str, list[dict[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            pid = p.get("id", "")
            ents = [_normalize_entity(e) for e in p.get("entities", [])]
            if ents:
                result[pid] = ents
    print(f"  Loaded gold entities: {len(result):,} passages from {path}")
    return result


def load_external_entity_frequencies(
    path: str, min_doc_freq: int
) -> dict[EntityKey, dict[str, str]]:
    """Stream external extraction JSONL and return entities in >= N documents.

    Args:
        path: Path to extraction JSONL.
        min_doc_freq: Minimum number of documents an entity must appear in.

    Returns:
        Mapping from EntityKey to entity dict for frequent entities.
    """
    entity_docs: dict[EntityKey, set[str]] = {}
    entity_form: dict[EntityKey, dict[str, str]] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            doc_id = p.get("doc_id", "")
            for raw_e in p.get("text_entities", []):
                e = _normalize_entity(raw_e)
                ekey = _entity_key(e)
                entity_docs.setdefault(ekey, set()).add(doc_id)
                # Keep longest surface form
                if ekey not in entity_form or len(e["text"]) > len(
                    entity_form[ekey]["text"]
                ):
                    entity_form[ekey] = e

    frequent = {
        ekey: entity_form[ekey]
        for ekey, docs in entity_docs.items()
        if len(docs) >= min_doc_freq
    }
    print(
        f"  External entities: {len(entity_form):,} unique, "
        f"{len(frequent):,} with doc_freq >= {min_doc_freq} from {path}"
    )
    return frequent


def augment_candidates_with_gold(
    candidates: dict[str, dict],
    doc_stats: dict[str, dict],
    gold_entities: dict[str, list[dict[str, str]]],
) -> set[EntityKey]:
    """Inject missing gold entities as FN candidates.

    For each passage in gold_entities, find entities not present in the
    extraction and add them as FN candidates.

    Args:
        candidates: Mutable candidates dict (modified in-place).
        doc_stats: Document statistics from aggregation step.
        gold_entities: {passage_id: [entity_dicts]} from gold standard.

    Returns:
        Set of EntityKeys that were injected (for audit trail).
    """
    injected: set[EntityKey] = set()
    injected_count = 0

    for pid, gold_ents in gold_entities.items():
        # Find which document this passage belongs to
        doc_id = None
        for did, doc in doc_stats.items():
            if pid in doc.get("passage_ids", []):
                doc_id = did
                break
        if doc_id is None:
            continue

        # Get existing entity keys for this passage from doc_stats
        existing_keys: set[EntityKey] = set()
        for ekey, info in doc_stats[doc_id]["entities"].items():
            if pid in info["passage_ids"]:
                existing_keys.add(ekey)

        # Find gold entities missing from extraction
        for e in gold_ents:
            ekey = _entity_key(e)
            if ekey in existing_keys:
                continue

            # Ensure candidates entry exists for this document
            if doc_id not in candidates:
                candidates[doc_id] = {
                    "fp_entities": {},
                    "fn_entities": {},
                    "fp_passages": {},
                    "fn_passages": {},
                }

            cand = candidates[doc_id]
            if ekey not in cand["fn_entities"]:
                cand["fn_entities"][ekey] = e
            cand["fn_passages"].setdefault(pid, [])
            if ekey not in cand["fn_passages"][pid]:
                cand["fn_passages"][pid].append(ekey)
                injected.add(ekey)
                injected_count += 1

    print(
        f"  Gold FN injection: {len(injected):,} unique entities, "
        f"{injected_count:,} passage-entity pairs"
    )
    return injected


def augment_candidates_with_external(
    candidates: dict[str, dict],
    doc_stats: dict[str, dict],
    external_entities: dict[EntityKey, dict[str, str]],
) -> set[EntityKey]:
    """Inject frequent external entities as FN candidates.

    Only injects an entity into a document if it appears in at least one
    passage of that document (relevance filter).

    Args:
        candidates: Mutable candidates dict (modified in-place).
        doc_stats: Document statistics from aggregation step.
        external_entities: {EntityKey: entity_dict} for frequent entities.

    Returns:
        Set of EntityKeys that were injected (for audit trail).
    """
    injected: set[EntityKey] = set()
    injected_count = 0

    for doc_id, doc in doc_stats.items():
        # Build set of entity keys present in this document
        doc_entity_keys: set[EntityKey] = set(doc["entities"].keys())

        # Find external entities that appear in this doc but are missing
        # from some passages
        for ekey, e in external_entities.items():
            if ekey not in doc_entity_keys:
                # Entity doesn't appear in this document at all — skip
                continue

            # Entity exists in doc — find passages where it's missing
            present_pids = doc["entities"][ekey]["passage_ids"]
            all_pids = set(doc["passage_ids"])
            missing_pids = all_pids - present_pids

            if not missing_pids:
                continue

            # Ensure candidates entry exists
            if doc_id not in candidates:
                candidates[doc_id] = {
                    "fp_entities": {},
                    "fn_entities": {},
                    "fp_passages": {},
                    "fn_passages": {},
                }

            cand = candidates[doc_id]
            if ekey not in cand["fn_entities"]:
                cand["fn_entities"][ekey] = e

            for pid in missing_pids:
                cand["fn_passages"].setdefault(pid, [])
                if ekey not in cand["fn_passages"][pid]:
                    cand["fn_passages"][pid].append(ekey)
                    injected.add(ekey)
                    injected_count += 1

    print(
        f"  External FN injection: {len(injected):,} unique entities, "
        f"{injected_count:,} passage-entity pairs"
    )
    return injected


def merge_candidates(
    within_doc: dict[str, dict], cross_doc: dict[str, dict]
) -> dict[str, dict]:
    """Merge cross-doc candidates into within-doc candidates.

    For docs in both: merge fn_entities and fn_passages (union).
    For docs only in cross_doc: add as new entries.
    For docs only in within_doc: keep unchanged.
    """
    if not cross_doc:
        return within_doc

    merged = dict(within_doc)

    for doc_id, cross in cross_doc.items():
        if doc_id not in merged:
            merged[doc_id] = cross
            continue

        existing = merged[doc_id]

        # Merge fn_entities (union)
        for ekey, form in cross["fn_entities"].items():
            if ekey not in existing["fn_entities"]:
                existing["fn_entities"][ekey] = form

        # Merge fn_passages (union of entity key lists per passage)
        for pid, entity_keys in cross["fn_passages"].items():
            if pid not in existing["fn_passages"]:
                existing["fn_passages"][pid] = entity_keys
            else:
                existing_set = set(existing["fn_passages"][pid])
                for ek in entity_keys:
                    if ek not in existing_set:
                        existing["fn_passages"][pid].append(ek)

    return merged


# Step C: LLM review


def _parse_review_output(
    raw: str,
    candidate_ekeys: list[EntityKey] | None = None,
) -> dict[EntityKey, str]:
    """Parse LLM review output into {(text_lower, type): verdict}.

    Supports multiple output formats:
    - Typed: "Entity [TYPE] | VERDICT"
    - Pipe (no type): "Entity Name | VERDICT" — matched against candidate_ekeys
    - Untyped fallback: defaults to PERSON type

    Args:
        raw: Raw LLM output string.
        candidate_ekeys: If provided, used to match untyped entity names
            to their (text_lower, type) keys. Required for pipe-no-type format.
    """
    # Build text→ekey lookup for pipe-no-type matching
    text_to_ekey: dict[str, EntityKey] = {}
    if candidate_ekeys:
        for ek in candidate_ekeys:
            text_to_ekey[ek[0]] = ek  # text_lower → (text_lower, type)
    results: dict[EntityKey, str] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if "|" not in line:
            continue
        parts = line.rsplit("|", 1)
        if len(parts) != 2:
            continue
        entity_part = parts[0].strip().strip("*").strip()
        verdict = parts[1].strip().upper()
        # Normalize list prefixes
        entity_part = re.sub(r"^[\d]+[.)]\s*", "", entity_part)
        entity_part = entity_part.lstrip("-*\u2022\u2013 ").strip()

        if not entity_part or verdict not in (
            "KEEP",
            "REMOVE",
            "FOUND",
            "ABSENT",
            "ALL",
            "NONE",
            "CORRECT",
        ):
            continue

        # Try typed format: "Entity [TYPE]" or "Entity [TYPE1, TYPE2]"
        m = _TYPED_ENTITY_RE.match(entity_part)
        if m:
            text = m.group(1).strip()
            etype = m.group(2).strip().upper()
            # Handle multi-type: take first valid type
            if "," in etype:
                etypes = [t.strip() for t in etype.split(",")]
                etype = next((t for t in etypes if t in VALID_ENTITY_TYPES), "PERSON")
            elif etype not in VALID_ENTITY_TYPES:
                etype = "PERSON"
            results[(text.lower(), etype)] = verdict
        else:
            # Pipe-no-type format: match by text against candidate keys
            text_lower = entity_part.lower()
            if text_lower in text_to_ekey:
                results[text_to_ekey[text_lower]] = verdict
            else:
                # Ultimate fallback: default to PERSON
                results[(text_lower, "PERSON")] = verdict
    return results


async def review_passage(
    client,
    model: str,
    passage: dict,
    fp_entities: list[dict[str, str]],
    fn_entities: list[dict[str, str]],
    max_retries: int = 3,
    prompt_dir: str | None = None,
) -> dict[EntityKey, str]:
    """Send a single passage for FP/FN review.

    Returns {(text_lower, type): verdict}.
    """
    # Load prompts from directory if provided, else use hardcoded defaults
    if prompt_dir is not None:
        loaded = _load_refine_prompts(prompt_dir)
        fp_prompt = loaded.get("refine_fp", FP_SYSTEM_PROMPT)
        fn_prompt = loaded.get("refine_fn", FN_SYSTEM_PROMPT)
        combined_prompt = loaded.get("refine_combined", COMBINED_SYSTEM_PROMPT)
    else:
        fp_prompt = FP_SYSTEM_PROMPT
        fn_prompt = FN_SYSTEM_PROMPT
        combined_prompt = COMBINED_SYSTEM_PROMPT

    title = passage.get("title", "")
    section = passage.get("section", "")
    contents = passage.get("contents", "")

    has_fp = len(fp_entities) > 0
    has_fn = len(fn_entities) > 0

    # Build prompt showing entities as "Entity [TYPE]"
    if has_fp and has_fn:
        system = combined_prompt
        user_msg = f"Title: {title}"
        if section:
            user_msg += f"\nSection: {section}"
        user_msg += f"\n\nPassage:\n{contents}"
        user_msg += "\n\nVerify these entities (KEEP if genuine named entity, REMOVE if generic term):\n"
        user_msg += "\n".join(f"- {_format_entity_for_prompt(e)}" for e in fp_entities)
        user_msg += "\n\nCheck if these entities appear in this passage:\n"
        user_msg += "\n".join(f"- {_format_entity_for_prompt(e)}" for e in fn_entities)
    elif has_fp:
        system = fp_prompt
        user_msg = f"Title: {title}"
        if section:
            user_msg += f"\nSection: {section}"
        user_msg += f"\n\nPassage:\n{contents}"
        user_msg += "\n\nVerify these entities (KEEP if genuine named entity, REMOVE if generic term):\n"
        user_msg += "\n".join(f"- {_format_entity_for_prompt(e)}" for e in fp_entities)
    else:
        system = fn_prompt
        user_msg = f"Title: {title}"
        if section:
            user_msg += f"\nSection: {section}"
        user_msg += f"\n\nPassage:\n{contents}"
        user_msg += "\n\nCheck if these entities appear in this passage:\n"
        user_msg += "\n".join(f"- {_format_entity_for_prompt(e)}" for e in fn_entities)

    for attempt in range(max_retries):
        _record_request()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=500,
                timeout=600.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            return _parse_review_output(raw)
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_cause = (
                f" caused_by={type(exc.__cause__).__name__}: {exc.__cause__}"
                if exc.__cause__
                else ""
            )
            pid = passage.get("id", "?")
            if attempt < max_retries - 1:
                _record_retry()
                wait = 2**attempt
                print(
                    f"  [retry {attempt + 1}/{max_retries}] {pid}: "
                    f"{exc_type}: {exc}{exc_cause}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
            else:
                _record_error(exc_type)
                print(
                    f"  [error] {pid}: {exc_type}: {exc}{exc_cause}",
                    file=sys.stderr,
                )
                return {}


async def review_document(
    pool: EndpointPool,
    model: str,
    doc_id: str,
    doc_candidates: dict,
    passages_by_id: dict[str, dict],
    prompt_dir: str | None = None,
) -> dict[str, dict]:
    """Review all flagged passages for one document. Returns refinements per pid.

    Returns:
        {pid: {"fp_removed": [entity_dict, ...], "fn_added": [entity_dict, ...]}}
    """
    fp_passages = doc_candidates["fp_passages"]
    fn_passages = doc_candidates["fn_passages"]
    fp_entities = doc_candidates["fp_entities"]
    fn_entities = doc_candidates["fn_entities"]

    all_pids = set(fp_passages.keys()) | set(fn_passages.keys())

    tasks = []
    pid_order = []
    for pid in all_pids:
        passage = passages_by_id.get(pid)
        if passage is None:
            continue

        fp_ents = [fp_entities[ekey] for ekey in fp_passages.get(pid, [])]
        fn_ents = [fn_entities[ekey] for ekey in fn_passages.get(pid, [])]

        client = pool.next()
        tasks.append(
            review_passage(
                client,
                model,
                passage,
                fp_ents,
                fn_ents,
                prompt_dir=prompt_dir,
            )
        )
        pid_order.append(pid)

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks)
    refinements: dict[str, dict] = {}

    for pid, verdicts in zip(pid_order, results):
        fp_removed: list[dict[str, str]] = []
        fn_added: list[dict[str, str]] = []

        for ekey in fp_passages.get(pid, []):
            verdict = verdicts.get(ekey, "")
            if verdict == "REMOVE":
                fp_removed.append(fp_entities[ekey])

        for ekey in fn_passages.get(pid, []):
            verdict = verdicts.get(ekey, "")
            if verdict == "FOUND":
                fn_added.append(fn_entities[ekey])

        if fp_removed or fn_added:
            refinements[pid] = {"fp_removed": fp_removed, "fn_added": fn_added}

    return refinements


async def _llm_call_with_retry(
    client,
    model: str,
    system: str,
    user_msg: str,
    label: str = "?",
    max_retries: int = 3,
    candidate_ekeys: list[EntityKey] | None = None,
) -> dict[EntityKey, str]:
    """Make an LLM call with retry logic, return parsed verdicts."""
    for attempt in range(max_retries):
        _record_request()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=500,
                timeout=600.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_review_output(raw, candidate_ekeys=candidate_ekeys)
            # Diagnostic: log first 5 raw responses to stdout
            if _total_requests <= 5:
                print(
                    f"  [diag] {label}: raw={raw!r:.200s} "
                    f"parsed={dict(list(parsed.items())[:3])}"
                )
                sys.stdout.flush()
            return parsed
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_cause = (
                f" caused_by={type(exc.__cause__).__name__}: {exc.__cause__}"
                if exc.__cause__
                else ""
            )
            if attempt < max_retries - 1:
                _record_retry()
                wait = 2**attempt
                print(
                    f"  [retry {attempt + 1}/{max_retries}] {label}: "
                    f"{exc_type}: {exc}{exc_cause}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
            else:
                _record_error(exc_type)
                print(
                    f"  [error] {label}: {exc_type}: {exc}{exc_cause}",
                    file=sys.stderr,
                )
                return {}
    return {}


CONFLICT_SYSTEM_PROMPT = (
    "You resolve entity type conflicts in a document. "
    "The same text was detected with different entity types in different "
    "passages. Decide which type is correct based on the document context.\n\n"
    "Answer with exactly one line: the entity text followed by the correct "
    "type in brackets.\n"
    "Example: Empire [ORGANIZATION] | CORRECT\n\n"
    "Only output one line, nothing else."
)


async def review_document_by_type(
    pool: EndpointPool,
    model: str,
    doc_id: str,
    doc_candidates: dict,
    doc_full_text: str,
    doc_entity_info: dict[EntityKey, dict],
    all_passage_ids: list[str],
    prompt_dir: str | None = None,
) -> dict[str, dict]:
    """Review candidates for one document — one entity per LLM call.

    Three kinds of review:
    - FP: entity detected < undetected -> maybe wrong (verify prompt)
    - FN: entity detected >= undetected -> maybe missed some (check prompt)
    - Type conflict: same text tagged with different types -> which is right

    Each entity gets its own LLM call with a type-specific prompt.
    3 possible answers: ALL / NONE / KEEP.

    Args:
        pool: Endpoint pool for round-robin LLM access.
        model: Model name for LLM calls.
        doc_id: Document ID (for logging).
        doc_candidates: Candidate dict from identify_candidates_occurrence.
        doc_full_text: Full document text for context.
        doc_entity_info: {EntityKey: {form, detected_count, text_occurrences,
            passage_ids}} for entities in this document.
        all_passage_ids: All passage IDs in this document.
        prompt_dir: Directory with type-specific prompt files.

    Returns:
        {pid: {"fp_removed": [entity_dict, ...], "fn_added": [entity_dict, ...]}}
    """
    fp_entities = doc_candidates["fp_entities"]
    fn_entities = doc_candidates["fn_entities"]
    type_conflicts = doc_candidates.get("type_conflicts", {})

    tasks = []
    task_ekeys: list[EntityKey] = []
    task_directions: list[str] = []  # "fp", "fn", "conflict"

    # Cache loaded prompts per (direction, type)
    _prompt_cache: dict[tuple[str, str], str] = {}

    def _get_prompt(direction: str, etype: str) -> str:
        key = (direction, etype)
        if key not in _prompt_cache:
            if prompt_dir:
                _prompt_cache[key] = _load_type_specific_prompt(
                    prompt_dir, direction, etype
                )
            else:
                _prompt_cache[key] = (
                    FP_SYSTEM_PROMPT if direction == "verify" else FN_SYSTEM_PROMPT
                )
        return _prompt_cache[key]

    # FP entities: one call per entity
    for ekey, form in fp_entities.items():
        system = _get_prompt("verify", ekey[1])
        user_msg = (
            f"Document:\n{doc_full_text}\n\n"
            f"Entity to review:\n- {_format_entity_for_prompt(form)}"
        )
        tasks.append(
            _llm_call_with_retry(
                pool.next(),
                model,
                system,
                user_msg,
                label=f"{doc_id}/{form['text']}/fp",
                candidate_ekeys=[ekey],
            )
        )
        task_ekeys.append(ekey)
        task_directions.append("fp")

    # FN entities: one call per entity
    for ekey, form in fn_entities.items():
        system = _get_prompt("check", ekey[1])
        user_msg = (
            f"Document:\n{doc_full_text}\n\n"
            f"Entity to review:\n- {_format_entity_for_prompt(form)}"
        )
        tasks.append(
            _llm_call_with_retry(
                pool.next(),
                model,
                system,
                user_msg,
                label=f"{doc_id}/{form['text']}/fn",
                candidate_ekeys=[ekey],
            )
        )
        task_ekeys.append(ekey)
        task_directions.append("fn")

    # Type conflicts: one call per conflicting text
    for text_lower, conflict_ekeys in type_conflicts.items():
        forms = [
            doc_entity_info[ek]["form"]
            for ek in conflict_ekeys
            if ek in doc_entity_info
        ]
        if len(forms) < 2:
            continue
        types_found = ", ".join(f["type"] for f in forms)
        entity_text = forms[0]["text"]
        user_msg = (
            f"Document:\n{doc_full_text}\n\n"
            f'The text "{entity_text}" was detected with these types: '
            f"{types_found}\n"
            f"Which type is correct? Answer with:\n"
            f"{entity_text} [CORRECT_TYPE] | CORRECT"
        )
        tasks.append(
            _llm_call_with_retry(
                pool.next(),
                model,
                CONFLICT_SYSTEM_PROMPT,
                user_msg,
                label=f"{doc_id}/{entity_text}/conflict",
            )
        )
        task_ekeys.append(conflict_ekeys[0])
        task_directions.append("conflict")

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks)

    all_verdicts: dict[EntityKey, str] = {}
    conflict_resolutions: dict[str, str] = {}  # text_lower -> correct_type

    for ekey, direction, verdicts in zip(task_ekeys, task_directions, results):
        if direction == "conflict":
            for vkey, verdict in verdicts.items():
                if verdict == "CORRECT":
                    conflict_resolutions[vkey[0]] = vkey[1]
        else:
            all_verdicts.update(verdicts)

    # Apply 3-answer logic:
    #   ALL  -> entity is real everywhere: add to missing passages
    #   NONE -> entity is not real: remove from detected passages
    #   KEEP -> no change
    refinements: dict[str, dict] = {}
    all_pids = set(all_passage_ids)

    for ekey, form in {**fp_entities, **fn_entities}.items():
        verdict = all_verdicts.get(ekey, "KEEP")
        info = doc_entity_info.get(ekey)
        if info is None or verdict == "KEEP":
            continue
        detected_pids = info["passage_ids"]

        if verdict == "ALL":
            for pid in all_pids - detected_pids:
                refinements.setdefault(pid, {"fp_removed": [], "fn_added": []})
                refinements[pid]["fn_added"].append(form)
        elif verdict == "NONE":
            for pid in detected_pids:
                refinements.setdefault(pid, {"fp_removed": [], "fn_added": []})
                refinements[pid]["fp_removed"].append(form)

    # Apply type conflict resolutions:
    # Replace wrong-type detections with correct-type detections
    for text_lower, correct_type in conflict_resolutions.items():
        for ekey in type_conflicts.get(text_lower, []):
            info = doc_entity_info.get(ekey)
            if info is None or ekey[1] == correct_type:
                continue
            wrong_form = info["form"]
            corrected = {"text": wrong_form["text"], "type": correct_type}
            for pid in info["passage_ids"]:
                refinements.setdefault(pid, {"fp_removed": [], "fn_added": []})
                refinements[pid]["fp_removed"].append(wrong_form)
                refinements[pid]["fn_added"].append(corrected)

    return refinements


# Apply refinements and write output


def apply_refinement(
    passage: dict,
    refinement: dict,
    cross_doc_entities: set[EntityKey] | None = None,
    gold_entities: set[EntityKey] | None = None,
    external_entities: set[EntityKey] | None = None,
) -> dict:
    """Apply FP removals and FN additions to a passage's entity fields.

    All entities are canonical dicts: {"text": str, "type": str, "types": [str]}.

    Args:
        passage: The passage dict to modify.
        refinement: {fp_removed: [entity_dict, ...], fn_added: [entity_dict, ...]}.
        cross_doc_entities: Set of (text_lower, type) tuples that came from
            cross-doc detection. Used to split fn_added into fn_added vs
            cross_doc_added.
        gold_entities: Set of (text_lower, type) tuples injected from gold
            standard. Tracked as gold_fn_added in audit trail.
        external_entities: Set of (text_lower, type) tuples injected from
            external extraction. Tracked as external_fn_added in audit trail.
    """
    fp_removed_keys: set[EntityKey] = {
        _entity_key(e) for e in refinement.get("fp_removed", [])
    }
    fn_added: list[dict[str, str]] = refinement.get("fn_added", [])

    # Classify fn_added by source for audit trail
    all_source_sets = {
        "gold": gold_entities or set(),
        "cross_doc": cross_doc_entities or set(),
        "external": external_entities or set(),
    }
    gold_fn: list[dict[str, str]] = []
    cross_fn: list[dict[str, str]] = []
    external_fn: list[dict[str, str]] = []
    within_fn: list[dict[str, str]] = []

    for e in fn_added:
        ekey = _entity_key(e)
        if all_source_sets["gold"] and ekey in all_source_sets["gold"]:
            gold_fn.append(e)
        elif all_source_sets["cross_doc"] and ekey in all_source_sets["cross_doc"]:
            cross_fn.append(e)
        elif all_source_sets["external"] and ekey in all_source_sets["external"]:
            external_fn.append(e)
        else:
            within_fn.append(e)

    # Remove FP entities from text_entities
    old_text = passage.get("text_entities", [])
    # Normalize all entities to canonical format (both "type" and "types")
    old_text = [normalize_entity(e) for e in old_text]
    new_text = [e for e in old_text if _entity_key(e) not in fp_removed_keys]

    # For multi-label entities where only one type was FP-removed,
    # update the "types" list on the remaining entry to remove it
    fp_removed_texts: dict[str, set[str]] = {}
    for ekey in fp_removed_keys:
        fp_removed_texts.setdefault(ekey[0], set()).add(ekey[1])
    for e in new_text:
        text_lower = e["text"].lower()
        if text_lower in fp_removed_texts:
            removed_types = fp_removed_texts[text_lower]
            updated_types = [t for t in e["types"] if t not in removed_types]
            if updated_types:
                e["types"] = sorted(set(updated_types))

    # Add FN entities to text_entities (deduplicated on (text_lower, type))
    # Normalize FN entities to ensure both "type" and "types" fields
    existing_keys: set[EntityKey] = {_entity_key(e) for e in new_text}
    for e in fn_added:
        ne = normalize_entity(e)
        ekey = _entity_key(ne)
        if ekey not in existing_keys:
            new_text.append(ne)
            existing_keys.add(ekey)

    passage["text_entities"] = new_text

    # Rebuild entities = doc_entities ∪ text_entities (typed dicts)
    title = passage.get("title", "")
    seen: set[EntityKey] = set()
    entities: list[dict] = []
    if title:
        title_key: EntityKey = (title.lower(), "WORK")
        seen.add(title_key)
        entities.append({"text": title, "type": "WORK", "types": ["WORK"]})
    for e in new_text:
        ekey = _entity_key(e)
        if ekey not in seen:
            seen.add(ekey)
            entities.append(e)
    passage["entities"] = entities

    # Build refinement record with source splits
    ref = dict(refinement)
    has_splits = cross_fn or gold_fn or external_fn
    if has_splits:
        ref["fn_added"] = within_fn
    if cross_fn:
        ref["cross_doc_added"] = cross_fn
    if gold_fn:
        ref["gold_fn_added"] = gold_fn
    if external_fn:
        ref["external_fn_added"] = external_fn
    passage["refinement"] = ref
    return passage


# Main pipeline


async def run(args: argparse.Namespace) -> None:
    """Async entry point."""
    prompt_dir = getattr(args, "prompt_dir", None)
    documents_path = getattr(args, "documents", None)
    use_occurrence = not getattr(args, "legacy_thresholds", False)
    t0_total = time.time()

    # Step A: Aggregation
    if use_occurrence and documents_path:
        doc_stats = aggregate_document_stats_v2(args.input, documents_path)
    elif use_occurrence:
        doc_stats = aggregate_document_stats_v2(args.input)
    else:
        doc_stats = aggregate_document_stats(args.input, args.min_doc_passages)

    # Step B: Identify candidates
    if use_occurrence:
        candidates = identify_candidates_occurrence(doc_stats)
    else:
        fp_thresh = args.fp_threshold / 100.0
        fn_thresh = args.fn_threshold / 100.0
        candidates = identify_candidates(doc_stats, fp_thresh, fn_thresh)

    # Step B1b: Gold entity injection (optional)
    gold_entity_set: set[EntityKey] = set()
    if getattr(args, "gold_entities", None):
        print("\nStep B1b: Gold entity FN injection...")
        gold_ents = load_gold_entities(args.gold_entities)
        gold_entity_set = augment_candidates_with_gold(candidates, doc_stats, gold_ents)

    # Step B1c: External entity injection (optional)
    external_entity_set: set[EntityKey] = set()
    if getattr(args, "external_entities", None):
        print("\nStep B1c: External entity FN injection...")
        ext_ents = load_external_entity_frequencies(
            args.external_entities, getattr(args, "external_min_doc_freq", 5)
        )
        external_entity_set = augment_candidates_with_external(
            candidates, doc_stats, ext_ents
        )

    # Step B2: Cross-document candidates (optional)
    cross_doc_entity_set: set[EntityKey] = set()  # for audit trail in apply_refinement
    precomputed_crossdoc = getattr(args, "precomputed_crossdoc", None)
    if args.cross_document and precomputed_crossdoc:
        # Load precomputed candidates from fast_crossdoc.py output
        print(f"\n  Loading precomputed cross-doc candidates: {precomputed_crossdoc}")
        with open(precomputed_crossdoc, encoding="utf-8") as f:
            raw_cands = json.load(f)
        # Convert "text||type" keys back to (text, type) tuples
        cross_cands: dict[str, dict] = {}
        input_doc_ids = set(doc_stats.keys())
        for doc_id, cand in raw_cands.items():
            if doc_id not in input_doc_ids:
                continue  # skip docs not in this shard
            fn_entities: dict[EntityKey, dict[str, str]] = {}
            for key_str, form in cand["fn_entities"].items():
                text, etype = key_str.rsplit("||", 1)
                ekey: EntityKey = (text, etype)
                fn_entities[ekey] = form
            fn_passages: dict[str, list[EntityKey]] = {}
            for pid, keys in cand["fn_passages"].items():
                fn_passages[pid] = [
                    (k.rsplit("||", 1)[0], k.rsplit("||", 1)[1]) for k in keys
                ]
            cross_cands[doc_id] = {
                "fp_entities": {},
                "fn_entities": fn_entities,
                "fp_passages": {},
                "fn_passages": fn_passages,
            }
        print(
            f"  Loaded {len(cross_cands):,} docs with cross-doc candidates "
            f"(filtered to {len(input_doc_ids):,} input docs)"
        )
        for doc_cands in cross_cands.values():
            cross_doc_entity_set.update(doc_cands["fn_entities"].keys())
        candidates = merge_candidates(candidates, cross_cands)
    elif args.cross_document:
        # Compute cross-doc candidates from scratch
        cross_doc_corpus = getattr(args, "cross_doc_corpus", None)
        if cross_doc_corpus:
            print(f"\n  Loading cross-doc corpus: {cross_doc_corpus}")
            if use_occurrence:
                corpus_stats = aggregate_document_stats_v2(
                    cross_doc_corpus, documents_path
                )
            else:
                corpus_stats = aggregate_document_stats(cross_doc_corpus, 1)
            cross_doc_source = corpus_stats
            cross_doc_target = set(doc_stats.keys())
        else:
            cross_doc_source = doc_stats
            cross_doc_target = None

        cross_cands = cross_document_candidates(
            cross_doc_source,
            min_freq=args.cross_doc_min_freq,
            iqr_k=args.cross_doc_iqr_k,
            min_cond_prob=args.cross_doc_min_prob / 100.0,
            top_percentile=args.cross_doc_top_pct,
            max_per_doc=args.cross_doc_max_per_doc,
            min_cooccur=args.cross_doc_min_cooccur,
            stats_only=args.stats_only,
            target_doc_ids=cross_doc_target,
        )
        # Collect cross-doc entity keys for audit trail
        for doc_cands in cross_cands.values():
            cross_doc_entity_set.update(doc_cands["fn_entities"].keys())
        candidates = merge_candidates(candidates, cross_cands)

    if args.stats_only:
        print("\n--- Within-Document Stats ---")
        for doc_id, cand in sorted(
            candidates.items(),
            key=lambda x: len(x[1]["fp_passages"]) + len(x[1]["fn_passages"]),
            reverse=True,
        )[:20]:
            doc = doc_stats[doc_id]
            n_fp = len(cand["fp_passages"])
            n_fn = len(cand["fn_passages"])
            n_fp_ents = len(cand["fp_entities"])
            n_fn_ents = len(cand["fn_entities"])
            print(
                f"  {doc['title'][:50]:50s}  "
                f"passages={doc['total_passages']:4d}  "
                f"fp={n_fp_ents:4d}/{n_fp:4d}p  "
                f"fn={n_fn_ents:4d}/{n_fn:4d}p"
            )
        elapsed = time.time() - t0_total
        print(f"\nStats completed in {elapsed:.1f}s")
        return

    # Step C: LLM review
    urls = [u.strip() for u in args.vllm_urls.split(",") if u.strip()]
    pool = EndpointPool(urls)

    print("\nStep C: LLM review...")
    print(f"  Model: {args.model}")
    print(f"  Endpoints: {', '.join(urls)}")
    print(
        f"  Mode: {'occurrence-based (v2)' if use_occurrence else 'legacy thresholds'}"
    )

    # Resume support
    progress_path = args.output + ".progress.jsonl"
    completed_docs: set[str] = set()
    if args.resume and os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_docs.add(json.loads(line)["doc_id"])
        print(f"  Resuming: {len(completed_docs):,} documents already done")

    # Load all passages into memory (needed for passage lookup by id)
    print("  Loading passages into memory...")
    t_load = time.time()
    passages_by_id: dict[str, dict] = {}
    passage_order: list[str] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            pid = p.get("id", "")
            passages_by_id[pid] = p
            passage_order.append(pid)
    print(f"  Loaded {len(passages_by_id):,} passages in {time.time() - t_load:.1f}s")

    doc_ids_to_process = [did for did in candidates if did not in completed_docs]
    if args.limit:
        doc_ids_to_process = doc_ids_to_process[: args.limit]

    total_docs = len(doc_ids_to_process)
    print(f"  Documents to review: {total_docs:,}")

    all_refinements: dict[str, dict] = {}  # pid -> refinement
    processed_docs = 0
    t0 = time.time()

    # Count LLM calls for diagnostics
    if use_occurrence:
        # In occurrence mode: one call per entity + one per type conflict
        total_llm_calls = 0
        for doc_id in doc_ids_to_process:
            dc = candidates[doc_id]
            total_llm_calls += len(dc["fp_entities"]) + len(dc["fn_entities"])
            total_llm_calls += len(dc.get("type_conflicts", {}))
        print(f"  Total per-entity LLM calls: {total_llm_calls:,}")
    else:
        total_passage_calls = 0
        for doc_id in doc_ids_to_process:
            dc = candidates[doc_id]
            pids = set(dc["fp_passages"].keys()) | set(dc["fn_passages"].keys())
            total_passage_calls += len(pids)
        print(f"  Total passage-level LLM calls: {total_passage_calls:,}")

    # Process in chunks
    progress_mode = "a" if args.resume else "w"
    with open(progress_path, progress_mode, encoding="utf-8") as prog_f:
        for i in range(0, len(doc_ids_to_process), args.chunk_size):
            chunk_doc_ids = doc_ids_to_process[i : i + args.chunk_size]

            if use_occurrence:
                tasks = [
                    review_document_by_type(
                        pool,
                        args.model,
                        doc_id,
                        candidates[doc_id],
                        doc_stats[doc_id].get("full_text", ""),
                        doc_stats[doc_id].get("entities", {}),
                        doc_stats[doc_id].get("passage_ids", []),
                        prompt_dir=prompt_dir,
                    )
                    for doc_id in chunk_doc_ids
                ]
            else:
                tasks = [
                    review_document(
                        pool,
                        args.model,
                        doc_id,
                        candidates[doc_id],
                        passages_by_id,
                        prompt_dir=prompt_dir,
                    )
                    for doc_id in chunk_doc_ids
                ]

            results = await asyncio.gather(*tasks)

            for doc_id, doc_refinements in zip(chunk_doc_ids, results):
                all_refinements.update(doc_refinements)
                prog_f.write(
                    json.dumps({"doc_id": doc_id, "refined": len(doc_refinements)})
                    + "\n"
                )
                processed_docs += 1

            prog_f.flush()

            elapsed = time.time() - t0
            rate = processed_docs / elapsed if elapsed > 0 else 0
            eta = (total_docs - processed_docs) / rate if rate > 0 else 0
            print(
                f"  {processed_docs:>8,} / {total_docs:,} docs  "
                f"({processed_docs / total_docs * 100:5.1f}%)  "
                f"{rate:,.1f} docs/s  "
                f"refinements={len(all_refinements):,}  "
                f"ETA {eta / 60:.1f}min"
            )

    # Write output
    print(f"\nWriting output: {args.output}")
    t_write = time.time()
    total_written = 0
    total_refined = 0
    total_fp_removed = 0
    total_fn_added = 0
    total_cross_doc_added = 0
    total_gold_fn_added = 0
    total_external_fn_added = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for pid in passage_order:
            passage = passages_by_id[pid]
            if pid in all_refinements:
                passage = apply_refinement(
                    dict(passage),
                    all_refinements[pid],
                    cross_doc_entities=cross_doc_entity_set or None,
                    gold_entities=gold_entity_set or None,
                    external_entities=external_entity_set or None,
                )
                total_refined += 1
                ref = passage.get("refinement", {})
                total_fp_removed += len(ref.get("fp_removed", []))
                total_fn_added += len(ref.get("fn_added", []))
                total_cross_doc_added += len(ref.get("cross_doc_added", []))
                total_gold_fn_added += len(ref.get("gold_fn_added", []))
                total_external_fn_added += len(ref.get("external_fn_added", []))
            fout.write(json.dumps(passage, ensure_ascii=False) + "\n")
            total_written += 1

    elapsed_total = time.time() - t0_total
    print(f"  {total_written:,} passages written in {time.time() - t_write:.1f}s")
    print(f"\nRefinement complete in {elapsed_total:.1f}s:")
    print(f"  {processed_docs:,} documents reviewed")
    print(f"  {total_refined:,} passages modified")
    print(f"  {total_fp_removed:,} false-positive entities removed")
    print(f"  {total_fn_added:,} false-negative entities added (within-doc)")
    if total_cross_doc_added:
        print(f"  {total_cross_doc_added:,} cross-document entities added")
    if total_gold_fn_added:
        print(f"  {total_gold_fn_added:,} gold-standard FN entities added")
    if total_external_fn_added:
        print(f"  {total_external_fn_added:,} external FN entities added")
    print(f"  Output: {args.output}")

    _print_error_summary()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Refine entity annotations using document-level statistics"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to passages_enriched.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input_dir>/passages_refined.jsonl)",
    )
    parser.add_argument(
        "--vllm-urls",
        default=None,
        help="Comma-separated vLLM endpoint URLs for round-robin",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="Single vLLM endpoint URL (alias for --vllm-urls with one URL)",
    )
    parser.add_argument(
        "--model",
        default=VLLM_MODEL,
        help=f"Model name (default: {VLLM_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Documents per processing chunk (default: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from progress file",
    )
    parser.add_argument(
        "--fp-threshold",
        type=float,
        default=10,
        help="FP threshold: entity doc-ratio below this %% (default: 10)",
    )
    parser.add_argument(
        "--fn-threshold",
        type=float,
        default=80,
        help="FN threshold: entity doc-ratio above this %% (default: 80)",
    )
    parser.add_argument(
        "--min-doc-passages",
        type=int,
        default=5,
        help="Skip documents with fewer passages (default: 5)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Run aggregation + stats only, no LLM calls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of documents to refine (0=all)",
    )
    parser.add_argument(
        "--documents",
        default=None,
        help="Path to documents.jsonl for metadata zones (infobox, categories, appearances)",
    )
    parser.add_argument(
        "--legacy-thresholds",
        action="store_true",
        help="Use legacy percentage-based thresholds instead of occurrence-based rules",
    )

    # Cross-document arguments
    parser.add_argument(
        "--cross-document",
        action="store_true",
        help="Enable cross-document entity detection",
    )
    parser.add_argument(
        "--precomputed-crossdoc",
        default=None,
        help=(
            "Path to precomputed cross-doc candidates JSON (from fast_crossdoc.py). "
            "Skips the co-occurrence matrix computation and loads candidates directly."
        ),
    )
    parser.add_argument(
        "--cross-doc-corpus",
        default=None,
        help=(
            "Path to a larger extraction JSONL for building the co-occurrence "
            "probability matrix. Candidates are only generated for documents "
            "in --input, but probabilities use this corpus for volume."
        ),
    )
    parser.add_argument(
        "--cross-doc-min-freq",
        type=int,
        default=2,
        help="Skip entities in fewer than N docs (default: 2)",
    )
    parser.add_argument(
        "--cross-doc-min-cooccur",
        type=int,
        default=2,
        help="Min absolute co-occurrence count for candidates (default: 2)",
    )
    parser.add_argument(
        "--cross-doc-iqr-k",
        type=float,
        default=1.5,
        help="Tukey's fence multiplier for outlier removal (default: 1.5)",
    )
    parser.add_argument(
        "--cross-doc-min-prob",
        type=float,
        default=5.0,
        help="Floor for adaptive threshold in %% (default: 5.0)",
    )
    parser.add_argument(
        "--cross-doc-top-pct",
        type=float,
        default=95.0,
        help="Percentile of P(C|E) for adaptive threshold (default: 95.0)",
    )
    parser.add_argument(
        "--cross-doc-max-per-doc",
        type=int,
        default=30,
        help="Max cross-doc candidates per document (default: 30)",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory with prompt files (refine_fp.txt, refine_fn.txt, refine_combined.txt). If absent, uses hardcoded defaults.",
    )
    parser.add_argument(
        "--gold-entities",
        default=None,
        help="Gold standard JSONL path. Injects missing gold entities as FN candidates.",
    )
    parser.add_argument(
        "--external-entities",
        default=None,
        help="External extraction JSONL path. Injects frequent entities as FN candidates.",
    )
    parser.add_argument(
        "--external-min-doc-freq",
        type=int,
        default=5,
        help="Minimum document frequency for external entity injection (default: 5).",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.input), "passages_refined.jsonl"
        )

    # Resolve endpoint URLs: --vllm-urls > --vllm-url > default
    if args.vllm_urls is None:
        args.vllm_urls = args.vllm_url or "http://localhost:8000/v1"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
