#!/usr/bin/env python3
"""Span-level entity utilities for NER evaluation.

Provides functions to find entity spans (character offsets) in text,
merge passage-level spans to document-level, and resolve overlaps.

Supports multi-label entities: the same text position can carry multiple
entity types. Overlap resolution is per-type (longest wins within a type),
while different types at the same position coexist.

All spans use the format:
    {"text": str, "type": str, "start": int, "end": int, "zone": str}

where zone is one of "title", "section", "content".
"""

import re
from typing import TypedDict

__all__ = [
    "SpanDict",
    "find_entity_spans",
    "merge_passage_spans_to_document",
    "resolve_overlaps",
]


class SpanDict(TypedDict):
    """Entity span with character offsets."""

    text: str
    type: str
    start: int
    end: int
    zone: str


def find_entity_spans(
    text: str,
    entities: list[dict],
    zone: str,
) -> list[SpanDict]:
    """Find all non-overlapping occurrences of entities in text.

    Entities are processed longest-first so that "New York City" is found
    before "New York". Within each entity type, all non-overlapping matches
    are returned. The same text position can be matched by different entity
    types (multi-label).

    Args:
        text: Source text to search in.
        entities: List of {"text": str, "type": str} dicts.
        zone: Zone label ("title", "section", or "content").

    Returns:
        List of SpanDict with character offsets, sorted by start position.
    """
    if not text or not entities:
        return []

    seen: set[tuple[str, str]] = set()
    unique_entities: list[dict] = []
    for e in entities:
        key = (e["text"].lower(), e["type"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    # Longest first, so that a longer name claims its positions before a shorter one inside it
    unique_entities.sort(key=lambda e: len(e["text"]), reverse=True)

    # Per-type occupied positions: same position can be claimed by different types
    occupied_by_type: dict[str, set[int]] = {}
    spans: list[SpanDict] = []

    for entity in unique_entities:
        etype = entity["type"]
        if etype not in occupied_by_type:
            occupied_by_type[etype] = set()
        occupied = occupied_by_type[etype]

        pattern = re.compile(re.escape(entity["text"]), re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            # Only positions claimed by the same type block a match
            match_positions = set(range(start, end))
            if match_positions & occupied:
                continue
            occupied.update(match_positions)
            spans.append(
                SpanDict(
                    text=match.group(),
                    type=etype,
                    start=start,
                    end=end,
                    zone=zone,
                )
            )

    spans.sort(key=lambda s: (s["start"], s["type"]))
    return spans


def merge_passage_spans_to_document(
    passage_spans: list[dict],
    passage_offsets: list[dict],
) -> list[SpanDict]:
    """Map passage-level spans to document-level offsets.

    For each passage's spans:
    - Content spans: offset by passage's content_start
    - Title spans: mapped to document title offset (union across passages)
    - Section spans: mapped to section offset (union across passages sharing section)

    Deduplicates and resolves overlaps at document level.

    Args:
        passage_spans: List of dicts, each with:
            - passage_id: str
            - content_spans: list[SpanDict]
            - title_spans: list[SpanDict]
            - section_spans: list[SpanDict]
        passage_offsets: List of PassageOffset dicts (from document_reconstruction).

    Returns:
        List of document-level SpanDicts, sorted by start, overlaps resolved.
    """
    offset_map: dict[str, dict] = {}
    for po in passage_offsets:
        offset_map[po["passage_id"]] = po

    doc_spans: list[SpanDict] = []
    seen: set[tuple[int, int, str]] = set()

    for ps in passage_spans:
        pid = ps["passage_id"]
        if pid not in offset_map:
            continue
        po = offset_map[pid]

        for span in ps.get("content_spans", []):
            doc_start = po["content_start"] + span["start"]
            doc_end = po["content_start"] + span["end"]
            key = (doc_start, doc_end, span["type"])
            if key not in seen:
                seen.add(key)
                doc_spans.append(
                    SpanDict(
                        text=span["text"],
                        type=span["type"],
                        start=doc_start,
                        end=doc_end,
                        zone="content",
                    )
                )

        # The title is shared by all passages of a document, so its spans are
        # deduplicated through `seen`
        if po["title_start"] >= 0 and po["title_end"] > po["title_start"]:
            for span in ps.get("title_spans", []):
                doc_start = po["title_start"] + span["start"]
                doc_end = po["title_start"] + span["end"]
                key = (doc_start, doc_end, span["type"])
                if key not in seen:
                    seen.add(key)
                    doc_spans.append(
                        SpanDict(
                            text=span["text"],
                            type=span["type"],
                            start=doc_start,
                            end=doc_end,
                            zone="title",
                        )
                    )

        # Likewise for a section heading shared by several passages
        if po["section_start"] >= 0 and po["section_end"] > po["section_start"]:
            for span in ps.get("section_spans", []):
                doc_start = po["section_start"] + span["start"]
                doc_end = po["section_start"] + span["end"]
                key = (doc_start, doc_end, span["type"])
                if key not in seen:
                    seen.add(key)
                    doc_spans.append(
                        SpanDict(
                            text=span["text"],
                            type=span["type"],
                            start=doc_start,
                            end=doc_end,
                            zone="section",
                        )
                    )

    return resolve_overlaps(doc_spans)


def resolve_overlaps(spans: list[SpanDict]) -> list[SpanDict]:
    """Resolve overlapping spans, allowing multi-label at the same position.

    Spans of different types may coexist at the same character positions
    (multi-label). Overlaps are only resolved between spans of the same
    type, where the longest span wins (greedy longest-first strategy).

    Exact duplicates (same start, end, and type) are removed.

    Args:
        spans: List of SpanDicts, possibly overlapping.

    Returns:
        SpanDicts sorted by start position, with same-type overlaps resolved
        but cross-type overlaps preserved.
    """
    if not spans:
        return []

    seen: set[tuple[int, int, str]] = set()
    unique_spans: list[SpanDict] = []
    for span in spans:
        key = (span["start"], span["end"], span["type"])
        if key not in seen:
            seen.add(key)
            unique_spans.append(span)

    by_type: dict[str, list[SpanDict]] = {}
    for span in unique_spans:
        by_type.setdefault(span["type"], []).append(span)

    result: list[SpanDict] = []
    for entity_type, type_spans in by_type.items():
        # Sort by length desc, then start asc for deterministic tie-breaking
        sorted_spans = sorted(
            type_spans,
            key=lambda s: (-(s["end"] - s["start"]), s["start"]),
        )
        occupied: set[int] = set()
        for span in sorted_spans:
            positions = set(range(span["start"], span["end"]))
            if positions & occupied:
                continue
            occupied.update(positions)
            result.append(span)

    result.sort(key=lambda s: (s["start"], s["type"]))
    return result
