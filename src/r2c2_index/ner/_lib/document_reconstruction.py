#!/usr/bin/env python3
"""Reconstruct full documents from passage-level JSONL.

Deterministic reconstruction: passages are grouped by doc_id and ordered by
the numeric sequence suffix of their id (chunk order). The canonical format is:

    {title}\n\n{section_1}\n\n{contents_1}\n\n{contents_2}\n\n{section_2}\n\n{contents_3}...

- Title appears once at top
- Each unique section heading appears once before its first passage
- Lead section (empty heading) has no heading marker
- Consecutive passages separated by \n\n

Usage:
    python document_reconstruction.py \
        --input data/sample_1000.jsonl \
        --output data/reconstructed_documents.jsonl
"""

import argparse
import json
import re  # noqa: E402 — used by _clean_markup below
from collections import defaultdict
from dataclasses import asdict, dataclass

__all__ = [
    "PassageOffset",
    "MetadataOffsets",
    "reconstruct_document",
    "reconstruct_document_with_metadata",
    "reconstruct_all",
]

SEPARATOR = "\n\n"


@dataclass
class PassageOffset:
    """Character offsets for a passage's components in the reconstructed document.

    Attributes:
        passage_id: Original passage ID.
        doc_id: Document ID.
        title_start: Start offset of the document title.
        title_end: End offset of the document title.
        section_start: Start offset of the section heading (-1 if lead section).
        section_end: End offset of the section heading (-1 if lead section).
        content_start: Start offset of the passage content.
        content_end: End offset of the passage content.
    """

    passage_id: str
    doc_id: str
    title_start: int
    title_end: int
    section_start: int
    section_end: int
    content_start: int
    content_end: int


def reconstruct_document(
    passages: list[dict],
) -> tuple[str, list[PassageOffset]]:
    """Reconstruct a document from its passages.

    Args:
        passages: List of passage dicts (must share the same doc_id),
            each with keys: id, doc_id, title, section, contents.

    Returns:
        Tuple of (reconstructed_text, list of PassageOffset).
    """
    if not passages:
        return "", []

    # The sequence suffix is compared as a number: as a string, 10 sorts before 2.
    def _sort_key(p: dict) -> tuple[str, int]:
        pid = p["id"]
        if "__" in pid:
            prefix, seq = pid.rsplit("__", 1)
            try:
                return (prefix, int(seq))
            except ValueError:
                pass
        return (pid, 0)

    sorted_passages = sorted(passages, key=_sort_key)

    doc_id = sorted_passages[0].get("doc_id", "")
    title = sorted_passages[0].get("title", "")

    parts: list[str] = []
    offsets: list[PassageOffset] = []
    cursor = 0

    parts.append(title)
    title_start = 0
    title_end = len(title)
    cursor = title_end

    seen_sections: dict[str, tuple[int, int]] = {}  # section -> (start, end)

    for p in sorted_passages:
        section = p.get("section", "")
        contents = p.get("contents", "")

        section_start = -1
        section_end = -1

        if section and section not in seen_sections:
            parts.append(SEPARATOR)
            cursor += len(SEPARATOR)
            section_start = cursor
            parts.append(section)
            cursor += len(section)
            section_end = cursor
            seen_sections[section] = (section_start, section_end)
        elif section and section in seen_sections:
            # A section heading appears once in the document, so later passages
            # of the same section share the offset of its first occurrence.
            section_start, section_end = seen_sections[section]

        parts.append(SEPARATOR)
        cursor += len(SEPARATOR)
        content_start = cursor
        parts.append(contents)
        cursor += len(contents)
        content_end = cursor

        offsets.append(
            PassageOffset(
                passage_id=p["id"],
                doc_id=doc_id,
                title_start=title_start,
                title_end=title_end,
                section_start=section_start,
                section_end=section_end,
                content_start=content_start,
                content_end=content_end,
            )
        )

    doc_text = "".join(parts)
    return doc_text, offsets


_WIKI_MARKUP_RE = re.compile(r"'{2,3}")


def _clean_markup(text: str) -> str:
    """Strip wiki bold/italic markers."""
    return _WIKI_MARKUP_RE.sub("", text).strip()


@dataclass
class MetadataOffsets:
    """Character offsets for document metadata sections in the reconstructed text.

    Attributes:
        infobox_start: Start offset of infobox section (-1 if absent).
        infobox_end: End offset of infobox section (-1 if absent).
        categories_start: Start offset of categories section (-1 if absent).
        categories_end: End offset of categories section (-1 if absent).
        appearances_start: Start offset of appearances section (-1 if absent).
        appearances_end: End offset of appearances section (-1 if absent).
    """

    infobox_start: int = -1
    infobox_end: int = -1
    categories_start: int = -1
    categories_end: int = -1
    appearances_start: int = -1
    appearances_end: int = -1


def reconstruct_document_with_metadata(
    passages: list[dict],
    doc_record: dict | None = None,
) -> tuple[str, list[PassageOffset], MetadataOffsets]:
    """Reconstruct a document including metadata for BIO evaluation.

    Extends reconstruct_document by appending infobox values, categories,
    and appearances after the passage content. This produces a deterministic
    text that covers all entity-bearing fields.

    The metadata is appended in a fixed order:
        {passage_text}\\n\\n{infobox_values}\\n\\n{categories}\\n\\n{appearances}

    Args:
        passages: List of passage dicts (must share the same doc_id).
        doc_record: Optional document metadata dict from documents.jsonl.
            If None, behaves identically to reconstruct_document.

    Returns:
        Tuple of (doc_text, passage_offsets, metadata_offsets).
    """
    doc_text, offsets = reconstruct_document(passages)
    meta_offsets = MetadataOffsets()

    if doc_record is None:
        return doc_text, offsets, meta_offsets

    parts: list[str] = [doc_text]
    cursor = len(doc_text)

    # Infobox: flatten param values into "key: value" lines
    infobox = doc_record.get("infobox")
    if infobox and infobox.get("params"):
        ib_lines = []
        for key, value in infobox["params"].items():
            if isinstance(value, str) and value.strip():
                ib_lines.append(f"{key}: {_clean_markup(value)}")
        if ib_lines:
            parts.append(SEPARATOR)
            cursor += len(SEPARATOR)
            meta_offsets.infobox_start = cursor
            ib_text = "\n".join(ib_lines)
            parts.append(ib_text)
            cursor += len(ib_text)
            meta_offsets.infobox_end = cursor

    categories = doc_record.get("categories", [])
    if categories:
        parts.append(SEPARATOR)
        cursor += len(SEPARATOR)
        meta_offsets.categories_start = cursor
        cat_text = "\n".join(categories)
        parts.append(cat_text)
        cursor += len(cat_text)
        meta_offsets.categories_end = cursor

    appearances = doc_record.get("appearances", [])
    if appearances:
        parts.append(SEPARATOR)
        cursor += len(SEPARATOR)
        meta_offsets.appearances_start = cursor
        app_text = "\n".join(appearances)
        parts.append(app_text)
        cursor += len(app_text)
        meta_offsets.appearances_end = cursor

    doc_text = "".join(parts)
    return doc_text, offsets, meta_offsets


def reconstruct_all(
    passages_path: str,
) -> dict[str, tuple[str, list[PassageOffset], list[dict]]]:
    """Reconstruct all documents from a passages JSONL file.

    Args:
        passages_path: Path to passages JSONL.

    Returns:
        Dict mapping doc_id -> (doc_text, offsets, passages).
    """
    doc_passages: dict[str, list[dict]] = defaultdict(list)

    with open(passages_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id", record["id"].rsplit("__", 1)[0])
            doc_passages[doc_id].append(record)

    results: dict[str, tuple[str, list[PassageOffset], list[dict]]] = {}
    for doc_id, plist in doc_passages.items():
        doc_text, offsets = reconstruct_document(plist)
        results[doc_id] = (doc_text, offsets, plist)

    return results


def main() -> None:
    """CLI entry point: reconstruct documents and write to JSONL."""
    parser = argparse.ArgumentParser(
        description="Reconstruct documents from passage-level JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to passages JSONL (with id, doc_id, title, section, contents).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for reconstructed documents JSONL.",
    )
    parser.add_argument(
        "--documents",
        default=None,
        help="Path to documents.jsonl (metadata: infobox, categories, appearances). "
        "If provided, metadata is appended to the reconstructed text.",
    )

    args = parser.parse_args()

    doc_meta: dict[str, dict] = {}
    if args.documents:
        print(f"Loading document metadata from {args.documents}...")
        with open(args.documents, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc_meta[rec["doc_id"]] = rec
        print(f"  {len(doc_meta)} documents loaded")

    print(f"Reading passages from {args.input}...")
    all_docs = reconstruct_all(args.input)
    print(f"Reconstructed {len(all_docs)} documents.")

    with open(args.output, "w", encoding="utf-8") as out:
        for doc_id, (_, offsets, passages) in sorted(all_docs.items()):
            doc_record = doc_meta.get(doc_id)
            if doc_record is not None:
                doc_text, offsets, meta = reconstruct_document_with_metadata(
                    passages, doc_record
                )
            else:
                doc_text, offsets = reconstruct_document(passages)
                meta = MetadataOffsets()

            first = passages[0] if passages else {}
            record = {
                "doc_id": doc_id,
                "doc_text": doc_text,
                "title": first.get("title", ""),
                "passage_offsets": [asdict(o) for o in offsets],
                "metadata_offsets": asdict(meta),
                "categories": first.get("categories", []),
                "references": first.get("references", []),
                "source": first.get("source", ""),
                "n_passages": len(passages),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Written to {args.output}")
    if doc_meta:
        n_with_meta = sum(1 for d in all_docs if d in doc_meta)
        print(f"  {n_with_meta}/{len(all_docs)} documents enriched with metadata")


if __name__ == "__main__":
    main()
