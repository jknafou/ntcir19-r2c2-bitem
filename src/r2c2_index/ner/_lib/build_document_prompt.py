#!/usr/bin/env python3
"""Build XML-zoned document text for LLM prompting.

Renders a document's metadata (infobox, categories, appearances) and
passage text into a structured XML format that gives the LLM full
document context for entity extraction.

The XML format is only an input to the LLM and is neither stored nor
evaluated. BIO evaluation uses document_reconstruction.py, which produces a
deterministic plain-text concatenation.

Usage:
    # Single document preview:
    python build_document_prompt.py --doc-id 255020 \
        --passages data/sample_1000.jsonl \
        --documents data/documents.jsonl

    # As library:
    from build_document_prompt import build_document_text, load_documents_index
    docs = load_documents_index("data/documents.jsonl")
    text = build_document_text(docs["255020"], passages_for_doc)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

__all__ = ["build_document_text", "load_documents_index"]

_WIKI_MARKUP_RE = re.compile(r"'{2,3}")


def _clean(text: str) -> str:
    """Strip wiki bold/italic markers."""
    return _WIKI_MARKUP_RE.sub("", text).strip()


def build_document_text(
    doc_record: dict,
    passages: list[dict],
) -> str:
    """Render a document as XML-zoned text for LLM prompting.

    Zones produced:
        <TITLE>         — document title
        <INFOBOX>       — infobox key-value pairs (if present)
        <PASSAGE>       — one per passage, with section/type attributes
        <CATEGORIES>    — category list (if present)
        <APPEARANCES>   — appearance/reference list (if present)

    Args:
        doc_record: Document metadata dict from documents.jsonl
            (doc_id, title, source, infobox, categories, appearances, ...).
        passages: List of passage dicts for this document, each with
            id, contents, title, section, source, doc_id.

    Returns:
        XML-zoned text string for LLM input.
    """
    parts: list[str] = []

    title = doc_record.get("title", "")
    if title:
        parts.append(f"<TITLE>\n{_clean(title)}\n</TITLE>")

    infobox = doc_record.get("infobox")
    if infobox and infobox.get("params"):
        ib_lines = []
        template = infobox.get("template", "")
        if template:
            ib_lines.append(f"Template: {template}")
        for key, value in infobox["params"].items():
            if isinstance(value, str) and value.strip():
                ib_lines.append(f"{key}: {_clean(value)}")
        if ib_lines:
            parts.append("<INFOBOX>\n" + "\n".join(ib_lines) + "\n</INFOBOX>")

    # Passages are sorted by id so that the output is deterministic.
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

    for p in sorted_passages:
        section = p.get("section", "")
        ptype = p.get("type", "prose")
        contents = _clean(p.get("contents", ""))
        attrs = []
        if section:
            attrs.append(f'section="{section}"')
        if ptype and ptype != "prose":
            attrs.append(f'type="{ptype}"')
        attr_str = " " + " ".join(attrs) if attrs else ""
        parts.append(f"<PASSAGE{attr_str}>\n{contents}\n</PASSAGE>")

    categories = doc_record.get("categories", [])
    if categories:
        parts.append("<CATEGORIES>\n" + "\n".join(categories) + "\n</CATEGORIES>")

    appearances = doc_record.get("appearances", [])
    if appearances:
        parts.append("<APPEARANCES>\n" + "\n".join(appearances) + "\n</APPEARANCES>")

    return "\n\n".join(parts)


def load_documents_index(path: str) -> dict[str, dict]:
    """Load documents.jsonl and index by doc_id.

    Args:
        path: Path to documents.jsonl.

    Returns:
        Dict mapping doc_id -> document record dict.
    """
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index[record["doc_id"]] = record
    return index


def load_passages_by_doc(path: str) -> dict[str, list[dict]]:
    """Load passages.jsonl grouped by doc_id.

    Args:
        path: Path to passages.jsonl.

    Returns:
        Dict mapping doc_id -> list of passage dicts.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id", record["id"].rsplit("__", 1)[0])
            groups[doc_id].append(record)
    return dict(groups)


def main() -> None:
    """CLI entry point: preview XML-zoned document text."""
    parser = argparse.ArgumentParser(
        description="Build XML-zoned document text for LLM prompting."
    )
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document ID to render.",
    )
    parser.add_argument(
        "--passages",
        required=True,
        help="Path to passages.jsonl.",
    )
    parser.add_argument(
        "--documents",
        required=True,
        help="Path to documents.jsonl.",
    )
    args = parser.parse_args()

    docs = load_documents_index(args.documents)
    if args.doc_id not in docs:
        print(
            f"ERROR: doc_id {args.doc_id} not found in {args.documents}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_passages = load_passages_by_doc(args.passages)
    passages = all_passages.get(args.doc_id, [])
    if not passages:
        print(f"WARNING: No passages found for doc_id {args.doc_id}", file=sys.stderr)

    doc_record = docs[args.doc_id]
    text = build_document_text(doc_record, passages)

    print(f"--- Document {args.doc_id}: {doc_record.get('title', '?')} ---")
    print(f"--- {len(passages)} passages, {len(text)} chars ---")
    print()
    print(text)


if __name__ == "__main__":
    main()
