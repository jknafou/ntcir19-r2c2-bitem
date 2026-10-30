"""Universal chunker: structured JSON → passages.jsonl.

Reads structured_docs.jsonl (produced by collection-specific parsers)
and writes passages.jsonl, documents.jsonl and corpus_meta.json. The
passage format is the one the downstream tools read (Pyserini indexing,
NER extraction, evaluation).

Passage format, one per line:
    {"id": "<doc_id>__<chunk_idx>", "contents": "...", "type": "...",
     "ner": true, "title": "...", "section": "...", "source": "...",
     "doc_id": "...", "raw_char_offset": 0, "raw_char_length": 0,
     "word_count": 0}

Usage:
    python chunker.py \
        --input structured_docs.jsonl \
        --output-dir ../data/chunks-v2 \
        --target-words 200 --max-words 300
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

__all__ = [
    "split_sentences",
    "chunk_sentences",
    "render_infobox_prose",
    "render_quote",
    "make_document_record",
    "chunk_structured_doc",
    "build_norm_doc_text",
    "_merge_small_passages",
    "main",
]

# Sentence splitting

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\(])')

_ABBREV = {
    "Dr",
    "Mr",
    "Mrs",
    "Ms",
    "St",
    "Jr",
    "Sr",
    "Prof",
    "Gen",
    "Gov",
    "Vol",
    "vs",
    "etc",
    "Inc",
    "Ltd",
    "Corp",
    "Dept",
    "Univ",
    "approx",
    "i.e",
    "e.g",
    "ca",
    "ft",
    "lb",
    "oz",
    "no",
    "No",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at paragraph, line, and sentence boundaries.

    Single-newline-separated lines (e.g. cast lists, bulleted lists) are
    treated as individual items. Double newlines separate paragraphs.
    Within a paragraph, sentences are split at .!? boundaries.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    sentences: list[str] = []

    for para in paragraphs:
        # Split on single newlines first — each line may be a list item
        lines = para.split("\n")
        for line in lines:
            line = " ".join(line.split())
            if not line:
                continue
            parts = _SENT_RE.split(line)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)

    return sentences


# Chunking


def chunk_sentences(
    sentences: list[str],
    target_words: int = 200,
    max_words: int = 300,
) -> list[str]:
    """Group sentences into chunks targeting a word count range.

    Args:
        sentences: List of sentences to chunk.
        target_words: Flush buffer when this count is reached.
        max_words: Hard ceiling — flush before exceeding.

    Returns:
        List of chunk strings.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buf_wc = 0

    for sent in sentences:
        words = sent.split()
        sent_wc = len(words)

        # A sentence longer than max_words is split at word boundaries
        if sent_wc > max_words:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buf_wc = 0
            for i in range(0, sent_wc, target_words):
                chunks.append(" ".join(words[i : i + target_words]))
            continue

        if buf_wc + sent_wc > max_words and buffer:
            chunks.append(" ".join(buffer))
            buffer = list(words)
            buf_wc = sent_wc
        else:
            buffer.extend(words)
            buf_wc += sent_wc

        if buf_wc >= target_words:
            chunks.append(" ".join(buffer))
            buffer = []
            buf_wc = 0

    # A short remainder joins the last chunk instead of standing alone
    if buffer:
        text = " ".join(buffer)
        if chunks and buf_wc < target_words // 3:
            chunks[-1] += " " + text
        else:
            chunks.append(text)

    return chunks


# Rendering helpers (structured → prose for chunking)


# Infobox fields to skip in prose rendering
_SKIP_INFOBOX_KEYS = {"image", "imagebg", "img", "photo", "caption", "image_size"}

# Placeholder values commonly used in unfilled Wookieepedia infobox fields
_PLACEHOLDER_VALUES = {"new", "unknown", "tbd", "none", "n/a"}


def render_infobox_prose(infobox: dict | None) -> str:
    """Render infobox as a compact prose summary.

    Only non-empty fields are included. Lists are joined with commas.
    Skips image fields, captions, and placeholder values.
    """
    if not infobox or not infobox.get("params"):
        return ""

    params = infobox["params"]
    parts: list[str] = []

    for key, value in params.items():
        if not value:
            continue
        if key.lower() in _SKIP_INFOBOX_KEYS:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v)
        if not value:
            continue
        if isinstance(value, str) and value.lower().strip() in _PLACEHOLDER_VALUES:
            continue
        parts.append(f"{key}: {value}")

    if not parts:
        return ""
    return "; ".join(parts) + "."


def render_quote(quote: dict) -> str:
    """Render a quote as inline prose: "text" — speaker."""
    text = quote.get("text", "").strip()
    speaker = quote.get("speaker", "").strip()
    if not text:
        return ""
    if speaker:
        return f'"{text}" \u2014 {speaker}'
    return f'"{text}"'


def _render_list_content(text: str) -> str:
    """Detect list-like content and join items with semicolons.

    If the text has multiple newline-separated short lines (avg < 15 words,
    no sentence-ending punctuation), treat as a list and join with " ; ".
    Otherwise return as-is for normal prose processing.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return text

    # Check if this looks like a list: short lines, few sentence endings
    avg_words = sum(len(l.split()) for l in lines) / len(lines)
    lines_with_period = sum(1 for l in lines if l.rstrip().endswith((".", "!", "?")))

    if avg_words < 15 and lines_with_period < len(lines) * 0.3:
        return " ; ".join(lines)
    return text


# Section tree walking


def _walk_sections(
    sections: list[dict],
    parent_path: str = "",
) -> list[tuple[str, str, list[dict], list[dict]]]:
    """Depth-first walk of section tree.

    Yields (section_path, content, quotes, tables) tuples.
    Subsection path uses " > " separator: "History > Clone Wars".
    """
    results: list[tuple[str, str, list[dict], list[dict]]] = []
    for section in sections:
        heading = section.get("heading", "")
        path = f"{parent_path} > {heading}" if parent_path else heading
        content = section.get("content", "")
        quotes = section.get("quotes", [])
        tables = section.get("tables", [])
        results.append((path, content, quotes, tables))

        subsections = section.get("subsections", [])
        if subsections:
            results.extend(_walk_sections(subsections, path))

    return results


# Main chunking logic


def _merge_small_passages(
    passages: list[dict],
    min_words: int = 50,
    max_words: int = 300,
) -> list[dict]:
    """Merge consecutive small passages to reduce empty/tiny chunk rate.

    Passages under min_words are merged with the previous passage if the
    combined result stays under max_words. A small passage with no
    previous passage, or one that would not fit, is kept as it is.

    Args:
        passages: List of passage dicts with "contents" field.
        min_words: Passages under this word count are candidates for merging.
        max_words: Combined passage must not exceed this.

    Returns:
        New list of passage dicts with small passages merged.
    """
    if len(passages) <= 1:
        return passages

    merged: list[dict] = []
    for p in passages:
        wc = len(p["contents"].split())
        if wc >= min_words or not merged:
            merged.append(dict(p))  # shallow copy
            continue

        prev = merged[-1]
        prev_wc = len(prev["contents"].split())
        if prev_wc + wc <= max_words:
            prev["contents"] = prev["contents"] + " " + p["contents"]
        else:
            merged.append(dict(p))

    return merged


def _render_table_text(table: dict) -> str:
    """Render a wiki table as semicolon-delimited text.

    Each row becomes: "Header1: val1 ; Header2: val2 ; ..."
    Rows are separated by newlines.
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    if not rows:
        return ""

    lines: list[str] = []
    if table.get("caption"):
        lines.append(table["caption"])
    for row in rows:
        row_parts = []
        for i, cell in enumerate(row):
            if i < len(headers) and headers[i]:
                row_parts.append(f"{headers[i]}: {cell}")
            else:
                row_parts.append(str(cell))
        lines.append(" ; ".join(row_parts))

    return "\n".join(lines)


# Maximum words for a single table passage before splitting
_TABLE_MAX_WORDS = 1000


def make_document_record(doc: dict) -> dict | None:
    """Build a document-level record with all metadata for NER + MCP retrieval.

    This is the single source of truth for document metadata. NER runs on
    this record (all pieces), and the MCP tool retrieve_doc_w_meta returns it.

    Args:
        doc: Structured document dict from parser.

    Returns:
        Document record dict, or None if redirect.
    """
    if doc.get("is_redirect"):
        return None

    infobox = doc.get("infobox")
    clean_infobox = None
    if infobox and infobox.get("params"):
        clean_params = {
            k: v
            for k, v in infobox["params"].items()
            if not (isinstance(v, str) and v.lower().strip() in _PLACEHOLDER_VALUES)
            and k.lower() not in _SKIP_INFOBOX_KEYS
        }
        if clean_params:
            clean_infobox = {
                "template": infobox.get("template", ""),
                "params": clean_params,
            }

    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "source": doc["source"],
        "infobox": clean_infobox,
        "categories": doc.get("categories", []),
        "appearances": doc.get("appearances", []),
        "sources_refs": doc.get("sources_refs", []),
        "interlang": doc.get("interlang", {}),
    }


def chunk_structured_doc(
    doc: dict,
    target_words: int = 200,
    max_words: int = 300,
    min_words: int = 50,
) -> list[dict]:
    """Chunk a structured document into lean passages.

    Passages contain only retrieval content — no document-level metadata.
    Metadata (infobox, categories, appearances) lives in documents.jsonl
    and is accessed via doc_id linkage.

    Passage types:
        - type="prose": natural language text, chunked if long
        - type="table": tabular data kept as single passage, split only if >1000w

    Args:
        doc: Structured document dict.
        target_words: Target words per prose chunk.
        max_words: Maximum words per prose chunk.
        min_words: Prose passages under this are merged with neighbors.

    Returns:
        List of lean passage dicts.
    """
    if doc.get("is_redirect"):
        return []

    doc_id = doc["doc_id"]
    title = doc["title"]
    source = doc["source"]

    passages: list[dict] = []
    chunk_idx = 0

    def _make_passage(
        text: str, section: str, ptype: str = "prose", ner: bool = True
    ) -> dict | None:
        nonlocal chunk_idx
        text = text.strip()
        if not text:
            return None
        chunk_idx += 1
        return {
            "id": f"{doc_id}__{chunk_idx:05d}",
            "contents": text,
            "type": ptype,
            "ner": ner,
            "title": title,
            "section": section,
            "source": source,
            "doc_id": doc_id,
        }

    # Lead section (prose)
    lead = doc.get("lead", "").strip()
    if lead:
        lead_sentences = split_sentences(lead)
        if lead_sentences:
            lead_chunks = chunk_sentences(lead_sentences, target_words, max_words)
            for chunk_text in lead_chunks:
                p = _make_passage(chunk_text, "", "prose")
                if p:
                    passages.append(p)

    # Sections (depth-first walk)
    sections = doc.get("sections", [])
    skip = {
        "see also",
        "references",
        "external links",
        "further reading",
        "notes",
        "bibliography",
        "footnotes",
        "citations",
        "notes and references",
        "sources and notes",
        "appearances",
        "sources",
        "contents",
        "cover gallery",
        "editions",
    }

    for section_path, content, quotes, tables in _walk_sections(sections):
        leaf_heading = section_path.split(" > ")[-1] if section_path else ""
        if leaf_heading.lower().strip() in skip:
            continue

        # Prose content + quotes → type="prose", chunked
        prose_parts: list[str] = []

        if content.strip():
            content_text = _render_list_content(content.strip())
            prose_parts.append(content_text)

        for q in quotes:
            rendered = render_quote(q)
            if rendered:
                prose_parts.append(rendered)

        if prose_parts:
            prose_text = "\n\n".join(prose_parts)
            sentences = split_sentences(prose_text)
            if sentences:
                prose_chunks = chunk_sentences(sentences, target_words, max_words)
                for chunk_text in prose_chunks:
                    p = _make_passage(chunk_text, section_path, "prose")
                    if p:
                        passages.append(p)

        # Tables → type="table", kept as single passages
        for table in tables:
            table_text = _render_table_text(table)
            if not table_text.strip():
                continue

            table_wc = len(table_text.split())
            if table_wc <= _TABLE_MAX_WORDS:
                p = _make_passage(table_text, section_path, "table")
                if p:
                    passages.append(p)
            else:
                # A very large table is split at row boundaries
                lines = table_text.split("\n")
                buf: list[str] = []
                buf_wc = 0
                for line in lines:
                    lwc = len(line.split())
                    if buf_wc + lwc > _TABLE_MAX_WORDS and buf:
                        p = _make_passage("\n".join(buf), section_path, "table")
                        if p:
                            passages.append(p)
                        buf = [line]
                        buf_wc = lwc
                    else:
                        buf.append(line)
                        buf_wc += lwc
                if buf:
                    p = _make_passage("\n".join(buf), section_path, "table")
                    if p:
                        passages.append(p)

    # Top-level tables (extracted globally by parser)
    for table in doc.get("tables", []):
        table_text = _render_table_text(table)
        if not table_text.strip():
            continue

        table_wc = len(table_text.split())
        if table_wc <= _TABLE_MAX_WORDS:
            p = _make_passage(table_text, "", "table")
            if p:
                passages.append(p)
        else:
            lines = table_text.split("\n")
            buf: list[str] = []
            buf_wc = 0
            for line in lines:
                lwc = len(line.split())
                if buf_wc + lwc > _TABLE_MAX_WORDS and buf:
                    p = _make_passage("\n".join(buf), "", "table")
                    if p:
                        passages.append(p)
                    buf = [line]
                    buf_wc = lwc
                else:
                    buf.append(line)
                    buf_wc += lwc
            if buf:
                p = _make_passage("\n".join(buf), "", "table")
                if p:
                    passages.append(p)

    # Only small prose passages are merged; tables stay whole
    prose_passages = [p for p in passages if p["type"] == "prose"]
    table_passages = [p for p in passages if p["type"] == "table"]

    prose_passages = _merge_small_passages(prose_passages, min_words, max_words)
    passages = prose_passages + table_passages

    # Merging leaves gaps in the passage IDs, so they are renumbered
    for i, p in enumerate(passages):
        p["id"] = f"{doc_id}__{i + 1:05d}"

    # Compute offsets into normalized document text
    _add_offsets(passages)

    return passages


# Offset computation (post-hoc, after chunking + merging)


def _add_offsets(passages: list[dict]) -> None:
    """Add raw_char_offset, raw_char_length, word_count to each passage.

    Offsets point into a "normalized document text" defined as all passage
    contents joined by ``\\n``. This text is constructed post-hoc after
    chunking and merging, guaranteeing that each passage's ``contents``
    is a literal substring at its reported offset.
    """
    cursor = 0
    for p in passages:
        text = p["contents"]
        p["raw_char_offset"] = cursor
        p["raw_char_length"] = len(text)
        p["word_count"] = len(text.split())
        cursor += len(text) + 1  # +1 for \n separator


def build_norm_doc_text(passages: list[dict]) -> str:
    """Build the normalized document text from final passage contents.

    Returns the concatenation of all passage contents separated by
    newlines. This is the text that ``get_raw_document`` returns and
    that ``raw_char_offset`` indexes into.
    """
    return "\n".join(p["contents"] for p in passages)


# CLI


def main() -> None:
    """CLI entry point for universal chunker."""
    ap = argparse.ArgumentParser(
        description="Universal chunker: structured JSON → passages.jsonl",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to structured_docs.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for passages.jsonl and corpus_meta.json.",
    )
    ap.add_argument(
        "--target-words",
        type=int,
        default=200,
        help="Target words per chunk (default: 200).",
    )
    ap.add_argument(
        "--max-words",
        type=int,
        default=300,
        help="Maximum words per chunk (default: 300).",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "passages.jsonl"

    total_docs = 0
    total_chunks = 0
    source_counts: dict[str, int] = {}

    doc_file = out_dir / "documents.jsonl"

    print(f"Chunking (target={args.target_words}w, max={args.max_words}w)")
    print(f"Input: {input_path}")
    print(f"Output: {out_file} + {doc_file}")
    print()

    with (
        open(input_path, encoding="utf-8") as f_in,
        open(out_file, "w", encoding="utf-8") as f_out,
        open(doc_file, "w", encoding="utf-8") as f_doc,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON: {e}", file=sys.stderr)
                continue

            # Lean passages (content only, with offsets)
            passages = chunk_structured_doc(
                doc,
                target_words=args.target_words,
                max_words=args.max_words,
            )

            for p in passages:
                f_out.write(json.dumps(p, ensure_ascii=False) + "\n")

            # Document-level record (metadata + norm_doc_text)
            doc_record = make_document_record(doc)
            if doc_record:
                doc_record["norm_doc_text"] = build_norm_doc_text(passages)
                doc_record["passage_count"] = len(passages)
                f_doc.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            total_docs += 1
            src = doc.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + len(passages)
            total_chunks += len(passages)

    meta = {"total": total_chunks, "total_docs": total_docs, **source_counts}
    meta_file = out_dir / "corpus_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {total_docs} documents → {total_chunks} passages")
    print(f"Passages: {out_file}")
    print(f"Documents: {doc_file}")
    print(f"Metadata: {meta_file}")
    print(f"Counts: {json.dumps(source_counts)}")


if __name__ == "__main__":
    main()
