"""Structured chunker: wikitext → parse → chunk → passages.jsonl.

Uses collection-specific parsers (mwparserfromhell) for template
extraction, then the universal chunker for passage generation.

Output format per line:
    {"id": "<doc_id>__<chunk_idx>", "contents": "...", "title": "...",
     "section": "...", "source": "...", "doc_id": "...",
     "categories": [...], "references": [...]}

Usage:
    python chunk_structured.py \
        --input ../data/r2c2_corpus_v1.1/wikitext/wookieepedia_wikitext:wookieepedia_wikitext \
        --input ../data/r2c2_corpus_v1.1/wikitext/wikipedia_wikitext.tar.gz:wikipedia_wikitext \
        --output-dir ../data/chunks-v2

    # Validate mode (no output, just metrics):
    python chunk_structured.py \
        --input archive.tar.gz:wookieepedia_wikitext \
        --validate 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .chunker import chunk_structured_doc, make_document_record  # noqa: F401
from .parsers.base import iter_wikitext, parse_header
from .parsers.wikipedia_parser import WikipediaParser
from .parsers.wookieepedia_parser import WookieepediaParser

__all__ = ["main"]

_PARSERS = {
    "wookieepedia": WookieepediaParser(),
    "wookieepedia_wikitext": WookieepediaParser(),
    "wikipedia": WikipediaParser(),
    "wikipedia_wikitext": WikipediaParser(),
}


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def validate_documents(
    inputs: list[tuple[Path, str]],
    n: int,
    target_words: int = 200,
    max_words: int = 300,
) -> None:
    """Validate N random documents: parse + chunk, print before/after comparison."""
    print(f"Validating {n} random documents per source...")
    print()

    for input_path, source in inputs:
        parser = _PARSERS.get(source)
        if parser is None:
            print(f"WARN: No parser for source '{source}', skipping", file=sys.stderr)
            continue

        reservoir: list[tuple[str, str]] = []
        count = 0
        for filename, content in iter_wikitext(input_path):
            count += 1
            if len(reservoir) < n:
                reservoir.append((filename, content))
            else:
                j = random.randint(0, count - 1)
                if j < n:
                    reservoir[j] = (filename, content)

        print(f"=== {source}: {len(reservoir)} / {count} documents sampled ===")
        print()

        total_passages = 0
        total_words = 0
        empty_count = 0

        for filename, content in reservoir:
            parsed = parse_header(content)
            if parsed is None:
                print(f"  SKIP: {filename} (invalid header)")
                continue

            doc_id, title, lang, body = parsed
            doc = parser.parse(doc_id, title, lang, body, source)
            passages = chunk_structured_doc(doc, target_words, max_words)

            passage_words = sum(_count_words(p["contents"]) for p in passages)
            total_passages += len(passages)
            total_words += passage_words
            if not passages:
                empty_count += 1

            print(f"  [{doc_id}] {title}")
            print(
                f"    Raw: {len(body):,} bytes | Passages: {len(passages)} | Words: {passage_words}"
            )
            if doc.get("infobox"):
                tpl = doc["infobox"].get("template", "?")
                nf = len(doc["infobox"].get("params", {}))
                print(f"    Infobox: {tpl} ({nf} fields)")
            if passages:
                p0 = passages[0]
                print(
                    f"    First passage ({_count_words(p0['contents'])}w): {p0['contents'][:120]}..."
                )
            print()

        print(
            f"  Summary: {total_passages} passages, {total_words} words, {empty_count} empty docs"
        )
        if total_passages:
            print(f"  Avg words/passage: {total_words / total_passages:.0f}")
        print()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Structured chunker: wikitext → parse → chunk → passages.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chunk_structured.py \\
      --input ../data/r2c2_corpus_v1.1/wikitext/wookieepedia_wikitext:wookieepedia_wikitext \\
      --input ../data/r2c2_corpus_v1.1/wikitext/wikipedia_wikitext.tar.gz:wikipedia_wikitext \\
      --output-dir ../data/chunks-v2
        """,
    )
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="PATH:SOURCE",
        help="Input path and source label separated by ':'. "
        "Path can be a directory of .wikitext files or a .tar.gz archive. "
        "Source selects the parser (wookieepedia_wikitext, wikipedia_wikitext). "
        "Can be specified multiple times.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for passages.jsonl and corpus_meta.json (required unless --validate).",
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
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of documents per source (0=unlimited, for testing).",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "When set with --limit N>0, pick the N documents per source by "
            "reservoir sampling (uniform over the full stream) seeded by "
            "this integer, instead of taking the first N in archive order. "
            "Use this for demo runs so the sub-corpus is representative of "
            "the full corpus rather than just A-prefix titles."
        ),
    )
    ap.add_argument(
        "--validate",
        type=int,
        default=0,
        metavar="N",
        help="Validate N random documents (print before/after comparison, no output).",
    )

    args = ap.parse_args()

    inputs: list[tuple[Path, str]] = []
    for spec in args.input:
        if ":" not in spec:
            print(f"ERROR: --input must be PATH:SOURCE, got: {spec}", file=sys.stderr)
            sys.exit(1)
        idx = spec.rfind(":")
        path_str, source = spec[:idx], spec[idx + 1 :]
        path = Path(path_str)
        if not path.exists():
            print(f"ERROR: Input path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        if source not in _PARSERS:
            print(
                f"ERROR: Unknown source '{source}'. "
                f"Available: {', '.join(_PARSERS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        inputs.append((path, source))

    if args.validate:
        validate_documents(inputs, args.validate, args.target_words, args.max_words)
        return

    if not args.output_dir:
        print(
            "ERROR: --output-dir is required (unless using --validate)", file=sys.stderr
        )
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "passages.jsonl"
    doc_file = out_dir / "documents.jsonl"

    total_docs = 0
    total_chunks = 0
    total_errors = 0
    source_counts: dict[str, int] = {}
    error_log_file = out_dir / "chunk_errors.jsonl"

    print(f"Chunking (target={args.target_words}w, max={args.max_words}w)")
    print(f"Output: {out_file} + {doc_file}")
    print()

    with (
        open(out_file, "w", encoding="utf-8") as f_out,
        open(doc_file, "w", encoding="utf-8") as f_doc,
        open(error_log_file, "w", encoding="utf-8") as f_err,
    ):
        for input_path, source in inputs:
            parser = _PARSERS[source]
            print(
                f"Processing {input_path} (source={source}, parser={parser.collection})..."
            )
            doc_count = 0
            chunk_count = 0
            error_count = 0

            # Build the iterator over docs to process for this source.
            # Three modes:
            #   1) no --limit       : stream the full archive
            #   2) --limit N        : take the first N docs in archive order
            #   3) --limit N --random-seed S : reservoir-sample N uniformly
            #      over the full stream, seeded by S (reproducible).
            if args.limit and args.random_seed is not None:
                rng = random.Random(args.random_seed)
                reservoir: list[tuple[str, str]] = []
                seen = 0
                for fname, content in iter_wikitext(input_path):
                    seen += 1
                    if len(reservoir) < args.limit:
                        reservoir.append((fname, content))
                    else:
                        j = rng.randint(0, seen - 1)
                        if j < args.limit:
                            reservoir[j] = (fname, content)
                print(
                    f"  reservoir-sampled {len(reservoir)} / {seen} docs "
                    f"(seed={args.random_seed})"
                )
                doc_iter = iter(reservoir)
            else:
                doc_iter = iter_wikitext(input_path)

            for _filename, content in doc_iter:
                if args.limit and doc_count >= args.limit:
                    break

                parsed = parse_header(content)
                if parsed is None:
                    error_count += 1
                    continue

                doc_id, title, lang, body = parsed

                try:
                    doc = parser.parse(doc_id, title, lang, body, source)

                    # Document-level record (metadata for NER + MCP)
                    doc_record = make_document_record(doc)
                    if doc_record:
                        f_doc.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

                    # Lean passages (content only)
                    passages = chunk_structured_doc(
                        doc,
                        target_words=args.target_words,
                        max_words=args.max_words,
                    )
                    for p in passages:
                        f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
                    doc_count += 1
                    chunk_count += len(passages)
                except Exception as e:
                    error_record = {
                        "doc_id": doc_id,
                        "source": source,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "file": _filename,
                    }
                    f_err.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    f_err.flush()
                    error_count += 1

                if doc_count % 10000 == 0 and doc_count > 0:
                    print(f"  ... {doc_count} docs, {chunk_count} chunks")

            source_counts[source] = chunk_count
            total_docs += doc_count
            total_chunks += chunk_count
            total_errors += error_count
            print(
                f"  {source}: {doc_count} docs → {chunk_count} chunks ({error_count} errors)"
            )

    meta = {"total": total_chunks, "total_docs": total_docs, **source_counts}
    meta_file = out_dir / "corpus_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"Done. {total_docs} documents → {total_chunks} passages")
    print(f"Passages: {out_file}")
    print(f"Documents: {doc_file}")
    print(f"Metadata: {meta_file}")
    print(f"Counts: {json.dumps(source_counts)}")
    if total_errors:
        print(f"Errors: {total_errors} documents failed — see {error_log_file}")


if __name__ == "__main__":
    main()
