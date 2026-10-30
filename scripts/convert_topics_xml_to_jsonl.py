#!/usr/bin/env python3
"""Convert an R2C2 topics XML file (organizer format) to JSONL.

The NTCIR-19 R2C2 organizers publish their topics files (test set + dry-run
topics) as plain XML at Waseda Box. Each question is a top-level
`<question>...</question>` block with a `<qID>` and a `<q>` child:

    <question>
    <qID>0001</qID>
    <q>What did Saw Gerrera use as a torturing device in a Star Wars spinoff movie from 2016?</q>
    </question>

This script reads such an XML file from stdin (or from a path) and emits one
JSON object per line on stdout matching the shape the inference pipeline
expects::

    {"question_id": "0001", "question_text": "What did Saw Gerrera ..."}

Usage::

    python3 scripts/convert_topics_xml_to_jsonl.py r2c2topics.txt > topics.jsonl
    curl -sL "$BOX_URL" | python3 scripts/convert_topics_xml_to_jsonl.py > topics.jsonl

The XML is not strictly well-formed (no single root element), so we wrap it
in a synthetic root before parsing.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET

# The organizers' published XML uses bare ``&`` in question text (e.g.
# "Fast & the Furious") which is not strictly well-formed. Escape any ``&``
# that is not already a known entity reference before parsing.
_BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")


def convert(xml_text: str) -> str:
    """Convert R2C2 topics XML text to JSONL lines.

    Args:
        xml_text: Raw XML content (organizer format, may have multiple top-level
            `<question>` blocks without an enclosing root element).

    Returns:
        Newline-joined JSONL string (one `{"question_id", "question_text"}`
        object per line). No trailing newline.

    Raises:
        ValueError: If a `<question>` block is missing `<qID>` or `<q>`.
    """
    escaped = _BARE_AMP_RE.sub("&amp;", xml_text)
    wrapped = f"<root>{escaped}</root>"
    root = ET.fromstring(wrapped)
    rows = []
    for q in root.findall("question"):
        qid_node = q.find("qID")
        text_node = q.find("q")
        if qid_node is None or text_node is None:
            raise ValueError("Malformed <question> block: missing <qID> or <q>")
        qid = (qid_node.text or "").strip()
        text = (text_node.text or "").strip()
        if not qid or not text:
            raise ValueError(f"Empty <qID> or <q> for question_id={qid!r}")
        rows.append(
            json.dumps(
                {"question_id": qid, "question_text": text},
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    """CLI entry point.

    Args:
        argv: ``sys.argv[1:]``. Optionally a single positional path; otherwise
            reads stdin.

    Returns:
        Process exit code (0 on success).
    """
    if len(argv) > 1:
        sys.stderr.write("usage: convert_topics_xml_to_jsonl.py [INPUT_XML]\n")
        return 2
    if argv:
        with open(argv[0], encoding="utf-8") as f:
            xml_text = f.read()
    else:
        xml_text = sys.stdin.read()
    sys.stdout.write(convert(xml_text))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
