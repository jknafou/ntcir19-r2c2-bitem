"""Base parser protocol and shared utilities for wikitext parsing.

Holds the logic every collection shares: header parsing, category
extraction, wiki link resolution, archive iteration.
Collection-specific parsers implement the WikitextParser protocol.
"""

from __future__ import annotations

import html
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "WikitextParser",
    "StructuredDoc",
    "Section",
    "Quote",
    "Table",
    "parse_header",
    "extract_categories",
    "extract_interlang",
    "resolve_wikilinks",
    "strip_refs",
    "replace_br_tags",
    "strip_html_tags",
    "strip_html_comments",
    "strip_file_links",
    "strip_magic_words",
    "unescape_html",
    "normalize_whitespace",
    "clean_wikitext_basic",
    "iter_wikitext_from_tar",
    "iter_wikitext_from_dir",
    "iter_wikitext",
    "HEADING_RE",
    "SKIP_SECTIONS",
    "parse_sections_nested",
]

# Type definitions

Quote = dict[str, str]  # {"text": ..., "speaker": ..., "source": ...}

Table = dict[str, Any]  # {"caption": ..., "headers": [...], "rows": [[...]]}

Section = dict[str, Any]
# {
#     "heading": str,
#     "level": int,
#     "content": str,
#     "quotes": list[Quote],
#     "tables": list[Table],
#     "subsections": list[Section],
# }

StructuredDoc = dict[str, Any]
# make_empty_doc lists the fields


@runtime_checkable
class WikitextParser(Protocol):
    """Protocol for collection-specific wikitext parsers."""

    collection: str
    """Identifier for the collection this parser handles."""

    def parse(
        self,
        doc_id: str,
        title: str,
        lang: str,
        body: str,
        source: str,
    ) -> StructuredDoc:
        """Parse raw wikitext body into structured document JSON.

        Args:
            doc_id: Document identifier.
            title: Document title.
            lang: Language code.
            body: Raw wikitext body (after metadata header lines).
            source: Corpus source label.

        Returns:
            Structured document dict (see StructuredDoc type).
        """
        ...


# Regex patterns (shared across parsers)

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_REF_TAG_RE = re.compile(r"<ref[^>/]*?/?>.*?</ref>|<ref[^>]*?/?>", re.I | re.S)
_HTML_TAG_RE = re.compile(r"</?([a-z][a-z0-9]*)\b[^>]*>", re.I)
_INTERNAL_LINK_RE = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[([a-z]{3,6}://[^\s\]]+)(?:\s+([^\]]+))?\]")
_CATEGORY_RE = re.compile(r"\[\[Category:([^\]]+)\]\]", re.I)
_FILE_RE = re.compile(r"\[\[(File|Image):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", re.I)
_MAGIC_WORDS_RE = re.compile(r"__[A-Z]+__")
_INTERLANG_RE = re.compile(r"\{\{Interlang\s*\n(.*?)\}\}", re.S | re.I)

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

SKIP_SECTIONS = {
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
}

# Sections containing document references (wiki links / templates)
REFERENCE_SECTIONS = {"appearances", "sources"}


# Header / metadata parsing


def _parse_meta_line(s: str) -> str:
    """Parse a metadata line from a wikitext file header."""
    s = re.sub(r"^#+\s*", "", s.strip())
    s = re.sub(r"^(qid|id|title|lang)\s*[:=]\s*", "", s, flags=re.I)
    return s.strip(" \t")


def parse_header(raw: str) -> tuple[str, str, str, str] | None:
    """Parse wikitext file header into (doc_id, title, lang, body).

    The R2C2 format has 3 header lines:
        # ID: <id>
        # TITLE: <title>
        # LANG: <lang>
        <body>

    Args:
        raw: Full file content.

    Returns:
        (doc_id, title, lang, body) or None if invalid.
    """
    lines = raw.splitlines()
    if len(lines) < 4:
        return None

    doc_id = _parse_meta_line(lines[0])
    title = _parse_meta_line(lines[1])
    lang = _parse_meta_line(lines[2])

    # Normalize doc_id: strip leading non-alnum, keep only alnum/underscore/dash
    doc_id = re.sub(r"^[^A-Za-z0-9]+", "", doc_id)
    doc_id = re.sub(r"[^A-Za-z0-9_-]", "", doc_id)

    body = "\n".join(lines[3:])

    if not doc_id or not body.strip():
        return None

    return doc_id, title, lang, body


# Text cleaning utilities (building blocks for parsers)


def strip_html_comments(text: str) -> str:
    """Remove HTML comments <!-- ... -->."""
    return _COMMENT_RE.sub("", text)


def strip_refs(text: str) -> str:
    """Remove <ref>...</ref> and <ref .../> tags."""
    return _REF_TAG_RE.sub("", text)


def replace_br_tags(text: str) -> str:
    """Replace <br>, <br/>, <br /> with ' ; ' to preserve list delimiters."""
    return re.sub(r"<br\s*/?\s*>", " ; ", text, flags=re.I)


def strip_html_tags(text: str) -> str:
    """Remove HTML tags but keep their content."""
    return _HTML_TAG_RE.sub("", text)


def strip_file_links(text: str) -> str:
    """Remove [[File:...]] and [[Image:...]] links."""
    return _FILE_RE.sub("", text)


def strip_magic_words(text: str) -> str:
    """Remove magic words like __NOTOC__, __TOC__, etc."""
    return _MAGIC_WORDS_RE.sub("", text)


def unescape_html(text: str) -> str:
    """Unescape HTML entities: &mdash; → —."""
    return html.unescape(text)


def resolve_wikilinks(text: str) -> str:
    """Resolve internal wiki links: [[Target|Display]] → Display, [[Target]] → Target.

    Also removes orphan ]] closers from malformed/nested links.
    """

    def _repl(m: re.Match) -> str:
        if m.group(1) and m.group(2):
            return m.group(2)
        return m.group(3) or ""

    text = _INTERNAL_LINK_RE.sub(_repl, text)
    text = re.sub(r"\]\]", "", text)
    return text


def resolve_external_links(text: str) -> str:
    """Resolve [url label] → label, [url] → url."""

    def _repl(m: re.Match) -> str:
        return m.group(2) or m.group(1)

    return _EXTERNAL_LINK_RE.sub(_repl, text)


def strip_bold_italic(text: str) -> str:
    """Remove wiki bold/italic markup ('' and ''')."""
    return re.sub(r"'{2,3}", "", text)


def strip_list_markers(text: str) -> str:
    """Remove bullet/numbered/indent list markers at line start."""
    return re.sub(r"^[*#:]+\s*", "", text, flags=re.MULTILINE)


def normalize_whitespace(text: str) -> str:
    """Normalize trailing spaces and excessive blank lines."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_wikitext_basic(text: str) -> str:
    """Basic wikitext cleaning: refs, HTML, file links, bold/italic, whitespace.

    Does not remove templates or categories — those are handled by
    collection-specific parsers before calling this.
    """
    text = strip_html_comments(text)
    text = strip_refs(text)
    text = strip_file_links(text)
    text = strip_html_tags(text)
    text = strip_magic_words(text)
    text = resolve_wikilinks(text)
    text = resolve_external_links(text)
    # Orphan pipe from links
    text = re.sub(r"\s*\|\s*", " ", text)
    text = strip_bold_italic(text)
    # Empty parens left by template removal
    text = re.sub(r"\(\s*\)", "", text)
    text = unescape_html(text)
    text = strip_list_markers(text)
    text = normalize_whitespace(text)
    return text


# Extraction utilities


def extract_categories(body: str) -> tuple[list[str], str]:
    """Extract [[Category:...]] from body.

    Args:
        body: Raw wikitext body.

    Returns:
        (categories, cleaned_body) — list of category names and body
        with category links removed.
    """
    categories = [c.strip() for c in _CATEGORY_RE.findall(body)]
    cleaned = _CATEGORY_RE.sub("", body)
    return categories, cleaned


def extract_interlang(body: str) -> tuple[dict[str, str], str]:
    """Extract {{Interlang|...}} template from body.

    Args:
        body: Raw wikitext body.

    Returns:
        (interlang_dict, cleaned_body) — dict of lang→title mappings
        and body with Interlang template removed.
    """
    interlang: dict[str, str] = {}
    match = _INTERLANG_RE.search(body)
    if match:
        for line in match.group(1).splitlines():
            line = line.strip().strip("|").strip()
            if "=" in line:
                lang, _, title = line.partition("=")
                lang = lang.strip()
                title = title.strip()
                if lang and title:
                    interlang[lang] = title
        body = body[: match.start()] + body[match.end() :]
    return interlang, body


def is_redirect(body: str) -> bool:
    """Check if body is a #REDIRECT page."""
    return bool(re.match(r"^\s*#REDIRECT\s*\[\[", body, re.I))


# Section parsing (nested tree)


def parse_sections_flat(body: str) -> list[tuple[str, int, str]]:
    """Split raw wikitext into flat (heading, level, body) tuples.

    Lead section (before first heading) has heading="" and level=0.

    Args:
        body: Raw wikitext body.

    Returns:
        List of (heading_text, level, body_text) tuples.
    """
    sections: list[tuple[str, int, str]] = []
    last_end = 0
    current_heading = ""
    current_level = 0

    for match in HEADING_RE.finditer(body):
        section_body = body[last_end : match.start()]
        # Always append — even empty-body headings must appear so that
        # subsections get correct parent attribution in the nested tree.
        sections.append((current_heading, current_level, section_body))

        heading_raw = match.group(2).strip()
        heading_raw = resolve_wikilinks(heading_raw)
        heading_raw = strip_bold_italic(heading_raw)
        # Strip inline templates from headings (e.g. {{SectionFlag|1}})
        heading_raw = re.sub(r"\{\{[^}]+\}\}", "", heading_raw).strip()

        current_heading = heading_raw
        current_level = len(match.group(1))  # == → 2, === → 3, etc.
        last_end = match.end()

    remaining = body[last_end:]
    if remaining.strip():
        sections.append((current_heading, current_level, remaining))

    return sections


def parse_sections_nested(body: str) -> tuple[str, list[Section]]:
    """Parse wikitext into a nested section tree.

    Args:
        body: Raw wikitext body.

    Returns:
        (lead_text, sections) — the lead section text (before first heading)
        and a nested list of section dicts.
    """
    flat = parse_sections_flat(body)
    if not flat:
        return "", []

    lead = ""
    sections: list[Section] = []

    # Extract lead (level 0, no heading)
    start_idx = 0
    if flat[0][1] == 0 and flat[0][0] == "":
        lead = flat[0][2]
        start_idx = 1

    def _build_tree(
        items: list[tuple[str, int, str]], idx: int, parent_level: int
    ) -> tuple[list[Section], int]:
        result: list[Section] = []
        while idx < len(items):
            heading, level, text = items[idx]
            if level <= parent_level:
                break
            section: Section = {
                "heading": heading,
                "level": level,
                "content": text,
                "quotes": [],
                "tables": [],
                "subsections": [],
            }
            idx += 1
            subsections, idx = _build_tree(items, idx, level)
            section["subsections"] = subsections
            result.append(section)
        return result, idx

    sections, _ = _build_tree(
        flat, start_idx, 1
    )  # level 1 doesn't exist, so all ==+ are children
    return lead, sections


# File iteration (directories and tar.gz archives)


def iter_wikitext_from_dir(dir_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (filename, content) from a directory of .wikitext files."""
    for f in sorted(dir_path.rglob("*.wikitext")):
        try:
            yield f.name, f.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"[WARN] Could not read {f}: {e}")


def iter_wikitext_from_tar(tar_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (filename, content) from a tar.gz archive of .wikitext files."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".wikitext"):
                f = tar.extractfile(member)
                if f:
                    try:
                        content = f.read().decode("utf-8", errors="ignore")
                        yield Path(member.name).name, content
                    except Exception as e:
                        print(f"[WARN] Failed reading {member.name}: {e}")


def iter_wikitext(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (filename, content) from a path (directory or tar.gz)."""
    if path.is_dir():
        yield from iter_wikitext_from_dir(path)
    elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        yield from iter_wikitext_from_tar(path)
    else:
        raise ValueError(f"Unsupported input: {path} (expected directory or .tar.gz)")


# Utility: make empty structured doc


def make_empty_doc(
    doc_id: str,
    title: str,
    lang: str,
    source: str,
    raw_byte_count: int,
    is_redir: bool = False,
) -> StructuredDoc:
    """Create an empty structured document with all required fields.

    Collection-specific parsers should call this, then populate fields.
    """
    return {
        "doc_id": doc_id,
        "title": title,
        "lang": lang,
        "source": source,
        "infobox": None,
        "lead": "",
        "sections": [],
        "tables": [],
        "categories": [],
        "appearances": [],
        "sources_refs": [],
        "interlang": {},
        "is_redirect": is_redir,
        "raw_byte_count": raw_byte_count,
    }
