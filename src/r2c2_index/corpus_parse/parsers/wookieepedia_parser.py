"""Wookieepedia-specific wikitext parser.

Uses mwparserfromhell for template extraction (infoboxes, quotes, dialogue)
and regex for section boundaries.

Template handling:
- Infobox templates (Character, DroidSeries, Planet, IndividualShip, etc.)
  → extracted to doc.infobox
- {{Quote|text|speaker|source}} → section quotes[]
- {{Dialogue|speaker1|line1|speaker2|line2}} → section quotes[]
- {{Top|...}}, {{Char-stub}}, {{Expand}}, {{Youmay|...}} → dropped (editorial)
- {{Interlang|...}} → doc.interlang
- {{Film|I}}, {{TCW|episode}}, etc. → resolved to readable text
- {{1st}}, {{Flash}}, {{Mo}}, {{Ab|...}} → dropped (annotation tags)
- <ref>...</ref> → stripped (no retrieval value)
"""

from __future__ import annotations

import re
from typing import Any

try:
    import mwparserfromhell
except ImportError:
    mwparserfromhell = None  # type: ignore[assignment]

from .base import (
    Section,
    StructuredDoc,
    clean_wikitext_basic,
    extract_categories,
    extract_interlang,
    is_redirect,
    make_empty_doc,
    parse_sections_flat,
    resolve_wikilinks,
    strip_bold_italic,
    strip_refs,
)

__all__ = ["WookieepediaParser"]

# Infobox detection, by exclusion

# Templates that are not infoboxes (editorial, navigation, annotation).
# Any lead-position template not in this set is treated as an infobox candidate.
_NOT_INFOBOX_TEMPLATES = {
    # Editorial / annotation
    "era",
    "eras",
    "top",
    "youmay",
    "otheruses",
    "about",
    "for",
    "redirect",
    "char-stub",
    "stub",
    "expand",
    "cleanup",
    "update",
    "more sources",
    "conjecture",
    "unreferenced",
    "verify",
    "citation needed",
    # Annotation tags
    "1st",
    "1stm",
    "1stid",
    "flash",
    "mo",
    "po",
    "c",
    "nc",
    "imo",
    "ab",
    "abbr",
    "nihongo",
    "nowrap",
    # Navigation / metadata
    "interlang",
    "defaultsort",
    "featured",
    "good article",
    "reflist",
    "scroll box",
    "scrollbox",
    "mediacat",
    "swarchive",
    "webarchive",
    "anchor",
    "color",
    "clear",
    "clr",
    "main",
    "see",
    "further",
    # Quote / dialogue (handled separately)
    "quote",
    "dialogue",
    "dialog",
    # Media title templates
    "film",
    "tcw",
    "rebels",
    "resistance",
    "tbb",
    "mando",
    "boba",
    "andor",
    "ahsoka",
    "tales",
    "acolyte",
    "obi-wan",
    "visions",
    # Database / blog links
    "db",
    "blog",
    # Short description
    "short description",
}

# Templates to silently drop (editorial directives, annotation tags)
_DROP_TEMPLATES = {
    "top",
    "youmay",
    "char-stub",
    "stub",
    "expand",
    "cleanup",
    "update",
    "more sources",
    "conjecture",
    "1st",
    "1stm",
    "1stid",
    "flash",
    "mo",
    "po",
    "c",
    "nc",
    "imo",
    "mediacat",
    "scroll box",
    "scrollbox",
    "reflist",
    "swarchive",
    "webarchive",
    "ab",
    "abbr",
    "nihongo",
    "nowrap",
    "anchor",
    "color",
    "clear",
    "clr",
    "main",
    "see",
    "for",
    "about",
    "further",
    "unreferenced",
    "verify",
    "citation needed",
}

# Film/media templates that resolve to titles
_FILM_MAP = {
    "i": "Star Wars: Episode I The Phantom Menace",
    "ii": "Star Wars: Episode II Attack of the Clones",
    "iii": "Star Wars: Episode III Revenge of the Sith",
    "iv": "Star Wars: Episode IV A New Hope",
    "v": "Star Wars: Episode V The Empire Strikes Back",
    "vi": "Star Wars: Episode VI Return of the Jedi",
    "vii": "Star Wars: Episode VII The Force Awakens",
    "viii": "Star Wars: Episode VIII The Last Jedi",
    "ix": "Star Wars: Episode IX The Rise of Skywalker",
}


# Template processing helpers


def _template_name_lower(template: Any) -> str:
    """Get lowercase template name, stripped of whitespace."""
    return str(template.name).strip().lower()


def _template_params(template: Any) -> dict[str, str]:
    """Extract template parameters as a dict.

    Positional params get keys "1", "2", etc.
    Named params get their actual names.
    Values have wikilinks resolved and bold/italic stripped.
    """
    params: dict[str, str] = {}
    for param in template.params:
        name = str(param.name).strip()
        value = str(param.value).strip()
        value = strip_refs(value)
        value = resolve_wikilinks(value)
        value = strip_bold_italic(value)
        value = value.strip()
        if value:
            params[name] = value
    return params


def _is_infobox(template_name: str) -> bool:
    """Check if template is an infobox candidate (exclusion-based).

    Any template outside the known non-infobox set is treated as a potential
    infobox. This catches Celestialbody, Starshipclass, Company, Duel,
    Device, and any future Wookieepedia infobox templates without needing
    an explicit allowlist.
    """
    if not template_name:
        return False
    # Magic words with colons (DEFAULTSORT:..., DISPLAYTITLE:..., etc.)
    if ":" in template_name:
        return False
    if template_name in _NOT_INFOBOX_TEMPLATES:
        return False
    if template_name in _DROP_TEMPLATES:
        return False
    # Check first word for exclusion (handles multi-word template names)
    first_word = template_name.split()[0] if template_name else ""
    if first_word in _NOT_INFOBOX_TEMPLATES or first_word in _DROP_TEMPLATES:
        return False
    # Templates with possessives are not infoboxes
    if template_name.endswith("'s"):
        return False
    # Very short names (1-2 chars) are usually annotation tags, not infoboxes
    if len(template_name) <= 2:
        return False
    return True


def _parse_infobox(template: Any) -> dict[str, Any]:
    """Parse an infobox template into structured dict."""
    params = _template_params(template)

    # Convert list-like fields (equipment, armament, affiliation with * bullets)
    processed: dict[str, Any] = {}
    for key, value in params.items():
        if "\n" in value or value.startswith("*"):
            # Multi-line list field
            items = [
                line.strip().lstrip("*").strip()
                for line in value.splitlines()
                if line.strip() and line.strip() != "*"
            ]
            items = [i for i in items if i]
            processed[key] = items if len(items) > 1 else (items[0] if items else value)
        else:
            processed[key] = value

    return {
        "template": str(template.name).strip(),
        "params": processed,
    }


def _parse_quote(template: Any) -> dict[str, str] | None:
    """Parse {{Quote|text|speaker|source}} into quote dict."""
    import html as _html

    params = _template_params(template)
    text = _html.unescape(params.get("1", "")).strip()
    if not text:
        return None
    return {
        "text": text,
        "speaker": _html.unescape(params.get("2", "")),
        "source": params.get("3", ""),
    }


def _parse_dialogue(template: Any) -> list[dict[str, str]]:
    """Parse {{Dialogue|speaker1|line1|speaker2|line2|...}} into quote list."""
    params = _template_params(template)
    quotes: list[dict[str, str]] = []

    # Dialogue alternates: 1=speaker1, 2=line1, 3=speaker2, 4=line2, ...
    i = 1
    while True:
        speaker = params.get(str(i), "").strip()
        line = params.get(str(i + 1), "").strip()
        if not speaker and not line:
            break
        if line:
            quotes.append(
                {
                    "text": line,
                    "speaker": speaker,
                    "source": "",
                }
            )
        i += 2

    return quotes


def _resolve_film_template(template: Any) -> str:
    """Resolve {{Film|I}} → 'Star Wars: Episode I The Phantom Menace'."""
    params = _template_params(template)
    num = params.get("1", "").strip().lower()
    return _FILM_MAP.get(num, f"Star Wars Episode {params.get('1', '?')}")


def _resolve_media_template(template: Any) -> str:
    """Resolve media shorthand templates to readable titles.

    Handles {{TCW|episode}}, {{Rebels|episode}}, {{TBB|episode}}, etc.
    """
    name = _template_name_lower(template)
    params = _template_params(template)
    title_arg = params.get("1", "").strip()

    media_names = {
        "tcw": "The Clone Wars",
        "rebels": "Rebels",
        "resistance": "Resistance",
        "tbb": "The Bad Batch",
        "mando": "The Mandalorian",
        "boba": "The Book of Boba Fett",
        "andor": "Andor",
        "ahsoka": "Ahsoka",
        "tales": "Tales of the Jedi",
        "acolyte": "The Acolyte",
        "obi-wan": "Obi-Wan Kenobi",
        "visions": "Visions",
    }

    series = media_names.get(name, name)
    if title_arg:
        return f"{series}: {title_arg}"
    return series


# Section-level template processing


def _process_section_body(raw_body: str) -> tuple[str, list[dict], list[dict]]:
    """Process raw section body: extract templates, clean text.

    Args:
        raw_body: Raw wikitext section body.

    Returns:
        (cleaned_content, quotes, tables)
    """
    quotes: list[dict] = []
    tables: list[dict] = []

    if mwparserfromhell is None:
        # Fallback: basic cleaning without template extraction
        cleaned = _strip_templates_regex(raw_body)
        cleaned = clean_wikitext_basic(cleaned)
        return cleaned, quotes, tables

    wikicode = mwparserfromhell.parse(raw_body)

    for template in wikicode.filter_templates(recursive=True):
        name = _template_name_lower(template)

        if name in ("quote",):
            q = _parse_quote(template)
            if q:
                quotes.append(q)
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        elif name in ("dialogue", "dialog"):
            dialogue_quotes = _parse_dialogue(template)
            quotes.extend(dialogue_quotes)
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        elif name in ("film",):
            resolved = _resolve_film_template(template)
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in _DROP_TEMPLATES:
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        elif name in (
            "tcw",
            "rebels",
            "resistance",
            "tbb",
            "mando",
            "boba",
            "andor",
            "ahsoka",
            "tales",
            "acolyte",
            "obi-wan",
            "visions",
        ):
            resolved = _resolve_media_template(template)
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in ("db",):
            # Database link template: extract display text
            params = _template_params(template)
            display = params.get("3", params.get("2", ""))
            try:
                wikicode.replace(template, display)
            except ValueError:
                pass

        elif name in ("blog",):
            params = _template_params(template)
            display = params.get("3", params.get("2", ""))
            try:
                wikicode.replace(template, display)
            except ValueError:
                pass

        elif name.endswith("'s") or name == "'s":
            # Possessive template: {{Ship's}} → 's
            try:
                wikicode.replace(template, "'s")
            except ValueError:
                pass

    text = str(wikicode)
    # Templates not handled above are dropped.
    text = _strip_templates_regex(text)
    text = clean_wikitext_basic(text)

    return text, quotes, tables


# Regex fallback for template removal
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")


def _strip_templates_regex(text: str) -> str:
    """Strip templates using iterative regex (fallback when mwparserfromhell unavailable)."""
    for _ in range(10):
        new = _TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new
    return text


# Table extraction

_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)


def _extract_tables(body: str) -> tuple[list[dict], str]:
    """Extract wiki tables from body.

    Args:
        body: Raw wikitext.

    Returns:
        (tables, cleaned_body) — list of table dicts and body without tables.
    """
    tables: list[dict] = []

    for match in _TABLE_RE.finditer(body):
        table_text = match.group(0)
        table = _parse_wiki_table(table_text)
        if table and table.get("rows"):
            tables.append(table)

    cleaned = _TABLE_RE.sub("", body)
    return tables, cleaned


def _parse_wiki_table(table_text: str) -> dict | None:
    """Parse a wiki table into {caption, headers, rows}."""
    lines = table_text.splitlines()
    caption = ""
    headers: list[str] = []
    rows: list[list[str]] = []
    current_row: list[str] = []

    for line in lines:
        line = line.strip()
        if line.startswith("{|") or line.startswith("|}"):
            continue
        if line.startswith("|+"):
            caption = line[2:].strip()
            caption = resolve_wikilinks(caption)
        elif line.startswith("!"):
            cells = re.split(r"!!", line[1:])
            for cell in cells:
                cell = cell.strip()
                # Remove sort keys and styling
                if "|" in cell:
                    cell = cell.split("|")[-1].strip()
                cell = resolve_wikilinks(cell)
                cell = strip_bold_italic(cell)
                headers.append(cell.strip())
        elif line.startswith("|-"):
            if current_row:
                rows.append(current_row)
                current_row = []
        elif line.startswith("|"):
            cells = re.split(r"\|\|", line[1:])
            for cell in cells:
                cell = cell.strip()
                if "|" in cell:
                    cell = cell.split("|")[-1].strip()
                cell = resolve_wikilinks(cell)
                cell = strip_bold_italic(cell)
                cell = strip_refs(cell)
                current_row.append(cell.strip())

    if current_row:
        rows.append(current_row)

    if not rows:
        return None

    return {"caption": caption, "headers": headers, "rows": rows}


# Reference section extraction


def _extract_reference_titles(section_body: str) -> list[str]:
    """Extract document titles from Appearances/Sources sections.

    Handles both wiki links ([[Title]]) and templates ({{TCW|Episode}}).
    """
    titles: list[str] = []

    # Wiki link targets
    link_re = re.compile(r"\[\[([^|\]]+?)(?:\|[^\]]+)?\]\]")
    for m in link_re.finditer(section_body):
        target = m.group(1).strip()
        if not target.lower().startswith(("category:", "file:", "image:")):
            titles.append(target)

    # Template titles (first positional arg)
    if mwparserfromhell is not None:
        wikicode = mwparserfromhell.parse(section_body)
        for template in wikicode.filter_templates(recursive=True):
            name = _template_name_lower(template)
            if name in _DROP_TEMPLATES:
                continue
            params = _template_params(template)
            title_arg = params.get("1", "").strip()
            if title_arg and len(title_arg) > 2:
                titles.append(title_arg)

    return titles


# Main parser class


class WookieepediaParser:
    """Wookieepedia-specific wikitext parser."""

    collection: str = "wookieepedia"

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
            Structured document dict.
        """
        raw_byte_count = len(body.encode("utf-8"))

        if is_redirect(body):
            return make_empty_doc(doc_id, title, lang, source, raw_byte_count, True)

        doc = make_empty_doc(doc_id, title, lang, source, raw_byte_count)

        # 1. Extract categories (before section parsing)
        categories, body = extract_categories(body)
        doc["categories"] = categories

        # 2. Extract interlang
        interlang, body = extract_interlang(body)
        doc["interlang"] = interlang

        # 3. Tables are extracted per section (step 6), so the list starts empty.
        doc["tables"] = []

        # 4. Extract infobox from body (before section parsing)
        infobox, body = self._extract_infobox(body)
        doc["infobox"] = infobox

        # 5. Parse sections (flat, with levels)
        flat_sections = parse_sections_flat(body)

        # 6. Process each section
        lead_text = ""
        sections: list[Section] = []
        appearances: list[str] = []
        sources_refs: list[str] = []

        for heading, level, raw_body in flat_sections:
            heading_lower = heading.lower().strip()

            if level == 0 and heading == "":
                lead_tables, raw_body = _extract_tables(raw_body)
                doc["tables"] = lead_tables
                content, quotes, tables = _process_section_body(raw_body)
                lead_text = content
                if quotes:
                    doc["_lead_quotes"] = quotes
                continue

            # Reference sections → extract titles
            if heading_lower in ("appearances", "non-canon appearances"):
                appearances.extend(_extract_reference_titles(raw_body))
                continue
            if heading_lower in ("sources", "non-canon sources"):
                sources_refs.extend(_extract_reference_titles(raw_body))
                continue

            # Sections with no retrieval value
            if heading_lower in (
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
            ):
                continue

            # Extract tables from section body before template processing
            section_tables, raw_body = _extract_tables(raw_body)
            content, quotes, tpl_tables = _process_section_body(raw_body)
            section: Section = {
                "heading": heading,
                "level": level,
                "content": content,
                "quotes": quotes,
                "tables": section_tables + tpl_tables,
                "subsections": [],
            }
            sections.append(section)

        doc["lead"] = lead_text
        doc["sections"] = self._nest_sections(sections)
        doc["appearances"] = appearances
        doc["sources_refs"] = sources_refs

        # Move lead quotes to first section if available
        lead_quotes = doc.pop("_lead_quotes", [])
        if lead_quotes and doc["sections"]:
            doc["sections"][0]["quotes"] = lead_quotes + doc["sections"][0].get(
                "quotes", []
            )

        return doc

    def _extract_infobox(self, body: str) -> tuple[dict | None, str]:
        """Extract the first infobox template from the lead section only.

        Only searches before the first heading (==) to avoid grabbing
        templates from Appearances/Sources sections.

        Returns:
            (infobox_dict, cleaned_body) or (None, body) if no infobox found.
        """
        if mwparserfromhell is None:
            return None, body

        # Only search in the lead (before first heading)
        import re as _re

        heading_match = _re.search(r"^==[^=]", body, _re.MULTILINE)
        if heading_match:
            lead_part = body[: heading_match.start()]
        else:
            lead_part = body

        try:
            wikicode = mwparserfromhell.parse(lead_part)
        except Exception:
            return None, body

        for template in wikicode.filter_templates(recursive=False):
            name = _template_name_lower(template)
            if _is_infobox(name):
                infobox = _parse_infobox(template)
                try:
                    wikicode.remove(template)
                except ValueError:
                    pass
                # Reconstruct full body: cleaned lead + rest
                if heading_match:
                    return infobox, str(wikicode) + body[heading_match.start() :]
                return infobox, str(wikicode)

        return None, body

    @staticmethod
    def _nest_sections(flat: list[Section]) -> list[Section]:
        """Convert flat list of sections into nested tree by level.

        Sections at level N become subsections of the preceding section at level N-1.
        """
        if not flat:
            return []

        result: list[Section] = []
        stack: list[Section] = []

        for section in flat:
            level = section["level"]

            # Pop stack until we find a parent at a lower level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            if stack:
                stack[-1]["subsections"].append(section)
            else:
                result.append(section)

            stack.append(section)

        return result
