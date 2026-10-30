"""Wikipedia-specific wikitext parser.

Uses mwparserfromhell for template extraction and regex for section boundaries.

Key differences from Wookieepedia:
- Infoboxes: {{Infobox person|...}}, {{Infobox settlement|...}} (standard naming)
- No Appearances/Sources sections
- Tables more common (demographics, statistics)
- {{Short description|...}} → extract to short_description field
- {{Convert|N|unit1|unit2}} → resolve to "N unit1"
- {{Birth date and age|Y|M|D}} → resolve to date string
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
    strip_html_comments,
    strip_html_tags,
    strip_refs,
)
from .base import replace_br_tags as replace_br_tags  # noqa: F811

__all__ = ["WikipediaParser"]


# Templates to silently drop
_DROP_TEMPLATES = {
    "reflist",
    "refn",
    "efn",
    "notelist",
    "anchor",
    "clear",
    "clr",
    "main",
    "see",
    "see also",
    "further",
    "for",
    "about",
    "redirect",
    "other uses",
    "distinguish",
    "hatnote",
    "cleanup",
    "unreferenced",
    "citation needed",
    "cn",
    "verify",
    "vague",
    "when",
    "who",
    "where",
    "by whom",
    "failed verification",
    "dubious",
    "tone",
    "weasel",
    "peacock",
    "pov",
    "globalize",
    "update",
    "outdated",
    "use dmy dates",
    "use mdy dates",
    "use british english",
    "use american english",
    "good article",
    "featured article",
    "pp",
    "pp-move",
    "pp-semi",
    "pp-vandalism",
    "bots",
    "nobots",
    "commons category",
    "authority control",
    "defaultsort",
    "wikidata",
    "portal",
    "portal bar",
    "wikiquote",
    "wikisource",
    "wiktionary",
    "wikimedia",
    "sisterlinks",
    "col-begin",
    "col-end",
    "div col",
    "div col end",
    "columns-list",
    "flatlist",
    "startflatlist",
    "endflatlist",
    "nowrap",
    "spaces",
    "nbsp",
    "spaced ndash",
    "snd",
    "mdash",
    "ndash",
    "shy",
    "break",
    "clear",
}


# Template processing helpers


def _template_name_lower(template: Any) -> str:
    """Get lowercase template name, stripped of whitespace."""
    return str(template.name).strip().lower()


def _template_params(template: Any) -> dict[str, str]:
    """Extract template parameters as a dict."""
    params: dict[str, str] = {}
    for param in template.params:
        name = str(param.name).strip()
        value = str(param.value).strip()
        value = strip_refs(value)
        value = strip_html_comments(value)
        value = replace_br_tags(value)
        value = strip_html_tags(value)
        value = resolve_wikilinks(value)
        value = strip_bold_italic(value)
        value = value.strip()
        if value:
            params[name] = value
    return params


def _resolve_nested_templates(value_wikicode: Any) -> str:
    """Resolve nested templates in an infobox parameter value.

    Handles common Wikipedia templates like {{Birth date and age}},
    {{Convert}}, {{hlist}}, {{Plainlist}}, etc.
    """
    if mwparserfromhell is None:
        return str(value_wikicode).strip()

    # Work on a copy to avoid modifying the original
    wikicode = mwparserfromhell.parse(str(value_wikicode))

    for tpl in wikicode.filter_templates(recursive=True):
        name = _template_name_lower(tpl)
        params = _template_params(tpl)

        resolved = None
        if name in (
            "birth date and age",
            "birth date",
            "death date and age",
            "death date",
        ):
            resolved = _resolve_birth_date(tpl)
        elif name in ("convert",):
            resolved = _resolve_convert(tpl)
        elif name in ("hlist", "plainlist", "flatlist", "unbulleted list", "ubl"):
            # List templates: join positional args
            items = [params[str(i)] for i in range(1, 20) if str(i) in params]
            resolved = ", ".join(items)
        elif name in ("start date", "start date and age"):
            parts = [params.get(str(i), "") for i in range(1, 4) if params.get(str(i))]
            resolved = "-".join(parts)
        elif name in ("small", "smaller", "big", "large"):
            resolved = params.get("1", "")
        elif name in (
            "lang",
            "lang-fr",
            "lang-de",
            "lang-es",
            "lang-it",
            "lang-la",
            "lang-pt",
            "lang-ru",
            "lang-ar",
            "lang-zh",
        ):
            resolved = params.get("2", params.get("1", ""))
        elif name in ("nowrap",):
            resolved = params.get("1", "")
        elif name in ("url",):
            resolved = params.get("1", "")
        elif name in ("ill", "illm"):
            # Interlanguage link: return English title (param 1)
            resolved = params.get("1", "")
        elif name in ("nihongo",):
            # {{Nihongo|English|日本語|romanization}} → "English (日本語, romanization)"
            english = params.get("1", "")
            kanji = params.get("2", "")
            romaji = params.get("3", "")
            if english and kanji and romaji:
                resolved = f"{english} ({kanji}, {romaji})"
            elif english and kanji:
                resolved = f"{english} ({kanji})"
            else:
                resolved = english or kanji or ""
        elif name.startswith("lang") and name not in ("lang",) and "-" in name:
            # {{lang-xx|text}} variants not in the main list
            resolved = params.get("1", "")
        elif name in ("langx",):
            # {{langx|fa|text}} → text
            resolved = params.get("2", params.get("1", ""))
        elif name in ("zh",):
            # {{zh|first=t|t=食神}} → take t, s, or first available
            resolved = params.get("t", params.get("s", params.get("1", "")))
        elif name in ("film date",):
            # {{film date|df=y|1996|12|21|Hong Kong}} → date string
            parts = [params.get(str(i), "") for i in range(1, 5) if params.get(str(i))]
            # Filter out named params like df=y
            parts = [p for p in parts if not any(c == "=" for c in p)]
            resolved = " ".join(parts)
        elif name in ("marriage",):
            # {{marriage|Name|1939|1947|end=divorced}} → "Name (1939–1947, divorced)"
            spouse = params.get("1", "")
            year1 = params.get("2", "")
            year2 = params.get("3", "")
            end = params.get("end", params.get("reason", ""))
            if spouse and year1 and year2:
                resolved = f"{spouse} ({year1}\u2013{year2}"
                if end:
                    resolved += f", {end}"
                resolved += ")"
            elif spouse:
                resolved = spouse
            else:
                resolved = ""

        if resolved is not None:
            try:
                wikicode.replace(tpl, resolved)
            except ValueError:
                pass

    # Strip any remaining unresolved templates
    text = str(wikicode)
    for _ in range(5):
        new = _TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new
    return text.strip()


def _parse_infobox(template: Any) -> dict[str, Any]:
    """Parse an infobox template into structured dict."""
    processed: dict[str, Any] = {}

    for param in template.params:
        key = str(param.name).strip()
        raw_value = param.value

        value = _resolve_nested_templates(raw_value)
        value = strip_refs(value)
        value = strip_html_comments(value)
        value = replace_br_tags(value)
        value = strip_html_tags(value)
        value = resolve_wikilinks(value)
        value = strip_bold_italic(value)
        import html as _html

        value = _html.unescape(value)
        value = value.strip()

        if not value:
            continue

        if "\n" in value or value.startswith("*"):
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


def _resolve_convert(template: Any) -> str:
    """Resolve {{Convert|N|unit1|unit2}} → 'N unit1'."""
    params = _template_params(template)
    number = params.get("1", "")
    unit = params.get("2", "")
    if number and unit:
        return f"{number} {unit}"
    return number


def _resolve_birth_date(template: Any) -> str:
    """Resolve {{Birth date and age|Y|M|D}} or {{Birth date|Y|M|D}} → date string."""
    params = _template_params(template)
    year = params.get("1", "")
    month = params.get("2", "")
    day = params.get("3", "")
    parts = [p for p in [year, month, day] if p]
    return "-".join(parts) if parts else ""


def _resolve_lang(template: Any) -> str:
    """Resolve {{Lang|xx|text}} → text."""
    params = _template_params(template)
    return params.get("2", params.get("1", ""))


# Section-level template processing


# Regex fallback for template removal
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")


def _strip_templates_regex(text: str) -> str:
    """Strip templates using iterative regex."""
    for _ in range(10):
        new = _TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new
    return text


def _process_section_body(raw_body: str) -> tuple[str, list[dict], list[dict]]:
    """Process raw section body: extract templates, clean text.

    Returns:
        (cleaned_content, quotes, tables)
    """
    quotes: list[dict] = []
    tables: list[dict] = []

    if mwparserfromhell is None:
        cleaned = _strip_templates_regex(raw_body)
        cleaned = clean_wikitext_basic(cleaned)
        return cleaned, quotes, tables

    wikicode = mwparserfromhell.parse(raw_body)

    for template in wikicode.filter_templates(recursive=True):
        name = _template_name_lower(template)

        if name in ("quote", "cquote", "quote box", "blockquote"):
            params = _template_params(template)
            text = params.get("1", params.get("text", "")).strip()
            if text:
                quotes.append(
                    {
                        "text": text,
                        "speaker": params.get("2", params.get("author", "")),
                        "source": params.get("3", params.get("source", "")),
                    }
                )
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        elif name in ("convert",):
            resolved = _resolve_convert(template)
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in (
            "birth date and age",
            "birth date",
            "death date and age",
            "death date",
        ):
            resolved = _resolve_birth_date(template)
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in (
            "lang",
            "lang-fr",
            "lang-de",
            "lang-es",
            "lang-it",
            "lang-la",
            "lang-pt",
            "lang-ru",
            "lang-ar",
            "lang-zh",
        ):
            resolved = _resolve_lang(template)
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in ("ipa",):
            # IPA pronunciation: text is param 2 (param 1 is lang code)
            params = _template_params(template)
            try:
                wikicode.replace(template, params.get("2", params.get("1", "")))
            except ValueError:
                pass

        elif name in ("ill", "illm"):
            # Interlanguage link: return English title (param 1)
            params = _template_params(template)
            try:
                wikicode.replace(template, params.get("1", ""))
            except ValueError:
                pass

        elif name in ("nihongo",):
            # {{Nihongo|English|日本語|romanization}}
            params = _template_params(template)
            english = params.get("1", "")
            kanji = params.get("2", "")
            romaji = params.get("3", "")
            if english and kanji and romaji:
                resolved = f"{english} ({kanji}, {romaji})"
            elif english and kanji:
                resolved = f"{english} ({kanji})"
            else:
                resolved = english or kanji or ""
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in ("langx",):
            # {{langx|fa|text}} → text
            params = _template_params(template)
            try:
                wikicode.replace(template, params.get("2", params.get("1", "")))
            except ValueError:
                pass

        elif name.startswith("lang") and "-" in name:
            # {{lang-xx|text}} variants not in the main list
            params = _template_params(template)
            try:
                wikicode.replace(template, params.get("1", ""))
            except ValueError:
                pass

        elif name in ("zh",):
            params = _template_params(template)
            resolved = params.get("t", params.get("s", params.get("1", "")))
            try:
                wikicode.replace(template, resolved)
            except ValueError:
                pass

        elif name in ("film date",):
            params = _template_params(template)
            parts = [params.get(str(i), "") for i in range(1, 5) if params.get(str(i))]
            try:
                wikicode.replace(template, " ".join(parts))
            except ValueError:
                pass

        elif name in ("cast listing",):
            # Expand inner text (bulleted list of actors) inline
            try:
                inner = str(template.params[0].value) if template.params else ""
                wikicode.replace(template, inner)
            except (ValueError, IndexError):
                pass

        elif name in _DROP_TEMPLATES:
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        elif name.startswith("infobox"):
            # Infoboxes appearing mid-section (rare) — just remove
            try:
                wikicode.remove(template)
            except ValueError:
                pass

    text = str(wikicode)
    text = _strip_templates_regex(text)
    text = clean_wikitext_basic(text)

    return text, quotes, tables


# Table extraction (shared with Wookieepedia)

_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)


def _extract_tables(body: str) -> tuple[list[dict], str]:
    """Extract wiki tables from body."""
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


# Main parser class


class WikipediaParser:
    """Wikipedia-specific wikitext parser."""

    collection: str = "wikipedia"

    def parse(
        self,
        doc_id: str,
        title: str,
        lang: str,
        body: str,
        source: str,
    ) -> StructuredDoc:
        """Parse raw wikitext body into structured document JSON."""
        raw_byte_count = len(body.encode("utf-8"))

        if is_redirect(body):
            return make_empty_doc(doc_id, title, lang, source, raw_byte_count, True)

        doc = make_empty_doc(doc_id, title, lang, source, raw_byte_count)

        categories, body = extract_categories(body)
        doc["categories"] = categories

        interlang, body = extract_interlang(body)
        doc["interlang"] = interlang

        short_desc = ""
        if mwparserfromhell is not None:
            try:
                wikicode = mwparserfromhell.parse(body)
                for template in wikicode.filter_templates(recursive=False):
                    if _template_name_lower(template) in ("short description",):
                        params = _template_params(template)
                        short_desc = params.get("1", "")
                        try:
                            wikicode.remove(template)
                        except ValueError:
                            pass
                        break
                body = str(wikicode)
            except Exception:
                pass

        if short_desc:
            doc["short_description"] = short_desc

        # Tables are extracted per section; doc["tables"] holds only those of
        # the lead section.
        doc["tables"] = []

        infobox, body = self._extract_infobox(body)
        doc["infobox"] = infobox

        flat_sections = parse_sections_flat(body)

        lead_text = ""
        sections: list[Section] = []

        for heading, level, raw_body in flat_sections:
            heading_lower = heading.lower().strip()

            # Lead section
            if level == 0 and heading == "":
                lead_tables, raw_body = _extract_tables(raw_body)
                doc["tables"] = lead_tables
                content, quotes, tables = _process_section_body(raw_body)
                lead_text = content
                continue

            # Skip sections
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

        return doc

    def _extract_infobox(self, body: str) -> tuple[dict | None, str]:
        """Extract the first infobox template from body."""
        if mwparserfromhell is None:
            return None, body

        try:
            wikicode = mwparserfromhell.parse(body)
        except Exception:
            return None, body

        for template in wikicode.filter_templates(recursive=False):
            name = _template_name_lower(template)
            if name.startswith("infobox"):
                infobox = _parse_infobox(template)
                try:
                    wikicode.remove(template)
                except ValueError:
                    pass
                return infobox, str(wikicode)

        return None, body

    @staticmethod
    def _nest_sections(flat: list[Section]) -> list[Section]:
        """Convert flat list of sections into nested tree by level."""
        if not flat:
            return []

        result: list[Section] = []
        stack: list[Section] = []

        for section in flat:
            level = section["level"]
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            if stack:
                stack[-1]["subsections"].append(section)
            else:
                result.append(section)
            stack.append(section)

        return result
