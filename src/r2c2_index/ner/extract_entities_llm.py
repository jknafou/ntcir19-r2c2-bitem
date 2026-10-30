#!/usr/bin/env python3
"""Extract named entities from passages using an LLM.

Two modes that need only the passages file:

  Per-passage (default): each passage is sent individually to the LLM.
  Per-document (--per-document): all passages of a document are sent together
    in one LLM call for better cross-passage context. Entities are then
    assigned to individual passages via case-insensitive substring matching.

Requirements:
    pip install openai

Usage:
    # Per-passage extraction:
    python3 src/r2c2_index/ner/extract_entities_llm.py \\
        --input data/passages.jsonl \\
        --output data/passages_enriched.jsonl \\
        --vllm-url http://localhost:8000/v1

    # Per-document extraction:
    python3 src/r2c2_index/ner/extract_entities_llm.py \\
        --input data/passages.jsonl \\
        --output data/passages_enriched.jsonl \\
        --per-document \\
        --vllm-url http://localhost:8000/v1

    # Several endpoints, used round-robin:
    python3 src/r2c2_index/ner/extract_entities_llm.py \\
        --input data/passages.jsonl \\
        --output data/passages_enriched.jsonl \\
        --vllm-urls http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1

    # Resume after interruption:
    python3 src/r2c2_index/ner/extract_entities_llm.py \\
        --input data/passages.jsonl \\
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# The helper modules live in _lib/ and are imported by bare name.
# Span utilities are optional: without them the output carries no offsets.
_LIB_DIR = str(Path(__file__).resolve().parent / "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
try:
    from document_reconstruction import reconstruct_document
    from span_utils import find_entity_spans

    _HAS_SPAN_UTILS = True
except ImportError:
    _HAS_SPAN_UTILS = False

# Optional: only --document-level needs the document prompt builder.
try:
    from build_document_prompt import (  # noqa: E402
        build_document_text,
        load_documents_index,
    )

    _HAS_DOC_PROMPT = True
except ImportError:
    _HAS_DOC_PROMPT = False


VLLM_MODEL = "Qwen/Qwen3-32B-FP8"

# Entity type definitions and system prompts

_ENTITY_TYPES = """\
Entity types (use these exact labels):
- PERSON: Real people with unambiguous names (e.g. "Marie Curie", "Anthony Mackie"). \
Exclude ambiguous first-only names like "Anthony" or "Smith".
- CHARACTER: Fictional characters with unambiguous names (e.g. "Luke Skywalker", "Yoda"). \
Exclude ambiguous short forms like "Luke".
- LOCATION: Named geographic or fictional places (e.g. "Brazil", "Tatooine", "New York City"). \
Exclude generic references like "downtown" or "the city".
- ORGANIZATION: Named organizations and abbreviations of named entities \
(e.g. "FBI", "NASA", "Jedi Order"). Exclude generic references like "the government".
- EVENT: Named historical or fictional events (e.g. "American Civil War", "Battle of Endor"). \
Exclude generic references like "the war" or "the battle".
- WORK: Titles of creative works — films, books, games, songs, albums \
(e.g. "Star Wars", "A New Hope"). Exclude generic references like "the movie".
- DATE: Specific dates and years only (e.g. "1978", "July 4, 1776", "19 BBY"). \
Exclude relative references like "last year" or decades like "the 1960s"."""

_AMBIGUITY_RULE = """\

AMBIGUITY RULE (critical): Only tag names that are unambiguous on their own. \
"Anthony Mackie" is valid, but "Anthony" alone is NEVER valid even if it's the \
only form in the passage. Single-name entities are valid only when globally \
unambiguous (Yoda, Madonna, Beyoncé, Chewbacca, Shakespeare). \
If the name could plausibly refer to 10+ different people, exclude it."""

_FORMAT_INSTRUCTIONS = """\

IMPORTANT: Always include the main subject of the passage. Biographical passages \
often start with the subject's name — do NOT skip it because it seems obvious.

OUTPUT FORMAT: One entity per line in the format: Entity [TYPE]
Copy entity text EXACTLY as it appears in the passage — no normalization.
If no named entities are found, output NONE."""

_WIKIPEDIA_EXAMPLES = """

EXAMPLES:

Passage: "Anthony Mackie was born in New Orleans, Louisiana, in 1978. He studied at the Juilliard School and later starred in Captain America: The Winter Soldier."
Output:
Anthony Mackie [PERSON]
New Orleans [LOCATION]
Louisiana [LOCATION]
1978 [DATE]
Juilliard School [ORGANIZATION]
Captain America: The Winter Soldier [WORK]

Passage: "The American Civil War began in 1861 when Confederate forces attacked Fort Sumter in South Carolina."
Output:
American Civil War [EVENT]
1861 [DATE]
Fort Sumter [LOCATION]
South Carolina [LOCATION]

Passage: "The building was constructed in the early 20th century and has undergone several renovations."
Output:
NONE"""

_WOOKIEEPEDIA_EXAMPLES = """

EXAMPLES:

Passage: "Chewbacca accompanied Han Solo on the Millennium Falcon during the Battle of Yavin in 0 BBY. The Rebel Alliance had established their base on Yavin 4."
Output:
Chewbacca [CHARACTER]
Han Solo [CHARACTER]
Millennium Falcon [WORK]
Battle of Yavin [EVENT]
0 BBY [DATE]
Rebel Alliance [ORGANIZATION]
Yavin 4 [LOCATION]

Passage: "Darth Vader ordered the Imperial Navy to blockade the Hoth system. The Sith Lord's fleet included several Star Destroyers."
Output:
Darth Vader [CHARACTER]
Imperial Navy [ORGANIZATION]
Hoth [LOCATION]

Passage: "The Wookiees are a species of tall, hairy humanoids from the planet Kashyyyk."
Output:
Kashyyyk [LOCATION]"""

# Extra rules for fictional-universe corpora (e.g. Wookieepedia)
_FICTIONAL_UNIVERSE_RULES = """\

Additional rules for this fictional-universe text:
- Do NOT include species names (e.g. "Wookiee", "Hutt", "Twi'lek", "Trandoshan", "Human")
- Do NOT include Force-user titles or traditions used generically \
(e.g. "Jedi", "Sith", "Padawan", "Dark Lord"). But DO include \
specific organizations like "Jedi Order", "Sith Empire"
- Do NOT include generic weapon/vehicle types \
(e.g. "lightsaber", "blaster", "star destroyer", "speeder")
- Do NOT include generic political/military ranks \
(e.g. "Moff", "Grand Admiral", "Senator", "Chancellor") unless part \
of a full name like "Grand Moff Tarkin\""""

_ZONE_INSTRUCTIONS = """\

The text is divided into zones: [TITLE], [SECTION], [CONTENT]. \
Extract entities from ALL zones. The title and section heading often contain \
important entities — do not skip them."""

SYSTEM_PROMPT_BASE = (
    "You extract named entities from text and classify each by type.\n\n"
    + _ENTITY_TYPES
    + _AMBIGUITY_RULE
    + _FORMAT_INSTRUCTIONS
    + _ZONE_INSTRUCTIONS
    + _WIKIPEDIA_EXAMPLES
)

_SYSTEM_PROMPT_WOOKIEEPEDIA = (
    "You extract named entities from text and classify each by type.\n\n"
    + _ENTITY_TYPES
    + _FICTIONAL_UNIVERSE_RULES
    + _AMBIGUITY_RULE
    + _FORMAT_INSTRUCTIONS
    + _ZONE_INSTRUCTIONS
    + _WOOKIEEPEDIA_EXAMPLES
)

# Sources absent from this mapping use SYSTEM_PROMPT_BASE.
_SOURCE_PROMPTS: dict[str, str] = {
    "wookieepedia": _SYSTEM_PROMPT_WOOKIEEPEDIA,
}

VALID_ENTITY_TYPES = {
    "PERSON",
    "CHARACTER",
    "LOCATION",
    "ORGANIZATION",
    "EVENT",
    "WORK",
    "DATE",
}


def get_system_prompt(source: str, prompt_dir: str | None = None) -> str:
    """Return the appropriate system prompt for a given source collection.

    Args:
        source: Source collection name (e.g. 'wookieepedia').
        prompt_dir: If provided, load prompt from this directory instead
            of using hardcoded defaults.

    Returns:
        System prompt string.
    """
    if prompt_dir is not None:
        prompt_path = Path(prompt_dir)
        if source == "wookieepedia":
            f = prompt_path / "extract_wookieepedia.txt"
        else:
            f = prompt_path / "extract_wikipedia.txt"
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
    return _SOURCE_PROMPTS.get(source, SYSTEM_PROMPT_BASE)


def get_docl_system_prompt(source: str, prompt_dir: str | None = None) -> str:
    """Return the document-level system prompt for a given source.

    Args:
        source: Source collection name (e.g. 'wookieepedia').
        prompt_dir: If provided, load prompt from this directory.

    Returns:
        System prompt string for document-level extraction.
    """
    if prompt_dir is not None:
        prompt_path = Path(prompt_dir)
        if source == "wookieepedia":
            f = prompt_path / "extract_wookieepedia_docl.txt"
        else:
            f = prompt_path / "extract_wikipedia_docl.txt"
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
    # Fallback to passage-level prompts with zone instructions
    return get_system_prompt(source, prompt_dir)


# Strip wiki bold/italic markers: '''bold''' or ''italic''
_WIKI_MARKUP_RE = re.compile(r"'{2,3}")


# Endpoint pool, imported here because _lib/ is on sys.path only after the block above.
from llm_utils import EndpointPool  # noqa: E402

# Text helpers


def clean_wiki_markup(text: str) -> str:
    """Strip basic wiki formatting (bold/italic markers)."""
    return _WIKI_MARKUP_RE.sub("", text)


# Regex for typed entity output: "Entity Text [TYPE]" or "Entity Text [TYPE1, TYPE2]"
_TYPED_ENTITY_RE = re.compile(r"^(.+?)\s*\[([^\]]+)\]\s*$")


def parse_entity_list(raw: str) -> list[str]:
    """Parse LLM output into a deduplicated list of untyped entities.

    Handles common formatting variations: numbered lists, bullet points,
    markdown bold, trailing punctuation.
    """
    entities: list[str] = []
    seen: set[str] = set()
    for line in raw.strip().split("\n"):
        # Strip list prefixes: "1. ", "- ", "* ", "• "
        e = line.strip()
        e = re.sub(r"^[\d]+[.)]\s*", "", e)  # numbered
        e = e.lstrip("-*•– ").strip()
        # Strip markdown bold
        e = e.strip("*").strip()
        if not e or len(e) < 2:
            continue
        if e.upper() == "NONE":
            continue
        e_lower = e.lower()
        if e_lower not in seen:
            seen.add(e_lower)
            entities.append(e)
    return entities


def parse_typed_entity_list(raw: str) -> list[dict]:
    """Parse LLM output into a deduplicated list of typed entity dicts.

    Supports single-type ("Entity [TYPE]") and multi-type
    ("Entity [TYPE1, TYPE2]") formats. Falls back to untyped parsing
    if no typed entities are found.

    Returns:
        List of {"text": str, "types": [str]} dicts. Each entity carries
        a sorted list of one or more types.
    """
    entities: list[dict] = []
    # Dedup key: text_lower only. Types are merged if same text seen again.
    seen: dict[str, int] = {}  # text_lower -> index in entities
    has_typed = False

    for line in raw.strip().split("\n"):
        e = line.strip()
        e = re.sub(r"^[\d]+[.)]\s*", "", e)  # numbered
        e = e.lstrip("-*•– ").strip()
        e = e.strip("*").strip()
        if not e or len(e) < 2:
            continue
        if e.upper() == "NONE":
            continue

        # Try typed format: "Entity [TYPE]" or "Entity [TYPE1, TYPE2]"
        m = _TYPED_ENTITY_RE.match(e)
        if m:
            text = m.group(1).strip()
            type_str = m.group(2).upper()
            etypes = [t.strip() for t in type_str.split(",")]
            etypes = [t for t in etypes if t in VALID_ENTITY_TYPES]
            if not etypes:
                etypes = ["PERSON"]  # fallback for unknown types
            has_typed = True
            key = text.lower()
            if key in seen:
                idx = seen[key]
                existing = set(entities[idx]["types"])
                existing.update(etypes)
                entities[idx]["types"] = sorted(existing)
            else:
                seen[key] = len(entities)
                entities.append({"text": text, "types": sorted(set(etypes))})
        else:
            # Untyped lines are handled by the fallback below.
            pass

    # Fallback: if no typed entities found, parse as untyped with PERSON default
    if not has_typed:
        for line in raw.strip().split("\n"):
            e = line.strip()
            e = re.sub(r"^[\d]+[.)]\s*", "", e)
            e = e.lstrip("-*•– ").strip()
            e = e.strip("*").strip()
            if not e or len(e) < 2:
                continue
            if e.upper() == "NONE":
                continue
            key = e.lower()
            if key not in seen:
                seen[key] = len(entities)
                entities.append({"text": e, "types": ["PERSON"]})

    return entities


# LLM interaction


async def extract_entities_for_passage(
    client,  # AsyncOpenAI
    model: str,
    passage: dict,
    max_retries: int = 3,
    prompt_dir: str | None = None,
) -> list[dict[str, str]]:
    """Send a passage to the LLM and extract typed named entities."""
    source = passage.get("source", "")
    title = passage.get("title", "")
    section = passage.get("section", "")
    contents = clean_wiki_markup(passage.get("contents", ""))

    system_prompt = get_system_prompt(source, prompt_dir)

    # Zone markers let the model attribute entities to title, section or content.
    parts = [f"[TITLE]\n{title}"]
    if section:
        parts.append(f"[SECTION]\n{section}")
    parts.append(f"[CONTENT]\n{contents}")
    user_msg = "\n\n".join(parts)

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=300,
                timeout=600.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            return parse_typed_entity_list(raw)
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(
                    f"  [retry {attempt + 1}] {passage.get('id', '?')}: {exc}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
            else:
                print(
                    f"  [error] {passage.get('id', '?')}: {exc}",
                    file=sys.stderr,
                )
                return []


# Per-document LLM interaction


async def extract_entities_for_document(
    client,  # AsyncOpenAI
    model: str,
    passages: list[dict],
    max_retries: int = 3,
    prompt_dir: str | None = None,
) -> list[dict[str, str]]:
    """Send all passages of a document to the LLM in one call.

    Concatenates passage texts with section markers so the LLM sees
    full document context. Returns a single deduplicated typed entity list
    for the whole document.
    """
    title = passages[0].get("title", "")
    source = passages[0].get("source", "")
    system_prompt = get_system_prompt(source, prompt_dir)

    parts = [f"Title: {title}"]
    for p in passages:
        section = p.get("section", "")
        contents = clean_wiki_markup(p.get("contents", ""))
        if section:
            parts.append(f"\n--- Section: {section} ---\n{contents}")
        else:
            parts.append(f"\n---\n{contents}")
    user_msg = "\n".join(parts)

    # Scale max_tokens by number of passages (more text = more entities)
    max_tokens = min(300 * len(passages), 4096)
    # Scale timeout by document size: 30s per passage, minimum 600s.
    # The high floor lets vLLM queue requests without the client timing out.
    timeout = max(600.0, 30.0 * len(passages))

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            return parse_typed_entity_list(raw)
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2**attempt
                doc_id = passages[0].get("doc_id", "?")
                print(
                    f"  [retry {attempt + 1}] doc {doc_id}: {exc}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
            else:
                doc_id = passages[0].get("doc_id", "?")
                print(
                    f"  [error] doc {doc_id}: {exc}",
                    file=sys.stderr,
                )
                return []


# Document-level LLM interaction (XML-zoned, uses documents.jsonl metadata)


async def extract_entities_for_document_level(
    client,  # AsyncOpenAI
    model: str,
    doc_record: dict,
    passages: list[dict],
    max_retries: int = 3,
    prompt_dir: str | None = None,
) -> list[dict[str, str]]:
    """Send full document (XML-zoned) to the LLM in one call.

    Uses build_document_text to render title, infobox, passages, categories,
    and appearances as XML zones. Returns a single deduplicated entity list.

    Args:
        client: AsyncOpenAI client.
        model: Model name.
        doc_record: Document metadata dict from documents.jsonl.
        passages: List of passage dicts for this document.
        max_retries: Number of retries on failure.
        prompt_dir: Optional prompt directory.

    Returns:
        List of typed entity dicts.
    """
    source = doc_record.get("source", passages[0].get("source", ""))
    system_prompt = get_docl_system_prompt(source, prompt_dir)
    user_msg = build_document_text(doc_record, passages)

    n_passages = len(passages)
    max_tokens = min(300 * max(n_passages, 1), 4096)
    timeout = max(600.0, 30.0 * n_passages)

    doc_id = doc_record.get("doc_id", "?")

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            return parse_typed_entity_list(raw)
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(
                    f"  [retry {attempt + 1}] doc {doc_id}: {exc}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
            else:
                print(
                    f"  [error] doc {doc_id}: {exc}",
                    file=sys.stderr,
                )
                return []


# Passage enrichment


def expand_multilabel_entities(entities: list[dict]) -> list[dict]:
    """Expand multi-label entities into single-type entries.

    Converts {"text": "Empire", "types": ["LOCATION", "ORGANIZATION"]}
    into two entries: {"text": "Empire", "type": "LOCATION", "types": [...]}
    and {"text": "Empire", "type": "ORGANIZATION", "types": [...]}.

    Also handles entities that carry "type" and no "types" key.

    Args:
        entities: List of entity dicts (multi-label or single-label).

    Returns:
        List of single-type entity dicts with both "type" and "types" fields.
    """
    result: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for e in entities:
        types = e.get("types", [e["type"]] if "type" in e else ["PERSON"])
        text = e["text"]
        for t in types:
            key = (text.lower(), t)
            if key not in seen:
                seen.add(key)
                result.append({"text": text, "type": t, "types": sorted(types)})
    return result


def build_enriched_passage(passage: dict, llm_entities: list[dict]) -> dict:
    """Assemble the enriched passage with zone-separated entity fields.

    Accepts multi-label entities ({"text", "types": [...]}) and expands
    them into per-type entries for downstream processing.
    """
    expanded = expand_multilabel_entities(llm_entities)

    title = passage.get("title", "")
    section = passage.get("section", "")
    contents = clean_wiki_markup(passage.get("contents", ""))

    title_lower = title.lower()
    section_lower = section.lower()
    contents_lower = contents.lower()
    full_lower = f"{title_lower} {section_lower} {contents_lower}"

    # Assign entities to zones via substring matching
    seen: set[tuple[str, str]] = set()
    title_entities: list[dict] = []
    section_entities: list[dict] = []
    content_entities: list[dict] = []

    for e in expanded:
        key = (e["text"].lower(), e["type"])
        if key in seen:
            continue
        e_lower = e["text"].lower()
        if e_lower not in full_lower:
            continue
        seen.add(key)
        if e_lower in title_lower:
            title_entities.append(e)
        if e_lower in section_lower:
            section_entities.append(e)
        if e_lower in contents_lower:
            content_entities.append(e)

    # text_entities: deduplicated union of all zones
    union_seen: set[tuple[str, str]] = set()
    text_entities: list[dict] = []
    for e in title_entities + section_entities + content_entities:
        key = (e["text"].lower(), e["type"])
        if key not in union_seen:
            union_seen.add(key)
            text_entities.append(e)

    result = {
        "id": passage["id"],
        "contents": contents,
        "title": title,
        "section": section,
        "source": passage.get("source", ""),
        "doc_id": passage.get("doc_id", ""),
        "title_entities": title_entities,
        "section_entities": section_entities,
        "content_entities": content_entities,
        "text_entities": text_entities,
        "entities": text_entities,
        "doc_entities": [],
    }

    if _HAS_SPAN_UTILS:
        result["content_spans"] = find_entity_spans(
            contents, content_entities, "content"
        )
        result["title_spans"] = find_entity_spans(title, title_entities, "title")
        result["section_spans"] = find_entity_spans(
            section, section_entities, "section"
        )

    return result


# Document-level entity propagation


def propagate_document_entities(filepath: str) -> None:
    """Propagate typed entities across all passages of the same document.

    After LLM extraction, each passage has text_entities for its own text.
    This pass computes the union of all text_entities per doc_id, then sets
    each passage's `entities` field to: title + all document-wide entities.

    The `text_entities` field is preserved as-is (passage-level).
    """
    print("\nPropagating document-level entities...")
    t0 = time.time()

    # Pass 1: collect all typed entities per doc_id
    # Store {doc_id: {(text_lower, type): entity_dict}}
    doc_entity_map: dict[str, dict[tuple[str, str], dict[str, str]]] = {}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id", "")
            if doc_id not in doc_entity_map:
                doc_entity_map[doc_id] = {}
            for e in doc.get("text_entities", []):
                key = (e["text"].lower(), e["type"])
                # Keep the first-seen form
                if key not in doc_entity_map[doc_id]:
                    doc_entity_map[doc_id][key] = e

    # Pass 2: rewrite with propagated entities
    tmp = filepath + ".tmp"
    total = 0
    total_before = 0
    total_after = 0

    with (
        open(filepath, encoding="utf-8") as fin,
        open(tmp, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id", "")

            old_entities = doc.get("entities", [])
            total_before += len(old_entities)

            seen: set[tuple[str, str]] = set()
            entities: list[dict[str, str]] = []

            for ent_key, ent_dict in doc_entity_map.get(doc_id, {}).items():
                if ent_key not in seen:
                    seen.add(ent_key)
                    entities.append(ent_dict)

            doc["entities"] = entities
            total_after += len(entities)
            total += 1

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    os.replace(tmp, filepath)

    elapsed = time.time() - t0
    avg_before = total_before / max(total, 1)
    avg_after = total_after / max(total, 1)
    print(f"  {total:,} passages in {elapsed:.1f}s")
    print(f"  {len(doc_entity_map):,} unique documents")
    print(f"  Avg entities per passage: {avg_before:.1f} -> {avg_after:.1f}")


# Chunk processing


async def process_chunk(
    pool: EndpointPool,
    model: str,
    passages: list[dict],
    prompt_dir: str | None = None,
) -> list[dict]:
    """Process a chunk of passages concurrently; return enriched dicts in order."""
    tasks = []
    for p in passages:
        client = pool.next()
        tasks.append(
            extract_entities_for_passage(client, model, p, prompt_dir=prompt_dir)
        )
    entity_lists = await asyncio.gather(*tasks)
    return [build_enriched_passage(p, ents) for p, ents in zip(passages, entity_lists)]


async def process_document_chunk(
    pool: EndpointPool,
    model: str,
    documents: list[list[dict]],
    prompt_dir: str | None = None,
) -> list[dict]:
    """Process a chunk of documents concurrently in per-document mode.

    Each document (list of passages) gets one LLM call. The returned
    entity list is then distributed to individual passages via
    build_enriched_passage (which applies substring validation).
    """
    tasks = []
    for doc_passages in documents:
        client = pool.next()
        tasks.append(
            extract_entities_for_document(
                client, model, doc_passages, prompt_dir=prompt_dir
            )
        )
    doc_entity_lists = await asyncio.gather(*tasks)

    results = []
    for doc_passages, doc_entities in zip(documents, doc_entity_lists):
        for p in doc_passages:
            # build_enriched_passage filters doc_entities to substring matches
            results.append(build_enriched_passage(p, doc_entities))
    return results


async def process_document_level_chunk(
    pool: EndpointPool,
    model: str,
    doc_records: list[dict],
    doc_passages: list[list[dict]],
    prompt_dir: str | None = None,
) -> list[dict]:
    """Process a chunk of documents using XML-zoned document-level mode.

    Each document gets one LLM call with full metadata (infobox, categories,
    appearances) rendered as XML zones. Entities are distributed to passages
    via substring matching.

    Args:
        pool: Endpoint pool.
        model: Model name.
        doc_records: List of document metadata dicts from documents.jsonl.
        doc_passages: List of passage lists (one per document).
        prompt_dir: Optional prompt directory.

    Returns:
        List of enriched passage dicts.
    """
    tasks = []
    for doc_record, passages in zip(doc_records, doc_passages):
        client = pool.next()
        tasks.append(
            extract_entities_for_document_level(
                client, model, doc_record, passages, prompt_dir=prompt_dir
            )
        )
    doc_entity_lists = await asyncio.gather(*tasks)

    results = []
    for passages, doc_entities in zip(doc_passages, doc_entity_lists):
        for p in passages:
            results.append(build_enriched_passage(p, doc_entities))
    return results


async def run(args: argparse.Namespace) -> None:
    """Async entry point."""
    # main() has already folded --vllm-url into args.vllm_urls.
    urls = [u.strip() for u in args.vllm_urls.split(",") if u.strip()]
    pool = EndpointPool(urls)

    if args.document_level:
        await _run_document_level(args, pool)
    elif args.per_document:
        await _run_per_document(args, pool)
    else:
        await _run_per_passage(args, pool)


async def _run_per_passage(args: argparse.Namespace, pool: EndpointPool) -> None:
    """Per-passage extraction: one LLM call per passage."""
    prompt_dir = getattr(args, "prompt_dir", None)
    # Resume support
    skip = 0
    mode = "w"
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            skip = sum(1 for _ in f)
        mode = "a"
        print(f"Resuming: skipping {skip:,} already-processed passages")

    # Count total
    print("Counting input passages...")
    total_input = 0
    with open(args.input, encoding="utf-8") as f:
        for _ in f:
            total_input += 1
    remaining = total_input - skip

    # Shuffle sampling
    sample_indices: set[int] | None = None
    if args.shuffle and args.limit and args.limit < remaining:
        sample_indices = set(random.sample(range(skip, total_input), args.limit))
        print(
            f"Total passages: {total_input:,}, random sample: {args.limit:,} passages"
        )
    else:
        print(f"Total passages: {total_input:,}, to process: {remaining:,}")

    print("Mode: per-passage")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}")
    print(
        f"Endpoints: {', '.join([u.strip() for u in args.vllm_urls.split(',') if u.strip()])}"
    )
    print()

    t0 = time.time()
    processed = 0
    total_entities = 0
    target = args.limit if args.limit else remaining

    with (
        open(args.input, encoding="utf-8") as fin,
        open(args.output, mode, encoding="utf-8") as fout,
    ):
        chunk: list[dict] = []
        line_idx = -1
        for line in fin:
            line_idx += 1

            # Skip already-processed lines (resume)
            if line_idx < skip:
                continue

            # Shuffle mode: only process selected indices
            if sample_indices is not None and line_idx not in sample_indices:
                continue

            line = line.strip()
            if not line:
                continue
            try:
                passage = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk.append(passage)

            if len(chunk) >= args.chunk_size:
                results = await process_chunk(
                    pool, args.model, chunk, prompt_dir=prompt_dir
                )
                for doc in results:
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total_entities += len(doc["text_entities"])
                fout.flush()
                processed += len(chunk)
                chunk = []

                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (target - processed) / rate if rate > 0 else 0
                avg_ents = total_entities / processed
                print(
                    f"  {processed:>10,} / {target:,}  "
                    f"({processed / target * 100:5.1f}%)  "
                    f"{rate:,.0f}/s  "
                    f"avg_entities={avg_ents:.1f}  "
                    f"ETA {eta / 3600:.1f}h"
                )

            # Sequential limit (no shuffle)
            if (
                sample_indices is None
                and args.limit
                and processed + len(chunk) >= args.limit
            ):
                needed = args.limit - processed
                chunk = chunk[:needed]
                break

        # Final partial chunk
        if chunk:
            results = await process_chunk(
                pool, args.model, chunk, prompt_dir=prompt_dir
            )
            for doc in results:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_entities += len(doc["text_entities"])
            fout.flush()
            processed += len(chunk)

    elapsed = time.time() - t0
    avg_ents = total_entities / max(processed, 1)
    rate = processed / elapsed if elapsed > 0 else 0
    print(
        f"\nLLM extraction done: {processed:,} passages in {elapsed:.1f}s ({rate:,.0f}/s)"
    )
    print(f"  Avg text_entities per passage: {avg_ents:.1f}")
    print(f"  Output: {args.output}")

    # Document-level entity propagation
    if not args.no_propagate:
        propagate_document_entities(args.output)


async def _run_per_document(args: argparse.Namespace, pool: EndpointPool) -> None:
    """Per-document extraction: one LLM call per document.

    Groups passages by doc_id, sends all passages of each document in
    one LLM call, then distributes entities to passages via substring
    matching in build_enriched_passage.
    """
    prompt_dir = getattr(args, "prompt_dir", None)
    print("Loading and grouping passages by document...")
    doc_passages: dict[str, list[dict]] = {}  # preserves insertion order
    total_input = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                passage = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_input += 1
            doc_id = passage.get("doc_id", passage.get("id", "unknown"))
            if doc_id not in doc_passages:
                doc_passages[doc_id] = []
            doc_passages[doc_id].append(passage)

    doc_ids = list(doc_passages.keys())
    n_docs = len(doc_ids)

    # Resume support
    skip_docs = 0
    mode = "w"
    if args.resume and os.path.exists(args.output):
        written_doc_ids: set[str] = set()
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                written_doc_ids.add(rec.get("doc_id", ""))
        skip_docs = len(written_doc_ids)
        mode = "a"
        print(f"Resuming: skipping {skip_docs:,} already-processed documents")
        doc_ids = [d for d in doc_ids if d not in written_doc_ids]

    # Apply limit (in documents)
    target_docs = n_docs - skip_docs
    if args.limit and args.limit < target_docs:
        doc_ids = doc_ids[: args.limit]
        target_docs = args.limit

    target_passages = sum(len(doc_passages[d]) for d in doc_ids)

    print(f"Total: {total_input:,} passages in {n_docs:,} documents")
    print(f"To process: {target_passages:,} passages in {target_docs:,} documents")
    print("Mode: per-document")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size} docs")
    print(
        f"Endpoints: {', '.join([u.strip() for u in args.vllm_urls.split(',') if u.strip()])}"
    )
    print()

    t0 = time.time()
    processed_docs = 0
    processed_passages = 0
    total_entities = 0

    with open(args.output, mode, encoding="utf-8") as fout:
        chunk: list[list[dict]] = []
        chunk_passage_count = 0

        for doc_id in doc_ids:
            passages = doc_passages[doc_id]
            chunk.append(passages)
            chunk_passage_count += len(passages)

            if len(chunk) >= args.chunk_size:
                results = await process_document_chunk(
                    pool, args.model, chunk, prompt_dir=prompt_dir
                )
                for enriched in results:
                    fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    total_entities += len(enriched["text_entities"])
                fout.flush()
                processed_docs += len(chunk)
                processed_passages += chunk_passage_count
                chunk = []
                chunk_passage_count = 0

                elapsed = time.time() - t0
                rate = processed_passages / elapsed if elapsed > 0 else 0
                doc_rate = processed_docs / elapsed if elapsed > 0 else 0
                eta = (target_passages - processed_passages) / rate if rate > 0 else 0
                avg_ents = total_entities / processed_passages
                print(
                    f"  {processed_docs:>6,} / {target_docs:,} docs  "
                    f"({processed_passages:,} passages)  "
                    f"{doc_rate:,.0f} docs/s  "
                    f"{rate:,.0f} pass/s  "
                    f"avg_entities={avg_ents:.1f}  "
                    f"ETA {eta / 3600:.1f}h"
                )

        # Final partial chunk
        if chunk:
            results = await process_document_chunk(
                pool, args.model, chunk, prompt_dir=prompt_dir
            )
            for enriched in results:
                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                total_entities += len(enriched["text_entities"])
            fout.flush()
            processed_docs += len(chunk)
            processed_passages += chunk_passage_count

    elapsed = time.time() - t0
    avg_ents = total_entities / max(processed_passages, 1)
    rate = processed_passages / elapsed if elapsed > 0 else 0
    doc_rate = processed_docs / elapsed if elapsed > 0 else 0
    print(
        f"\nLLM extraction done (per-document): "
        f"{processed_docs:,} docs ({processed_passages:,} passages) "
        f"in {elapsed:.1f}s ({doc_rate:,.0f} docs/s, {rate:,.0f} pass/s)"
    )
    print(f"  Avg text_entities per passage: {avg_ents:.1f}")
    print(f"  Output: {args.output}")

    # Document-level entity propagation
    if not args.no_propagate:
        propagate_document_entities(args.output)


async def _run_document_level(args: argparse.Namespace, pool: EndpointPool) -> None:
    """Document-level extraction with XML-zoned metadata from documents.jsonl.

    Groups passages by doc_id, loads document metadata, builds XML-zoned input
    with infobox/categories/appearances, sends one LLM call per document.
    """
    prompt_dir = getattr(args, "prompt_dir", None)

    if not _HAS_DOC_PROMPT:
        print(
            "ERROR: --document-level requires build_document_prompt.py "
            "(not found in NER-tagging/scripts/)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.documents:
        print(
            "ERROR: --document-level requires --documents <path/to/documents.jsonl>",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading document metadata from {args.documents}...")
    doc_index = load_documents_index(args.documents)
    print(f"  {len(doc_index)} documents loaded")

    print("Loading and grouping passages by document...")
    doc_passages: dict[str, list[dict]] = {}
    total_input = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                passage = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_input += 1
            doc_id = passage.get("doc_id", passage.get("id", "unknown"))
            if doc_id not in doc_passages:
                doc_passages[doc_id] = []
            doc_passages[doc_id].append(passage)

    # Only process documents that have metadata
    doc_ids = [d for d in doc_passages if d in doc_index]
    n_docs = len(doc_ids)
    n_no_meta = len(doc_passages) - n_docs
    if n_no_meta > 0:
        print(f"  WARNING: {n_no_meta} documents have no metadata in {args.documents}")

    # Resume support
    skip_docs = 0
    mode = "w"
    if args.resume and os.path.exists(args.output):
        written_doc_ids: set[str] = set()
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                written_doc_ids.add(rec.get("doc_id", ""))
        skip_docs = len(written_doc_ids)
        mode = "a"
        print(f"Resuming: skipping {skip_docs:,} already-processed documents")
        doc_ids = [d for d in doc_ids if d not in written_doc_ids]

    # Apply limit
    target_docs = len(doc_ids)
    if args.limit and args.limit < target_docs:
        doc_ids = doc_ids[: args.limit]
        target_docs = args.limit

    target_passages = sum(len(doc_passages[d]) for d in doc_ids)

    print(f"Total: {total_input:,} passages in {n_docs + n_no_meta:,} documents")
    print(f"To process: {target_passages:,} passages in {target_docs:,} documents")
    print("Mode: document-level (XML-zoned with metadata)")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size} docs")
    print(
        f"Endpoints: {', '.join([u.strip() for u in args.vllm_urls.split(',') if u.strip()])}"
    )
    print()

    t0 = time.time()
    processed_docs = 0
    processed_passages = 0
    total_entities = 0

    with open(args.output, mode, encoding="utf-8") as fout:
        chunk_records: list[dict] = []
        chunk_passages: list[list[dict]] = []
        chunk_passage_count = 0

        for doc_id in doc_ids:
            passages = doc_passages[doc_id]
            chunk_records.append(doc_index[doc_id])
            chunk_passages.append(passages)
            chunk_passage_count += len(passages)

            if len(chunk_records) >= args.chunk_size:
                results = await process_document_level_chunk(
                    pool,
                    args.model,
                    chunk_records,
                    chunk_passages,
                    prompt_dir=prompt_dir,
                )
                for enriched in results:
                    fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    total_entities += len(enriched["text_entities"])
                fout.flush()
                processed_docs += len(chunk_records)
                processed_passages += chunk_passage_count
                chunk_records = []
                chunk_passages = []
                chunk_passage_count = 0

                elapsed = time.time() - t0
                rate = processed_passages / elapsed if elapsed > 0 else 0
                doc_rate = processed_docs / elapsed if elapsed > 0 else 0
                eta = (target_passages - processed_passages) / rate if rate > 0 else 0
                avg_ents = total_entities / processed_passages
                print(
                    f"  {processed_docs:>6,} / {target_docs:,} docs  "
                    f"({processed_passages:,} passages)  "
                    f"{doc_rate:,.0f} docs/s  "
                    f"{rate:,.0f} pass/s  "
                    f"avg_entities={avg_ents:.1f}  "
                    f"ETA {eta / 3600:.1f}h"
                )

        # Final partial chunk
        if chunk_records:
            results = await process_document_level_chunk(
                pool,
                args.model,
                chunk_records,
                chunk_passages,
                prompt_dir=prompt_dir,
            )
            for enriched in results:
                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                total_entities += len(enriched["text_entities"])
            fout.flush()
            processed_docs += len(chunk_records)
            processed_passages += chunk_passage_count

    elapsed = time.time() - t0
    avg_ents = total_entities / max(processed_passages, 1)
    rate = processed_passages / elapsed if elapsed > 0 else 0
    doc_rate = processed_docs / elapsed if elapsed > 0 else 0
    print(
        f"\nLLM extraction done (document-level): "
        f"{processed_docs:,} docs ({processed_passages:,} passages) "
        f"in {elapsed:.1f}s ({doc_rate:,.0f} docs/s, {rate:,.0f} pass/s)"
    )
    print(f"  Avg text_entities per passage: {avg_ents:.1f}")
    print(f"  Output: {args.output}")

    # Document-level entity propagation
    if not args.no_propagate:
        propagate_document_entities(args.output)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract named entities from passages using an LLM"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to passages.jsonl (source passages)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input_dir>/passages_enriched.jsonl)",
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
        help="Passages per processing chunk (default: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--per-document",
        action="store_true",
        help="Per-document mode: send all passages of each document in one LLM call",
    )
    parser.add_argument(
        "--no-propagate",
        action="store_true",
        help="Skip document-level entity propagation",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly sample passages (use with --limit for representative testing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of passages (0=all, for testing)",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory with prompt files (extract_wikipedia.txt, extract_wookieepedia.txt). If absent, uses hardcoded defaults.",
    )
    parser.add_argument(
        "--document-level",
        action="store_true",
        help="Document-level mode: use XML-zoned input with metadata from documents.jsonl. Requires --documents.",
    )
    parser.add_argument(
        "--documents",
        default=None,
        help="Path to documents.jsonl (metadata: infobox, categories, appearances). Required for --document-level.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.input), "passages_enriched.jsonl"
        )

    # Resolve endpoint URLs: --vllm-urls > --vllm-url > default
    if args.vllm_urls is None:
        args.vllm_urls = args.vllm_url or "http://localhost:8000/v1"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
