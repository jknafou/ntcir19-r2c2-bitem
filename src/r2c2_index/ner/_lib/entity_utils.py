"""Shared entity normalization and expansion utilities.

Canonical entity format throughout the pipeline:
    {"text": str, "type": str, "types": [str]}

- "types" is the canonical multi-label list (sorted, deduplicated).
- "type" is the primary type, kept for consumers that read a single label.
- Both fields are present after normalization.
"""

from __future__ import annotations

__all__ = [
    "ENTITY_TYPES",
    "normalize_entity",
    "expand_entities",
    "entity_key",
    "format_entity_for_prompt",
]

ENTITY_TYPES = (
    "PERSON",
    "CHARACTER",
    "LOCATION",
    "ORGANIZATION",
    "EVENT",
    "WORK",
    "DATE",
)


def normalize_entity(e: str | dict) -> dict:
    """Normalize any entity format to canonical {"text", "type", "types"}.

    Handles:
    - Plain string → {"text": str, "type": "PERSON", "types": ["PERSON"]}
    - Single-label {"text", "type"} → adds "types": [type]
    - Multi-label {"text", "types": [...]} → adds "type": types[0]
    - Mixed {"text", "type", "types"} → ensures consistency

    Returns:
        Dict with "text", "type", and "types" fields always present.
    """
    if isinstance(e, str):
        return {"text": e, "type": "PERSON", "types": ["PERSON"]}

    text = e.get("text", "")

    if "types" in e and e["types"]:
        types = sorted(set(e["types"]))
        # "type" field: use existing if valid, else first from types
        primary = e.get("type", types[0])
        if primary not in types:
            primary = types[0]
        return {"text": text, "type": primary, "types": types}

    # Single-label format, or no type at all
    t = e.get("type", "PERSON")
    return {"text": text, "type": t, "types": [t]}


def expand_entities(entities: list[dict]) -> list[dict]:
    """Expand multi-label entities into per-type entries.

    {"text": "Empire", "types": ["LOCATION", "ORGANIZATION"]} becomes:
    [{"text": "Empire", "type": "LOCATION", "types": ["LOCATION"]},
     {"text": "Empire", "type": "ORGANIZATION", "types": ["ORGANIZATION"]}]

    Each expanded entry is independently normalized (both keys present).
    Deduplicates on (text_lower, type).

    Args:
        entities: List of entity dicts (any format).

    Returns:
        List of single-type entity dicts with "text", "type", "types".
    """
    result: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for e in entities:
        ne = normalize_entity(e)
        for t in ne["types"]:
            key = (ne["text"].lower(), t)
            if key not in seen:
                seen.add(key)
                result.append({"text": ne["text"], "type": t, "types": [t]})
    return result


def entity_key(e: dict) -> tuple[str, str]:
    """Return the dedup/comparison key for an entity dict.

    Returns (text_lower, type) tuple. For multi-label entities, call
    expand_entities first to get one key per type.
    """
    ne = normalize_entity(e) if "type" not in e else e
    return (ne["text"].lower(), ne["type"])


def format_entity_for_prompt(e: dict) -> str:
    """Format an entity dict for display in LLM prompts.

    Multi-label: "Empire [LOCATION, ORGANIZATION]"
    Single-label: "Empire [LOCATION]"
    """
    ne = normalize_entity(e)
    types_str = ", ".join(ne["types"])
    return f"{ne['text']} [{types_str}]"
