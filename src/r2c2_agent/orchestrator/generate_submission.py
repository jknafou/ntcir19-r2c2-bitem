"""Trace → passages + nuggets extraction helpers.

Two pure functions are exposed for the worker's post-trace path
(:func:`r2c2_agent.orchestrator.worker._run_post_trace`):

- :func:`extract_passages_from_trace` — citation-first hybrid
  ranking over passages observed in tool results.
- :func:`extract_nuggets_from_trace` — primary regex match on
  ``NUGGET[passage_id]: text`` tags, with a reasoning-text synthesiser
  fallback for traces where the LLM emitted bullets instead of tags.

Submission ZIP emission lives in
:mod:`r2c2_agent.orchestrator.ensemble_aggregator` (PR/AC files +
:mod:`r2c2_agent.cli` zip-pack); this module is helpers only.
"""

from __future__ import annotations

import json as _json
import re

__all__ = [
    "extract_nuggets_from_trace",
    "extract_passages_from_trace",
]


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting markers from text."""
    for marker in ("**", "__", "*", "_", "`"):
        text = text.replace(marker, "")
    return text.strip()


def extract_passages_from_trace(trace: dict, *, limit: int = 20) -> list[dict]:
    """Extract and rank passages from a trace, cited passages first.

    Citation-first hybrid ranking:
    1. Cited passages first (``NUGGET[…]`` tags + free-text passage_id
       mentions), sorted by citation count desc, then MCP score desc.
    2. Uncited passages, sorted by MCP score desc.
    3. Tiebreak: first appearance ascending.
    4. ``score=None`` passages rank last within their tier.

    Args:
        trace: Rollout trace dict.
        limit: Max passages to return. ``0`` = all.

    Returns:
        List of dicts with ``passage_id``, ``doc_id``, ``passage_text``,
        ``title``, ``section``, ``score``, ``citation_count``.
    """
    citation_count: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for idx, msg in enumerate(trace.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content", "") or "") + (msg.get("reasoning", "") or "")
        for pid in re.findall(r"NUGGET\[([^\]]+)\]", text):
            pid = pid.strip()
            citation_count[pid] = citation_count.get(pid, 0) + 1
            if pid not in first_seen:
                first_seen[pid] = idx
        for pid in re.findall(r"(?:passage\s+)?([A-Z]?\d+__\d+)", text):
            citation_count[pid] = citation_count.get(pid, 0) + 1
            if pid not in first_seen:
                first_seen[pid] = idx

    passage_data: dict[str, dict] = {}

    for idx, msg in enumerate(trace.get("messages", [])):
        if msg.get("role") != "tool":
            continue
        try:
            data = _json.loads(msg.get("content", ""))
        except (_json.JSONDecodeError, TypeError):
            continue
        items = []
        if isinstance(data, dict):
            for key in ("passages", "results"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            if not items and "passage_id" in data:
                items = [data]
        elif isinstance(data, list):
            items = data
        for p in items:
            pid = p.get("passage_id", "")
            if not pid:
                continue
            if pid not in passage_data:
                passage_data[pid] = {
                    "text": p.get("text", p.get("passage_text", "")),
                    "title": p.get("title", ""),
                    "section": p.get("section", ""),
                    "score": p.get("score"),
                    "doc_id": p.get(
                        "doc_id", pid.rsplit("__", 1)[0] if "__" in pid else pid
                    ),
                }
            if pid not in first_seen:
                first_seen[pid] = idx

    all_pids = set(citation_count.keys()) | set(passage_data.keys())
    ranked = []
    for pid in all_pids:
        cc = citation_count.get(pid, 0)
        score = passage_data.get(pid, {}).get("score")
        score_val = score if score is not None else float("-inf")
        is_cited = 1 if cc > 0 else 0
        ranked.append((pid, -is_cited, -cc, -score_val, first_seen.get(pid, 999)))

    ranked.sort(key=lambda x: (x[1], x[2], x[3], x[4]))

    passages = []
    top = ranked if limit == 0 else ranked[:limit]
    for pid, *_ in top:
        pdata = passage_data.get(pid, {})
        text = pdata.get("text", f"[passage text not available for {pid}]")
        doc_id = pdata.get("doc_id", pid.rsplit("__", 1)[0] if "__" in pid else pid)
        passages.append(
            {
                "passage_id": pid,
                "doc_id": doc_id,
                "title": pdata.get("title", ""),
                "section": pdata.get("section", ""),
                "passage_text": re.sub(r"\s+", " ", text).strip(),
                "score": pdata.get("score"),
                "citation_count": citation_count.get(pid, 0),
            }
        )

    return passages


def extract_nuggets_from_trace(trace: dict) -> list[dict]:
    """Extract nuggets from the trace.

    Primary: ``NUGGET[passage_id]: text`` tags in reasoning/content.
    Fallback: synthesise nuggets from ``passage_id`` references in
    reasoning paired with nearby bullet claims.

    Args:
        trace: Rollout trace dict.

    Returns:
        List of dicts with ``passage_id`` and ``nugget_text`` (≤ 10).
    """
    nuggets = []
    seen_texts: set[str] = set()

    # Primary: NUGGET[passage_id]: text
    for msg in trace.get("messages", []):
        if msg.get("role") == "assistant":
            text = (msg.get("content", "") or "") + (msg.get("reasoning", "") or "")
            matches = re.findall(r"NUGGET\[([^\]]+)\]:\s*(.+?)(?:\n|$)", text)
            for pid, ntext in matches:
                ntext = _strip_markdown(ntext).replace(";", ",").replace("\n", " ")
                if ntext and ntext not in seen_texts:
                    seen_texts.add(ntext)
                    nuggets.append({"passage_id": pid, "nugget_text": ntext})

    if nuggets:
        return nuggets[:10]

    # Fallback: synthesise from reasoning bullets under a passage context.
    _META_RE = re.compile(
        r"^(let me|i need|i should|i'll|i have|i found|i can see|"
        r"my plan|looking at|reviewing|"
        r"good|great|perfect|interesting|now i|the search|processing|"
        r"here's what|not enough|partial|time to|so |the question|however|"
        r"based on|putting|my conclusion|after reviewing|the most common|"
        r"for |clearly states|"
        r"passage\s+(says|describes|mentions|states|also|shows|indicates|confirms)|"
        r"the\s+grep\s+search|according to\s+passage)",
        re.IGNORECASE,
    )
    _ENTITY_META_RE = re.compile(
        r"^[A-Z][\w\s:'-]+:\s*(?:Referenced|Mentioned|Noted|Listed|Described)\b",
        re.IGNORECASE,
    )
    _BARE_QUOTE_RE = re.compile(r'^"[^"]{3,50}"$')
    _TRAILING_FRAG_RE = re.compile(r"\s*\((?:from|confirmed from|see|per)\b.*$")

    current_pid: str | None = None
    for msg in trace.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        reasoning = msg.get("reasoning", "")
        if not reasoning:
            continue
        for line in reasoning.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            pid_match = re.findall(r"([A-Z]?\d+__\d+)", stripped)
            if pid_match and re.match(
                r"^(?:from\s+)?(?:passage\s+)?[A-Z]?\d+__\d+",
                stripped,
                re.IGNORECASE,
            ):
                current_pid = pid_match[0]
                continue
            if current_pid and re.match(r"^[-*]\s+", stripped):
                claim = _strip_markdown(stripped.lstrip("-* "))
                claim = re.sub(r"^\d+[.)]\s*", "", claim)
                claim = claim.replace(";", ",")
                claim = _TRAILING_FRAG_RE.sub("", claim).rstrip(" ,")
                if (
                    _META_RE.match(claim)
                    or _ENTITY_META_RE.match(claim)
                    or _BARE_QUOTE_RE.match(claim)
                ):
                    continue
                if len(claim) < 15 or claim in seen_texts:
                    continue
                seen_texts.add(claim)
                nuggets.append({"passage_id": current_pid, "nugget_text": claim})
                if len(nuggets) >= 10:
                    break
            elif pid_match:
                claim = re.sub(
                    r"\(?(?:passage\s+)?[A-Z]?\d+__\d+(?:\s*\([^)]*\))?\)?[:\s]*",
                    "",
                    stripped,
                )
                claim = re.sub(r"^from\s+", "", claim, flags=re.IGNORECASE)
                claim = re.sub(r"^\d+[.)]\s*", "", claim)
                claim = re.sub(r"^[-*]\s+", "", claim)
                claim = _strip_markdown(claim).rstrip(":").strip().replace(";", ",")
                claim = _TRAILING_FRAG_RE.sub("", claim).rstrip(" ,")
                if (
                    _META_RE.match(claim)
                    or _ENTITY_META_RE.match(claim)
                    or _BARE_QUOTE_RE.match(claim)
                ):
                    continue
                if len(claim) < 15 or claim in seen_texts:
                    continue
                seen_texts.add(claim)
                nuggets.append({"passage_id": pid_match[0], "nugget_text": claim})
                if len(nuggets) >= 10:
                    break
            else:
                current_pid = None
        if len(nuggets) >= 10:
            break

    return nuggets[:10]
