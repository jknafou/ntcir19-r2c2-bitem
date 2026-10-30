"""Post-processing pipeline: trace → PR/AC submission files.

Deterministic code-only transformation (no LLM calls).

Steps:
  1. Extract cited passages from trace (thinking + tool results)
  2. Rank by citation frequency (ties broken by first appearance)
  3. Format PR entries (semicolon-separated)
  4. Format AC result (XML-tagged nuggets)
  5. Validate PR↔AC consistency
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from r2c2_agent.agent.interfaces.types import (
    ACResult,
    CitedPassage,
    Nugget,
    PREntry,
    QuestionTrace,
)
from r2c2_agent.agent.prompt import NUGGET_PATTERN

__all__ = [
    "extract_cited_passages",
    "format_ac_file",
    "format_ac_result",
    "format_pr_file",
    "rank_passages",
    "validate_submission",
]

log = logging.getLogger(__name__)

# Regex for passage_id references in free text (e.g. "passage ABC-123")
_PASSAGE_ID_RE = re.compile(r"\b([A-Za-z0-9]+-[A-Za-z0-9._-]+)\b")


# Extract cited passages from a trace


def extract_cited_passages(trace: QuestionTrace) -> list[CitedPassage]:
    """Walk a trace and build a citation frequency map.

    Three citation sources:
      - NUGGET[id] tags in thinking/reasoning tokens
      - Tool calls to get_passage / search-like tools (tool results)
      - Passage IDs mentioned in thinking text

    Args:
        trace: Completed question trace with messages.

    Returns:
        List of CitedPassage objects with frequency and first-appearance data.
    """
    freq: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    passage_data: dict[str, dict[str, str]] = {}  # id -> {doc_id, text}

    for idx, msg in enumerate(trace.messages):
        role = msg.get("role", "")

        # NUGGET tags and free-text mentions in reasoning
        reasoning = msg.get("reasoning", "")
        if isinstance(reasoning, str) and reasoning:
            for pid, _ in NUGGET_PATTERN.findall(reasoning):
                pid = pid.strip()
                freq[pid] = freq.get(pid, 0) + 1
                if pid not in first_seen:
                    first_seen[pid] = idx

            for match in _PASSAGE_ID_RE.finditer(reasoning):
                pid = match.group(1)
                # Any hyphenated token matches the pattern, so count only
                # ids already seen in a tool result.
                if pid in passage_data:
                    freq[pid] = freq.get(pid, 0) + 1
                    if pid not in first_seen:
                        first_seen[pid] = idx

        # NUGGET tags can also appear in assistant content.
        if role == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                for pid, _ in NUGGET_PATTERN.findall(content):
                    pid = pid.strip()
                    freq[pid] = freq.get(pid, 0) + 1
                    if pid not in first_seen:
                        first_seen[pid] = idx

        # Tool results containing passage data
        if role == "tool":
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue

            _extract_passages_from_tool_result(
                data, idx, freq, first_seen, passage_data
            )

    # The MCP score rides on each CitedPassage through setattr, because
    # CitedPassage is frozen and its interface has no score field.
    result: list[CitedPassage] = []
    for pid, count in freq.items():
        pdata = passage_data.get(pid, {})
        cp = CitedPassage(
            passage_id=pid,
            citation_count=count,
            first_appearance=first_seen.get(pid, 0),
            doc_id=pdata.get("doc_id", ""),
            text=pdata.get("text", ""),
        )
        # rank_passages reads this score for the hybrid ranking.
        object.__setattr__(cp, "_mcp_score", pdata.get("score"))
        result.append(cp)

    return result


def _extract_passages_from_tool_result(
    data: Any,
    msg_idx: int,
    freq: dict[str, int],
    first_seen: dict[str, int],
    passage_data: dict[str, dict[str, str]],
) -> None:
    """Extract passage info from a parsed tool result JSON.

    Handles both list-of-passages and single-passage formats.
    """
    passages_to_process: list[dict[str, Any]] = []

    if isinstance(data, dict):
        if "passages" in data and isinstance(data["passages"], list):
            passages_to_process = data["passages"]
        elif "passage_id" in data:
            passages_to_process = [data]

    for p in passages_to_process:
        pid = p.get("passage_id", "")
        if not pid:
            continue

        freq[pid] = freq.get(pid, 0) + 1
        if pid not in first_seen:
            first_seen[pid] = msg_idx

        if pid not in passage_data:
            passage_data[pid] = {
                "doc_id": p.get("doc_id", ""),
                "text": p.get("text", ""),
                "score": p.get("score"),  # MCP cross-encoder score (float or None)
            }


# Rank passages


def rank_passages(
    cited: list[CitedPassage],
    max_passages: int = 20,
) -> list[CitedPassage]:
    """Rank passages with a citation-first hybrid.

    Ranking rule:
    1. Cited passages first (citation_count > 0), sorted by
       citation_count desc, then MCP score desc
    2. Uncited passages, sorted by MCP score desc
    3. Tiebreak: first_appearance ascending
    4. score=None passages rank last within their tier

    Args:
        cited: List of CitedPassage objects (with _mcp_score stashed).
        max_passages: Maximum passages to keep (PR limit = 20).

    Returns:
        Sorted and truncated list with top passages.
    """

    def _sort_key(p: CitedPassage) -> tuple:
        score = getattr(p, "_mcp_score", None)
        # -inf makes a missing score sort last within its tier.
        score_val = score if score is not None else float("-inf")
        is_cited = 1 if p.citation_count > 0 else 0
        return (-is_cited, -p.citation_count, -score_val, p.first_appearance)

    sorted_passages = sorted(cited, key=_sort_key)
    return sorted_passages[:max_passages]


# Format PR entries


def format_pr_entries(
    question_id: str,
    ranked_passages: list[CitedPassage],
) -> list[PREntry]:
    """Convert ranked passages to PR submission entries.

    Args:
        question_id: The question ID (e.g. "Q042").
        ranked_passages: Passages sorted by rank.

    Returns:
        List of PREntry objects with rank starting at 1.
    """
    entries: list[PREntry] = []
    for rank_idx, passage in enumerate(ranked_passages, start=1):
        text = _sanitize_passage_text(passage.text)
        entries.append(
            PREntry(
                question_id=question_id,
                rank=rank_idx,
                doc_id=passage.doc_id,
                passage_text=text,
            )
        )
    return entries


def _sanitize_passage_text(text: str) -> str:
    """Collapse newlines and excess whitespace in passage text.

    Args:
        text: Raw passage text.

    Returns:
        Cleaned text suitable for PR file.
    """
    cleaned = re.sub(r"[\n\r\t]+", " ", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


# Format AC result


def format_ac_result(
    trace: QuestionTrace,
    pr_run_name: str,
    passage_rank_map: dict[str, int],
    max_nuggets: int = 10,
) -> ACResult:
    """Build an AC result from a trace and its PR passage rankings.

    Nuggets whose passage didn't make the PR top 20 are dropped.

    Args:
        trace: Completed question trace.
        pr_run_name: Name of the PR run (e.g. "BITEM-PG-1").
        passage_rank_map: Maps passage_id → PR rank (1-based).
        max_nuggets: Maximum nuggets to include.

    Returns:
        ACResult with numbered nuggets referencing PR passages.
    """
    numbered_nuggets: list[Nugget] = []
    seen_texts: set[str] = set()

    for nugget in trace.nuggets:
        # A nugget must point at a ranked PR passage.
        if nugget.passage_id not in passage_rank_map:
            continue

        norm_text = nugget.text.strip().lower()
        if norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)

        pr_rank = passage_rank_map[nugget.passage_id]
        numbered_nuggets.append(
            Nugget(
                passage_id=nugget.passage_id,
                text=nugget.text.strip(),
                num=len(numbered_nuggets) + 1,
                pr_run_name=pr_run_name,
                passage_rank=pr_rank,
            )
        )

        if len(numbered_nuggets) >= max_nuggets:
            break

    return ACResult(
        answer=trace.draft_answer,
        confidence=trace.final_confidence,
        nuggets=numbered_nuggets,
        passage_rank_map=passage_rank_map,
    )


# File formatting


def format_pr_file(
    all_entries: dict[str, list[PREntry]],
) -> str:
    """Format all PR entries into the submission file content.

    Format: QuestionID;PassageRank;DocID;PassageText
    Sorted by question ID, then by rank.

    Args:
        all_entries: Maps question_id → list of PREntry.

    Returns:
        Complete PR file content as string.
    """
    lines: list[str] = []
    for qid in sorted(all_entries.keys()):
        for entry in sorted(all_entries[qid], key=lambda e: e.rank):
            lines.append(
                f"{entry.question_id};{entry.rank};{entry.doc_id};{entry.passage_text}"
            )
    return "\n".join(lines) + "\n" if lines else ""


def format_ac_file(
    all_results: dict[str, ACResult],
) -> str:
    """Format all AC results into the submission file content.

    Format per question:
      <qid>
      answer;confidence
      nuggetNum;prRunName;passageRank;nuggetText
      </qid>

    Args:
        all_results: Maps question_id → ACResult.

    Returns:
        Complete AC file content as string.
    """
    blocks: list[str] = []
    for qid in sorted(all_results.keys()):
        result = all_results[qid]
        lines: list[str] = [f"<{qid}>"]
        lines.append(f"{result.answer};{result.confidence}")
        for nugget in result.nuggets:
            lines.append(
                f"{nugget.num};{nugget.pr_run_name};{nugget.passage_rank};{nugget.text}"
            )
        lines.append(f"</{qid}>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n" if blocks else ""


# Validate PR↔AC consistency


def validate_submission(
    pr_entries: dict[str, list[PREntry]],
    ac_results: dict[str, ACResult],
    pr_run_name: str,
) -> list[str]:
    """Validate PR and AC submissions for consistency.

    Checks:
      - Every AC nugget's PassageKey references valid PR passage
      - ≤20 PR passages per question
      - Confidence ∈ [0, 100] integer
      - ≤10 nuggets per question
      - No empty answers

    Args:
        pr_entries: Maps question_id → list of PREntry.
        ac_results: Maps question_id → ACResult.
        pr_run_name: Expected PR run name.

    Returns:
        List of error strings. Empty list = valid.
    """
    errors: list[str] = []

    for qid, entries in pr_entries.items():
        if len(entries) > 20:
            errors.append(f"{qid}: {len(entries)} PR passages (max 20)")

    for qid, result in ac_results.items():
        if not result.answer.strip():
            errors.append(f"{qid}: empty answer")

        if not (0 <= result.confidence <= 100):
            errors.append(f"{qid}: confidence {result.confidence} out of range [0,100]")

        if len(result.nuggets) > 10:
            errors.append(f"{qid}: {len(result.nuggets)} nuggets (max 10)")

        # Each nugget must name this PR run and a rank that exists in it.
        pr_ranks = {e.rank for e in pr_entries.get(qid, [])}
        for nugget in result.nuggets:
            if nugget.pr_run_name != pr_run_name:
                errors.append(
                    f"{qid}: nugget {nugget.num} references run "
                    f"'{nugget.pr_run_name}' (expected '{pr_run_name}')"
                )
            if nugget.passage_rank not in pr_ranks:
                errors.append(
                    f"{qid}: nugget {nugget.num} references PR rank "
                    f"{nugget.passage_rank} which doesn't exist"
                )

    return errors
