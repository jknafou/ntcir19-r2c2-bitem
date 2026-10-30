#!/usr/bin/env python3
"""Lightweight HTTP worker that wraps the agentic loop for distributed orchestration.

Exposes endpoints for processing bundles (Tier 2 traces) and questions (RL rollouts).
Runs on a GPU node alongside vLLM + ES + MCP, all on localhost.

Endpoints:
    GET  /health          — worker status (ready/busy/error)
    GET  /info            — GPU type, model, available tools
    POST /calibrate       — run GPU calibration against local vLLM
    POST /process-bundle  — compose question from bundle + run agentic loop
    POST /process-question — run agentic loop on a pre-composed question
"""

from __future__ import annotations

# Re-use compose_question and helpers from tier2 script (imported as module)
import importlib.util
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiohttp import web
from openai import AsyncOpenAI

from r2c2_agent.agent.config import LoopConfig
from r2c2_agent.agent.interfaces.types import Question, QuestionTrace
from r2c2_agent.agent.llm.client import OpenAILLMClient
from r2c2_agent.agent.llm.verifier_client import VerifierClient
from r2c2_agent.agent.loop import answer_question
from r2c2_agent.agent.loop_hooks import DraftContext, DraftDecision
from r2c2_agent.agent.mcp.client import MCPHTTPClient

__all__: list[str] = []

log = logging.getLogger(__name__)

# Schemas for the server-side handlers.
#
# vLLM's tool-call parser emits structured `tool_calls` only for functions it
# has an OpenAI-format schema for. The server-side handlers live outside MCP,
# so `list_tools()` does not surface them and their schemas are appended here.
# They mirror the argument contracts of the handlers in
# src/r2c2_agent/orchestrator/pipeline_handlers.py.

_HANDLER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_evidence",
            "description": (
                "Commit a stance for a passage and optionally up to 3 "
                "nuggets. Call this after reading a passage to record "
                "whether it supports, counter-evidences, is irrelevant to, "
                "or provides context for the question, and to attach short "
                "quoted spans as nuggets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": (
                            "Passage id exactly as returned by a retrieval tool."
                        ),
                    },
                    "stance": {
                        "type": "string",
                        "enum": [
                            "supporting",
                            "counter_evidence",
                            "irrelevant",
                            "context",
                        ],
                        "description": "Stance of the passage w.r.t. the question.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "≤400 chars reasoning for the stance decision.",
                    },
                    "nuggets": {
                        "type": "array",
                        "maxItems": 3,
                        "description": (
                            "Up to 3 nugget objects. Each carries a concise "
                            "claim, polarity (affirming|infirming), and a "
                            "quoted_span taken verbatim from the passage."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {
                                    "type": "string",
                                    "description": (
                                        "Concise factual claim in your own words."
                                    ),
                                },
                                "polarity": {
                                    "type": "string",
                                    "enum": ["affirming", "infirming"],
                                    "description": (
                                        "affirming = the claim is a positive "
                                        "assertion; infirming = the claim rules "
                                        "something out."
                                    ),
                                },
                                "quoted_span": {
                                    "type": "string",
                                    "description": (
                                        "Verbatim substring from the passage."
                                    ),
                                },
                            },
                            "required": ["claim", "polarity", "quoted_span"],
                        },
                    },
                    "supersede_by": {
                        "type": "string",
                        "description": "Nugget id to supersede (optional).",
                    },
                },
                "required": ["pid", "stance", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Emit the final answer and terminate the loop. Call this "
                "exactly once per question. If no passage supports an "
                "answer, call with refusal=true instead of fabricating one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Final answer text. Ignored when refusal=true.",
                    },
                    "refusal": {
                        "type": "boolean",
                        "description": (
                            "True iff the retrieved corpus does not "
                            "support an answer. Canonical refusal text "
                            "will be substituted server-side."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief reasoning for the answer or refusal.",
                    },
                    "citation_pids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of pids that ground the answer. "
                            "Omit to let the server derive citations from "
                            "the accepted supporting nuggets."
                        ),
                    },
                },
                "required": ["rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_nuggets",
            "description": (
                "List nuggets recorded in the current session with their "
                "reference passage. Read-only; does not mutate the DB. "
                "Returns up to 20 nuggets ranked by "
                "(status_priority, -created_at_turn). Use to re-ground "
                "mid-session without re-reading call history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status_in": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "verified_supporting",
                                "verified_infirming",
                                "bogus",
                                "reflected",
                                "refusal",
                                "unclassified",
                            ],
                        },
                        "description": (
                            "Filter by status; default includes all non-refusal."
                        ),
                    },
                    "passage_id": {
                        "type": "string",
                        "description": "Filter to nuggets on a specific pid.",
                    },
                    "polarity": {
                        "type": "string",
                        "enum": ["affirming", "infirming"],
                        "description": "Filter by nugget polarity.",
                    },
                    "created_this_session": {
                        "type": "boolean",
                        "description": (
                            "Phase-2 only: if true, return only nuggets "
                            "created since the current Phase-2 session "
                            "began. No-op in Phase 1."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_nugget_bogus",
            "description": (
                "Mark a previously-recorded nugget as bogus. The nugget "
                "stays in the DB for the audit trail but is excluded from "
                "the PR/AC submission and from confidence calibration. Use "
                "when later evidence contradicts or invalidates a claim "
                "you earlier committed, AND you do NOT have a tighter "
                "replacement (for replacement, use record_evidence with "
                "supersede_by instead)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nugget_id": {
                        "type": "string",
                        "description": (
                            "Nugget id (e.g. n_<8hex>); get from list_nuggets."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why the nugget is now considered wrong (≤200 chars)."
                        ),
                    },
                },
                "required": ["nugget_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_record_stance",
            "description": (
                "Bulk-demote up to 20 passages to stance='irrelevant' in "
                "one call. PHASE 2 ONLY — unavailable in Phase 1 (use "
                "record_evidence per-pid instead). Attached nuggets on "
                "demoted pids cascade to status='bogus' automatically. "
                "Idempotent: a pid already 'rejected' is reported in "
                "'skipped' but does not abort the batch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1..20 passage ids from the Phase 2 DB handoff."
                        ),
                    },
                    "stance": {
                        "type": "string",
                        "enum": ["irrelevant"],
                        "description": (
                            "Only 'irrelevant' is supported for batch ops."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why these pids are being demoted (≤200 chars)."
                        ),
                    },
                },
                "required": ["pids", "stance", "reason"],
            },
        },
    },
]

# Module state, set during startup.

_mcp_client: MCPHTTPClient | None = None
_tools: list[dict[str, Any]] = []
_model_name: str = ""
_llm_url: str = ""
_mcp_url: str = ""
_context_budget: int = 124518  # 95% of 131072 (Qwen3.5 128K context)
_token_budget: int = 124518  # per-trace soft budget — overridden by TOKEN_BUDGET env
_active_requests: int = 0
_total_processed: int = 0
_errors: int = 0

# CP2 verifier state. ``_verifier_client`` is ``None`` when the verifier is
# unavailable (empty ``VERIFIER_URL``, init failure, or soft-fail). When
# ``None``, the worker runs CP0-only (no reflection injection).
_verifier_client: VerifierClient | None = None
_verifier_url: str = ""

# CP2 failure-mode classification and static self-correction thoughts.
#
# A heuristic classifier detects why the draft is wrong, then a predefined
# static thought is injected. The injected text is hardcoded, never produced
# at runtime, so it cannot leak the gold answer.

_STATIC_THOUGHTS: dict[str, str] = {
    "NO_RETRIEVAL": (
        "Wait, I answered without properly searching the knowledge base. "
        "I need to formulate a search query and retrieve relevant passages "
        "before I can answer this question."
    ),
    "BAD_QUERY": (
        "Wait, I searched but the retrieved passages don't seem to contain "
        "the information I need. I should try reformulating my search query "
        "with different terms — my search terms might not match how the "
        "information is stored in the index."
    ),
    "WRONG_EXTRACTION": (
        "Wait, the passages I retrieved do contain relevant information, "
        "but I may have extracted the wrong entity or value. Let me re-read "
        "the passages more carefully, paying close attention to which entity "
        "actually matches what the question is asking for."
    ),
    "BAD_REASONING": (
        "Wait, I think I have the right data but my reasoning or computation "
        "might be wrong. Let me go through the values step by step and redo "
        "the comparison or calculation carefully."
    ),
    "INCOMPLETE_AGG": (
        "Wait, I may not have found all the items yet. The question asks for "
        "a complete list or count, and I might be missing some. Let me search "
        "with broader or different query terms to find any remaining items."
    ),
    "CONDITION_IGNORED": (
        "Wait, the question has a specific condition or constraint that I may "
        "have overlooked. Let me re-read the question carefully and check "
        "whether my answer actually satisfies all the stated requirements."
    ),
    "GENERIC": (
        "Wait, I should reconsider my answer. Let me re-examine the retrieved "
        "passages more carefully and check whether my reasoning actually "
        "follows from the evidence."
    ),
}

# Second-tier templates: used on 2nd injection (injections_so_far == 1).
# Each has a {keyword} slot filled with a dynamic, contamination-checked value.
_TIER2_TEMPLATES: dict[str, str] = {
    "NO_RETRIEVAL": (
        "Wait, I still haven't searched the knowledge base properly. "
        "I need to search specifically for '{keyword}' to find relevant "
        "passages before answering."
    ),
    "BAD_QUERY": (
        "Wait, my previous searches didn't find the right information. "
        "I should try searching specifically for '{keyword}' — that's a "
        "key term from the question I haven't found in any passage yet."
    ),
    "WRONG_EXTRACTION": (
        "Wait, I found relevant passages but extracted the wrong information. "
        "I need to look specifically at the '{keyword}' field or relationship "
        "— that's what the question is actually asking about."
    ),
    "BAD_REASONING": (
        "Wait, I need to redo my calculation. The question asks about "
        "'{keyword}' — let me re-extract the numeric values from the passages "
        "and compute this step by step."
    ),
    "INCOMPLETE_AGG": (
        "Wait, my list is still incomplete. I should search specifically for "
        "'{keyword}' to find the missing items."
    ),
    "CONDITION_IGNORED": (
        "Wait, I keep ignoring the condition '{keyword}' in the question. "
        "I must filter my answer to only include results that satisfy this "
        "constraint."
    ),
    "GENERIC": (
        "Wait, I should try a different approach. Let me search for "
        "'{keyword}' and reconsider my answer based on what I find."
    ),
}


# Same-mode escalation thoughts (tiers 3-5 for non-multi-hop traces).
# When the same failure mode fires 3+ times, these provide progressively
# more directive hints specific to that mode, instead of cycling through
# unrelated failure modes.  Key: failure_mode -> [esc1, esc2, esc3].
_SAME_MODE_ESCALATION: dict[str, list[str]] = {
    "NO_RETRIEVAL": [
        (
            "Wait, I keep trying to answer from memory instead of searching. "
            "I must stop generating an answer and issue a search query RIGHT "
            "NOW — even a simple one using the main entity from the question."
        ),
        (
            "Wait, I have attempted multiple answers without retrieving any "
            "evidence. Let me break the question down: what is the core entity "
            "being asked about? I will search for that entity by name."
        ),
        (
            "Wait, I absolutely cannot answer this without evidence. Let me "
            "take the most distinctive proper noun from the question and use "
            "it as my search query — nothing else will work."
        ),
    ],
    "BAD_QUERY": [
        (
            "Wait, I keep searching with similar terms and not finding the "
            "right information. Let me try a completely different approach — "
            "use very different search terms or try searching by entity "
            "instead of by keyword."
        ),
        (
            "Wait, my search strategy is fundamentally not working. Let me "
            "think about what specific document or entity would contain this "
            "information and search for THAT directly."
        ),
        (
            "Wait, I have exhausted my usual search strategies. Let me try "
            "the most specific query I can construct from the question "
            "itself, using exact names and phrases verbatim."
        ),
    ],
    "WRONG_EXTRACTION": [
        (
            "Wait, I keep extracting the wrong value from the passages. Let "
            "me slow down and re-read each passage sentence by sentence, "
            "matching every entity to the EXACT constraint in the question."
        ),
        (
            "Wait, I am consistently picking the wrong entity from the "
            "evidence. The passages likely contain multiple similar items — "
            "I need to identify which one satisfies ALL conditions in the "
            "question, not just some of them."
        ),
        (
            "Wait, I have misread the passages multiple times now. Let me "
            "list every candidate entity I see in the passages and "
            "explicitly check each one against the question's criteria "
            "before choosing."
        ),
    ],
    "BAD_REASONING": [
        (
            "Wait, I keep getting the computation wrong. Let me write out "
            "every number from the passages explicitly, then perform the "
            "arithmetic one step at a time without skipping anything."
        ),
        (
            "Wait, my calculation keeps producing the wrong result. Let me "
            "re-extract the raw values from the passages — I may be using a "
            "wrong number. Then I will redo the operation from scratch."
        ),
        (
            "Wait, I have failed this computation multiple times. Let me "
            "verify: am I using the right operation (sum vs. average vs. "
            "difference)? Am I comparing in the right direction (greater "
            "vs. lesser, earlier vs. later)? Let me re-read the question "
            "to confirm what is actually being asked."
        ),
    ],
    "INCOMPLETE_AGG": [
        (
            "Wait, my list is still incomplete. Let me try broader search "
            "terms — perhaps a category name or a more general description "
            "instead of specific item names."
        ),
        (
            "Wait, I am still missing items. Let me try a completely "
            "different angle: search for the parent category, collection, "
            "or group that contains all the items, rather than searching "
            "for individual items one by one."
        ),
        (
            "Wait, I have searched multiple times and my list is still "
            "short. Let me count exactly what I have, check each passage "
            "for items I may have overlooked, and try one final search "
            "with the broadest possible terms."
        ),
    ],
    "CONDITION_IGNORED": [
        (
            "Wait, I keep ignoring a key constraint in the question. Let me "
            "re-read the question and explicitly write out every condition "
            "or filter it specifies before evaluating my answer."
        ),
        (
            "Wait, my answer still violates a condition in the question. "
            "Let me check my candidate answer against EACH condition: does "
            "it satisfy the time constraint? The attribute constraint? The "
            "comparison constraint? If not, I must find a different answer."
        ),
        (
            "Wait, I have repeatedly failed to satisfy the question's "
            "constraints. Let me re-search with the constraint as part of "
            "my query — include the filter words (dates, thresholds, "
            "attributes) directly in my search terms."
        ),
    ],
    "GENERIC": [
        (
            "Wait, my approach is not working. Let me reconsider the "
            "question from scratch — what TYPE of information is being "
            "asked for, and where would that information most likely be "
            "found in the knowledge base?"
        ),
        (
            "Wait, I need to try a fundamentally different strategy. "
            "Instead of refining my current approach, let me start over: "
            "new search terms, re-read the question, and consider whether "
            "I am even looking for the right kind of answer."
        ),
        (
            "Wait, I have tried multiple approaches without success. Let "
            "me focus on the single most important entity or fact in the "
            "question and search for exactly that, then build my answer "
            "strictly from what I find."
        ),
    ],
}

# Multi-hop escalation templates (levels 3-5, injections_so_far >= 2).
# Multi-hop chains: subject → (hop1) → intermediate → (hop2) → answer.
# The intermediate entity is safe to name: it equals the gold answer in
# 0 of 910 chains.
_MULTIHOP_ESCALATION: list[dict[str, Any]] = [
    {
        # Level 3 (inj=2): chain awareness — tell model this is multi-step
        "template": (
            "Wait, this is a multi-step question. I need to first find "
            "'{keyword_subject}' to identify an intermediate entity, then "
            "search for THAT entity to find the final answer. I may have "
            "stopped at the first hop."
        ),
        "keywords": ["subject"],  # uses nugget[0].entity
    },
    {
        # Level 4 (inj=3): name the intermediate entity explicitly
        "template": (
            "Wait, I found that '{keyword_subject}' is linked to "
            "'{keyword_intermediate}'. Now I need to search specifically "
            "for '{keyword_intermediate}' and look at its "
            "'{keyword_hop2_attr}' to find the final answer."
        ),
        "keywords": ["subject", "intermediate", "hop2_attr"],
    },
    {
        # Level 5 (inj=4): maximum help — spell out the full chain
        "template": (
            "Wait, let me trace the reasoning chain step by step. The "
            "question asks about '{keyword_subject}'. From the evidence, "
            "'{keyword_subject}' leads to '{keyword_intermediate}' via "
            "the '{keyword_hop1_attr}' relationship. Now I need to find "
            "what '{keyword_intermediate}' has for '{keyword_hop2_attr}'. "
            "That is the answer — not '{keyword_intermediate}' itself."
        ),
        "keywords": ["subject", "intermediate", "hop1_attr", "hop2_attr"],
    },
]


def _extract_multihop_keywords(
    nuggets: list[dict[str, Any]],
    expected_answer: str,
) -> dict[str, str] | None:
    """Extract multi-hop chain keywords from nuggets.

    Returns dict with keys: subject, intermediate, hop1_attr, hop2_attr.
    Returns None if chain structure is not 2-nugget or contamination detected.
    """
    if len(nuggets) < 2:
        return None

    subject = nuggets[0].get("entity", "")
    intermediate = nuggets[0].get("value", "")  # = nuggets[1].entity
    hop1_attr = nuggets[0].get("attribute", "").replace("_", " ")
    hop2_attr = nuggets[1].get("attribute", "").replace("_", " ")

    if not subject or not intermediate:
        return None

    # The intermediate must not be the gold answer.
    if _is_keyword_contaminated(intermediate, expected_answer):
        return None

    return {
        "subject": subject,
        "intermediate": intermediate,
        "hop1_attr": hop1_attr or "relationship",
        "hop2_attr": hop2_attr or "attribute",
    }


def _is_keyword_contaminated(keyword: str, expected_answer: str) -> bool:
    """Check if a keyword leaks the gold answer (bidirectional substring + token overlap)."""
    kw_lower = keyword.lower().strip()
    ans_lower = expected_answer.lower().strip()
    if not kw_lower or not ans_lower:
        return False
    # Bidirectional substring
    if kw_lower in ans_lower or ans_lower in kw_lower:
        return True
    # Token overlap: >50% of keyword tokens appear in answer
    kw_tokens = set(kw_lower.split())
    ans_tokens = set(ans_lower.split())
    if kw_tokens and len(kw_tokens & ans_tokens) / len(kw_tokens) > 0.5:
        return True
    return False


def _extract_tier2_keyword(
    failure_mode: str,
    ctx: DraftContext,
    nuggets: list[dict[str, Any]],
    question: str,
) -> str | None:
    """Extract a dynamic keyword for the second-tier hint (pure Python, no LLM)."""
    import re

    passage_text = " ".join(ctx.retrieved_passages).lower()

    if failure_mode == "NO_RETRIEVAL":
        # Use nugget entity (question subject, not the answer)
        if nuggets:
            entity = nuggets[0].get("entity", "")
            if entity and len(entity) > 2:
                return entity
        # Fall back: longest capitalized phrase from question
        caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
        multi = [c for c in caps if len(c.split()) >= 2]
        return multi[0] if multi else (caps[0] if caps else None)

    elif failure_mode == "BAD_QUERY":
        # Question entity absent from passages
        caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
        seen: set[str] = set()
        unique = []
        for c in caps:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        unused = [c for c in unique if c.lower() not in passage_text]
        if unused:
            return unused[0]
        # Nugget entity not in passages
        for n in nuggets:
            entity = n.get("entity", "")
            if entity and entity.lower() not in passage_text:
                return entity
        return None

    elif failure_mode == "WRONG_EXTRACTION":
        # Nugget attribute name (the relationship, not the value)
        if nuggets:
            attr = nuggets[0].get("attribute", "")
            if attr:
                return attr.replace("_", " ")
            # Entity in passages but not in draft
            draft_lower = (ctx.draft_answer or "").lower()
            for n in nuggets:
                entity = n.get("entity", "")
                if (
                    entity
                    and entity.lower() in passage_text
                    and entity.lower() not in draft_lower
                ):
                    return entity
        return None

    elif failure_mode == "BAD_REASONING":
        # Operation/comparison word from question
        ops = re.findall(
            r"\b(difference|average|sum|total|count|older|younger|"
            r"earlier|later|longer|shorter|higher|lower|more|fewer|"
            r"years between|how many years|how much)\b",
            question,
            re.IGNORECASE,
        )
        return ops[0] if ops else None

    elif failure_mode == "INCOMPLETE_AGG":
        # Nugget entity missing from draft
        draft_lower = (ctx.draft_answer or "").lower()
        for n in nuggets:
            entity = n.get("entity", "")
            if entity and entity.lower() not in draft_lower:
                return entity
        return None

    elif failure_mode == "CONDITION_IGNORED":
        # The condition clause itself (temporal, numeric)
        cond = re.findall(
            r"\b((?:before|after|since|until|between|during|prior to)"
            r"\s+[\w\s,]+?\d{4})\b",
            question,
            re.IGNORECASE,
        )
        if cond:
            return cond[0].strip()
        cond2 = re.findall(
            r"\b((?:at least|at most|more than|less than|greater than|fewer than)"
            r"\s+\d+)\b",
            question,
            re.IGNORECASE,
        )
        return cond2[0].strip() if cond2 else None

    elif failure_mode == "GENERIC":
        # Best effort: nugget attribute or longest question entity
        if nuggets:
            attr = nuggets[0].get("attribute", "")
            if attr:
                return attr.replace("_", " ")
        caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)
        return caps[0] if caps else None

    return None


def _classify_failure_mode(
    ctx: DraftContext,
    qtype: str,
    nuggets: list[dict[str, Any]],
) -> str:
    """Classify why the model's draft answer is wrong (heuristic-only).

    Uses DraftContext fields (question, draft_answer, retrieved_passages,
    injections_so_far) and bundle nuggets to determine the failure mode.
    No LLM call needed — pure Python heuristics.

    Returns one of: NO_RETRIEVAL, BAD_QUERY, WRONG_EXTRACTION,
    BAD_REASONING, INCOMPLETE_AGG, CONDITION_IGNORED, GENERIC.
    """
    import re

    # retrieved_passages contains recent tool-result strings (may be empty
    # if the model never searched or has no results yet).
    has_passages = bool(ctx.retrieved_passages)

    # Combine all passage text for evidence-in-passages check.
    all_passage_text = " ".join(ctx.retrieved_passages).lower()

    # Extract nugget key phrases (capitalized words > 3 chars) as proxy
    # for "gold evidence entities" (the actual gold answer is not here —
    # just surrounding entities from the evidence).
    nugget_phrases: list[str] = []
    for n in nuggets:
        text = n.get("text", "")
        for w in text.split():
            clean = re.sub(r"[^\w]", "", w)
            if len(clean) > 3 and clean[0].isupper():
                nugget_phrases.append(clean.lower())

    evidence_in_passages = (
        any(phrase in all_passage_text for phrase in nugget_phrases)
        if nugget_phrases
        else False
    )

    # Rule 1: NO_RETRIEVAL
    if not has_passages:
        return "NO_RETRIEVAL"
    if (
        qtype == "multi_hop"
        and len(ctx.retrieved_passages) <= 1
        and ctx.injections_so_far == 0
    ):
        return "NO_RETRIEVAL"

    # Rule 2: INCOMPLETE_AGG
    if qtype == "aggregation":
        draft = ctx.draft_answer or ""
        draft_items = len(re.split(r"[,;]\s*|\n\d+[.)]\s*", draft))
        nugget_count = len(nuggets)
        if nugget_count > 0 and draft_items < nugget_count:
            return "INCOMPLETE_AGG"

    # Rule 3: BAD_REASONING (PP/comparison with data present)
    if qtype in ("post_processing", "comparison") and evidence_in_passages:
        return "BAD_REASONING"

    # Rule 4: CONDITION_IGNORED
    condition_markers = re.findall(
        r"\b(before|after|since|until|between|during|at least|at most|"
        r"more than|less than|greater than|fewer than|"
        r"older than|younger than|prior to)\b",
        ctx.question,
        re.IGNORECASE,
    )
    if condition_markers:
        return "CONDITION_IGNORED"

    # Rule 5: WRONG_EXTRACTION (evidence present, wrong answer)
    if evidence_in_passages:
        return "WRONG_EXTRACTION"

    # Rule 6: BAD_QUERY (searched but evidence absent)
    if has_passages and not evidence_in_passages:
        return "BAD_QUERY"

    return "GENERIC"


def _make_on_draft(
    expected_answer: str,
    question_text: str,
    bundle: dict[str, Any] | None = None,
) -> Callable[[DraftContext], Awaitable[DraftDecision]] | None:
    """Build an on_draft callback: local gold-match + static thought injection.

    1. Compare draft to gold locally via ``check_answer_match`` (instant).
    2. If match → no-op.
    3. If mismatch → classify the failure mode (heuristic, no LLM call)
       → inject the corresponding static thought from ``_STATIC_THOUGHTS``.

    The injected text is hardcoded, so it cannot leak the gold answer.
    Classification is pure Python heuristics, with no LLM call.

    Args:
        expected_answer: The gold reference answer.
        question_text: The original user question.
        bundle: Full bundle dict (needed for the comparison matcher
            and for extracting nuggets for the classifier).

    Returns:
        An async callback, or ``None`` if no expected answer available.
    """
    if not expected_answer:
        return None

    tier2 = _load_tier2_module()
    qtype = (bundle or {}).get("question_type", "")
    ca = (bundle or {}).get("comparison_answer", {}) or {}
    ca_kind = ca.get("kind", "")
    raw_nuggets = (bundle or {}).get("nuggets", [])[:5]

    async def _on_draft(ctx: DraftContext) -> DraftDecision:
        """Local-match + failure-mode-classified static thought CP2 callback."""
        matches = tier2.check_answer_match(
            expected_answer, ctx.draft_answer, qtype, bundle=bundle
        )

        if matches:
            return DraftDecision(
                inject=False,
                reflection_text="",
                abort=False,
                verdict_info={
                    "verdict": "ok",
                    "confidence": 1.0,
                    "reason": "local_match: draft matches gold",
                    "decision": "no_op_local_match",
                },
            )

        # The draft does not match gold: classify why and pick the hint tier.
        failure_mode = _classify_failure_mode(ctx, qtype, raw_nuggets)
        hint_tier = 1
        keyword: str | None = None
        keyword_contaminated = False
        thought: str | None = None

        if ctx.injections_so_far == 0:
            # First injection → tier-1 static thought
            thought = _STATIC_THOUGHTS.get(failure_mode, _STATIC_THOUGHTS["GENERIC"])
        elif ctx.injections_so_far == 1:
            # Second injection → try tier-2 with dynamic keyword
            keyword = _extract_tier2_keyword(
                failure_mode,
                ctx,
                raw_nuggets,
                question_text,
            )
            if keyword and not _is_keyword_contaminated(keyword, expected_answer):
                template = _TIER2_TEMPLATES.get(
                    failure_mode, _TIER2_TEMPLATES["GENERIC"]
                )
                thought = template.format(keyword=keyword)
                hint_tier = 2
            else:
                keyword_contaminated = keyword is not None
                keyword = None
        elif qtype == "multi_hop" and ctx.injections_so_far >= 2:
            # Multi-hop tiers 3-5: escalating chain hints
            mh_keywords = _extract_multihop_keywords(raw_nuggets, expected_answer)
            esc_idx = ctx.injections_so_far - 2  # 0, 1, 2 for inj 2, 3, 4
            if mh_keywords and esc_idx < len(_MULTIHOP_ESCALATION):
                esc = _MULTIHOP_ESCALATION[esc_idx]
                try:
                    thought = esc["template"].format(
                        keyword_subject=mh_keywords["subject"],
                        keyword_intermediate=mh_keywords["intermediate"],
                        keyword_hop1_attr=mh_keywords["hop1_attr"],
                        keyword_hop2_attr=mh_keywords["hop2_attr"],
                    )
                    hint_tier = 3 + esc_idx
                    keyword = mh_keywords.get("intermediate")
                except KeyError:
                    thought = None  # template mismatch, fall through
        elif ctx.injections_so_far >= 2:
            # Non-multi-hop tiers 3-5: progressively more directive hints
            # for the same failure mode, rather than cycling through
            # unrelated modes.
            esc_idx = ctx.injections_so_far - 2  # 0, 1, 2 for inj 2, 3, 4
            escalations = _SAME_MODE_ESCALATION.get(
                failure_mode, _SAME_MODE_ESCALATION["GENERIC"]
            )
            if esc_idx < len(escalations):
                thought = escalations[esc_idx]
                hint_tier = 3 + esc_idx
            # else: thought stays None, falls through to final fallback

        # Final fallback: GENERIC static thought (better than nothing).
        if thought is None:
            thought = _STATIC_THOUGHTS["GENERIC"]

        log.info(
            "CP2 inject: fm=%s tier=%d qtype=%s inj=%d kw=%s",
            failure_mode,
            hint_tier,
            qtype,
            ctx.injections_so_far + 1,
            repr(keyword) if keyword else "none",
        )

        return DraftDecision(
            inject=True,
            reflection_text=thought,
            abort=False,
            verdict_info={
                "verdict": "mismatch",
                "confidence": 1.0,
                "reason": (
                    f"local_mismatch: draft != gold ({qtype}/{ca_kind or 'default'})"
                ),
                "decision": "inject_reflection",
                "hint_source": (f"tier{hint_tier}_{failure_mode}"),
                "failure_mode": failure_mode,
                "hint_tier": hint_tier,
                "keyword": keyword,
                "keyword_contaminated": keyword_contaminated,
                "injection_index": ctx.injections_so_far + 1,
            },
        )

    return _on_draft


# generate_tier2_traces.py is a script, loaded by path to reuse compose_question.
_tier2_mod: Any = None


def _load_tier2_module() -> Any:
    """Import generate_tier2_traces.py as a module to reuse compose_question."""
    global _tier2_mod
    if _tier2_mod is not None:
        return _tier2_mod
    project = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "generate_tier2_traces",
        project / "scripts" / "synth" / "generate_tier2_traces.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot find generate_tier2_traces.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _tier2_mod = mod
    return mod


# Handlers


async def health(request: web.Request) -> web.Response:
    """Return worker health status."""
    status = "busy" if _active_requests > 0 else "ready"
    return web.json_response(
        {
            "status": status,
            "active_requests": _active_requests,
            "total_processed": _total_processed,
            "errors": _errors,
        }
    )


async def info(request: web.Request) -> web.Response:
    """Return worker info: model, GPU, tools."""
    gpu = os.environ.get("GPU_INFO", "unknown")
    return web.json_response(
        {
            "model": _model_name,
            "gpu": gpu,
            "tools": [t.get("function", t).get("name", "?") for t in _tools],
            "context_budget": _context_budget,
            "llm_url": _llm_url,
            "mcp_url": _mcp_url,
        }
    )


# Hot-reload admin endpoint.
#
# Gated on ENABLE_HOT_RELOAD=1. Never registered in production SLURM launch.sh.
# Owned state (``_mcp_client``, ``_tools``, ``_model_name``, ``_llm_url``,
# vLLM) is deliberately untouched — ``on_startup`` constructs those once and
# they must survive reload. Only pipeline-logic modules (accessed via lazy
# import inside ``answer_question_pipeline`` and friends) are popped; the
# next request then re-imports with the freshly rsynced code.

HOT_RELOAD_MODULES: list[str] = [
    "pipeline_handlers",
    "r2c2_agent.agent.loop",
    "r2c2_agent.agent.loop_phase2",
    "r2c2_agent.agent.verifier",
    "r2c2_agent.agent.postprocess",
    "r2c2_agent.agent.nli_verifier",
    "nugget_reconciler",
    "r2c2_agent.agent.question_classifier",
    "r2c2_agent.agent.confidence_calibrator",
    "r2c2_agent.agent.prompt",
]


async def admin_reload(request: web.Request) -> web.Response:
    """Pop hot-reloadable pipeline modules from ``sys.modules``.

    The next inbound ``/process-question`` request re-imports them from
    disk (lazy imports inside ``answer_question_pipeline`` and handlers),
    picking up whatever code was just rsynced over.

    Does not touch ``_mcp_client``, ``_tools``, ``_model_name``, vLLM, or
    any other ``on_startup``-owned state — those survive reload.

    Gated at route-registration time on ``ENABLE_HOT_RELOAD=1``; this
    handler is not reachable in production.

    Returns:
        JSON ``{"ok": true, "reloaded": [<popped-module-names>]}``.
    """
    popped: list[str] = []
    for mod in HOT_RELOAD_MODULES:
        if sys.modules.pop(mod, None) is not None:
            popped.append(mod)
    log.info("admin.reload popped=%d modules=%s", len(popped), popped)
    return web.json_response({"ok": True, "reloaded": popped})


async def admin_inject_state(request: web.Request) -> web.Response:
    """Crafted-state live testing endpoint.

    Body JSON:
        state          — serialized ``PipelineState`` (pydantic dump).
        entry_point    — one of ``force_deliver``, ``handle_submit_answer``,
                         ``phase2_synthesis``.
        entry_args     — kwargs for the entry. Required keys vary:
            force_deliver        → ``reason`` (str), optional
                                   ``preferred_answer`` (str|None)
            handle_submit_answer → ``args`` dict (see ``handle_submit_answer``)
            phase2_synthesis     → ``parent_reason`` (str), default
                                   ``token_budget_watermark``.
        bind_finalizer_clients — default True; wires the worker's real LLM +
                                 MCP clients so the ladder can call them.

    Behaviour:
        1. Deserialize ``state`` into a ``PipelineState`` via
           ``PipelineState.model_validate``.
        2. Reset the finalizer registry to the worker's real clients
           (reusing ``pipeline_handlers.set_finalizer_clients``); session
           messages are seeded with a minimal (system, user) pair as a
           prompt-frame anchor for the finalizer ladder.
        3. Dispatch to the selected entry point.
        4. Return the envelope + post-run DB + telemetry.

    On any exception returns ``{"error": "...", "traceback": "..."}``
    with HTTP 500, so a 5xx always carries a body.

    Gated at route-registration time on ``ENABLE_HOT_RELOAD=1``: this
    endpoint is dev-only.
    """
    import traceback as _tb

    # Best-effort JSON parse — malformed bodies get a structured error
    # instead of a stack trace leak.
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response(
            {"error": f"bad_json: {type(exc).__name__}: {exc}"}, status=400
        )
    if not isinstance(body, dict):
        return web.json_response({"error": "body_must_be_object"}, status=400)

    state_dict = body.get("state")
    entry_point = body.get("entry_point")
    entry_args = body.get("entry_args") or {}
    bind_clients = bool(body.get("bind_finalizer_clients", True))

    if not isinstance(state_dict, dict):
        return web.json_response(
            {"error": "missing_or_invalid_state (expected object)"},
            status=400,
        )
    if entry_point not in {
        "force_deliver",
        "handle_submit_answer",
        "phase2_synthesis",
    }:
        return web.json_response(
            {
                "error": (
                    "invalid_entry_point (expected force_deliver | "
                    "handle_submit_answer | phase2_synthesis)"
                ),
            },
            status=400,
        )
    if not isinstance(entry_args, dict):
        return web.json_response({"error": "entry_args_must_be_object"}, status=400)

    # Import lazily so admin_reload can hot-swap pipeline_handlers between
    # inject_state calls without this endpoint caching a stale reference.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        import r2c2_agent.orchestrator.pipeline_handlers as _ph
        from r2c2_agent.orchestrator.pipeline_handlers import PipelineState
    except Exception as exc:  # noqa: BLE001
        return web.json_response(
            {
                "error": f"pipeline_handlers_import_failed: {exc}",
                "traceback": _tb.format_exc(),
            },
            status=500,
        )

    # Deserialize the state.
    try:
        db = PipelineState.model_validate(state_dict)
    except Exception as exc:  # noqa: BLE001
        return web.json_response(
            {
                "error": f"state_validation_failed: {type(exc).__name__}: {exc}",
                "traceback": _tb.format_exc(),
            },
            status=400,
        )

    # Wire the finalizer clients.
    # force_deliver's ladder reads the finalizer registry via contextvars,
    # so binding here (on the request's asyncio task) is sufficient — no
    # other request can see our clients.
    llm_client_bound: Any = None
    if bind_clients:
        assert _mcp_client is not None, "MCP client not initialized"
        llm_client_bound = OpenAILLMClient(
            base_url=_llm_url,
            model=_model_name,
            temperature=float(body.get("temperature", 0.0)),
        )
        _ph.set_finalizer_clients(llm_client_bound, _mcp_client)
        # Minimal session frame as a prompt-frame anchor for the
        # finalizer ladder. Caller can override by passing
        # ``session_messages`` in the body (list of {role, content}).
        session_msgs = body.get("session_messages")
        if not isinstance(session_msgs, list):
            session_msgs = [
                {"role": "system", "content": "You are an R2C2 QA assistant."},
                {"role": "user", "content": db.question},
            ]
        _ph.set_finalizer_session_messages(session_msgs)
        # Stash the tool catalogue for the recursive submission cycle
        # (_reenter_agentic_loop) so it can pass tool_choice="required"
        # with the catalogue — OpenAI/vLLM rejects the constraint without
        # a tools list.
        try:
            _ph.set_finalizer_tools(_tools)
        except AttributeError:
            # Older pipeline_handlers without set_finalizer_tools — quiet
            # degrade so hot-reload during rolling deploys keeps working.
            pass
    else:
        _ph.clear_finalizer_clients()

    # Snapshot the pre-call counters for the telemetry diff.
    pre_finalizer = int(getattr(db, "finalizer_llm_calls_used", 0))
    pre_ass = int(getattr(db, "answer_search_support_calls_used", 0))

    t0 = time.monotonic()
    envelope: dict[str, Any] | None = None
    try:
        if entry_point == "force_deliver":
            reason = entry_args.get("reason", "admin_inject_state_probe")
            preferred = entry_args.get("preferred_answer")
            envelope = _ph.force_deliver(db, reason, preferred_answer=preferred)
        elif entry_point == "handle_submit_answer":
            # args can be nested under entry_args["args"] or flat.
            sa_args = entry_args.get("args")
            if not isinstance(sa_args, dict):
                sa_args = {
                    k: v
                    for k, v in entry_args.items()
                    if k in {"answer", "refusal", "rationale", "citation_pids"}
                }
            envelope = _ph.handle_submit_answer(db, sa_args)
        else:  # phase2_synthesis
            parent_reason = entry_args.get("parent_reason", "token_budget_watermark")
            from r2c2_agent.agent.loop_phase2 import run_phase2_loop

            question = Question(
                qid=db.question_id,
                text=db.question,
                question_type=db.question_type,
            )
            trace = QuestionTrace(
                question_id=db.question_id,
                question_text=db.question,
            )
            assert _mcp_client is not None, "MCP client not initialized"
            assert llm_client_bound is not None, (
                "phase2_synthesis requires bind_finalizer_clients=True"
            )
            await run_phase2_loop(
                db,
                question,
                llm_client=llm_client_bound,
                mcp_client=_mcp_client,
                tools=_tools,
                trace=trace,
                parent_reason=parent_reason,
            )
            # The loop force-delivers on any non-commit exit; echo the
            # memoised envelope.
            envelope = dict(getattr(db, "force_deliver_envelope", {}) or {})
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        log.exception("admin_inject_state: %s raised", entry_point)
        return web.json_response(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _tb.format_exc(),
                "telemetry": {
                    "elapsed_s": round(elapsed, 3),
                    "entry_point": entry_point,
                },
            },
            status=500,
        )
    finally:
        # Never leak the finalizer registry to a subsequent request
        # handler on the same task (aiohttp can reuse the task).
        try:
            _ph.clear_finalizer_clients()
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.monotonic() - t0
    try:
        db_after = db.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        db_after = {
            "dump_error": f"{type(exc).__name__}: {exc}",
        }

    telemetry = {
        "elapsed_s": round(elapsed, 3),
        "entry_point": entry_point,
        "finalizer_calls_used": int(
            getattr(db, "finalizer_llm_calls_used", 0) - pre_finalizer
        ),
        "ass_calls_used": int(
            getattr(db, "answer_search_support_calls_used", 0) - pre_ass
        ),
        "llm_url": _llm_url,
        "model": _model_name,
    }
    return web.json_response(
        {
            "envelope": envelope or {},
            "db_after": db_after,
            "telemetry": telemetry,
            "error": None,
        }
    )


# /admin/set_flags — runtime feature-flag toggle.
#
# Both whitelisted flags are read per-call via ``os.getenv``/``os.environ.get``
# by their respective modules (pipeline_handlers, nugget_reconciler), so
# mutating ``os.environ`` here takes effect on the next flag check — no
# worker relaunch needed.
#
# The whitelist is deliberately narrow, only the two flags below, so this
# is not an open-ended env-var API.
#
# Gated on ENABLE_HOT_RELOAD=1 (same gate as admin_reload /
# admin_inject_state), so it is dev-only and unreachable in production.

_ALLOWED_FLAG_KEYS: frozenset[str] = frozenset(
    {
        "ASS_RECURSIVE_SUBMIT",
        "R2C2_DECOUPLED_SCORING",
    }
)

# Canonical value set. Case-sensitive on purpose — keeps the admin surface
# tight. "True"/"False" are the Python repr() forms we tolerate; "TRUE" /
# "yes" / "on" are rejected so callers learn to stick to "0"/"1".
_ALLOWED_FLAG_VALUES: frozenset[str] = frozenset(
    {"0", "1", "true", "false", "True", "False"}
)


async def admin_set_flags(request: web.Request) -> web.Response:
    """Toggle feature-flag env vars on a live worker at runtime.

    Body JSON (all keys optional; at most the whitelisted keys allowed):
        ASS_RECURSIVE_SUBMIT   — "0" | "1" | "true" | "false" | "True" | "False"
        R2C2_DECOUPLED_SCORING — same allowed values.

    Behaviour:
        1. Parse body; reject malformed JSON / non-object with 400.
        2. Reject any key outside the whitelist with 400.
        3. Reject any value that is not a string in the allowed set with 400.
        4. Mutate ``os.environ`` for each provided (key, value). The flag
           readers call ``os.getenv`` per invocation, so the next call
           observes the new value.
        5. Log the old + new values at INFO level for forensic replay.
        6. Return ``{"updated": [...], "current": {...}}`` with the
           current environment value of every whitelisted flag (not just
           those set by this call) — callers verify state from ``current``.

    Returns:
        200 with JSON body on success; 400 for any validation error.

    Gated at route-registration time on ``ENABLE_HOT_RELOAD=1``: this
    endpoint is dev-only.
    """
    # Malformed body → 400 (never a 5xx crash).
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response(
            {"error": f"bad_json: {type(exc).__name__}: {exc}"}, status=400
        )
    if not isinstance(body, dict):
        return web.json_response({"error": "body_must_be_object"}, status=400)

    # Whitelist enforcement: any stray key aborts the whole request so
    # the caller learns about typos immediately (rather than silently
    # ignoring them and letting drift accumulate).
    extra_keys = [k for k in body.keys() if k not in _ALLOWED_FLAG_KEYS]
    if extra_keys:
        return web.json_response(
            {
                "error": "unknown_flag_keys",
                "unknown": sorted(extra_keys),
                "allowed": sorted(_ALLOWED_FLAG_KEYS),
            },
            status=400,
        )

    # Type + value enforcement: strings only, canonical set only.
    for k, v in body.items():
        if not isinstance(v, str):
            return web.json_response(
                {
                    "error": "value_must_be_string",
                    "key": k,
                    "got_type": type(v).__name__,
                },
                status=400,
            )
        if v not in _ALLOWED_FLAG_VALUES:
            return web.json_response(
                {
                    "error": "invalid_flag_value",
                    "key": k,
                    "value": v,
                    "allowed": sorted(_ALLOWED_FLAG_VALUES),
                },
                status=400,
            )

    # Snapshot old values before mutation so the forensic log captures
    # the transition accurately.
    old_values: dict[str, str | None] = {
        k: os.environ.get(k) for k in _ALLOWED_FLAG_KEYS
    }

    updated: list[str] = []
    for k, v in body.items():
        os.environ[k] = v
        updated.append(k)

    current: dict[str, str | None] = {k: os.environ.get(k) for k in _ALLOWED_FLAG_KEYS}

    # One-line INFO log per call — structured enough for grep-based replay.
    log.info(
        "admin.set_flags updated=%s old=%s new=%s",
        sorted(updated),
        {k: old_values[k] for k in sorted(_ALLOWED_FLAG_KEYS)},
        {k: current[k] for k in sorted(_ALLOWED_FLAG_KEYS)},
    )

    return web.json_response(
        {
            "updated": sorted(updated),
            "current": current,
        }
    )


async def calibrate_endpoint(request: web.Request) -> web.Response:
    """Run GPU calibration against the local vLLM and return the result.

    Accepts JSON body:
        {"target_tokens": 100000, "max_concurrency": 12}

    Returns the calibration result dict from calibrate_gpu.calibrate().
    """
    try:
        body = await request.json()
        target_tokens = body.get("target_tokens", 100_000)
        max_concurrency = body.get("max_concurrency", 12)
        timeout_per_request = body.get("timeout_per_request", 300.0)

        _cal_spec = importlib.util.spec_from_file_location(
            "calibrate_gpu",
            Path(__file__).resolve().parent / "calibrate_gpu.py",
        )
        assert _cal_spec is not None and _cal_spec.loader is not None
        _cal_mod = importlib.util.module_from_spec(_cal_spec)
        _cal_spec.loader.exec_module(_cal_mod)  # type: ignore[union-attr]
        calibrate = _cal_mod.calibrate

        result = await calibrate(
            url=_llm_url,
            target_tokens=target_tokens,
            max_concurrency=max_concurrency,
            timeout_per_request=timeout_per_request,
        )
        return web.json_response(result)
    except Exception as e:
        log.exception("calibrate_endpoint failed")
        return web.json_response({"error": str(e)}, status=500)


async def process_bundle(request: web.Request) -> web.Response:
    """Process a bundle: compose question + run agentic loop."""
    global _active_requests, _total_processed, _errors
    _active_requests += 1
    start = time.monotonic()
    try:
        body = await request.json()
        bundle = body["bundle"]

        # Compose question using tier2 module
        tier2 = _load_tier2_module()
        compose_client = AsyncOpenAI(base_url=_llm_url, api_key="EMPTY", timeout=None)
        question_text = await tier2.compose_question(
            compose_client, _model_name, bundle
        )
        if not question_text:
            _errors += 1
            return web.json_response({"error": "compose_failed"}, status=500)

        # PP question-op validation: reject if composed question doesn't
        # mention the transform operation (e.g., Q says "list the runtimes"
        # but expected answer is a mean). Prevents training on misleading Q/A.
        if bundle.get("question_type") == "post_processing":
            _OP_KEYWORDS: dict[str, list[str]] = {
                "mean": ["average", "mean", "typical"],
                "sum": ["total", "sum", "combined", "altogether", "overall"],
                "year_delta": [
                    "difference",
                    "years between",
                    "how many years",
                    "how long",
                ],
                "selection_max": [
                    "highest",
                    "largest",
                    "most",
                    "maximum",
                    "latest",
                    "longest",
                    "greatest",
                    "which",
                ],
            }
            transform_op = bundle.get("transform_op", "")
            keywords = _OP_KEYWORDS.get(transform_op, [])
            q_lower = question_text.lower()
            if keywords and not any(kw in q_lower for kw in keywords):
                log.warning(
                    "PP question-op mismatch: op=%s but Q has none of %s — rejecting %s",
                    transform_op,
                    keywords,
                    bundle.get("bundle_id"),
                )
                _errors += 1
                return web.json_response(
                    {"error": "pp_question_op_mismatch"},
                    status=500,
                )

        compose_time = time.monotonic() - start
        expected = tier2.expected_answer_from_bundle(bundle)

        # CP2: local gold-match + tailored reflection (no verifier model
        # call needed). See _make_on_draft docstring.
        on_draft = _make_on_draft(
            expected_answer=expected or "",
            question_text=question_text,
            bundle=bundle,
        )

        cb = body.get("context_budget", _context_budget)
        config = LoopConfig(
            max_iterations=body.get("max_iterations", 10),
            context_budget=cb,
            token_budget=body.get("token_budget", cb),  # default = context_budget
            tool_timeout=body.get("tool_timeout", 900.0),  # safety net; see LoopConfig
            concurrency=1,
            on_draft=on_draft,
        )

        question = Question(
            qid=f"t2_{bundle['bundle_id']}",
            text=question_text,
            question_type=bundle["question_type"],
        )

        llm_client = OpenAILLMClient(base_url=_llm_url, model=_model_name)
        assert _mcp_client is not None, "MCP client not initialized"
        pipeline = os.environ.get("AGENTIC_PIPELINE", "unified")
        if pipeline == "unified":
            from r2c2_agent.agent.loop import answer_question_pipeline

            trace: QuestionTrace = await answer_question_pipeline(
                question=question,
                llm_client=llm_client,
                mcp_client=_mcp_client,
                tools=_tools,
                config=config,
            )
        else:
            trace = await answer_question(
                question=question,
                llm_client=llm_client,
                mcp_client=_mcp_client,
                tools=_tools,
                config=config,
            )

        elapsed = time.monotonic() - start

        # no_tool_call traces are not RAG — force miss regardless of answer
        if trace.termination == "no_tool_call":
            answer_match = False
        else:
            answer_match = tier2.check_answer_match(
                expected, trace.draft_answer, bundle["question_type"], bundle=bundle
            )

        # CP2 audit trail, stashed on the trace by loop.answer_question.
        self_correction_injections = getattr(trace, "self_correction_injections", 0)
        verifier_verdicts = getattr(trace, "verifier_verdicts", [])

        # Pull the pipeline state off the trace so telemetry + provenance
        # fields come out of the shared helpers. Bundle-mode traces
        # must emit the same record shape as question-mode so
        # downstream training loaders can consume both without
        # per-entry-point branching.
        pipeline_db = getattr(trace, "m11_db", None)
        record = {
            "question_id": question.qid,
            "question_type": bundle["question_type"],
            "question_text": question_text,
            "expected_answer": expected,
            "gold_passage_ids": bundle["gold_passage_ids"],
            "gold_nuggets": [
                {"passage_id": n["passage_id"], "text": n["text"]}
                for n in bundle["nuggets"]
            ],
            "difficulty": bundle["difficulty_estimate"],
            "tier": 2,
            "generation_method": "llm_122b_trace",
            "bundle_id": bundle["bundle_id"],
            "messages": trace.messages,
            "trace_metadata": {
                "compose_time_seconds": round(compose_time, 2),
                "iterations": trace.iterations,
                "tool_calls": sum(1 for m in trace.messages if m.get("role") == "tool"),
                "wall_time_seconds": round(elapsed, 2),
                "answer_match": answer_match,
                "confidence": trace.raw_confidence,
                "termination": trace.termination,
                "trace_pending": False,
                "self_correction_injections": self_correction_injections,
                "verifier_verdicts": verifier_verdicts,
            },
            **_pipeline_telemetry_fields(
                pipeline_db,
                regex_fallback_stats=getattr(trace, "m11_regex_fallback_stats", None),
                turn_timings=getattr(trace, "m11_turn_timings", None),
                question_type=bundle["question_type"],
            ),
            **_provenance_fields(),
        }

        # Normalize Answer: line in trace for R2C2 format compliance.
        # Uses expected_answer when available (gold-answer data-prep mode).
        try:
            from r2c2_agent.shared.answer_normalization import normalize_answer_in_trace

            normalize_answer_in_trace(record)
        except ImportError:
            pass  # shared module not on path — skip normalization

        _total_processed += 1
        return web.json_response(record)

    except Exception as e:
        _errors += 1
        log.exception("process_bundle failed")
        return web.json_response(
            {"error": str(e), "elapsed": round(time.monotonic() - start, 2)},
            status=500,
        )
    finally:
        _active_requests -= 1


# Provenance helpers: computed once at worker boot, stamped onto every record.

_PROMPT_VERSION_HASH: str = ""
_TOOL_SCHEMA_HASH: str = ""


def _compute_provenance_hashes(tools: list[dict] | None, prompt: str) -> None:
    """Populate module-level provenance hashes used in record stamps.

    Called once by `on_startup` after the tool list is finalised and the
    system prompt is assembled. The resulting hashes are SHA256 prefixes
    of the exact JSON / text the model sees — any change to tools or
    prompt produces a new hash, so downstream training loaders can
    group traces by (model_name, tool_schema_hash, prompt_version_hash)
    and refuse to mix incompatible cohorts.
    """
    import hashlib
    import json as _json

    global _TOOL_SCHEMA_HASH, _PROMPT_VERSION_HASH
    tools_json = _json.dumps(tools or [], sort_keys=True, ensure_ascii=False)
    _TOOL_SCHEMA_HASH = hashlib.sha256(tools_json.encode("utf-8")).hexdigest()[:16]
    _PROMPT_VERSION_HASH = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[
        :16
    ]
    log.info(
        "provenance: model=%s tool_schema_hash=%s prompt_version_hash=%s",
        _model_name,
        _TOOL_SCHEMA_HASH,
        _PROMPT_VERSION_HASH,
    )


def _provenance_fields() -> dict[str, str]:
    """Return the per-record provenance stamp."""
    return {
        "model_name": _model_name or "",
        "tool_schema_hash": _TOOL_SCHEMA_HASH,
        "prompt_version_hash": _PROMPT_VERSION_HASH,
    }


def _pipeline_telemetry_fields(
    pipeline_db: Any,
    *,
    qtype_stats: dict[str, Any] | None = None,
    regex_fallback_stats: dict[str, Any] | None = None,
    turn_timings: list[float] | None = None,
    question_type: str = "",
) -> dict[str, Any]:
    """Bundle the full pipeline telemetry stamp for a record.

    Called from every handler that emits a trace record so every trace
    carries the identical shape regardless of which entry point produced
    it.
    """
    return {
        "m11_pipeline": _serialize_pipeline_state(pipeline_db),
        "m11_classifier": {
            "question_type": question_type or "",
            "stats": dict(qtype_stats or {}),
        },
        "m11_regex_fallback_stats": dict(regex_fallback_stats or {}),
        "m11_turn_timings": list(turn_timings or []),
    }


def _synthesize_force_deliver_envelope(pipeline_db: Any) -> dict[str, Any]:
    """Reconstruct a minimal force_deliver envelope from db state.

    Defense-in-depth: if ``db.answer.exit_reason == "force_deliver"`` but
    ``db.force_deliver_envelope`` is empty, the envelope-write in
    ``_force_deliver_impl`` never ran. This helper rebuilds the
    required fields (``ladder_path``, ``chosen_answer_tier``,
    ``force_deliver_reason``, ``committed``, ``answer``, ``confidence``,
    ``exit_reason``, ``citation_pid``, ``citation_pids``) from
    authoritative db state so every serialised orphan trace still
    carries a non-empty envelope for downstream training / reward /
    audit consumers. The ladder_path defaults to ``"unknown_orphan"``
    when the db mirror (``finalizer_ladder_path``) is also unset.
    """
    ans = getattr(pipeline_db, "answer", None)
    draft = (getattr(ans, "draft_answer", "") or "") if ans is not None else ""
    conf = float(getattr(ans, "confidence", 0.0) or 0.0) if ans is not None else 0.0
    tier_emitted = getattr(pipeline_db, "finalizer_tier_emitted", None)
    try:
        chosen_tier = int(tier_emitted) if tier_emitted is not None else 0
    except (TypeError, ValueError):
        chosen_tier = 0
    ladder_path = getattr(pipeline_db, "finalizer_ladder_path", None)
    if not ladder_path:
        # Best-effort default — mirrors ``pre_ladder_tier{N}`` when
        # _force_deliver_impl set the tier but the envelope-write aborted.
        if chosen_tier in (1, 2):
            ladder_path = f"pre_ladder_tier{chosen_tier}"
        else:
            ladder_path = "unknown_orphan"
    citation_pid = ""
    for n in getattr(pipeline_db, "nuggets", {}).values():
        if getattr(n, "status", "") == "verified_supporting":
            citation_pid = getattr(n, "passage_id", "") or ""
            if citation_pid:
                break
    rationale = getattr(ans, "rationale", "") or "" if ans is not None else ""
    return {
        "committed": True,
        "answer": draft,
        "confidence": round(conf, 3),
        "exit_reason": "force_deliver",
        "force_deliver_reason": (rationale or "orphan_envelope_recovered")[:400],
        "ladder_path": ladder_path,
        "chosen_answer_tier": chosen_tier,
        "citation_pid": citation_pid,
        "citation_pids": [citation_pid] if citation_pid else [],
        "budget": {},
        "session_status": {},
        "synthesised_from_db": True,
    }


def _serialize_pipeline_state(pipeline_db: Any) -> dict[str, Any]:
    """Dump the PipelineState state needed for downstream trace audit.

    The DB is populated in memory during the agentic loop, then only
    ``adapter_stats`` surfaces in the per-qid record. This helper captures
    the full state (global event log, per-nugget NLI scores/bands, passage
    history, answer state, force_deliver envelope, Phase-2 session stack,
    budget counters) so the trace audit can see what the pipeline actually did.

    Pydantic ``model_dump()`` gives us clean JSON-serializable dicts for
    every model (TransitionEvent, Nugget, Passage, AnswerState).
    """
    if pipeline_db is None:
        return {}
    try:
        # Defense-in-depth: if db.answer.exit_reason == "force_deliver" but
        # the envelope is empty, force_deliver's envelope-write never ran
        # (e.g. an exception in _force_deliver_impl between setting
        # ``exit_reason = "force_deliver"`` and copying the envelope). Reconstruct
        # a minimal envelope from authoritative db state so every record
        # with termination="force_deliver" carries ``ladder_path``,
        # ``chosen_answer_tier``, ``force_deliver_reason`` for trace-audit
        # contracts. The caller's original envelope wins when populated —
        # this only fires on the orphan path.
        _fde = dict(getattr(pipeline_db, "force_deliver_envelope", {}) or {})
        _ans = getattr(pipeline_db, "answer", None)
        if not _fde and _ans is not None and _ans.exit_reason == "force_deliver":
            _fde = _synthesize_force_deliver_envelope(pipeline_db)
        return {
            "events": [e.model_dump(mode="json") for e in pipeline_db.events],
            "nuggets": [n.model_dump(mode="json") for n in pipeline_db.nuggets.values()],
            "passages_db": {
                pid: p.model_dump(mode="json") for pid, p in pipeline_db.passages.items()
            },
            "answer_state": pipeline_db.answer.model_dump(mode="json"),
            "force_deliver_envelope": _fde,
            "phase2": {
                "active": bool(getattr(pipeline_db, "phase2_active", False)),
                "session_start_turn": int(
                    getattr(pipeline_db, "phase2_session_start_turn", 0) or 0
                ),
            },
            "budget": {
                "calls_used": int(getattr(pipeline_db, "calls_used", 0) or 0),
                "calls_total": int(getattr(pipeline_db, "calls_total", 0) or 0),
                "nuggets_recorded": int(getattr(pipeline_db, "nuggets_recorded", 0) or 0),
                "current_turn": int(getattr(pipeline_db, "current_turn", 0) or 0),
                "terminated": bool(getattr(pipeline_db, "terminated", False)),
                "answer_submit_retries": int(
                    getattr(pipeline_db, "answer_submit_retries", 0) or 0
                ),
                "refusal_validator_rejects": int(
                    getattr(pipeline_db, "refusal_validator_rejects", 0) or 0
                ),
                "calls_used_on_record_evidence": int(
                    getattr(pipeline_db, "calls_used_on_record_evidence", 0) or 0
                ),
            },
            "pid_reject_streak": dict(getattr(pipeline_db, "pid_reject_streak", {}) or {}),
            "search_queries": list(getattr(pipeline_db, "search_queries", []) or []),
        }
    except Exception:  # noqa: BLE001
        log.exception("pipeline state serialization failed")
        return {"error": "serialization_failed"}


async def _run_post_trace(
    q_data: dict[str, Any],
    messages: list[dict[str, Any]],
    draft_answer: str,
    raw_confidence: int,
    iterations: int,
    termination: str,
    start: float,
    rollout_idx: int,
    qid: str,
    pipeline_db: Any = None,
    qtype_stats: dict[str, Any] | None = None,
    regex_fallback_stats: dict[str, Any] | None = None,
    turn_timings: list[float] | None = None,
) -> dict[str, Any]:
    """Run the post-trace pipeline (reconcile + calibrate) on an existing trace.

    Used by both /process-question (after agentic loop) and /process-trace
    (rerun only the post-trace pipeline on an existing trace, skipping the
    expensive agentic loop). The input is a trace in its message-list form
    plus the draft answer/confidence from the loop.

    When ``pipeline_db`` is provided (unified pipeline path), the reconciler
    + calibrator are bypassed. The PipelineState already carries the final
    PR / AC / answer / confidence — we run it through the adapter
    (``apply_adapter``) which flattens to legacy-shape records.

    Returns a submission-ready record identical to process_question's output.
    """
    # Bypass path: the pipeline state already holds the final submission state.
    if pipeline_db is not None:
        from r2c2_agent.orchestrator.nugget_reconciler import (
            apply_adapter,  # noqa: E402
        )

        try:
            ranked_passages, clean_nuggets, adapter_stats = apply_adapter(pipeline_db)
            # Log the adapter outcome so an operator can see whether
            # submit_answer(refusal=True) produced any passages or nuggets.
            log.info(
                "pipeline adapter qid=%s ranked_passages=%d nuggets=%d stats=%s",
                qid,
                len(ranked_passages),
                len(clean_nuggets),
                {k: v for k, v in adapter_stats.items() if not k.startswith("_")},
            )
            try:
                from r2c2_agent.agent.obs import emit_event as _emit_event

                _emit_event(
                    "pipeline.adapter",
                    level="quiet",
                    qid=qid,
                    ranked_passages=len(ranked_passages),
                    nuggets=len(clean_nuggets),
                    stats={
                        k: v for k, v in adapter_stats.items() if not k.startswith("_")
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception:
            log.exception("%s: apply_adapter failed", qid)
            ranked_passages, clean_nuggets, adapter_stats = [], [], {}

        # The pipeline's confidence is authoritative; no calibrator pass.
        elapsed = time.monotonic() - start
        return {
            "question_id": qid,
            "question_type": q_data.get("question_type", ""),
            "question_text": q_data["question_text"],
            "expected_answer": q_data.get("expected_answer", ""),
            "gold_passage_ids": q_data.get("gold_passage_ids", []),
            "rollout_idx": rollout_idx,
            "final_answer": draft_answer,
            "confidence": raw_confidence,  # already computed by the pipeline
            "raw_confidence": raw_confidence,
            "iterations": iterations,
            "tool_calls": sum(1 for m in messages if m.get("role") == "tool"),
            "messages": messages,
            "elapsed_seconds": round(elapsed, 2),
            "termination": termination,
            "reconciled_nuggets": clean_nuggets,
            "reconciled_passages": [
                {
                    "passage_id": p["passage_id"],
                    "doc_id": p.get("doc_id", ""),
                    "passage_text": p.get("passage_text", ""),
                    "passage_text_with_title": p.get("passage_text_with_title", ""),
                    "title": p.get("title", ""),
                    "section": p.get("section", ""),
                }
                for p in ranked_passages
            ],
            **_pipeline_telemetry_fields(
                pipeline_db,
                qtype_stats=qtype_stats,
                regex_fallback_stats=regex_fallback_stats,
                turn_timings=turn_timings,
                question_type=q_data.get("question_type", ""),
            ),
            **_provenance_fields(),
            "post_trace_stats": {
                "pipeline": "unified",
                "adapter_stats": adapter_stats,
                "reconciler_llm_calls": 0,
                "reconciler_mcp_calls": 0,
                "reconciler_steps": {},
                "calibrator_llm_calls": 0,
                "total_post_trace_llm_calls": 0,
            },
        }

    from r2c2_agent.orchestrator.generate_submission import (
        extract_nuggets_from_trace as _regex_nuggets,
    )
    from r2c2_agent.orchestrator.generate_submission import (
        extract_passages_from_trace as _extract_passages,
    )
    from r2c2_agent.orchestrator.nugget_reconciler import (  # noqa: E402
        reconcile_nuggets_and_passages,
    )
    from r2c2_agent.shared.answer_extraction import (
        extract_answer_and_confidence as _extract_ac,
    )

    trace_dict = {"messages": messages}
    raw_nuggets = _regex_nuggets(trace_dict)
    raw_passages = _extract_passages(trace_dict, limit=0)

    # Fallback: if draft_answer is empty, scan the last assistant message
    answer_text = draft_answer or ""
    if not answer_text:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                answer_text, _ = _extract_ac(msg["content"])
                break
    if not answer_text:
        log.warning("%s: no answer text, skipping reconciliation", qid)

    reconciled_passages = raw_passages
    reconciled_nuggets = raw_nuggets
    reconciler_stats: dict[str, Any] = {}
    if not raw_passages or not answer_text:
        log.info("%s: no passages in trace, skipping reconciliation", qid)
    else:
        try:
            async_client = AsyncOpenAI(
                base_url=_llm_url,
                api_key="unused",
                timeout=None,
            )
            (
                reconciled_passages,
                reconciled_nuggets,
                reconciler_stats,
            ) = await reconcile_nuggets_and_passages(
                client=async_client,
                model=_model_name,
                question=q_data["question_text"],
                answer=answer_text,
                raw_nuggets=raw_nuggets,
                passages=raw_passages,
                mcp_client=_mcp_client,
                question_type=q_data.get("question_type", ""),
            )
            log.info(
                "%s: reconciled %d→%d nuggets, %d→%d passages",
                qid,
                len(raw_nuggets),
                len(reconciled_nuggets),
                len(raw_passages),
                len(reconciled_passages),
            )
        except Exception:
            log.exception("%s: nugget reconciliation failed, using raw", qid)
            reconciler_stats = {}

    # Confidence calibration
    calibrated_confidence = raw_confidence
    calibrator_llm_calls = 0
    try:
        from r2c2_agent.orchestrator.confidence_calibrator import calibrate_confidence

        async_client = AsyncOpenAI(
            base_url=_llm_url,
            api_key="unused",
            timeout=None,
        )
        cal_result = await calibrate_confidence(
            client=async_client,
            model=_model_name,
            question=q_data["question_text"],
            original_answer=answer_text,
            nuggets=reconciled_nuggets,
        )
        calibrated_confidence = cal_result.calibrated_confidence
        calibrator_llm_calls = cal_result.llm_calls
        log.info(
            "%s: confidence %d -> %d (match=%s, llm_conf=%d)",
            qid,
            raw_confidence,
            calibrated_confidence,
            cal_result.match,
            cal_result.llm_confidence,
        )
    except Exception:
        log.exception("%s: confidence calibration failed", qid)

    elapsed = time.monotonic() - start
    return {
        "question_id": qid,
        "question_type": q_data.get("question_type", ""),
        "question_text": q_data["question_text"],
        "expected_answer": q_data.get("expected_answer", ""),
        "gold_passage_ids": q_data.get("gold_passage_ids", []),
        "rollout_idx": rollout_idx,
        "final_answer": draft_answer,
        "confidence": calibrated_confidence,
        "raw_confidence": raw_confidence,
        "iterations": iterations,
        "tool_calls": sum(1 for m in messages if m.get("role") == "tool"),
        "messages": messages,
        "elapsed_seconds": round(elapsed, 2),
        "termination": termination,
        "reconciled_nuggets": reconciled_nuggets,
        "reconciled_passages": [
            {
                "passage_id": p["passage_id"],
                "doc_id": p.get("doc_id", ""),
                "passage_text": p.get("passage_text", ""),
                "passage_text_with_title": p.get("passage_text_with_title", ""),
                "title": p.get("title", ""),
                "section": p.get("section", ""),
            }
            for p in reconciled_passages
        ],
        "post_trace_stats": {
            "reconciler_llm_calls": reconciler_stats.get("llm_calls", 0),
            "reconciler_mcp_calls": reconciler_stats.get("mcp_calls", 0),
            "reconciler_steps": reconciler_stats.get("steps", {}),
            "calibrator_llm_calls": calibrator_llm_calls,
            "total_post_trace_llm_calls": (
                reconciler_stats.get("llm_calls", 0) + calibrator_llm_calls
            ),
        },
    }


async def process_trace(request: web.Request) -> web.Response:
    """Re-run post-trace pipeline on an existing trace.

    Skips the agentic loop entirely. Expects a payload with the trace
    fields already populated (messages, draft_answer, etc.) — typically
    a record from a previous /process-question run.

    Payload:
        {
            "question": {"question_id", "question_text", "question_type", ...},
            "trace": {
                "messages": [...],
                "draft_answer": "...",
                "raw_confidence": int,
                "iterations": int,
                "termination": "...",
            },
            "rollout_idx": int (optional),
        }
    """
    global _active_requests, _total_processed, _errors
    _active_requests += 1
    start = time.monotonic()
    try:
        body = await request.json()
        q_data = body["question"]
        trace_data = body["trace"]
        qid = q_data.get("question_id", "?")

        record = await _run_post_trace(
            q_data=q_data,
            messages=trace_data.get("messages", []),
            draft_answer=trace_data.get("draft_answer", "")
            or trace_data.get("final_answer", ""),
            raw_confidence=int(trace_data.get("raw_confidence", 0) or 0),
            iterations=int(trace_data.get("iterations", 0) or 0),
            termination=trace_data.get("termination", "rerun"),
            start=start,
            rollout_idx=body.get("rollout_idx", 0),
            qid=qid,
        )

        _total_processed += 1
        return web.json_response(record)

    except Exception as e:
        _errors += 1
        log.exception("process_trace failed")
        return web.json_response(
            {"error": str(e), "elapsed": round(time.monotonic() - start, 2)},
            status=500,
        )
    finally:
        _active_requests -= 1


async def process_question(request: web.Request) -> web.Response:
    """Process a pre-composed question: run agentic loop, optionally N times.

    When ``body["ensemble"] >= 2``, run the agentic loop N sequential times
    on the same worker and return a list of N per-iteration records. Each
    iteration K subtracts the pids the agent explicitly judged in
    iterations 0..K-1 from the MCP's retrieval index (ensemble feed-
    forward — see ``pipeline_handlers.touched_pids`` + ExcludingMCPWrapper).

    Default ``ensemble=1`` preserves the legacy single-record response.
    """
    global _active_requests, _total_processed, _errors
    _active_requests += 1
    start = time.monotonic()
    try:
        body = await request.json()
        q_data = body["question"]
        temperature = body.get("temperature", 0.0)
        cb = body.get("context_budget", _context_budget)
        config = LoopConfig(
            max_iterations=body.get("max_iterations", 10),
            context_budget=cb,
            token_budget=body.get("token_budget", cb),  # default = context_budget
            tool_timeout=body.get("tool_timeout", 900.0),  # safety net; see LoopConfig
            concurrency=1,
        )

        ensemble_n = max(1, int(body.get("ensemble", 1)))
        rollout_idx = body.get("rollout_idx", 0)

        question = Question(
            qid=q_data["question_id"],
            text=q_data["question_text"],
            question_type=q_data.get("question_type", ""),
        )

        llm_client = OpenAILLMClient(
            base_url=_llm_url,
            model=_model_name,
            temperature=temperature,
        )

        assert _mcp_client is not None, "MCP client not initialized"

        # Record-replay harness — wrap the real clients with
        # recorders when RECORD_TRACES_DIR is set. Fixture:
        # ${RECORD_TRACES_DIR}/<qid>__<rollout_idx>.jsonl.
        _trace_writer: Any = None
        _record_seq: list[int] | None = None
        _record_t0: float | None = None
        _trace_dir = os.environ.get("RECORD_TRACES_DIR")
        if _trace_dir:
            try:
                scripts_dir = str(Path(__file__).resolve().parents[1])
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from debug.trace_recorder import (  # type: ignore[import-not-found]
                    JsonlWriter,
                    RecordingLLMClient,
                )

                fixture_path = Path(_trace_dir) / f"{question.qid}__{rollout_idx}.jsonl"
                _trace_writer = JsonlWriter(fixture_path)
                _record_seq = [0]
                _record_t0 = time.monotonic()
                llm_client = RecordingLLMClient(  # type: ignore[assignment]
                    llm_client,
                    _trace_writer,
                    qid=question.qid,
                    seq_counter=_record_seq,
                    t0=_record_t0,
                )
                log.info(
                    "record-replay: capturing qid=%s -> %s",
                    question.qid,
                    fixture_path,
                )
            except Exception:  # noqa: BLE001 — record path is opt-in, never block
                log.exception(
                    "record-replay: failed to install recorder for qid=%s",
                    question.qid,
                )
                _trace_writer = None

        pipeline = os.environ.get("AGENTIC_PIPELINE", "unified")
        qtype_stats: dict[str, Any] = {}
        # The R2C2 test set ships no question_type, so classify at ingest
        # to populate metadata for per-type metrics. Gated on the unified
        # pipeline to keep the legacy path untouched. Classification happens
        # once per qid, so all N ensemble iterations see the same
        # question_type.
        if pipeline == "unified" and not question.question_type:
            from r2c2_agent.orchestrator.question_classifier import (
                classify_question_type,
            )

            qtype, qtype_stats = await classify_question_type(llm_client, question.text)
            question = Question(
                qid=question.qid,
                text=question.text,
                question_type=qtype,
            )
            q_data["question_type"] = qtype
            log.info(
                "classified qid=%s type=%s source=%s",
                question.qid,
                qtype,
                qtype_stats.get("source"),
            )

        # Per-iter dispatch (node-independent ensemble). When the orchestrator
        # opts in with ensemble_mode="per_iter", the worker runs one iteration
        # and reconstructs the touched_pids_set from on-disk iter_0..K-1
        # records instead of looping N times in memory.
        ensemble_mode = body.get("ensemble_mode", "legacy")
        if ensemble_mode == "per_iter":
            target_idx = int(body.get("ensemble_idx", 0))
            prior_paths_raw = body.get("prior_iter_paths") or []
            strict_prior = bool(body.get("prior_iter_strict", False))
            # The orchestrator pre-resolves the prior touched-pid set on
            # its own filesystem and ships the authoritative list in the
            # body. The worker prefers that list over re-reading the iter
            # files because the iter outputs may not be visible on the
            # worker's BeeGFS mount: a cross-cluster race that otherwise
            # yields empty exclusion lists.
            prior_touched_from_body_raw = body.get("prior_touched_pids")
            prior_touched_from_body: list[str] | None = None
            if isinstance(prior_touched_from_body_raw, list):
                prior_touched_from_body = [
                    str(pid)
                    for pid in prior_touched_from_body_raw
                    if isinstance(pid, str) and pid
                ]

            from r2c2_agent.agent.ensemble import (
                MissingPriorIterError,
                PriorIterResult,
                load_prior_rollout_answer,
                load_touched_set,
            )

            try:
                prior_result = load_touched_set(
                    [Path(p) for p in prior_paths_raw],
                    qid=question.qid,
                    rollout_idx=rollout_idx,
                    strict=strict_prior,
                )
            except MissingPriorIterError as exc:
                # In strict mode the orchestrator-resolved list cannot
                # rescue us — propagate the failure.
                _errors += 1
                return web.json_response(
                    {
                        "error": f"missing_prior_iter: {exc}",
                        "question_id": question.qid,
                        "rollout_idx": rollout_idx,
                        "ensemble_idx": target_idx,
                        "ensemble_n": ensemble_n,
                        "termination": "missing_prior_iter",
                    },
                    status=409,
                )

            # Prefer the orchestrator-resolved set when (a) it's present
            # and (b) either the worker's read returned nothing OR every
            # path was missing on the worker's mount. Instead of running
            # with an empty exclusion, the worker uses the orchestrator's
            # authoritative set.
            if prior_touched_from_body is not None and (
                not prior_result.touched or prior_result.missing_paths
            ):
                if prior_touched_from_body and not prior_result.touched:
                    log.warning(
                        "per_iter qid=%s iter=%d: worker found 0 touched "
                        "pids on disk but orchestrator resolved %d — using "
                        "orchestrator set (cross-cluster path race rescue)",
                        question.qid,
                        target_idx,
                        len(prior_touched_from_body),
                    )
                prior_result = PriorIterResult(
                    touched=frozenset(prior_touched_from_body),
                    iters_found=prior_result.iters_found,
                    missing_paths=prior_result.missing_paths,
                )

            touched_pids_set: set[str] = set(prior_result.touched)
            iter_start = time.monotonic()
            if touched_pids_set:
                from r2c2_agent.agent.mcp.excluding_wrapper import ExcludingMCPWrapper

                iter_mcp = ExcludingMCPWrapper(_mcp_client, frozenset(touched_pids_set))
            else:
                iter_mcp = _mcp_client

            if _trace_writer is not None and _record_seq is not None:
                from debug.trace_recorder import (  # type: ignore[import-not-found]
                    RecordingMCPClient,
                )

                iter_mcp = RecordingMCPClient(
                    iter_mcp,
                    _trace_writer,
                    qid=question.qid,
                    seq_counter=_record_seq,
                    t0=_record_t0,
                )

            # Harvest a high-confidence committed answer from prior
            # iters and thread it onto LoopConfig so the pipeline seeds
            # ``db.prior_rollout_answer``. force_deliver's final-fallback
            # ladder tries this seed before falling to tier-5.
            try:
                prior_ans_result = load_prior_rollout_answer(
                    [Path(p) for p in prior_paths_raw],
                    qid=question.qid,
                    rollout_idx=rollout_idx,
                    strict=False,
                )
            except Exception:  # noqa: BLE001
                log.exception(
                    "per_iter load_prior_rollout_answer failed for qid=%s",
                    question.qid,
                )
                prior_ans_result = None

            if prior_ans_result and prior_ans_result.best_answer:
                from dataclasses import replace as _replace

                config = _replace(
                    config,
                    prior_rollout_answer=prior_ans_result.best_answer,
                    prior_rollout_confidence=prior_ans_result.best_confidence,
                    prior_rollout_source_iter=prior_ans_result.source_iter,
                )
                log.info(
                    "per_iter seeded prior_rollout_answer qid=%s iter=%s conf=%d",
                    question.qid,
                    prior_ans_result.source_iter,
                    prior_ans_result.best_confidence,
                )

            try:
                if pipeline == "unified":
                    from r2c2_agent.agent.loop import answer_question_pipeline

                    trace = await answer_question_pipeline(
                        question=question,
                        llm_client=llm_client,
                        mcp_client=iter_mcp,
                        tools=_tools,
                        config=config,
                    )
                else:
                    trace = await answer_question(
                        question=question,
                        llm_client=llm_client,
                        mcp_client=iter_mcp,
                        tools=_tools,
                        config=config,
                    )
            except Exception as iter_exc:  # noqa: BLE001
                _errors += 1
                log.exception(
                    "per_iter iteration %d failed for qid=%s",
                    target_idx,
                    question.qid,
                )
                err_rec = {
                    "question_id": question.qid,
                    "question_text": question.text,
                    "question_type": question.question_type,
                    "rollout_idx": rollout_idx,
                    "ensemble_idx": target_idx,
                    "ensemble_n": ensemble_n,
                    "ensemble_excluded_pids": sorted(touched_pids_set),
                    "ensemble_touched_delta": [],
                    "temperature": temperature,
                    "final_answer": "",
                    "confidence": 0.0,
                    "raw_confidence": 0,
                    "termination": "ensemble_error",
                    "error": str(iter_exc)[:400],
                    "iterations": 0,
                    "tool_calls": [],
                    "elapsed_seconds": round(time.monotonic() - iter_start, 2),
                    "prior_iter_missing_paths": list(prior_result.missing_paths),
                    "prior_iter_found": list(prior_result.iters_found),
                }
                _total_processed += 1
                return web.json_response(err_rec)

            pipeline_db = getattr(trace, "m11_db", None)
            iter_delta: set[str] = set()
            if pipeline_db is not None:
                from r2c2_agent.orchestrator.pipeline_handlers import (
                    touched_pids as _touched_pids_helper,
                )

                iter_delta = _touched_pids_helper(pipeline_db) - touched_pids_set

            record = await _run_post_trace(
                q_data=q_data,
                messages=trace.messages,
                draft_answer=trace.draft_answer,
                raw_confidence=trace.raw_confidence,
                iterations=trace.iterations,
                termination=trace.termination,
                start=iter_start,
                rollout_idx=rollout_idx,
                qid=question.qid,
                pipeline_db=pipeline_db,
                qtype_stats=qtype_stats,
                regex_fallback_stats=getattr(trace, "m11_regex_fallback_stats", None),
                turn_timings=getattr(trace, "m11_turn_timings", None),
            )
            record["ensemble_idx"] = target_idx
            record["ensemble_n"] = ensemble_n
            record["ensemble_excluded_pids"] = sorted(touched_pids_set)
            record["ensemble_touched_delta"] = sorted(iter_delta)
            record["temperature"] = temperature
            record["prior_iter_missing_paths"] = list(prior_result.missing_paths)
            record["prior_iter_found"] = list(prior_result.iters_found)
            _total_processed += 1
            return web.json_response(record)

        # Ensemble loop — runs N times for N>=1.
        records: list[dict[str, Any]] = []
        touched_pids_set = set()
        for ensemble_idx in range(ensemble_n):
            iter_start = time.monotonic()
            # Wrap MCP client with ExcludingMCPWrapper when we have a
            # non-empty touched-pid set; the wrapper forwards the CSV as
            # `exclude_ids` to search_passages so the server filters all
            # three retrieval legs.
            if touched_pids_set:
                from r2c2_agent.agent.mcp.excluding_wrapper import ExcludingMCPWrapper

                iter_mcp = ExcludingMCPWrapper(_mcp_client, frozenset(touched_pids_set))
            else:
                iter_mcp = _mcp_client

            if _trace_writer is not None and _record_seq is not None:
                from debug.trace_recorder import (  # type: ignore[import-not-found]
                    RecordingMCPClient,
                )

                iter_mcp = RecordingMCPClient(
                    iter_mcp,
                    _trace_writer,
                    qid=question.qid,
                    seq_counter=_record_seq,
                    t0=_record_t0,
                )

            try:
                if pipeline == "unified":
                    from r2c2_agent.agent.loop import answer_question_pipeline

                    trace: QuestionTrace = await answer_question_pipeline(
                        question=question,
                        llm_client=llm_client,
                        mcp_client=iter_mcp,
                        tools=_tools,
                        config=config,
                    )
                else:
                    trace = await answer_question(
                        question=question,
                        llm_client=llm_client,
                        mcp_client=iter_mcp,
                        tools=_tools,
                        config=config,
                    )
            except Exception as iter_exc:  # noqa: BLE001
                # Iteration crashed. Emit a partial record and stop —
                # iter K+1..N-1 would run with a stale touched-set so
                # they'd produce incomparable data.
                _errors += 1
                log.exception(
                    "ensemble iteration %d/%d failed for qid=%s",
                    ensemble_idx,
                    ensemble_n,
                    question.qid,
                )
                records.append(
                    {
                        "question_id": question.qid,
                        "question_text": question.text,
                        "question_type": question.question_type,
                        "rollout_idx": rollout_idx,
                        "ensemble_idx": ensemble_idx,
                        "ensemble_n": ensemble_n,
                        "ensemble_excluded_pids": sorted(touched_pids_set),
                        "ensemble_touched_delta": [],
                        "temperature": temperature,
                        "final_answer": "",
                        "confidence": 0.0,
                        "raw_confidence": 0,
                        "termination": "ensemble_error",
                        "error": str(iter_exc)[:400],
                        "iterations": 0,
                        "tool_calls": [],
                        "elapsed_seconds": round(time.monotonic() - iter_start, 2),
                    }
                )
                break

            # PipelineState carries the final PR/AC/answer/conf. We read
            # the pre-submit touched-pid snapshot via the helper and
            # compute this iteration's delta (new pids agent judged).
            pipeline_db = getattr(trace, "m11_db", None)
            iter_delta: set[str] = set()
            if pipeline_db is not None:
                from r2c2_agent.orchestrator.pipeline_handlers import (
                    touched_pids as _touched_pids_helper,
                )

                iter_delta = _touched_pids_helper(pipeline_db) - touched_pids_set

            record = await _run_post_trace(
                q_data=q_data,
                messages=trace.messages,
                draft_answer=trace.draft_answer,
                raw_confidence=trace.raw_confidence,
                iterations=trace.iterations,
                termination=trace.termination,
                start=iter_start,
                rollout_idx=rollout_idx,
                qid=question.qid,
                pipeline_db=pipeline_db,
                qtype_stats=qtype_stats,
                regex_fallback_stats=getattr(trace, "m11_regex_fallback_stats", None),
                turn_timings=getattr(trace, "m11_turn_timings", None),
            )
            record["ensemble_idx"] = ensemble_idx
            record["ensemble_n"] = ensemble_n
            record["ensemble_excluded_pids"] = sorted(touched_pids_set)
            record["ensemble_touched_delta"] = sorted(iter_delta)
            record["temperature"] = temperature
            records.append(record)

            # Grow the touched-set for the next iteration.
            touched_pids_set |= iter_delta

        _total_processed += 1
        # Single-iteration responses stay flat-dict-shaped for back-compat
        # with the pre-ensemble orchestrator. N>1 wraps in {"records": [...]}
        # so the orchestrator can route each record to its per-iter folder.
        if ensemble_n == 1:
            return web.json_response(records[0])
        return web.json_response({"records": records})

    except Exception as e:
        _errors += 1
        log.exception("process_question failed")
        return web.json_response(
            {"error": str(e), "elapsed": round(time.monotonic() - start, 2)},
            status=500,
        )
    finally:
        # Close the record-replay fixture file (if opened) so partial
        # crashes still leave a readable JSONL on disk. Defined earlier
        # inside the try-block; guard with try/except to avoid masking
        # any exception already in flight.
        try:
            _writer = locals().get("_trace_writer")
            if _writer is not None:
                _writer.close()
        except Exception:  # noqa: BLE001
            pass
        _active_requests -= 1


# Startup


async def on_startup(app: web.Application) -> None:
    """Connect to MCP, detect model, and initialize verifier on startup."""
    global _mcp_client, _tools, _model_name, _llm_url, _mcp_url
    global _context_budget, _token_budget
    global _verifier_client, _verifier_url

    _llm_url = os.environ.get("LLM_URL", "http://localhost:8000/v1")
    _mcp_url = os.environ.get("MCP_URL", "http://localhost:8765/mcp")
    _context_budget = int(os.environ.get("CONTEXT_BUDGET", "124518"))
    _token_budget = int(os.environ.get("TOKEN_BUDGET", "124518"))

    probe = AsyncOpenAI(base_url=_llm_url, api_key="EMPTY")
    try:
        models = await probe.models.list()
        _model_name = models.data[0].id if models.data else "unknown"
    except Exception:
        _model_name = "unknown"
    log.info("Model: %s", _model_name)

    _mcp_client = MCPHTTPClient(standard_url=_mcp_url, ner_url="")
    await _mcp_client.connect()
    _tools = await _mcp_client.list_tools()
    # Register the local format_answer tool (R2C2 answer formatting) — only
    # for the legacy ``answer_question`` loop. The unified driver does
    # not bind this tool (its dispatcher returns ``unknown_tool``), so
    # advertising it in the unified system prompt burns LLM turns on phantom
    # calls. Gated on AGENTIC_PIPELINE.
    _pipeline = os.environ.get("AGENTIC_PIPELINE", "unified")
    if _pipeline == "unified":
        _tools.extend(_HANDLER_TOOL_SCHEMAS)
        log.info(
            "pipeline: registered %d MCP tools + %d server-side handler "
            "schemas (record_evidence, submit_answer)",
            len(_tools) - len(_HANDLER_TOOL_SCHEMAS),
            len(_HANDLER_TOOL_SCHEMAS),
        )
        # Startup guard: the unified pipeline needs NLI. If NLI_URL is
        # unset, every record_evidence call fails open and the whole PR is
        # unusable. Log an error so operators catch a misconfigured
        # environment before the run spends GPU hours.
        nli_url = os.environ.get("NLI_URL", "").strip()
        if not nli_url:
            log.error(
                "AGENTIC_PIPELINE=unified but NLI_URL is unset — "
                "record_evidence will fail-open on every call. Check "
                "worker.sbatch NLI startup gate + set NLI_URL explicitly."
            )
    else:
        try:
            from r2c2_agent.shared.answer_normalization import FORMAT_ANSWER_TOOL_SPEC

            _tools.append(FORMAT_ANSWER_TOOL_SPEC)
            log.info("MCP connected, %d tools (+format_answer)", len(_tools) - 1)
        except ImportError:
            log.info(
                "MCP connected, %d tools (format_answer not available)", len(_tools)
            )

    # MCP smoke test: verify critical tools before accepting work.
    smoke_tests = [
        (
            "extract",
            {"pattern": "test", "passage_id": "Q1__00001", "context_sentences": 0},
        ),
        ("extract", {"pattern": "test", "doc_id": "Q1", "context_sentences": 0}),
        ("extract", {"passage_id": "Q1__00001"}),  # full-passage mode
        ("search_passages", {"query": "smoke test", "max_results": 1}),
    ]
    for tool_name, tool_args in smoke_tests:
        try:
            result = await _mcp_client.call_tool(tool_name, tool_args)
            is_error = isinstance(result, dict) and "error" in result
            # "not found" is OK (passage may not exist), but 400/500 errors are not
            err_text = str(result.get("error", "")) if is_error else ""
            if is_error and ("BadRequest" in err_text or "500" in err_text):
                log.error(
                    "MCP SMOKE TEST FAILED: %s(%s) -> %s",
                    tool_name,
                    tool_args,
                    err_text[:200],
                )
                raise SystemExit(
                    f"MCP smoke test failed: {tool_name} returned server error"
                )
            log.info(
                "MCP smoke OK: %s -> %s",
                tool_name,
                "error:not_found" if is_error else "ok",
            )
        except SystemExit:
            raise
        except Exception as e:
            log.error("MCP SMOKE TEST FAILED: %s(%s) -> %s", tool_name, tool_args, e)
            raise SystemExit(f"MCP smoke test failed: {tool_name} raised {e}")

    # Provenance hashes, which freeze the pipeline contract.
    # Compute SHA256-derived fingerprints of (a) the finalised tool
    # schema JSON and (b) the assembled system prompt. Every trace
    # record carries these hashes so a downstream training loader can
    # group cohort-compatible traces together and refuse to mix runs
    # with different tool schemas or prompt versions. Computed here,
    # after tools are appended and before the first request is served.
    try:
        from r2c2_agent.agent.prompt import build_system_prompt

        _system_prompt_text = build_system_prompt(pipeline=_pipeline)
    except Exception:  # noqa: BLE001
        log.exception("could not assemble system prompt for version hash")
        _system_prompt_text = ""
    _compute_provenance_hashes(_tools, _system_prompt_text)

    # CP2 verifier (soft-fail).
    # If VERIFIER_URL is empty or init fails, run CP0-only. The
    # verifier is launched in-job by slurm/orchestrator/worker.sbatch
    # as a second vLLM process on localhost, completely invisible to
    # the orchestrator's port registry.
    _verifier_url = os.environ.get("VERIFIER_URL", "").strip()
    if _verifier_url:
        verifier_model = os.environ.get("VERIFIER_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8")
        try:
            _verifier_client = VerifierClient(
                base_url=_verifier_url, model=verifier_model
            )
            log.info(
                "VerifierClient initialized (base_url=%s, model=%s)",
                _verifier_url,
                verifier_model,
            )
        except Exception as e:
            log.warning("VerifierClient init failed: %s — running CP0-only", e)
            _verifier_client = None
    else:
        log.info("VERIFIER_URL not set — running CP0-only (no CP2)")
        _verifier_client = None


def _configure_pipeline_logging() -> None:
    """Configure the pipeline logging — level gating + noise suppression
    + JSONL event sink.

    Honours ``PIPELINE_LOG_VERBOSITY`` (``quiet`` | ``normal`` | ``verbose``) for
    the pipeline's plain-text loggers. The verbosity env var is consumed at
    ``emit_event`` time in ``r2c2_agent.agent.obs`` for the structured channel;
    this function mirrors the gate on the plain-text side so ``quiet`` mode
    also silences ``log.info`` calls in ``r2c2_agent.agent.loop`` and
    ``pipeline_handlers``.

    Attaches a dedicated FileHandler to the ``pipeline.events`` logger
    when ``SLURM_JOB_ID`` is set (or ``PIPELINE_EVENTS_LOG`` is set to an
    explicit path), writing JSON lines to
    ``slurm/orchestrator/logs/pipeline_events_<job>.jsonl``.
    """
    verbosity = os.environ.get("PIPELINE_LOG_VERBOSITY", "normal").lower().strip()
    level_map = {
        "quiet": logging.WARNING,  # WARN+ only on the pipeline drivers
        "normal": logging.INFO,
        "verbose": logging.DEBUG,
    }
    pipeline_level = level_map.get(verbosity, logging.INFO)
    for name in ("r2c2_agent.agent.loop", "pipeline_handlers"):
        logging.getLogger(name).setLevel(pipeline_level)
    log.info(
        "pipeline log verbosity=%s (r2c2_agent.agent.loop + pipeline_handlers level=%s)",
        verbosity,
        logging.getLevelName(pipeline_level),
    )

    # Noise suppression. These emit INFO-level access lines on every
    # /mcp / ES / FastMCP round-trip and drown out semantic signal. The
    # MCP server process applies the same muting at its own bootstrap.
    for noisy in (
        "elasticsearch",
        "elastic_transport",
        "mcp.server.lowlevel",
        "mcp.server.streamable_http",
        "httpx",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # JSONL event sink. One line per emit_event call across the driver
    # and handlers. Path resolution:
    #   1. PIPELINE_EVENTS_LOG env var (absolute path, operator override)
    #   2. slurm/orchestrator/logs/pipeline_events_<SLURM_JOB_ID>.jsonl
    #   3. default: stderr (dev runs)
    explicit = os.environ.get("PIPELINE_EVENTS_LOG", "").strip()
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    events_log = logging.getLogger("pipeline.events")
    events_log.setLevel(logging.INFO)
    events_log.propagate = False  # keep JSONL out of the plain-text log
    target_path: Path | None = None
    if explicit:
        target_path = Path(explicit)
    elif job_id:
        slurm_dir = (
            Path(__file__).resolve().parents[3] / "slurm" / "orchestrator" / "logs"
        )
        slurm_dir.mkdir(parents=True, exist_ok=True)
        target_path = slurm_dir / f"pipeline_events_{job_id}.jsonl"
    if target_path is not None:
        handler = logging.FileHandler(target_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        events_log.addHandler(handler)
        log.info("pipeline.events JSONL sink: %s", target_path)
    else:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("[pipeline.events] %(message)s"))
        events_log.addHandler(stream)
        log.info("pipeline.events JSONL sink: stderr (no SLURM_JOB_ID)")


def main() -> None:
    """Entry point: start the worker HTTP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _configure_pipeline_logging()

    port = int(os.environ.get("WORKER_PORT", "9500"))
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get("/health", health)
    app.router.add_get("/info", info)
    app.router.add_post("/calibrate", calibrate_endpoint)
    app.router.add_post("/process-bundle", process_bundle)
    app.router.add_post("/process-question", process_question)
    app.router.add_post("/process-trace", process_trace)
    # Hot-reload is off by default; the production launch.sh never sets
    # ENABLE_HOT_RELOAD.
    if os.environ.get("ENABLE_HOT_RELOAD") == "1":
        app.router.add_post("/admin/reload", admin_reload)
        # /admin/inject_state rides the same dev-only gate — it bypasses
        # the normal agentic loop to force-fire a specific pipeline entry
        # point against crafted PipelineState. See admin_inject_state.
        app.router.add_post("/admin/inject_state", admin_inject_state)
        # /admin/set_flags rides the same dev-only gate — runtime toggle
        # of ASS_RECURSIVE_SUBMIT / R2C2_DECOUPLED_SCORING without a
        # worker relaunch. See admin_set_flags.
        app.router.add_post("/admin/set_flags", admin_set_flags)
        log.info(
            "admin.reload + admin.inject_state + admin.set_flags endpoints "
            "ENABLED (ENABLE_HOT_RELOAD=1)"
        )

    log.info("Starting worker on port %d", port)
    web.run_app(app, host="0.0.0.0", port=port, print=log.info)


if __name__ == "__main__":
    main()
