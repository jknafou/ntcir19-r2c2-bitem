"""R2C2 Unified MCP Server — hybrid 3-way retrieval.

Always runs in hybrid mode: BM25 + Dense (nomic) + KG (NPMI co-occurrence),
fused via 3-way RRF, reranked by cross-encoder. Degrades gracefully when
legs are unavailable.

Runs as a stateless HTTP MCP server on port 8765.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from r2c2_agent.mcp_server.extract_utils import bm25_score_sentences, rank_matches, split_sentences_simple
from mcp.server.fastmcp import Context, FastMCP

# Configuration

MCP_HOST = "0.0.0.0"
MCP_PORT = int(os.environ.get("MCP_PORT", "8765"))
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

# ES indexes
ES_INDEX_NER = os.environ.get("ES_INDEX_NER", "bitem-passages-ner")

# NLI entailment server, used by extract_from_passage as a fallback.
# There is no read timeout: the orchestrator bounds the wall-clock time per
# question, so a healthy but slow response is waited for. The connect and
# write timeouts make a dead service fail fast.
NLI_URL = os.environ.get("NLI_URL", "http://localhost:18500")
NLI_CONNECT_TIMEOUT_SECONDS = 30.0

# extract_from_passage tuning
EXTRACT_BM25_FALLBACK_THRESHOLD = 0.3
EXTRACT_TOO_BROAD_SCORE = 0.2
EXTRACT_NLI_TOPK = 10

# Dense indexes (FAISS) — paths must be set via env vars before MCP startup.
# Defaults assume the indexes were built into ``data/indexes/`` next to the repo.
_DEFAULT_INDEX_ROOT = os.environ.get("INDEX_ROOT", "data/indexes")
DENSE_INDEX_DIR = os.environ.get(
    "DENSE_INDEX_DIR",
    f"{_DEFAULT_INDEX_ROOT}/bitem-faiss-enriched",
)
LOAD_PLAIN_FAISS = os.environ.get("LOAD_PLAIN_FAISS", "0") == "1"
DENSE_INDEX_DIR_PLAIN = os.environ.get(
    "DENSE_INDEX_DIR_PLAIN",
    f"{_DEFAULT_INDEX_ROOT}/bitem-faiss-plain",
)

# KG graph
KG_PATH = os.environ.get(
    "KG_PATH",
    f"{_DEFAULT_INDEX_ROOT}/bitem-kg.json",
)

MAX_TEXT_LENGTH = 800
# search_passages returns a larger chunk than get_passage /
# extract_from_passage so the LLM can reason on a full paragraph without an
# extra round-trip. It stays bounded so that many searches in one session do
# not overflow the 131K-token context of the vLLM server.
SEARCH_MAX_TEXT_LENGTH = 2000

# Sentence-ending punctuation for soft truncation
_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _soft_truncate(text: str, max_len: int) -> str:
    """Truncate text at the end of the first sentence past max_len.

    Instead of cutting mid-sentence, finds the next sentence boundary
    after max_len and cuts there. Falls back to hard truncation if no
    sentence boundary is found within 2x max_len.

    Args:
        text: Full passage text.
        max_len: Target truncation point.

    Returns:
        Truncated text ending at a sentence boundary.
    """
    if len(text) <= max_len:
        return text
    match = _SENTENCE_END_RE.search(text, max_len)
    if match and match.end() <= max_len * 2:
        return text[: match.start() + 1]  # include the punctuation
    # No sentence boundary found — hard truncate
    return text[:max_len]


DEFAULT_MAX_RESULTS = 10
HARD_MAX_RESULTS = 100
DEFAULT_BM25_PER_QUERY = 100

# BM25 parameters (MS MARCO tuned)
BM25_K1 = 0.82
BM25_B = 0.68

# Cross-encoder model
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# Hybrid parameters
BM25_CANDIDATES = 100
DENSE_CANDIDATES = 100
KG_CANDIDATES = 100
RRF_K = 60
RERANK_POOL = 50
KG_ENTITY_SCORE_THRESHOLD = 0.7
KG_NPMI_2HOP_THRESHOLD = 0.3

logger = logging.getLogger("mcp-r2c2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Tool descriptions (mode-specific, shown to the LLM)

_ES_QUERY_SYNTAX = """\

## Query syntax (Elasticsearch query_string)

The `queries` parameter supports the full Elasticsearch query_string syntax.
Searchable fields: `contents` (passage text, default), `title`, `section`.

### Operators

| Syntax | Meaning | Example |
|--------|---------|---------|
| word1 word2 | OR — either word | lightsaber combat |
| +word | Required — must appear | +lightsaber +combat |
| -word | Excluded — must not appear | lightsaber -Dooku |
| AND / OR / NOT | Boolean operators | Nolan AND Inception |
| "words" | Exact phrase | "lightsaber combat" |
| (a OR b) AND c | Grouping | (Skywalker OR Solo) AND Jedi |
| word~ | Fuzzy match (typo tolerance) | Mandalorien~ |
| "words"~N | Proximity (within N words) | "Force lightning"~3 |
| wild* | Prefix / wildcard | lightsab* |

### Field-specific search

| Syntax | Meaning | Example |
|--------|---------|---------|
| title:"words" | Search the title field | title:"Anthony Mackie" |
| section:History | Search the section field | section:Filmography |
| title:word^N | Boost field matches | title:Nolan^3 AND Inception |

### Tips

- Use +required for precision: `+Mandalorian +armor +beskar`
- Use "exact phrase" for names: `"Return of the Jedi"`
- Use title: for entity lookups: `title:"Darth Vader"`
- Use fuzzy matching for uncertain spelling: `Kashyyyk~1`
- Separate sub-questions into newline-delimited queries for broader recall"""

_SEARCH_DESCRIPTION = f"""\
Search the R2C2 corpus using hybrid retrieval (BM25 + Dense + KG).

Combines three retrieval strategies via Reciprocal Rank Fusion, then reranks
with a cross-encoder for maximum recall and precision:
  1. BM25 keyword matching (Elasticsearch)
  2. Dense semantic similarity (nomic-embed-text-v1.5 + FAISS)
  3. Knowledge graph entity co-occurrence (NPMI)

Multiple newline-separated queries broaden BM25 and KG recall. The dense leg
uses the first query line for semantic embedding.

Degrades gracefully: if any leg is unavailable, the remaining legs still work.
The `retrieval_mode` field in results shows which legs contributed.
{_ES_QUERY_SYNTAX}"""


# Server & Lifespan


@dataclass
class ServerContext:
    """Lifespan context — always loads all resources for hybrid mode."""

    es: Any = None
    reranker: Any = None
    # Enriched FAISS (primary)
    faiss_index: Any = None
    id_map: Any = None  # numpy array (mmap)
    # Plain FAISS (optional, for ablation)
    faiss_index_plain: Any = None
    id_map_plain: Any = None
    # Shared
    embed_model: Any = None
    query_prefix: str = ""
    # KG
    kg_graph: dict[str, Any] = field(default_factory=dict)


# Pre-load heavy resources at module level (before FastMCP starts).
# This avoids blocking the async event loop during lifespan and enables
# stateless_http mode (each request gets a lightweight lifespan).


def _preload_resources() -> ServerContext:
    """Load FAISS, KG, embedding model, cross-encoder at import time."""
    import numpy as np

    ctx = ServerContext()

    # Cross-encoder
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder: %s (CPU)", RERANKER_MODEL)
    ctx.reranker = CrossEncoder(RERANKER_MODEL, device="cpu")

    # FAISS enriched index
    import faiss

    index_path = os.path.join(DENSE_INDEX_DIR, "faiss.index")
    if os.path.exists(index_path):
        logger.info("Loading enriched FAISS index from %s", index_path)
        ctx.faiss_index = faiss.read_index(index_path)
        logger.info("FAISS (enriched): %d vectors", ctx.faiss_index.ntotal)
        id_map_path = os.path.join(DENSE_INDEX_DIR, "id_map.npy")
        ctx.id_map = np.load(id_map_path, mmap_mode="r")
        logger.info("ID map (enriched): %d entries", len(ctx.id_map))
    else:
        logger.warning("FAISS index not found at %s", index_path)

    # Plain FAISS (optional)
    if LOAD_PLAIN_FAISS:
        plain_path = os.path.join(DENSE_INDEX_DIR_PLAIN, "faiss.index")
        if os.path.exists(plain_path):
            ctx.faiss_index_plain = faiss.read_index(plain_path)
            ctx.id_map_plain = np.load(
                os.path.join(DENSE_INDEX_DIR_PLAIN, "id_map.npy"), mmap_mode="r"
            )

    # Embedding model (nomic)
    from sentence_transformers import SentenceTransformer

    meta_path = os.path.join(DENSE_INDEX_DIR, "build_meta.json")
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    try:
        with open(meta_path, encoding="utf-8") as f:
            build_meta = json.load(f)
        ctx.query_prefix = build_meta.get("query_prefix", "search_query: ")
        model_name = build_meta.get("model", model_name)
    except (FileNotFoundError, json.JSONDecodeError):
        ctx.query_prefix = "search_query: "

    logger.info("Loading embedding model: %s (CPU)", model_name)
    ctx.embed_model = SentenceTransformer(
        model_name, device="cpu", trust_remote_code=True
    )
    logger.info(
        "Embedding model loaded (dim=%d)",
        ctx.embed_model.get_sentence_embedding_dimension(),
    )

    # KG graph
    if os.path.exists(KG_PATH):
        logger.info("Loading KG graph from %s", KG_PATH)
        with open(KG_PATH, encoding="utf-8") as f:
            raw_kg = json.load(f)
        adj: dict[str, list[tuple[str, float]]] = {}
        for edge in raw_kg.get("edges", []):
            e1, e2, npmi = edge["e1"].lower(), edge["e2"].lower(), edge["npmi"]
            adj.setdefault(e1, []).append((e2, npmi))
            adj.setdefault(e2, []).append((e1, npmi))
        for entity in adj:
            adj[entity].sort(key=lambda x: x[1], reverse=True)
        ctx.kg_graph = {
            "adj": adj,
            "meta": {
                "total_entities": raw_kg.get("total_entities", 0),
                "total_edges": raw_kg.get("total_edges", 0),
            },
        }
        del raw_kg
        logger.info(
            "KG loaded: %d entities, %d edges",
            len(adj),
            ctx.kg_graph["meta"]["total_edges"],
        )
    else:
        logger.warning("KG graph not found at %s", KG_PATH)

    return ctx


# Pre-load at import time (runs once, before server starts)
_SHARED_CTX = _preload_resources()

# The ES client lives at module scope so it is opened once per server process
# and not once per HTTP request. With ``stateless_http=False`` below, this
# avoids a reconnect, an index check and a count on every tool call.
_GLOBAL_ES: Any | None = None
_SESSION_COUNT = 0

# These upstream loggers emit INFO-level access lines on every /mcp
# round-trip, about 500 lines per question, which hides the server's own logs.
for _noisy in (
    "elasticsearch",
    "elastic_transport",
    "mcp.server.lowlevel",
    "mcp.server.streamable_http",
    "httpx",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Persistent lifespan — open ES once, reference pre-loaded resources.

    With ``stateless_http=False`` this runs once per server process. The
    global ``_GLOBAL_ES`` client stays alive for the server's lifetime so
    retrieval tools avoid per-request reconnect overhead.
    """
    global _GLOBAL_ES, _SESSION_COUNT

    from elasticsearch import AsyncElasticsearch

    logger.info("Connecting to Elasticsearch at %s (persistent)", ELASTICSEARCH_URL)
    _GLOBAL_ES = AsyncElasticsearch(ELASTICSEARCH_URL, request_timeout=120)

    if not await _GLOBAL_ES.indices.exists(index=ES_INDEX_NER):
        raise RuntimeError(f"ES index '{ES_INDEX_NER}' not found")

    total = (await _GLOBAL_ES.count(index=ES_INDEX_NER))["count"]

    # Copy pre-loaded resources + add the persistent ES connection
    ctx = ServerContext(
        es=_GLOBAL_ES,
        reranker=_SHARED_CTX.reranker,
        faiss_index=_SHARED_CTX.faiss_index,
        id_map=_SHARED_CTX.id_map,
        faiss_index_plain=_SHARED_CTX.faiss_index_plain,
        id_map_plain=_SHARED_CTX.id_map_plain,
        embed_model=_SHARED_CTX.embed_model,
        query_prefix=_SHARED_CTX.query_prefix,
        kg_graph=_SHARED_CTX.kg_graph,
    )

    _SESSION_COUNT += 1
    logger.info(
        "Server ready: %d passages, hybrid mode (persistent session #%d)",
        total,
        _SESSION_COUNT,
    )

    # FastMCP invokes ``app_lifespan`` once per session, not once per process,
    # even with ``stateless_http=False``. Closing the shared ``_GLOBAL_ES``
    # here would fail every concurrent tool call with ``RuntimeError: Session
    # is closed`` as soon as one session ends. The process owns the client and
    # closes it at interpreter shutdown, through the atexit hook registered
    # below.
    yield ctx


import atexit  # noqa: E402


def _atexit_close_es() -> None:
    """Close ``_GLOBAL_ES`` exactly once at process shutdown.

    atexit runs outside an event loop. If a loop is still running (e.g.
    uvicorn graceful shutdown path) we schedule the close as a future;
    otherwise we spin up a short-lived loop via ``asyncio.run``.
    """
    global _GLOBAL_ES
    if _GLOBAL_ES is None:
        return
    client = _GLOBAL_ES
    _GLOBAL_ES = None
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(client.close())
            else:
                loop.run_until_complete(client.close())
        except RuntimeError:
            asyncio.run(client.close())
        logger.info("Closed _GLOBAL_ES at atexit")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to close _GLOBAL_ES at atexit: %s", exc)


atexit.register(_atexit_close_es)


mcp_server = FastMCP(
    "R2C2-Hybrid",
    # Stateless mode replays the lifespan and reconnects to ES on every
    # request, which pushes median search latency (~16 s) towards the 30 s
    # tool timeout.
    stateless_http=False,
    json_response=True,
    lifespan=app_lifespan,
    host=MCP_HOST,
    port=MCP_PORT,
)


# Internal search helpers


async def _bm25_search(
    es: Any,
    query_list: list[str],
    corpus: str,
    per_query: int,
    es_index: str,
    exclude_set: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Multi-query BM25 search. Returns {passage_id: {source, bm25_score}}.

    Args:
        exclude_set: optional frozenset of passage_ids to subtract from the
            index (ensemble feed-forward). Added as an ES ``must_not.ids``
            clause; ES backfills to ``per_query`` post-exclusion so the
            visible pool size is preserved.
    """
    candidates: dict[str, dict[str, Any]] = {}
    for q in query_list:
        es_query: dict[str, Any] = {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": q,
                            "fields": ["contents", "title^2", "section^1.5"],
                            "default_operator": "OR",
                        }
                    }
                ],
            }
        }
        if corpus != "all":
            es_query["bool"]["filter"] = [{"term": {"source": corpus}}]
        if exclude_set:
            es_query["bool"]["must_not"] = [{"ids": {"values": list(exclude_set)}}]

        resp = await es.search(
            index=es_index,
            query=es_query,
            size=per_query,
            _source=[
                "contents",
                "title",
                "section",
                "source",
                "doc_id",
                "entities",
                "passage_index",
                "word_count",
            ],
        )
        for hit in resp["hits"]["hits"]:
            docid = hit["_id"]
            score = hit["_score"]
            if docid in candidates:
                candidates[docid]["bm25_score"] = max(
                    candidates[docid]["bm25_score"], score
                )
            else:
                candidates[docid] = {"source": hit["_source"], "bm25_score": score}

    return candidates


async def _dense_search_raw(
    ctx: ServerContext,
    query_list: list[str],
    top_k: int,
    corpus: str = "all",
    exclude_set: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Dense vector search with multi-query support.

    Runs FAISS search for each query, keeps max cosine per passage across queries.

    Args:
        exclude_set: optional frozenset of passage_ids to subtract from the
            index. FAISS ``IndexFlatIP`` has no native pre-filter, so we
            overfetch ``fetch_k + |exclude_set|`` and drop excluded pids
            during merge. On >10% underfill vs ``top_k`` we retry once with
            ``fetch_k *= 2`` (capped at ``4*top_k + |exclude|`` or
            ``faiss_idx.ntotal``). The underfill counter is logged in
            ``search_passages`` for observability.
    """
    import numpy as np

    faiss_idx = ctx.faiss_index
    id_map = ctx.id_map
    es_index = ES_INDEX_NER

    # Encode all queries at once
    full_queries = [
        (ctx.query_prefix + q if ctx.query_prefix else q) for q in query_list
    ]
    embeddings = ctx.embed_model.encode(
        full_queries,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vecs = np.array(embeddings, dtype=np.float32)

    # Search for each query, merge with max score per passage.
    # Overfetch by |exclude_set| so the post-exclusion pool still hits top_k.
    base_fetch = top_k * 3 if corpus != "all" else top_k
    n_excl = len(exclude_set) if exclude_set else 0
    fetch_k = base_fetch + n_excl

    def _faiss_round(fetch_k_local: int) -> dict[str, float]:
        acc: dict[str, float] = {}
        for i in range(len(query_list)):
            scores, indices = faiss_idx.search(query_vecs[i : i + 1], fetch_k_local)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(id_map):
                    continue
                pid = str(id_map[idx])
                if exclude_set and pid in exclude_set:
                    continue
                acc[pid] = max(acc.get(pid, -1.0), float(score))
        return acc

    max_scores = _faiss_round(fetch_k)
    # Underfill escalation: if post-exclusion pool <90% of top_k, retry
    # once with a larger fetch_k. Capped to avoid runaway on sparse queries.
    underfill_retries = 0
    if exclude_set and len(max_scores) < 0.9 * top_k:
        cap = min(4 * top_k + n_excl, faiss_idx.ntotal)
        escalated = min(fetch_k * 2, cap)
        if escalated > fetch_k:
            underfill_retries = 1
            max_scores = _faiss_round(escalated)
    # The return shape is {pid: {...}} and has no place for the counter, so
    # it is kept as a function attribute for ``search_passages`` to log.
    _dense_search_raw._last_underfill_retries = underfill_retries  # type: ignore[attr-defined]

    id_score_pairs = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    if not id_score_pairs:
        return {}

    passage_ids = [pid for pid, _ in id_score_pairs]
    resp = await ctx.es.mget(index=es_index, ids=passage_ids)

    doc_lookup: dict[str, dict[str, Any]] = {}
    for doc in resp["docs"]:
        if doc.get("found", False):
            doc_lookup[doc["_id"]] = doc["_source"]

    candidates: dict[str, dict[str, Any]] = {}
    for passage_id, score in id_score_pairs:
        if passage_id not in doc_lookup:
            continue
        src = doc_lookup[passage_id]
        if corpus != "all" and src.get("source", "unknown") != corpus:
            continue
        candidates[passage_id] = {"source": src, "dense_score": score}
        if len(candidates) >= top_k:
            break

    return candidates


def _build_passage_entry(
    passage_id: str,
    src: dict[str, Any],
    rank: int,
    score: float = 0.0,
    retrieval_mode: str = "hybrid",
    truncate: bool = True,
    truncation_type: str = "soft",
    include_title: bool = False,
    max_len: int | None = None,
) -> dict[str, Any]:
    """Build a passage result entry.

    Args:
        passage_id: ES document _id.
        src: ES _source dict.
        rank: 1-based rank in results.
        score: Final score (CE > RRF > BM25).
        retrieval_mode: Which legs contributed (hybrid, bm25+dense, bm25_only, etc).
        truncate: Whether to truncate text to ``max_len`` (falls back to
            ``MAX_TEXT_LENGTH`` when ``max_len`` is None).
        truncation_type: "soft" (default) truncates at the next sentence
            boundary after the cap; "hard" chops at the cap.
        include_title: If True, prepend "Title > Section — " to text so the
            passage is self-contained (useful when title carries key info).
        max_len: Override the truncation cap. ``search_passages`` passes
            ``SEARCH_MAX_TEXT_LENGTH`` (2000) for fuller hits while
            ``get_passage`` / ``extract_from_passage`` use the default
            ``MAX_TEXT_LENGTH`` (800).
    """
    text = src.get("contents", "")
    entities = src.get("entities", [])
    cap = max_len if max_len is not None else MAX_TEXT_LENGTH
    if truncate and len(text) > cap:
        if truncation_type == "soft":
            text = _soft_truncate(text, cap)
        else:
            text = text[:cap] + "..."
    if include_title:
        title = src.get("title", "")
        section = src.get("section", "")
        prefix = " > ".join(s for s in [title, section] if s)
        if prefix:
            text = f"{prefix} — {text}"
    entry: dict[str, Any] = {
        "passage_id": passage_id,
        "doc_id": src.get("doc_id", ""),
        "title": src.get("title", ""),
        "section": src.get("section", ""),
        "corpus_source": src.get("source", "unknown"),
        "text": text,
        "rank": rank,
        "score": round(float(score), 4),
        "retrieval_mode": retrieval_mode,
        "word_count": src.get("word_count", len(src.get("contents", "").split())),
    }
    if entities:
        entry["entities"] = entities
    return entry


# Fusion & KG helpers


def _rrf_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = RRF_K,
) -> dict[str, float]:
    """N-way Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (pid, _) in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _retrieval_mode(pid: str, bm25: dict, dense: dict, kg: dict) -> str:
    """Determine which legs contributed (spec-compliant values)."""
    has_bm25 = pid in bm25
    has_dense = pid in dense
    has_kg = pid in kg
    if has_bm25 and has_dense and has_kg:
        return "hybrid"
    if has_bm25 and has_dense:
        return "bm25+dense"
    if has_bm25 and has_kg:
        return "bm25+kg"
    if has_dense and has_kg:
        return "dense+kg"
    if has_bm25:
        return "bm25_only"
    if has_dense:
        return "dense_only"
    return "unknown"


async def _kg_extract_entities(
    es: Any,
    query: str,
    max_entities: int = 5,
) -> list[tuple[str, float]]:
    """Extract entities from query via fuzzy ES match on entities field."""
    resp = await es.search(
        index=ES_INDEX_NER,
        query={
            "multi_match": {"query": query, "fields": ["entities"], "fuzziness": "AUTO"}
        },
        size=max_entities,
        _source=["entities"],
    )
    extracted: list[tuple[str, float]] = []
    seen: set[str] = set()
    for hit in resp["hits"]["hits"]:
        if hit["_score"] < KG_ENTITY_SCORE_THRESHOLD:
            continue
        for entity in hit["_source"].get("entities", []):
            normed = entity.lower().strip()
            if normed not in seen:
                seen.add(normed)
                extracted.append((normed, hit["_score"]))
    return extracted[:max_entities]


async def _kg_search(
    ctx: ServerContext,
    query_list: list[str],
    top_k: int,
    corpus: str = "all",
    exclude_set: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """KG entity co-occurrence search.

    Three-tier fallback:
      - No entities pass confidence gate -> return empty (skip KG leg)
      - <3 neighbors -> use query entities only (no expansion)
      - Normal -> full 1-hop expansion (2-hop if multi-query)
    """
    adj = ctx.kg_graph.get("adj", {})
    if not adj:
        return {}

    # Step 1: Entity extraction from all queries (union)
    all_entities: dict[str, float] = {}
    for q in query_list:
        for entity, _score in await _kg_extract_entities(ctx.es, q):
            if entity not in all_entities:
                all_entities[entity] = 1.0

    if not all_entities:
        return {}  # Tier 1: no entities pass confidence gate

    # Step 2: KG neighbor expansion
    expanded: dict[str, float] = dict(all_entities)
    total_neighbors = 0
    for entity in list(all_entities):
        for neighbor, npmi in adj.get(entity, [])[:10]:
            if neighbor not in expanded:
                expanded[neighbor] = npmi
                total_neighbors += 1
            else:
                expanded[neighbor] = max(expanded[neighbor], npmi)

    # 2-hop for multi-query (tighter threshold)
    if len(query_list) > 1 and total_neighbors >= 3:
        hop1 = set(expanded) - set(all_entities)
        for entity in hop1:
            for neighbor, npmi in adj.get(entity, [])[:10]:
                if npmi < KG_NPMI_2HOP_THRESHOLD:
                    break
                if neighbor not in expanded:
                    expanded[neighbor] = npmi
                if len(expanded) >= 20:
                    break
            if len(expanded) >= 20:
                break

    if total_neighbors < 3:
        expanded = dict(all_entities)  # Tier 2: too few neighbors

    # Step 3: Retrieve passages via entity term filters, scored by sum-of-NPMIs
    should_clauses = [{"term": {"entities": e}} for e in list(expanded)[:50]]
    es_query: dict[str, Any] = {
        "bool": {"should": should_clauses, "minimum_should_match": 1}
    }
    if corpus != "all":
        es_query["bool"]["filter"] = [{"term": {"source": corpus}}]
    if exclude_set:
        es_query["bool"]["must_not"] = [{"ids": {"values": list(exclude_set)}}]

    resp = await ctx.es.search(
        index=ES_INDEX_NER,
        query=es_query,
        size=top_k,
        _source=[
            "contents",
            "title",
            "section",
            "source",
            "doc_id",
            "entities",
            "passage_index",
            "word_count",
        ],
    )

    candidates: dict[str, dict[str, Any]] = {}
    for hit in resp["hits"]["hits"]:
        pid = hit["_id"]
        src = hit["_source"]
        passage_entities = {e.lower() for e in src.get("entities", [])}
        kg_score = sum(expanded.get(e, 0.0) for e in passage_entities if e in expanded)
        candidates[pid] = {"source": src, "kg_score": kg_score}

    return candidates


# MCP Tools


@mcp_server.tool(description=_SEARCH_DESCRIPTION)
async def search_passages(
    queries: str,
    corpus: str = "all",
    max_results: int = DEFAULT_MAX_RESULTS,
    exclude_ids: str = "",
    ctx: Context = None,
) -> str:
    """Search the R2C2 corpus using hybrid retrieval (BM25 + Dense + KG).

    Args:
        queries: Search query (or multiple newline-separated queries for
                 broader recall). Dense and KG legs use the first query only.
        corpus: Filter by source — "all", "wikipedia", or "wookieepedia".
        max_results: Number of top results to return (1-100, default 10).
        exclude_ids: Comma-separated passage_ids to subtract from the
                 index (ensemble feed-forward). LLM-invisible — injected
                 by the orchestrator's ExcludingMCPWrapper. Excluded pids
                 behave as if they never existed in any leg's candidate
                 pool; the top-``max_results`` contract is preserved via
                 ES ``must_not`` (BM25 / KG) and FAISS overfetch + mask
                 (dense).

    Returns:
        JSON with ranked passages including passage_id, doc_id, title,
        section, corpus_source, text (soft-truncated at 2000 chars), score,
        retrieval_mode, word_count, and rank.
    """
    if not queries or not queries.strip():
        raise ValueError("Queries must not be empty")

    if corpus not in ("all", "wikipedia", "wookieepedia"):
        raise ValueError(
            f"Invalid corpus: {corpus}. Must be 'all', 'wikipedia', or 'wookieepedia'"
        )

    # Parse exclusion list once into a frozenset (per-request, never
    # stashed on shared state — concurrency-safe with multiple qids on
    # the same server instance).
    exclude_set: frozenset[str] = frozenset(
        p.strip() for p in (exclude_ids or "").split(",") if p.strip()
    )

    # qid is not part of the MCP schema, so operators cross-reference this
    # entry log with driver logs by (query, corpus, timestamp) ordering.
    logger.info(
        "mcp search_passages.enter: query=%r corpus=%s max_results=%d exclude=%d",
        queries.splitlines()[0][:80] if queries else "",
        corpus,
        max_results,
        len(exclude_set),
    )

    query_list = [q.strip() for q in queries.strip().splitlines() if q.strip()]
    if not query_list:
        raise ValueError("At least one non-empty query is required")

    capped = min(max(max_results, 1), HARD_MAX_RESULTS)
    main_query = query_list[0]
    lifespan = ctx.request_context.lifespan_context
    t0 = time.time()

    # 1. Run the three legs in parallel with graceful degradation.
    # If the shared ES client is found closed mid-request ("Session is
    # closed"), reopen it once and retry instead of returning an empty leg.
    async def _reopen_es_if_closed(exc: Exception) -> bool:
        global _GLOBAL_ES
        msg = str(exc).lower()
        if "session is closed" not in msg and "closedresourceerror" not in msg:
            return False
        from elasticsearch import AsyncElasticsearch

        try:
            _GLOBAL_ES = AsyncElasticsearch(ELASTICSEARCH_URL, request_timeout=120)
            lifespan.es = _GLOBAL_ES
            logger.warning("_GLOBAL_ES reopened after detected close")
            return True
        except Exception as reopen_exc:  # noqa: BLE001
            logger.exception("_GLOBAL_ES reopen failed: %s", reopen_exc)
            return False

    async def _safe_bm25() -> dict[str, dict[str, Any]]:
        for attempt in range(2):
            try:
                return await _bm25_search(
                    lifespan.es,
                    query_list,
                    corpus,
                    BM25_CANDIDATES,
                    ES_INDEX_NER,
                    exclude_set=exclude_set or None,
                )
            except Exception as exc:  # noqa: BLE001
                if attempt == 0 and await _reopen_es_if_closed(exc):
                    continue
                logger.exception("BM25 leg failed")
                return {}
        return {}

    async def _safe_dense() -> dict[str, dict[str, Any]]:
        try:
            return await _dense_search_raw(
                lifespan,
                query_list,
                DENSE_CANDIDATES,
                corpus,
                exclude_set=exclude_set or None,
            )
        except Exception:
            logger.exception("Dense leg failed")
            return {}

    async def _safe_kg() -> dict[str, dict[str, Any]]:
        try:
            return await _kg_search(
                lifespan,
                query_list,
                KG_CANDIDATES,
                corpus,
                exclude_set=exclude_set or None,
            )
        except Exception:
            logger.exception("KG leg failed")
            return {}

    bm25_cands, dense_cands, kg_cands = await asyncio.gather(
        _safe_bm25(),
        _safe_dense(),
        _safe_kg(),
    )

    # 2. 3-way RRF fusion
    bm25_ranked = [
        (pid, data["bm25_score"])
        for pid, data in sorted(
            bm25_cands.items(), key=lambda x: x[1]["bm25_score"], reverse=True
        )
    ]
    dense_ranked = [
        (pid, data["dense_score"])
        for pid, data in sorted(
            dense_cands.items(), key=lambda x: x[1]["dense_score"], reverse=True
        )
    ]
    kg_ranked = [
        (pid, data.get("kg_score", 0.0))
        for pid, data in sorted(
            kg_cands.items(), key=lambda x: x[1].get("kg_score", 0.0), reverse=True
        )
    ]

    rrf_scores = _rrf_fusion(
        [(pid, s) for pid, s in bm25_ranked],
        [(pid, s) for pid, s in dense_ranked],
        [(pid, s) for pid, s in kg_ranked],
    )

    # Merge source data from all legs
    all_cands: dict[str, dict[str, Any]] = {}
    for pid in rrf_scores:
        if pid in bm25_cands:
            all_cands[pid] = dict(bm25_cands[pid])
        if pid in dense_cands:
            if pid not in all_cands:
                all_cands[pid] = dict(dense_cands[pid])
            else:
                all_cands[pid]["dense_score"] = dense_cands[pid]["dense_score"]
        if pid in kg_cands:
            if pid not in all_cands:
                all_cands[pid] = dict(kg_cands[pid])
            else:
                all_cands[pid]["kg_score"] = kg_cands[pid].get("kg_score", 0.0)

    # 3. Top RERANK_POOL candidates for CE reranking
    top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:RERANK_POOL]
    top_pids = [pid for pid, _ in top_rrf]

    if top_pids and lifespan.reranker is not None:
        pairs = [
            (main_query, all_cands[pid]["source"].get("contents", ""))
            for pid in top_pids
        ]
        ce_scores = lifespan.reranker.predict(pairs)
        reranked = sorted(
            zip(top_pids, ce_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
    else:
        # Fallback: use RRF scores directly
        reranked = [(pid, rrf_scores[pid]) for pid in top_pids]

    # 4. Build passage entries
    # Text is soft-truncated at SEARCH_MAX_TEXT_LENGTH (2000 chars).
    # Untruncated hits cost about 10K tokens per search and overflow the vLLM
    # context after a dozen searches. 2000 chars is about 600 tokens per hit,
    # so a 10-hit search returns about 6K tokens; the driver's preflight
    # budget check catches the remaining runaway cases.
    passages: list[dict[str, Any]] = []
    for rank, (pid, score) in enumerate(reranked[:capped], start=1):
        mode = _retrieval_mode(pid, bm25_cands, dense_cands, kg_cands)
        passages.append(
            _build_passage_entry(
                pid,
                all_cands[pid]["source"],
                rank,
                score=float(score),
                retrieval_mode=mode,
                truncate=True,
                truncation_type="soft",
                max_len=SEARCH_MAX_TEXT_LENGTH,
            )
        )

    elapsed_ms = round((time.time() - t0) * 1000)

    dense_underfill_retries = getattr(_dense_search_raw, "_last_underfill_retries", 0)
    logger.info(
        "SEARCH queries=%d results=%d time=%dms bm25=%d dense=%d kg=%d "
        "exclude=%d dense_underfetch_retries=%d main_query=%r",
        len(query_list),
        len(passages),
        elapsed_ms,
        len(bm25_cands),
        len(dense_cands),
        len(kg_cands),
        len(exclude_set),
        dense_underfill_retries,
        main_query,
    )

    result: dict[str, Any] = {
        "main_query": main_query,
        "queries": query_list,
        "corpus": corpus,
        "method": "hybrid (BM25+Dense+KG RRF + CE rerank)",
        "total_results": len(passages),
        "search_time_ms": elapsed_ms,
        "bm25_candidates": len(bm25_cands),
        "dense_candidates": len(dense_cands),
        "kg_candidates": len(kg_cands),
        "excluded_count": len(exclude_set),
        "dense_underfetch_retries": dense_underfill_retries,
        "passages": passages,
    }

    return json.dumps(result, indent=2)


def _parse_passage_id(pid: str) -> tuple[str, int] | None:
    """Parse a passage_id of the form '{doc_id}__{NNNNN}'.

    Returns `(doc_id, chunk_idx)` or None if the format is invalid.
    """
    if not pid or "__" not in pid:
        return None
    doc_id, chunk_str = pid.rsplit("__", 1)
    try:
        return doc_id, int(chunk_str)
    except ValueError:
        return None


@mcp_server.tool()
async def get_passage(
    passage_ids: str,
    window: int = 0,
    include_title: bool = False,
    ctx: Context = None,
) -> str:
    """Fetch full passage content by ID(s), optionally with neighbor context.

    Args:
        passage_ids: Comma-separated passage IDs (max 5).
        window: Number of neighboring passages on each side of every center
            passage (0..2 inclusive, default 0). Neighbors are same-doc
            passages clamped at doc boundary. When window>0, each center
            passage expands into up to ``1 + 2*window`` entries.
        include_title: If true, prepend "Title > Section — " to the text.
            Applies only to center passages.

    Returns:
        JSON with the requested passages. Each entry carries
        ``is_center`` and ``is_neighbor`` flags. Missing pids are reported
        inline as ``{passage_id, error}`` records.
    """
    if not passage_ids or not passage_ids.strip():
        raise ValueError("Passage IDs must not be empty")

    ids = [pid.strip() for pid in passage_ids.split(",") if pid.strip()]
    if len(ids) > 5:
        raise ValueError("Maximum 5 passage IDs per request")

    logger.info(
        "mcp get_passage.enter: ids=%s window=%d",
        ",".join(ids),
        window,
    )

    window = max(0, min(int(window), 2))

    # Build the id-set we need to mget: every center + its neighbors.
    # Preserve order via an OrderedDict-style list.
    wanted: list[str] = []
    seen: set[str] = set()
    center_set: set[str] = set()
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        center_set.add(pid)
        wanted.append(pid)
        if window == 0:
            continue
        parsed = _parse_passage_id(pid)
        if parsed is None:
            continue
        doc_id, chunk_idx = parsed
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            nb = chunk_idx + offset
            if nb < 1:
                continue  # left boundary clamp (passages are 1-indexed)
            nb_id = f"{doc_id}__{nb:05d}"
            if nb_id not in seen:
                seen.add(nb_id)
                wanted.append(nb_id)

    lifespan = ctx.request_context.lifespan_context
    resp = await lifespan.es.mget(index=ES_INDEX_NER, ids=wanted)

    # Build a lookup doc for each requested id (missing ones become error rows)
    found_by_id: dict[str, dict[str, Any]] = {}
    for doc in resp["docs"]:
        if doc.get("found", False):
            found_by_id[doc["_id"]] = doc["_source"]

    passages: list[dict[str, Any]] = []
    rank = 1
    for pid in wanted:
        is_center = pid in center_set
        src = found_by_id.get(pid)
        if src is None:
            # Only emit "not found" rows for explicit centers — missing
            # neighbors are silent (they fall outside the doc, which is
            # expected at boundaries).
            if is_center:
                passages.append(
                    {
                        "passage_id": pid,
                        "error": "Not found",
                        "is_center": True,
                        "is_neighbor": False,
                    }
                )
            continue

        entry = _build_passage_entry(
            pid,
            src,
            rank,
            score=0.0,
            retrieval_mode="fetch",
            truncate=False,
            include_title=include_title and is_center,
        )
        entry["is_center"] = is_center
        entry["is_neighbor"] = not is_center
        passages.append(entry)
        rank += 1

    logger.info(
        "FETCH passage_ids=%s window=%d found=%d",
        ",".join(ids),
        window,
        len([p for p in passages if "error" not in p]),
    )

    return json.dumps(
        {
            "passages": passages,
            "count": len(passages),
            "window": window,
        },
        indent=2,
    )


async def _fetch_passage_text(lifespan: Any, passage_id: str) -> str | None:
    """Fetch only the ``contents`` field of a single passage. Returns None if absent."""
    try:
        resp = await lifespan.es.get(
            index=ES_INDEX_NER,
            id=passage_id,
            _source=["contents"],
        )
    except Exception:
        return None
    if not resp.get("found", False):
        return None
    return resp["_source"].get("contents", "") or ""


async def _call_nli(
    client: httpx.AsyncClient,
    sentence: str,
    predicate: str,
) -> float | None:
    """Call the NLI entailment server. Returns the score or None on error."""
    try:
        resp = await client.post(
            f"{NLI_URL.rstrip('/')}/check_entailment",
            json={"passage_text": sentence, "claim_text": predicate},
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("score", 0.0))
    except (httpx.HTTPError, ValueError, KeyError) as e:  # noqa: BLE001
        logger.warning("NLI call failed: %s", e)
        return None


@mcp_server.tool()
async def extract_from_passage(
    passage_id: str,
    predicate: str,
    max_spans: int = 5,
    ctx: Context = None,
) -> str:
    """Extract short spans from a single passage matching a natural-language predicate.

    Pipeline:
      1. Fetch the passage.
      2. Sentence-split (simple regex boundary).
      3. BM25 score each sentence against the predicate.
      4. If the top-1 BM25 score is below 0.3 and an NLI server is reachable,
         re-rank the top-10 BM25 candidates by NLI entailment against the
         predicate as the claim.
      5. Return top-N sentences with scores.

    Read-only. No mutation. Used by the agent loop to locate the exact
    supporting sentence inside a passage before calling ``record_evidence``.

    Args:
        passage_id: Passage ID (required).
        predicate: Natural-language predicate, 3..200 chars (required).
            Example: "who piloted Red Five".
        max_spans: Maximum spans to return (default 5, cap 10).

    Returns:
        JSON with ``matches`` (sorted top-N sentences), ``total_sentences_scanned``,
        and ``too_broad`` (true if >50% of sentences score > 0.2 BM25 — the
        predicate is too general to be useful).
    """
    if not passage_id or not passage_id.strip():
        raise ValueError("passage_id must not be empty")
    if predicate is None:
        raise ValueError("predicate is required")
    predicate = predicate.strip()
    if not (3 <= len(predicate) <= 200):
        raise ValueError("predicate must be 3..200 characters")

    max_spans = max(1, min(int(max_spans), 10))
    lifespan = ctx.request_context.lifespan_context
    t0 = time.time()

    logger.info(
        "mcp extract_from_passage.enter: pid=%s pred=%r max_spans=%d",
        passage_id,
        predicate[:80],
        max_spans,
    )

    text = await _fetch_passage_text(lifespan, passage_id.strip())
    if text is None:
        return json.dumps(
            {
                "passage_id": passage_id,
                "error": "Passage not found",
                "matches": [],
                "total_sentences_scanned": 0,
                "too_broad": False,
            },
            indent=2,
        )

    sentences = split_sentences_simple(text)
    if not sentences:
        return json.dumps(
            {
                "passage_id": passage_id,
                "matches": [],
                "total_sentences_scanned": 0,
                "too_broad": False,
            },
            indent=2,
        )

    bm25_scores = bm25_score_sentences(sentences, predicate)

    # "too_broad" signal: more than half the sentences score above a low floor.
    # Uses BM25 scores pre-NLI — the NLI fallback doesn't change the signal.
    nonzero = [s for s in bm25_scores if s > EXTRACT_TOO_BROAD_SCORE]
    too_broad = len(nonzero) > len(sentences) / 2

    top_bm25 = max(bm25_scores) if bm25_scores else 0.0
    used_nli = False

    # NLI fallback when BM25 alone is unconvincing.
    if top_bm25 < EXTRACT_BM25_FALLBACK_THRESHOLD:
        # Top-K BM25 candidates carry into the NLI pass
        candidate_indices = sorted(
            range(len(sentences)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:EXTRACT_NLI_TOPK]

        nli_scores: dict[int, float] = {}
        # Connect and write time out at 30 s so a dead service fails fast;
        # read=None waits without limit for a healthy but slow response.
        _nli_http_timeout = httpx.Timeout(
            connect=NLI_CONNECT_TIMEOUT_SECONDS,
            read=None,
            write=NLI_CONNECT_TIMEOUT_SECONDS,
            pool=60.0,
        )
        try:
            async with httpx.AsyncClient(timeout=_nli_http_timeout) as client:
                for idx in candidate_indices:
                    s = await _call_nli(client, sentences[idx], predicate)
                    if s is None:
                        # Abort NLI fallback on the first error — stay on BM25
                        nli_scores.clear()
                        break
                    nli_scores[idx] = s
        except (httpx.HTTPError, OSError) as e:  # noqa: BLE001
            logger.warning("NLI fallback skipped: %s", e)
            nli_scores = {}

        if nli_scores:
            used_nli = True
            # Re-rank: NLI where we have it, BM25 elsewhere (but NLI-scored
            # sentences always outrank un-NLI'd ones because we only scored
            # the top-K BM25, which is the plausible set).
            scores = [nli_scores.get(i, bm25_scores[i]) for i in range(len(sentences))]
        else:
            scores = bm25_scores
    else:
        scores = bm25_scores

    matches = rank_matches(sentences, scores, max_spans)

    elapsed_ms = round((time.time() - t0) * 1000)
    logger.info(
        "EXTRACT_FROM_PASSAGE pid=%s pred=%r sentences=%d top=%.3f nli=%s "
        "matches=%d time=%dms",
        passage_id,
        predicate,
        len(sentences),
        top_bm25,
        used_nli,
        len(matches),
        elapsed_ms,
    )

    return json.dumps(
        {
            "passage_id": passage_id,
            "predicate": predicate,
            "matches": matches,
            "total_sentences_scanned": len(sentences),
            "too_broad": too_broad,
            "used_nli_fallback": used_nli,
            "search_time_ms": elapsed_ms,
        },
        indent=2,
    )


@mcp_server.tool()
async def search_entities(
    query: str,
    entity_type: str = "",
    max_results: int = 20,
    ctx: Context = None,
) -> str:
    """Search for named entities in the corpus.

    Uses fuzzy matching on the NER index to find entity names. Useful for
    discovering entities before searching for passages about them.

    Args:
        query: Entity name or partial name to search for.
        entity_type: Optional filter by type (CHARACTER, LOCATION, EVENT, etc).
        max_results: Maximum results (1-50, default 20).

    Returns:
        JSON with matching entities, their types, and passage counts.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    capped = min(max(max_results, 1), 50)
    lifespan = ctx.request_context.lifespan_context

    logger.info(
        "mcp search_entities.enter: query=%r entity_type=%s max_results=%d",
        query.strip()[:80],
        entity_type or "any",
        capped,
    )

    # Fuzzy match on entities field
    es_query: dict[str, Any] = {
        "multi_match": {
            "query": query.strip(),
            "fields": ["entities"],
            "fuzziness": "AUTO",
        }
    }

    # Entity type filter
    if entity_type:
        es_query = {
            "bool": {
                "must": [es_query],
                "filter": [{"term": {"entity_types": entity_type.upper()}}],
            }
        }

    resp = await lifespan.es.search(
        index=ES_INDEX_NER,
        query=es_query,
        size=0,
        aggs={
            "top_entities": {
                "terms": {
                    "field": "entities",
                    "size": capped,
                    "order": {"_count": "desc"},
                }
            }
        },
    )

    entities = [
        {"entity": bucket["key"], "passage_count": bucket["doc_count"]}
        for bucket in resp["aggregations"]["top_entities"]["buckets"]
    ]

    return json.dumps(
        {"query": query, "entity_type": entity_type or "any", "entities": entities},
        indent=2,
    )


# Main

if __name__ == "__main__":
    if "--server" in sys.argv:
        print(f"Starting R2C2 Hybrid MCP server at http://{MCP_HOST}:{MCP_PORT}/mcp")
        print("Press Ctrl+C to stop.\n")
    mcp_server.run(transport="streamable-http")
