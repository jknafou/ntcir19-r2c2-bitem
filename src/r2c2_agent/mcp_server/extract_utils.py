"""Pure helpers for the `extract_from_passage` MCP tool.

Isolated from `server.py` so tests can import the logic without triggering
the heavy FAISS/reranker/KG preload at module import time.
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "split_sentences_simple",
    "bm25_score_sentences",
    "rank_matches",
]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# BM25 parameters (standard Okapi defaults; small corpora are insensitive).
_BM25_K1 = 1.5
_BM25_B = 0.75


def split_sentences_simple(text: str) -> list[str]:
    """Split text into sentences using a simple regex boundary.

    Uses `(?<=[.!?])\\s+(?=[A-Z])`. Returns non-empty sentences only.
    """
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize into alphanumeric runs."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def bm25_score_sentences(
    sentences: list[str],
    query: str,
    k1: float = _BM25_K1,
    b: float = _BM25_B,
) -> list[float]:
    """BM25 score each sentence against the query.

    Treats each sentence as a short document. Returns a list of floats
    aligned to `sentences` (score >= 0; can exceed 1 when multiple query
    terms match a short sentence strongly).

    Empty query or empty sentences return zero vectors.
    """
    if not sentences or not query.strip():
        return [0.0] * len(sentences)

    q_tokens = list(dict.fromkeys(_tokenize(query)))  # dedupe, preserve order
    if not q_tokens:
        return [0.0] * len(sentences)

    tokenized: list[list[str]] = [_tokenize(s) for s in sentences]
    doc_lens = [len(t) for t in tokenized]
    n_docs = len(tokenized)
    avgdl = sum(doc_lens) / n_docs if n_docs > 0 and sum(doc_lens) > 0 else 1.0

    df: dict[str, int] = {}
    for term in q_tokens:
        df[term] = sum(1 for toks in tokenized if term in toks)

    # IDF per term (standard BM25 IDF with +1 smoothing)
    idf: dict[str, float] = {}
    for term, df_t in df.items():
        idf[term] = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)

    scores: list[float] = []
    for toks, dl in zip(tokenized, doc_lens):
        if not toks:
            scores.append(0.0)
            continue
        score = 0.0
        for term in q_tokens:
            tf = toks.count(term)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (dl / avgdl if avgdl > 0 else 1.0))
            score += idf[term] * (tf * (k1 + 1)) / denom if denom > 0 else 0.0
        scores.append(score)

    return scores


def rank_matches(
    sentences: list[str],
    scores: list[float],
    max_spans: int,
) -> list[dict[str, Any]]:
    """Sort sentences by score descending and take top-N.

    Returns list of dicts with keys: text, sentence_idx, score, context.
    """
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    out: list[dict[str, Any]] = []
    for idx, score in indexed[:max_spans]:
        out.append(
            {
                "text": sentences[idx],
                "sentence_idx": idx,
                "score": round(float(score), 4),
                "context": None,
            }
        )
    return out
