#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""How the agent searched.

One measurement, descriptive only: the retrieval behaviour of the submitted PR
runs, read out of the recovered execution traces. What the agent asked for
(`search_passages` volume, query lines per call, `max_results`, corpus
confinement), what syntax it exercised in those queries, how it reformulated
within a pass, how its querying changed across the passes of one question as the
corpus it may retrieve from shrinks, what `search_entities` returned, and how
much of what came back was ever used.

No official scores are read and no run is re-scored. The measurement is over the
traces alone.

Two committed caches carry everything the tables need, so the document rebuilds
from the repository without the 173MB trace drop:

  analysis/search_calls.tsv    one row per search call     -- 2474 rows
  analysis/pass_passages.tsv   one row per (pass, passage) -- 14226 rows

Both are (re)written by `--trace <root>`, which points at the recovered drop and
reads `agent-bitem/data/production/r2c2-pr-2026-04-25/iter_*/test_traces.jsonl`.
Without `--trace` the script reads the caches and produces the same document.
`analysis/trace_calls.tsv` and `analysis/trace_passes.tsv` supply the per-pass
context.

Pure read; writes only analysis/search_behaviour.md and, with --trace, the two
caches above.

Usage:  python3 analysis/search_behaviour.py
"""

from __future__ import annotations

import collections
import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "analysis"
OUT = Path(__file__).resolve().with_suffix(".md")
REL = "agent-bitem/data/production/r2c2-pr-2026-04-25"

CALLS_TSV = ANA / "search_calls.tsv"
PASSAGES_TSV = ANA / "pass_passages.tsv"

CALL_COLS = [
    "qid", "ensemble_idx", "seq", "search_idx", "tool",
    "corpus_arg", "corpus_used", "max_results", "n_queries", "query_chars",
    "total_results", "search_time_ms", "bm25_candidates", "dense_candidates",
    "kg_candidates", "excluded_count", "dense_underfetch_retries", "has_reflect",
    "entity_type", "n_entities", "calls_used", "calls_remaining",
    "queries_json", "pids_json", "entities_json",
]
PASSAGE_COLS = [
    "qid", "ensemble_idx", "pid", "status", "n_search_hits", "best_rank",
    "first_search_idx", "n_reads", "in_roster",
]

# The server's own normalisation, copied so the reformulation numbers below are
# the ones the refusal gate and the duplicate-query nudge compute.
# pipeline_handlers.py:1205-1221 (_WORD_RE, _tokenize_lower, _jaccard_unigrams)
# and :5912 (_normalize_nugget_text).
_WORD_RE = re.compile(r"\w+")
MAX_JACCARD_CONSECUTIVE_QUERIES = 0.70   # pipeline_handlers.py:285
MIN_UNIQUE_QUERIES = 6                   # pipeline_handlers.py:282


def toks(s: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def jac(a: str, b: str) -> float:
    ta, tb = set(toks(a)), set(toks(b))
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def norm(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


# Cache build

def build_cache(trace_root: Path) -> None:
    base = trace_root.expanduser() / REL
    files = sorted(base.glob("iter_*/test_traces.jsonl"),
                   key=lambda p: int(p.parent.name.split("_")[1]))
    if not files:
        sys.exit(f"no traces under {base}")

    def jload(s):
        try:
            return json.loads(s) if isinstance(s, str) else (s or {})
        except Exception:
            return {}

    calls, passages = [], []
    for fp in files:
        for line in fp.open(encoding="utf-8"):
            d = json.loads(line)
            qid, eidx = d.get("question_id", ""), d.get("ensemble_idx", "")
            msgs = d.get("messages") or []
            reply = {m.get("tool_call_id"): (m.get("content") or "")
                     for m in msgs if m.get("role") == "tool"}

            seq = 0
            search_idx = 0
            hits: dict[str, list[int]] = defaultdict(list)     # pid -> ranks
            first_seen: dict[str, int] = {}
            reads: Counter[str] = Counter()
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = jload(fn.get("arguments"))
                    res = jload(reply.get(tc.get("id"), ""))
                    seq += 1

                    if name == "get_passage":
                        pids = args.get("passage_ids") or ""
                        for p in (pids.split(",") if isinstance(pids, str) else pids):
                            if p.strip():
                                reads[p.strip()] += 1
                    elif name == "extract_from_passage":
                        p = (args.get("passage_id") or "").strip()
                        if p:
                            reads[p] += 1

                    if name not in ("search_passages", "search_entities"):
                        continue
                    search_idx += 1

                    if name == "search_passages":
                        raw = args.get("queries")
                        qs = ([q.strip() for q in raw.split("\n") if q.strip()]
                              if isinstance(raw, str) else list(raw or []))
                        plist = res.get("passages") or []
                        pids = []
                        for p in plist:
                            pid = p.get("passage_id", "")
                            pids.append(pid)
                            hits[pid].append(int(p.get("rank") or 0))
                            first_seen.setdefault(pid, search_idx)
                        ents = []
                        etype = ""
                    else:
                        qs = [args.get("query") or ""]
                        pids = []
                        ents = [[e.get("entity", ""), e.get("passage_count", 0)]
                                for e in (res.get("entities") or [])]
                        etype = res.get("entity_type", args.get("entity_type", ""))

                    b = res.get("budget") or {}
                    calls.append({
                        "qid": qid, "ensemble_idx": eidx, "seq": seq,
                        "search_idx": search_idx, "tool": name,
                        "corpus_arg": args.get("corpus", ""),
                        "corpus_used": res.get("corpus", ""),
                        "max_results": args.get("max_results", ""),
                        "n_queries": len(qs),
                        "query_chars": sum(len(x) for x in qs),
                        "total_results": res.get("total_results", ""),
                        "search_time_ms": res.get("search_time_ms", ""),
                        "bm25_candidates": res.get("bm25_candidates", ""),
                        "dense_candidates": res.get("dense_candidates", ""),
                        "kg_candidates": res.get("kg_candidates", ""),
                        "excluded_count": res.get("excluded_count", ""),
                        "dense_underfetch_retries": res.get("dense_underfetch_retries", ""),
                        "has_reflect": int("reflect" in res),
                        "entity_type": etype, "n_entities": len(ents),
                        "calls_used": b.get("calls_used", ""),
                        "calls_remaining": b.get("calls_remaining", ""),
                        "queries_json": json.dumps(qs, ensure_ascii=False),
                        "pids_json": json.dumps(pids),
                        "entities_json": json.dumps(ents, ensure_ascii=False),
                    })

            pdb = (d.get("m11_pipeline") or {}).get("passages_db") or {}
            roster = {p.get("passage_id") for p in (d.get("reconciled_passages") or [])}
            for pid in set(hits) | set(pdb) | set(reads):
                if pid.startswith("__"):
                    continue     # the virtual __refusal__ anchor
                r = hits.get(pid) or []
                passages.append({
                    "qid": qid, "ensemble_idx": eidx, "pid": pid,
                    "status": (pdb.get(pid) or {}).get("status", ""),
                    "n_search_hits": len(r),
                    "best_rank": min(r) if r else "",
                    "first_search_idx": first_seen.get(pid, ""),
                    "n_reads": reads.get(pid, 0),
                    "in_roster": int(pid in roster),
                })

    for path, cols, rows in ((CALLS_TSV, CALL_COLS, calls),
                             (PASSAGES_TSV, PASSAGE_COLS, passages)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.name}  {len(rows)} rows")


# Helpers

def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path}; rerun with --trace <root of the trace drop>")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def i(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def dist(xs) -> str:
    """min / median / mean / max, on one line."""
    xs = list(xs)
    if not xs:
        return "n/a"
    return (f"{min(xs)} / {statistics.median(xs):.1f} / "
            f"{statistics.mean(xs):.2f} / {max(xs)}")


def hist_row(counter: Counter, keys, total: int) -> str:
    return " | ".join(f"{counter.get(k, 0)} ({100.0 * counter.get(k, 0) / total:.1f})"
                      for k in keys)


def pct(n, d) -> str:
    return f"{100.0 * n / d:.1f}" if d else "n/a"


# Analysis

# Lucene syntax the BM25 leg would honour: the leg is an ES ``query_string``
# over contents/title^2/section^1.5 (server.py:451-459), so all of this is live.
SYNTAX = [
    ("quoted phrase", re.compile(r'"[^"]+"')),
    ("`title:` field", re.compile(r"\btitle\s*:")),
    ("`section:` field", re.compile(r"\bsection\s*:")),
    ("other `field:`", re.compile(r"\b(?!title|section)[a-z_]+:(?![/\s])")),
    ("`+` required", re.compile(r"(?:^|\s)\+\S")),
    ("`-` excluded", re.compile(r"(?:^|\s)-\w")),
    ("AND / OR / NOT", re.compile(r"(?:^|\s)(AND|OR|NOT)(?:\s|$)")),
    ("wildcard `*` / `?`", re.compile(r"\w[*?]")),
    ("fuzzy `~`", re.compile(r"~")),
    ("parentheses", re.compile(r"\(.*\)")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="root of the recovered trace drop; rebuilds the caches")
    a = ap.parse_args()
    if a.trace:
        build_cache(Path(a.trace))

    C = read_tsv(CALLS_TSV)
    PP = read_tsv(PASSAGES_TSV)
    PASSES = read_tsv(ANA / "trace_passes.tsv")
    ALLCALLS = read_tsv(ANA / "trace_calls.tsv")

    for r in C:
        r["queries"] = json.loads(r["queries_json"])
        r["pids"] = json.loads(r["pids_json"])
        r["entities"] = json.loads(r["entities_json"])

    SP = [r for r in C if r["tool"] == "search_passages"]
    SE = [r for r in C if r["tool"] == "search_entities"]
    n_sp, n_se = len(SP), len(SE)

    keyf = lambda r: (r["qid"], i(r["ensemble_idx"]))
    passes = sorted({keyf(p) for p in PASSES})
    n_pass, n_q = len(passes), len({p[0] for p in passes})

    by_pass_sp: dict[tuple, list[dict]] = defaultdict(list)
    for r in SP:
        by_pass_sp[keyf(r)].append(r)
    for v in by_pass_sp.values():
        v.sort(key=lambda r: i(r["seq"]))
    by_pass_se: dict[tuple, list[dict]] = defaultdict(list)
    for r in SE:
        by_pass_se[keyf(r)].append(r)
    calls_per_pass = {k: i(p["tool_calls"]) for p in PASSES for k in [keyf(p)]}
    passidx = {keyf(p): i(p["ensemble_idx"]) for p in PASSES}

    L: list[str] = []
    w = L.append

    w("# How the agent searched\n")
    w(f"One measurement over the recovered traces of the submitted PR runs: "
      f"{n_pass} passes over {n_q} questions, "
      f"{len(ALLCALLS)} tool calls in total, of which {n_sp} are "
      f"`search_passages` and {n_se} are `search_entities`. "
      "Descriptive only — nothing here is scored, and no official result is read. "
      "Rebuilt by `/usr/bin/python3 analysis/search_behaviour.py` from "
      "`analysis/search_calls.tsv` and `analysis/pass_passages.tsv`, which "
      "`--trace` regenerates from the trace drop.\n")

    # 1. volume
    w("## 1. What was asked for\n")

    spc = Counter()
    for k in passes:
        spc[len(by_pass_sp.get(k, []))] += 1
    per_pass = [len(by_pass_sp.get(k, [])) for k in passes]
    share = [100.0 * len(by_pass_sp.get(k, [])) / max(1, calls_per_pass.get(k, 0))
             for k in passes if calls_per_pass.get(k, 0)]

    over = sum(1 for p in PASSES if i(p["calls_used"]) > 30)
    w(f"A pass issues {dist(per_pass)} `search_passages` calls "
      f"(min / median / mean / max) and {dist([i(p['tool_calls']) for p in PASSES])} "
      "tool calls of any kind. Searching is where the tool budget goes: "
      f"{statistics.mean(share):.1f} per cent of a pass's calls are searches. The "
      f"advertised budget is thirty calls, and {over} of {n_pass} passes went "
      "past it — the counter is shown to the model and used as the denominator "
      "of the refusal gate, but nothing halts a pass for exceeding it. The "
      "distribution is bimodal, and section 4 shows why: a first pass searches "
      "about four times, a later pass ten or more.\n")

    w("| searches in the pass | 0 | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 | 11+ |")
    w("|---|---|---|---|---|---|---|---|")
    buckets = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 10**6)]
    w("| passes | " + " | ".join(
        str(sum(v for kk, v in spc.items() if lo <= kk <= hi))
        for lo, hi in buckets) + " |\n")

    nq = Counter(i(r["n_queries"]) for r in SP)
    w("The tool takes a newline-separated string, and the prompt asks for one to "
      "four reformulations per call. All the lines feed all three legs — BM25 "
      "unions their hits at the max score (`server.py:451-491`), the dense leg "
      "embeds each and keeps the max cosine per passage (`:523-551`), the graph "
      "leg unions the entities extracted from each (`:740-745`). Only the "
      "cross-encoder rerank, which decides the order the agent actually sees, "
      "uses the first line alone (`:1000-1004`). The tool's own docstring says "
      "the opposite — \"Dense and KG legs use the first query only\" "
      "(`:832-833`) — and is stale.\n")
    w("| query lines in the call | 1 | 2 | 3 | 4 | 5 |")
    w("|---|---|---|---|---|---|")
    w("| calls (per cent) | " + hist_row(nq, [1, 2, 3, 4, 5], n_sp) + " |\n")

    qlines = [q for r in SP for q in r["queries"]]
    qlen_tok = [len(toks(q)) for q in qlines]
    qlen_ch = [len(q) for q in qlines]
    w(f"Those {len(qlines)} query lines are short: {dist(qlen_tok)} tokens and "
      f"{dist(qlen_ch)} characters. A whole call carries "
      f"{dist([i(r['query_chars']) for r in SP])} characters.\n")

    mr = Counter(i(r["max_results"]) for r in SP)
    mrk = sorted(mr)
    w("`max_results` is almost never moved off its default of ten.\n")
    w("| `max_results` | " + " | ".join(str(k) for k in mrk) + " |")
    w("|---" * (len(mrk) + 1) + "|")
    w("| calls (per cent) | " + hist_row(mr, mrk, n_sp) + " |\n")

    ca = Counter(r["corpus_arg"] or "(omitted)" for r in SP)
    cak = [k for k, _ in ca.most_common()]
    conf = [r for r in SP if r["corpus_arg"] in ("wikipedia", "wookieepedia")]
    w("The corpus argument confines a search to `wikipedia` or `wookieepedia`; "
      "omitting it means `all`, which is also what passing `all` means. The "
      f"agent confined a search to one half of the corpus {len(conf)} times in "
      f"{n_sp}, in {len({keyf(r) for r in conf})} passes over "
      f"{len({r['qid'] for r in conf})} questions.\n")
    w("| `corpus` | " + " | ".join(f"`{k}`" for k in cak) + " |")
    w("|---" * (len(cak) + 1) + "|")
    w("| calls (per cent) | " + hist_row(ca, cak, n_sp) + " |\n")

    rn = Counter(i(r["total_results"]) for r in SP)
    zero = [r for r in SP if i(r["total_results"]) == 0]
    short = [r for r in SP if 0 < i(r["total_results"]) < i(r["max_results"])]
    w(f"The search returned exactly the requested number of results in "
      f"{n_sp - len(zero) - len(short)} of {n_sp} calls, was never short, and "
      f"came back empty {len(zero)} times ({pct(len(zero), n_sp)} per cent).\n")
    rnk = sorted(rn)
    w("| results returned | " + " | ".join(str(k) for k in rnk) + " |")
    w("|---" * (len(rnk) + 1) + "|")
    w("| calls | " + " | ".join(str(rn[k]) for k in rnk) + " |\n")

    zc = [r for r in zero if r["corpus_arg"] in ("wikipedia", "wookieepedia")]
    zp = Counter(keyf(r) for r in zero)
    w(f"Those empty returns are not scattered. **All {len(conf)} corpus-confined "
      f"searches returned nothing, and all {len(zero)} empty returns are "
      "corpus-confined searches**; every search that left `corpus` at `all` "
      "returned results. The cause is a value mismatch. Each of the three legs "
      "filters on `{\"term\": {\"source\": corpus}}` (`server.py:465`, `:589`, "
      "`:785`), an exact match on a `keyword` field, but the indexer stores "
      "`wikipedia_wikitext` and `wookieepedia_wikitext` in that field — the "
      "value every reply hands back as `corpus_source`. So `wikipedia` matches "
      "nothing, and confining the search empties it. The prompt names the "
      "parameter and tells the model which three values the server accepts "
      "(`prompt.py:62-64`); two of the three silently return zero results. "
      f"The {len(zero)} calls fall in {len(zp)} passes over "
      f"{len({r['qid'] for r in zero})} questions, so the damage is contained, "
      "but where a pass reached for the parameter it lost the call and the "
      f"budget with it — the worst-hit pass is {zp.most_common(1)[0][0][0]} pass "
      f"{zp.most_common(1)[0][0][1]}, which spent {zp.most_common(1)[0][1]} of "
      f"its {len(by_pass_sp[zp.most_common(1)[0][0]])} searches this way.\n")

    # legs
    bm = [i(r["bm25_candidates"]) for r in SP]
    de = [i(r["dense_candidates"]) for r in SP]
    kg = [i(r["kg_candidates"]) for r in SP]
    kg_hit = [r for r in SP if i(r["kg_candidates"]) > 0]
    bm0 = sum(1 for x in bm if x == 0)
    w("Every reply names the method `hybrid (BM25+Dense+KG RRF + CE rerank)` and "
      "reports each leg's candidate count. The three legs did not contribute "
      "equally.\n")
    w("| leg | calls contributing nothing | candidates, min / median / mean / max |")
    w("|---|---|---|")
    w(f"| BM25 | {bm0} | {dist(bm)} |")
    w(f"| dense | {sum(1 for x in de if x == 0)} | {dist(de)} |")
    w(f"| graph | {sum(1 for x in kg if x == 0)} | {dist(kg)} |\n")
    bm0_free = [r for r in SP if i(r["bm25_candidates"]) == 0
                and r["corpus_arg"] not in ("wikipedia", "wookieepedia")]
    w(f"{bm0 - len(bm0_free)} of BM25's {bm0} blanks are the corpus-filter "
      f"failure above. The other {len(bm0_free)} all carry Lucene punctuation — "
      + ", ".join(f"`{r['queries'][0]}`" for r in bm0_free) +
      " — which is either a genuine miss or a parser rejection. The traces "
      "cannot tell the two apart, because the leg is wrapped in a bare `except` "
      "that logs and returns an empty dictionary (`server.py:904-920`): a query "
      "the parser refuses degrades the search to dense-only and the reply says "
      "only `bm25_candidates: 0`.\n")
    kgq = [q for r in kg_hit for q in r["queries"][:1]]
    w(f"The graph leg returned candidates in {len(kg_hit)} of {n_sp} searches "
      f"({pct(len(kg_hit), n_sp)} per cent), across "
      f"{len({keyf(r) for r in kg_hit})} passes and "
      f"{len({r['qid'] for r in kg_hit})} questions. Where it fired, the first "
      "query line was short and name-shaped — "
      f"{statistics.median([len(toks(q)) for q in kgq]):.0f} tokens at the "
      f"median against {statistics.median(qlen_tok):.0f} for query lines "
      "generally, with "
      + ", ".join(f"`{q}`" for q in kgq[:5]) +
      " among them. It contributed nothing to the other "
      f"{n_sp - len(kg_hit)} searches, so what the runs filed is in practice a "
      "two-leg fusion.\n")

    ms = [i(r["search_time_ms"]) for r in SP]
    w(f"A search took {dist(ms)} milliseconds, which at the median puts "
      f"{statistics.median(ms) / 1000.0:.1f} seconds of wall clock behind each "
      "of a pass's searches.\n")

    # 2. syntax
    w("## 2. Query syntax\n")
    w("The BM25 leg is an Elasticsearch `query_string` over `contents`, "
      "`title^2` and `section^1.5` with a default `OR` (`server.py:451-459`), so "
      "Lucene syntax — phrases, field prefixes, required and excluded terms, "
      "boolean operators, wildcards, fuzziness — is honoured. The prompt never "
      "mentions any of it, and one `reflect` nudge suggests a quoted phrase "
      "(`pipeline_handlers.py:5872`). What the agent actually wrote:\n")

    w("| construct | query lines | per cent of lines | passes using it |")
    w("|---|---|---|---|")
    line_owner = [(r, q) for r in SP for q in r["queries"]]
    for label, rx in SYNTAX:
        hits = [(r, q) for r, q in line_owner if rx.search(q)]
        w(f"| {label} | {len(hits)} | {pct(len(hits), len(qlines))} | "
          f"{len({keyf(r) for r, _ in hits})} |")
    w("")
    any_rx = [(r, q) for r, q in line_owner
              if any(rx.search(q) for _, rx in SYNTAX)]
    par = [(r, q) for r, q in line_owner if SYNTAX[-1][1].search(q)]
    disamb = [x for x in par if re.search(r"\((\d{4}[^)]*|[a-z][^)]*)\)", x[1])]
    w(f"{len(any_rx)} of {len(qlines)} lines ({pct(len(any_rx), len(qlines))} per "
      f"cent) carry any operator at all, in {len({keyf(r) for r, _ in any_rx})} "
      f"of {n_pass} passes. The rest are bags of words. The operators that would "
      "change what BM25 retrieves rather than merely how it scores — required "
      "and excluded terms, boolean operators, wildcards, fuzziness — are "
      "essentially unused: four lines in five thousand. What the counts do pick "
      "up is the agent writing Wikipedia article titles: of the "
      f"{len(par)} lines with parentheses, {len(disamb)} are a disambiguation "
      "suffix such as `Interstellar (film) plot` or `Bean (1997 film)`, which "
      "Lucene reads as a group and not as the title the agent meant.\n")

    qp = [(r, q) for r, q in line_owner if SYNTAX[0][1].search(q)]
    qp_first = [(r, q) for r, q in qp if r["queries"] and r["queries"][0] == q]
    tp = [(r, q) for r, q in line_owner if SYNTAX[1][1].search(q)]
    tf = [(r, q) for r, q in tp if r["queries"] and r["queries"][0] == q]
    w("This matters more than the counts suggest. A phrase operator or a field "
      "prefix is Lucene to BM25 and literal punctuation to the other two legs: "
      "the dense leg embeds the line as written, and the cross-encoder that "
      "sets the final order scores the first line as written against passage "
      f"text. {len(qp_first)} of the {len(qp)} quoted lines are the first line "
      f"of their call, and {len(tf)} of the {len(tp)} lines with a `title:` "
      "prefix are, so in those searches the reranker was handed a string whose "
      "punctuation carries no meaning for it.\n")

    # 3. within-pass reformulation
    w("## 3. Reformulation within a pass\n")
    w("The server keeps a flat list of every query line a pass has issued and "
      "uses it in two places: the refusal gate requires at least "
      f"{MIN_UNIQUE_QUERIES} distinct normalised lines and fails a pass with two "
      "or more consecutive pairs above a Jaccard of "
      f"{MAX_JACCARD_CONSECUTIVE_QUERIES:.2f}, and a `reflect` nudge fires on "
      "two byte-identical consecutive lines (`pipeline_handlers.py:282-285`, "
      ":5669-5720, :5863). The numbers below use that code's own tokeniser and "
      "normalisation, so they are the quantities those gates saw.\n")

    cons_line, cons_call, deltas = [], [], []
    exact_rep, near_rep, tot_lines_seq = 0, 0, 0
    uniq_per_pass, lines_per_pass = [], []
    for k in passes:
        rs = by_pass_sp.get(k, [])
        flat = [q for r in rs for q in r["queries"]]
        lines_per_pass.append(len(flat))
        uniq_per_pass.append(len({norm(q) for q in flat}))
        for x, y in zip(flat, flat[1:]):
            cons_line.append(jac(x, y))
        seen: list[str] = []
        for r in rs:
            for q in r["queries"]:
                tot_lines_seq += 1
                if seen:
                    if norm(q) in {norm(s) for s in seen}:
                        exact_rep += 1
                    elif max(jac(q, s) for s in seen) >= MAX_JACCARD_CONSECUTIVE_QUERIES:
                        near_rep += 1
            # Only lines from earlier calls count as ground already covered; the
            # sibling lines of one call share an anchor by construction.
            seen.extend(r["queries"])
        for x, y in zip(rs, rs[1:]):
            cons_call.append(jac(" ".join(x["queries"]), " ".join(y["queries"])))
            deltas.append(len(toks(" ".join(y["queries"])))
                          - len(toks(" ".join(x["queries"]))))

    def band(xs, lo, hi):
        return sum(1 for x in xs if lo <= x < hi)

    w(f"A pass issues {dist(lines_per_pass)} query lines of which "
      f"{dist(uniq_per_pass)} are distinct after normalisation. "
      f"{sum(1 for u in uniq_per_pass if u < MIN_UNIQUE_QUERIES)} of {n_pass} "
      f"passes never reached the {MIN_UNIQUE_QUERIES} distinct lines the refusal "
      "gate demands, which is a floor on how often refusal could be admitted "
      "at all.\n")

    w("| Jaccard with the previous | 0 (disjoint) | (0, 0.25) | [0.25, 0.50) | "
      "[0.50, 0.70) | ≥ 0.70 | of which identical |")
    w("|---|---|---|---|---|---|---|")
    for label, xs, ident in (
            ("consecutive query lines", cons_line,
             sum(1 for x in cons_line if x >= 0.999)),
            ("consecutive calls, lines pooled", cons_call,
             sum(1 for x in cons_call if x >= 0.999))):
        w(f"| {label} | {sum(1 for x in xs if x == 0)} | {band(xs, 1e-9, .25)} | "
          f"{band(xs, .25, .50)} | {band(xs, .50, .70)} | "
          f"{sum(1 for x in xs if x >= .70)} | {ident} |")
    w("")
    w(f"The median consecutive-line Jaccard is "
      f"{statistics.median(cons_line):.3f} and the mean "
      f"{statistics.mean(cons_line):.3f}, so successive lines share between a "
      "quarter and a third of their tokens: the entity is held fixed and the "
      "words around it are swapped. Whole calls sit higher, at a median of "
      f"{statistics.median(cons_call):.3f}, because a call repeats its own "
      f"anchor across its two or three lines. {sum(1 for x in cons_line if x >= .70)} "
      "consecutive line pairs "
      f"({pct(sum(1 for x in cons_line if x >= .70), len(cons_line))} per cent) "
      f"sit at or above the {MAX_JACCARD_CONSECUTIVE_QUERIES:.2f} the gate calls "
      "near-identical.\n")

    w(f"Repetition, measured against every line an earlier call in the same pass "
      f"issued rather than only against the previous line: {exact_rep} of "
      f"{tot_lines_seq} lines "
      f"({pct(exact_rep, tot_lines_seq)} per cent) are a normalised repeat of a "
      f"line issued by an earlier call in the same pass, and a further "
      f"{near_rep} ({pct(near_rep, tot_lines_seq)} per cent) are within a "
      f"Jaccard of {MAX_JACCARD_CONSECUTIVE_QUERIES:.2f} of one. Sibling lines "
      "within a single call are excluded from the comparison, since they share "
      f"an anchor by construction. Between them, {pct(exact_rep + near_rep, tot_lines_seq)} "
      "per cent of the query lines the agent wrote were ground an earlier call "
      "in the same pass had already covered.\n")

    longer = sum(1 for d in deltas if d > 0)
    shorter = sum(1 for d in deltas if d < 0)
    w(f"Queries do not systematically lengthen or shorten as a pass goes on: of "
      f"{len(deltas)} consecutive calls, {longer} are longer than their "
      f"predecessor, {shorter} shorter and {len(deltas) - longer - shorter} the "
      f"same length, for a mean change of {statistics.mean(deltas):+.2f} tokens. "
      "That is a dull result and it is worth stating plainly: within a pass the "
      "agent rephrases sideways rather than narrowing.\n")

    # 4. across passes
    w("## 4. Across the passes of one question\n")
    w("Every passage id a pass saw is withheld from the next pass on the same "
      "question, so pass *k+1* faces a strictly smaller corpus than pass *k*. "
      "The exclusion is reported back in every search reply as "
      "`excluded_count`, and it grows as expected. The question is whether the "
      "agent's querying responds to it. It does not — the agent is never told "
      "the corpus shrank, and it behaves as if it were opening the question "
      "fresh. What does change is volume: a first pass searches under four "
      "times, a fourth pass eleven. The elevated empty-return rate in the first "
      "row is the corpus-filter failure of section 1 and not the exclusion, "
      "which by definition cannot bite on a first pass.\n")

    maxi = max(passidx.values())
    rows = []
    for pi in range(0, maxi + 1):
        ks = [k for k in passes if passidx[k] == pi]
        if not ks:
            continue
        rs = [r for k in ks for r in by_pass_sp.get(k, [])]
        if not rs:
            continue
        fl = [q for r in rs for q in r["queries"]]
        rows.append((
            pi, len(ks),
            statistics.mean(len(by_pass_sp.get(k, [])) for k in ks),
            statistics.mean(len(toks(q)) for q in fl),
            statistics.median(i(r["excluded_count"]) for r in rs),
            100.0 * sum(1 for r in rs if i(r["total_results"]) == 0) / len(rs),
            100.0 * sum(1 for k in ks if by_pass_se.get(k)) / len(ks),
        ))
    w("| pass index | passes | searches per pass | tokens per query line | "
      "median `excluded_count` | empty returns (%) | passes using `search_entities` (%) |")
    w("|---|---|---|---|---|---|---|")
    for pi, nk, sc, tk, ex, zr, se_ in rows:
        w(f"| {pi} | {nk} | {sc:.2f} | {tk:.2f} | {ex:.0f} | {zr:.1f} | {se_:.1f} |")
    w("")

    first_j, first_ident, vocab_j = [], 0, []
    byq: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for k in passes:
        if by_pass_sp.get(k):
            byq[k[0]][passidx[k]] = by_pass_sp[k]
    for qid, d in byq.items():
        idxs = sorted(d)
        for x, y in zip(idxs, idxs[1:]):
            a1, b1 = " ".join(d[x][0]["queries"]), " ".join(d[y][0]["queries"])
            first_j.append(jac(a1, b1))
            first_ident += int(norm(a1) == norm(b1))
            va = {t for r in d[x] for q in r["queries"] for t in toks(q)}
            vb = {t for r in d[y] for q in r["queries"] for t in toks(q)}
            vocab_j.append(len(va & vb) / max(1, len(va | vb)))

    w(f"Comparing the opening search of pass *k* with the opening search of pass "
      f"*k+1* on the same question, over {len(first_j)} adjacent pairs: the "
      f"median Jaccard is {statistics.median(first_j):.3f} and the mean "
      f"{statistics.mean(first_j):.3f}, and {first_ident} pairs "
      f"({pct(first_ident, len(first_j))} per cent) open with a normalised "
      f"line-for-line identical query. "
      f"{sum(1 for x in first_j if x >= MAX_JACCARD_CONSECUTIVE_QUERIES)} pairs "
      f"({pct(sum(1 for x in first_j if x >= MAX_JACCARD_CONSECUTIVE_QUERIES), len(first_j))} "
      "per cent) open above the near-identical threshold. Over a whole pass the "
      f"vocabulary overlap between consecutive passes has median "
      f"{statistics.median(vocab_j):.3f}.\n")
    w("So the diversity the ensemble obtains across passes is manufactured by "
      "the exclusion, not by the agent asking differently. Pass two mostly "
      "re-asks pass one's question and is handed whatever the index has left.\n")

    # 5. entity search
    w("## 5. `search_entities`\n")
    se_passes = {keyf(r) for r in SE}
    empty = [r for r in SE if i(r["n_entities"]) == 0]
    posn = [i(r["seq"]) / max(1, calls_per_pass.get(keyf(r), 1)) for r in SE]
    et = Counter(r["entity_type"] or "(any)" for r in SE)
    w(f"The entity index was consulted {n_se} times in {len(se_passes)} of "
      f"{n_pass} passes and {len({r['qid'] for r in SE})} of {n_q} questions — "
      f"{pct(n_se, n_sp + n_se)} per cent of all search calls. The prompt asks "
      "for it only when a surface form is ambiguous, and it is used late rather "
      f"than to open: the median call sits {statistics.median(posn) * 100:.0f} per cent of the "
      f"way through its pass's tool calls, at call "
      f"{statistics.median([i(r['seq']) for r in SE]):.0f} of thirty.\n")
    etk = [k for k, _ in et.most_common()]
    emp = Counter((r["entity_type"] or "(any)") for r in SE if i(r["n_entities"]) == 0)
    w("| `entity_type` | " + " | ".join(f"`{k}`" for k in etk) + " |")
    w("|---" * (len(etk) + 1) + "|")
    w("| calls | " + " | ".join(str(et[k]) for k in etk) + " |")
    w("| of which empty | " + " | ".join(str(emp.get(k, 0)) for k in etk) + " |\n")
    VALID = {"PERSON", "CHARACTER", "LOCATION", "ORGANIZATION", "EVENT", "WORK", "DATE"}
    bad = [r for r in SE if r["entity_type"] not in VALID | {"", "any"}]
    w("The filter is a `term` on `entity_types` upper-cased "
      "(`server.py:1453-1459`), and the seven types the NER stage emits are "
      "PERSON, CHARACTER, LOCATION, ORGANIZATION, EVENT, WORK and DATE "
      "(`src/r2c2_index/ner/refine_entities.py:26`). `FILM` is not among them. The "
      f"agent asked for it {len(bad)} times and got nothing back every time — "
      "the tool's own docstring lists three types and an `etc`, which is an "
      "invitation to invent one, and an out-of-vocabulary type is not an error, "
      "it is an empty result.\n")

    ne = [i(r["n_entities"]) for r in SE]
    exact = [r for r in SE if any(norm(e[0]) == norm(r["queries"][0])
                                  for e in r["entities"])]
    top_exact = [r for r in SE if r["entities"]
                 and norm(r["entities"][0][0]) == norm(r["queries"][0])]
    sub = [r for r in SE if r["entities"] and not any(
        norm(e[0]) == norm(r["queries"][0]) for e in r["entities"])]
    w(f"It returned {dist(ne)} entities and came back empty {len(empty)} times "
      f"({pct(len(empty), n_se)} per cent). Of the {n_se - len(empty)} non-empty "
      f"replies, {len(exact)} contain the queried string as one of the returned "
      f"names and {len(top_exact)} rank it first. The remaining {len(sub)} "
      f"({pct(len(sub), n_se)} per cent of all calls) return ten names none of "
      "which is the thing asked for.\n")
    ex_rows = [(r["queries"][0], [e[0] for e in r["entities"]][:4]) for r in sub][:6]
    if ex_rows:
        w("| query | first names returned |")
        w("|---|---|")
        for q, names in ex_rows:
            w(f"| `{q}` | " + ", ".join(f"`{x}`" for x in names) + " |")
        w("")
    w("The failure has a shape, and it is not fuzziness alone. The query is a "
      "`multi_match` with `fuzziness: AUTO` over the `entities` field, but the "
      "reply is a terms aggregation over the `entities` field of whatever "
      "passages matched, ordered by document count (`server.py:1444-1479`). So "
      "the tool does not return the entity names that matched the query. It "
      "returns the entities that most often co-occur in passages that fuzzily "
      "matched it, and it returns them ranked by corpus frequency. When the "
      "query is a full distinctive name the two coincide and the tool looks "
      "like it works; when the query is one short token, `AUTO` admits every "
      "name within two edits and the aggregation hands back the commonest "
      "entities of that unrelated neighbourhood — years, capital cities, the "
      "cast of another franchise. Nothing in the reply distinguishes the two "
      "cases, so the agent reads `Tatort` and `Berlin` as evidence about "
      "Totoro.\n")

    # 6. yield
    w("## 6. Yield\n")
    ret = [r for r in PP if i(r["n_search_hits"]) > 0]
    read = [r for r in ret if i(r["n_reads"]) > 0]
    acc = [r for r in ret if r["status"] == "accepted"]
    acc_all = [r for r in PP if r["status"] == "accepted"]
    st = Counter(r["status"] or "(never classified)" for r in ret)
    delivered = sum(i(r["total_results"]) for r in SP)
    distinct_pp = [len([r for r in PP if keyf(r) == k and i(r["n_search_hits"]) > 0])
                   for k in passes]

    w(f"The {n_sp} searches delivered {delivered} passage slots, which collapse "
      f"to {len(ret)} distinct (pass, passage) pairs — a pass sees "
      f"{dist(distinct_pp)} distinct passages. What became of them:\n")
    stk = [k for k, _ in st.most_common()]
    w("| fate of a retrieved passage | " + " | ".join(stk) + " |")
    w("|---" * (len(stk) + 1) + "|")
    w("| pairs | " + " | ".join(str(st[k]) for k in stk) + " |")
    w("| per cent | " + " | ".join(pct(st[k], len(ret)) for k in stk) + " |\n")

    w(f"{len(read)} of {len(ret)} retrieved passages ({pct(len(read), len(ret))} "
      f"per cent) were ever opened with `get_passage` or `extract_from_passage`, "
      f"and {len(acc)} ({pct(len(acc), len(ret))} per cent) were accepted as "
      f"evidence. Read the other way, a search call earns "
      f"{len(acc) / n_sp:.2f} accepted passages, and the agent spends "
      f"{delivered / max(1, len(acc)):.0f} retrieved slots per passage it ends "
      f"up believing. {pct(st.get('(never classified)', 0), len(ret))} per cent "
      "of what came back never received a status at all: it was returned, it "
      "was never opened, and the pass ended without an opinion on it.\n")

    ro = [r for r in PP if i(r["in_roster"]) == 1]
    rst = Counter(r["status"] or "(never classified)" for r in ro)
    rsz = Counter()
    for r in ro:
        rsz[keyf(r)] += 1
    w(f"That is not the same as waste, because the PR subtask asks for a ranked "
      f"list of passages and not for evidence. The {len(ro)} roster slots the "
      f"{len(rsz)} passes that filed one produced — the other "
      f"{n_pass - len(rsz)} filed nothing — run "
      f"{statistics.median(rsz.values()):.0f} per pass "
      "against a cap of twenty, and they are ranked retrieval output: only "
      f"{rst.get('accepted', 0)} of them ({pct(rst.get('accepted', 0), len(ro))} "
      f"per cent) are passages the agent accepted; {sum(1 for r in ro if i(r['n_reads']) > 0)} "
      f"({pct(sum(1 for r in ro if i(r['n_reads']) > 0), len(ro))} per cent) were "
      "opened at all. Conversely "
      f"{sum(1 for r in acc_all if i(r['in_roster']) == 1)} of the {len(acc_all)} "
      "accepted passages made the roster, so acceptance essentially guarantees "
      "a slot and the other seventeen slots are filled by rank.\n")

    ranks = Counter()
    accranks = Counter()
    for r in ret:
        br = i(r["best_rank"], -1)
        if br > 0:
            ranks[br] += 1
            if r["status"] == "accepted":
                accranks[br] += 1
    w("Acceptance falls off steeply with rank, which is the argument for asking "
      "for ten results rather than twenty.\n")
    w("| best rank at which the passage was seen | 1 | 2 | 3 | 4 | 5 | 6–10 | 11–15 | 16–20 |")
    w("|---|---|---|---|---|---|---|---|---|")
    grp = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 15), (16, 20)]
    w("| retrieved | " + " | ".join(
        str(sum(v for kk, v in ranks.items() if lo <= kk <= hi)) for lo, hi in grp) + " |")
    w("| accepted | " + " | ".join(
        str(sum(v for kk, v in accranks.items() if lo <= kk <= hi)) for lo, hi in grp) + " |")
    w("| accepted (%) | " + " | ".join(
        pct(sum(v for kk, v in accranks.items() if lo <= kk <= hi),
            sum(v for kk, v in ranks.items() if lo <= kk <= hi)) for lo, hi in grp) + " |\n")

    only_db = [r for r in PP if i(r["n_search_hits"]) == 0 and r["status"] == "accepted"]
    w(f"A further {len(only_db)} accepted (pass, passage) pairs were never "
      "returned by a search in that pass — they arrive through the neighbour "
      "windows of `get_passage` and through the retrieval the orchestrator runs "
      f"on its own behalf, and they are {pct(len(only_db), len(acc) + len(only_db))} "
      "per cent of everything accepted.\n")

    # Why a first-pass roster stopped short.
    # The paper reports that the single-pass run filed fewer than twenty passages on 37 of the
    # 64 scored questions, 21 of them stopping at exactly ten. Ten is DEFAULT_MAX_RESULTS in
    # src/r2c2_agent/mcp_server/server.py, and the model could have asked for up to a hundred on any
    # call, so the question is whether the retrieval ran out or whether nobody asked for more.
    w("## Why an under-filled first-pass roster stopped where it did\n")
    roster = collections.Counter()
    for r in PP:
        if i(r["ensemble_idx"]) == 0 and r["in_roster"] == "1":
            roster[r["qid"]] += 1
    first = collections.defaultdict(list)
    for r in SP:
        if i(r["ensemble_idx"]) == 0:
            first[r["qid"]].append(r)
    at_ten = sorted(q for q, n in roster.items() if n == 10)
    one_search = [q for q in at_ten if len(first[q]) == 1]
    asked_ten = [q for q in at_ten if len(first[q]) == 1 and i(first[q][0]["max_results"]) == 10]
    got_ten = [q for q in asked_ten if i(first[q][0]["total_results"]) == 10]
    w(f"| questions whose first pass filed exactly ten | {len(at_ten)} |")
    w("|---|---|")
    w(f"| of those, the pass issued exactly one search | {len(one_search)} |")
    w(f"| of those, that search asked for the default ten | {len(asked_ten)} |")
    w(f"| of those, that search was given ten | {len(got_ten)} |\n")
    full = sorted(q for q, n in roster.items() if n >= 20)
    ns = sorted(len(first[q]) for q in full)
    med = ns[len(ns) // 2] if ns else 0
    w(f"Against the {len(full)} questions that filed a full twenty, whose first passes issued "
      f"{ns[0] if ns else 0} to {ns[-1] if ns else 0} searches with a median of {med}. The "
      "retrieval was not short on the ten-passage questions, it was asked once for its default "
      "page and never asked again.\n")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}  ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
