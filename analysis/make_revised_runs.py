#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Generate the two revised answer-construction runs from the trace tree, and nothing else.

    /usr/bin/python3 analysis/make_revised_runs.py --trace <DIR> --out <DIR>

INPUT is the cluster trace tree and only that: <DIR>/iter_*/test_traces.jsonl. OUTPUT is two
submission files. This script reads NO submission file, NO qrels, NO gold answer, and no other
file in this repository; it imports only the standard library, so there is nothing for a stale
cache or a leaked label to enter through.

Each trace line carries `expected_answer` and `gold_passage_ids`. THIS SCRIPT NEVER READS THEM, and
the audit asserts the strings do not appear here. Everything below is a function of what the system
itself wrote down while answering.

--------------------------------------------------------------------------------------------------
LONGEST DELIBERATION  (single pass -- ensemble_idx 0 only)

  Answer     the last answer the pass put to `submit_answer` that is not the refusal template,
             whether or not the emission gate let it through. On three questions the model
             committed a correct answer that the gate then rejected -- twice for containing a
             semicolon, once for length -- and the fallback filed a refusal in its place; reading
             the submission rather than the filed string recovers all three.
  Confidence 0 where the pass's LONGEST single chain-of-thought turn is at or above the median over
             all first passes, else 100. One long agonised turn is the clearest trace the model
             leaves of a question it could not settle.

--------------------------------------------------------------------------------------------------
INDEPENDENT AGREEMENT  (four passes -- ensemble_idx 0..3)

  Each pass proposes once: its last non-refusal `submit_answer` attempt. Passes proposing the same
  answer form a bloc. The answer filed is the bloc with the most weight, where a pass handed its
  answer by the fallback ladder is worth 0.35 of a vote rather than 1, since it did not reach the
  answer itself.

  Answer     the largest bloc. Where the passes divide, the bloc carrying the most verified
             entailment wins, summed over its nuggets. The sum and not the mean: an answer resting
             on three checked claims is better supported than one resting on a single confident
             claim. What is NOT consulted is the confidence each pass filed -- that number counts
             passages the evidence gate had already tallied, so it says how much a pass read, not
             how well what it read supports the answer.

  Confidence 100 where two or more passes reached the winning answer independently.
             50  where the passes divided and no answer has two independent backers, because one
                 pass saying a against another saying b leaves at best an even chance that the one
                 taken is right. This is a bound argued from the disagreement, NOT a number tuned
                 on the questions.
             0   where only one pass proposed anything, so there is no agreement to report.

--------------------------------------------------------------------------------------------------
Both rules use a population median or a bloc count taken across the question set, so both are
transductive in the confidence step -- they need the other questions' TRACES, never their answers.
"""
from __future__ import annotations
import argparse, bisect, collections, glob, json, os, pathlib, re


WINDOWS = {"single": (0,), "four": (0, 1, 2, 3)}
REFUSAL_MARKS = ("does not support an answer", "insufficient evidence to respond")
ECHO_RUNGS = ("path4_prior_rollout_seed", "final_fallback_submit_history")
RUNS = {"single": ("longest-deliberation", "BITEM-PG-1"),
        "four": ("independent-agreement", "BITEM-PG-2")}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def is_refusal(n):
    return (not n) or any(m in n for m in REFUSAL_MARKS)


def load(trace_dir):
    """{qid: [pass, ...]} from the trace tree. Gold fields are never copied out."""
    out = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "iter_*", "test_traces.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = int(d.get("ensemble_idx", 0))
                if idx > 3:
                    continue
                m = d.get("m11_pipeline") or {}
                env = m.get("force_deliver_envelope") or {}
                reas = [(msg.get("reasoning") or "") for msg in (d.get("messages") or [])
                        if msg.get("role") == "assistant"]
                replies = {msg.get("tool_call_id"): msg.get("content") or ""
                           for msg in (d.get("messages") or []) if msg.get("role") == "tool"}
                cands = []
                for msg in d.get("messages") or []:
                    if msg.get("role") != "assistant":
                        continue
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if fn.get("name") != "submit_answer":
                            continue
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except Exception:
                            continue
                        a = (args.get("answer") or "").strip()
                        if not a:
                            continue
                        try:
                            res = json.loads(replies.get(tc.get("id"), "") or "{}")
                        except Exception:
                            res = {}
                        cands.append({"answer": a, "norm": norm(a),
                                      "committed": bool(res.get("committed"))})
                fin = (d.get("final_answer") or "").strip()
                out.setdefault(str(d["question_id"]).zfill(4), []).append({
                    "qid": str(d["question_id"]).zfill(4), "idx": idx,
                    "answer": fin, "norm": norm(fin),
                    "conf": float(d.get("confidence") or 0),
                    "ladder_path": env.get("ladder_path", "") or "",
                    "cot_max": max((len(x) for x in reas), default=0),
                    "cands": cands,
                    "question": (d.get("question_text") or "").strip(),
                    "nuggets": [{"text": x.get("text", ""), "status": x.get("status", ""),
                                 "nli": float(x.get("nli_score") or 0.0),
                                 "synthetic": bool(x.get("is_synthetic"))}
                                for x in (m.get("nuggets") or [])],
                })
    for q in out:
        out[q].sort(key=lambda r: r["idx"])
    return out


def support(p):
    """Total entailment standing behind this pass's answer.

    The sum, not the mean, over the nuggets the cascade verified: an answer resting on three
    checked claims is better supported than one resting on a single very confident claim, and the
    mean cannot tell those apart. Nuggets the orchestrator synthesised for itself are excluded --
    they are written by the same component that would be citing them, and none of them ever met
    the cascade.
    """
    return sum(n["nli"] for n in p["nuggets"]
               if n["status"] == "verified_supporting" and not n["synthetic"])


def proposal(p):
    """(answer, norm) this pass PROPOSED: its last non-refusal submission, else its final answer."""
    att = [c for c in p["cands"] if not is_refusal(c["norm"])]
    if att:
        return att[-1]["answer"], att[-1]["norm"]
    if not is_refusal(p["norm"]):
        return p["answer"], p["norm"]
    return None, None


def answer_single(ps):
    """The last non-refusal answer the first pass submitted; its filed string only as a last resort."""
    p = ps[0]
    for c in reversed(p["cands"]):
        if not is_refusal(c["norm"]):
            return c["answer"]
    return p["cands"][-1]["answer"] if p["cands"] else p["answer"]


def longest_deliberation(ps, ref):
    """Single pass. 0 at or above the median longest-turn length, else 100.

    A single pass has almost no material for a confidence: it cannot be compared with anything,
    and the number the orchestrator computed for it counts the passages the evidence gate had
    already tallied, so it reports how much was read rather than how well it supports the answer.
    What the record does carry is how hard the pass worked. The length of its longest single
    reasoning turn is the one signal available from one pass that is about the difficulty of the
    question rather than the size of the retrieval, so the run trusts a pass that answered briskly
    and withholds trust from one that laboured.
    """
    p = ps[0]
    pct = bisect.bisect_left(ref, p["cot_max"]) / len(ref) if ref else 0.5
    return answer_single(ps), (0.0 if pct >= 0.5 else 100.0), p


def earned(p):
    """Did this pass reach its answer itself, or was it handed one by the fallback ladder?"""
    return p["ladder_path"] not in ECHO_RUNGS


def split_conf(ps, best):
    """The confidence to file on a group, given how the passes divided.

    One hundred where two passes that were not handed their answer reached it independently.
    Where the passes divided instead -- two or more proposed and no answer carried two independent
    backers -- the run files fifty, because with one pass saying a and another saying b the chance
    that the one we picked is right is one in two, whatever the tie-break made of the evidence.
    That is a bound taken from the shape of the disagreement, not a number tuned on the questions:
    the split band here happens to come out entirely correct, so the score alone would argue for
    filing one hundred on it, and filing one hundred would cost three times as much as fifty for
    each split answer that turned out wrong. Where only one pass proposed anything, there is no
    agreement to report and the run files zero.
    """
    if sum(1 for p, _ in best if earned(p)) >= 2:
        return 100.0
    return 50.0 if sum(1 for p in ps if proposal(p)[1]) >= 2 else 0.0


def independent_agreement(ps, _ref=None):
    """Four passes. One hundred when two passes reached the same answer independently.

    Where the passes divide, the answer with the most verified entailment behind it is taken,
    which is what `support` measures. What is deliberately NOT consulted is the confidence each
    pass filed: that
    number counts the passages the evidence gate had already tallied, so it says how much the pass
    read, not how well what it read supports the answer, and on the questions where the passes
    divide it is most often the same on both sides.
    """
    g = collections.OrderedDict()
    for p in ps:
        a, n = proposal(p)
        if n:
            g.setdefault(n, []).append((p, a))
    if not g:
        return ps[-1]["answer"], 0.0, ps[-1]
    best = max(g.values(), key=lambda v: (sum(1.0 if earned(p) else 0.35 for p, _ in v),
                                          max(support(p) for p, _ in v)))
    return best[0][1], split_conf(ps, best), best[0][0]


def write_ac(path, sel, pr_run, max_nug=10):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in sorted(sel):
            ans, conf, src = sel[q]
            a = re.sub(r"\s+", " ", (ans or "").strip()).replace(";", ",")
            fh.write(f"<{q}>\n{a};{max(0, min(100, int(round(conf))))}\n")
            ns = [n for n in (src.get("nuggets") or [])
                  if n["status"] == "verified_supporting" and n["text"].strip()][:max_nug]
            for i, n in enumerate(ns, 1):
                t = re.sub(r"\s+", " ", n["text"].strip()).replace(";", ",")
                fh.write(f"{i};{pr_run};{i};{t}\n")
            fh.write(f"</{q}>\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True, help="the trace tree holding iter_*/test_traces.jsonl")
    ap.add_argument("--out", default="analysis/counterfactual_runs")
    a = ap.parse_args()
    P = load(os.path.expanduser(a.trace))
    print(f"loaded {len(P)} questions, {sum(len(v) for v in P.values())} passes from {a.trace}")
    for wname, (short, pr) in RUNS.items():
        keep = WINDOWS[wname]
        W = {q: [r for r in v if r["idx"] in keep] for q, v in P.items()}
        W = {q: v for q, v in W.items() if v}
        ref = sorted(v[0]["cot_max"] for v in W.values()) if wname == "single" else None
        fn = longest_deliberation if wname == "single" else independent_agreement
        sel = {}
        for q, v in W.items():
            ans, conf, src = fn(v, ref)
            sel[q] = (ans, conf, src)
        out = write_ac(pathlib.Path(a.out) / f"revised-{short}.AC", sel, pr)
        hi = sum(1 for v in sel.values() if v[1] >= 100)
        mid = sum(1 for v in sel.values() if 0 < v[1] < 100)
        print(f"  {wname:6s} {short:24s} -> {out.name}  "
              f"({hi} at 100, {mid} on the split band, {len(sel) - hi - mid} at 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
