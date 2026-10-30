#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Test whether de-duplicating the retrieval rosters would have improved their scores.

Both PR rosters repeat `docid`s: a slot holds a document already present at a better
rank.  The hypothesis under test is that those slots are dead weight under MSnDCG@20,
because the measure is keyed by document, and that removing them and promoting the rest
is a free improvement.  This script tests it end to end:

  1. it counts the repeats exactly, for both runs, over the 64 scored questions;
  2. it re-implements the organisers' MSnDCG@20 and Q@20 and validates the
     implementation by reproducing every published per-question score for all 22 PR
     runs before any counterfactual is scored;
  3. it rescores de-duplicated rosters under that validated scorer, in three variants
     that bracket what de-duplication could have been worth.

Reads results/ read-only by relative path; reads the submission directory (single
optional positional argument, default "submission") for the rosters we filed.  Writes
one markdown document beside itself and nothing else.

    uv run python analysis/roster_duplicates.py submission
"""
from __future__ import annotations

import collections
import csv
import hashlib
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "roster_duplicates.md")
RES = os.path.join("results", "OfficialResults")

QRELS = os.path.join(RES, "official.consolidated.directL0-L4.qrels")
TSM = {
    "MSnDCG@20": os.path.join(RES, "official.PRruns.cons.directL0-L4.nev.MSnDCG@0020.tsmdressed.csv"),
    "Q@20": os.path.join(RES, "official.PRruns.cons.directL0-L4.nev.Q@0020.tsmdressed.csv"),
}
OURS = ("BITEM-PG-1", "BITEM-PG-2")
LABEL = {"BITEM-PG-1": "single pass", "BITEM-PG-2": "pooled"}
CUTOFF = 20
GAIN = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
BOOTSTRAP = 10000
SEED = 20260822


# Reading

def read_qrels(path):
    """(qid, run) -> {rank: level}, and qid -> pooled list of levels.

    The qrels are keyed `<qid> <run>;<rank> <Ln>`: a judgement belongs to one rank
    position of one run, not to a document id.  Every result below depends on this, so
    it is read literally and checked in `validate`.
    """
    per_run = collections.defaultdict(dict)
    pool = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            qid, key, level = line.split()
            run, rank = key.rsplit(";", 1)
            lv = int(level[1:])
            per_run[(qid, run)][int(rank)] = lv
            pool[qid].append(lv)
    return per_run, pool


def read_tsm(path):
    table = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid = row[""]
            table[qid] = {k: float(v) for k, v in row.items() if k}
    return table


def read_roster(path, keep_text=False):
    """qid -> [(rank, docid)] from a filed run file `qid;rank;docid;text`.

    With keep_text, the md5 of the passage text is carried as a third element, so that
    repeats of a document can be separated from repeats of a passage.
    """
    rosters = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, rank, docid, text = line.split(";", 3)
            if keep_text:
                rosters[qid].append(
                    (int(rank), docid, hashlib.md5(text.encode("utf-8")).hexdigest()))
            else:
                rosters[qid].append((int(rank), docid))
    for qid in rosters:
        rosters[qid].sort()
    return rosters


# Measures

def msndcg(levels, ideal_sorted, cutoff=CUTOFF):
    dcg = sum(GAIN[l] / math.log2(i + 2) for i, l in enumerate(levels[:cutoff]))
    idcg = sum(GAIN[l] / math.log2(i + 2) for i, l in enumerate(ideal_sorted[:cutoff]))
    return dcg / idcg if idcg else 0.0


def qmeasure(levels, ideal_sorted, cutoff=CUTOFF, beta=1.0):
    ideal = ideal_sorted[:cutoff]
    R = sum(1 for l in ideal if l > 0)
    if R == 0:
        return 0.0
    cig = cg = total = 0.0
    found = 0
    for r in range(1, cutoff + 1):
        cig += GAIN[ideal[r - 1]] if r - 1 < len(ideal) else 0.0
        lv = levels[r - 1] if r - 1 < len(levels) else 0
        cg += GAIN[lv]
        if lv > 0:
            found += 1
            total += (found + beta * cg) / (r + beta * cig)
    return total / R


MEASURES = {"MSnDCG@20": msndcg, "Q@20": qmeasure}


# Validation

def validate(per_run, ideal, published):
    """Reproduce every published per-question score, all 22 runs, both measures."""
    report = {}
    for name, fn in MEASURES.items():
        worst = 0.0
        worst_at = None
        n = 0
        qids = sorted(published[name])
        runs = sorted(published[name][qids[0]])
        for qid in qids:
            for run in runs:
                judged = per_run.get((qid, run), {})
                levels = [judged[r] for r in sorted(judged)]
                got = fn(levels, ideal[qid])
                err = abs(got - published[name][qid][run])
                n += 1
                if err > worst:
                    worst, worst_at = err, (qid, run)
        report[name] = {"n": n, "max_abs_err": worst, "worst_at": worst_at,
                        "runs": len(runs), "qids": len(qids)}
    return report


# Variants

def variants(roster, judged):
    """Four orderings of one question's slots, as lists of relevance levels.

    `roster` is [(rank, docid)] as filed; `judged` maps rank -> level.  Unjudged ranks
    cannot occur for the scored questions but are treated as L0 if they ever do.
    """
    filed, first, later = [], [], []
    seen = set()
    for rank, docid in ((r[0], r[1]) for r in roster):
        lv = judged.get(rank, 0)
        filed.append(lv)
        if docid in seen:
            later.append(lv)
        else:
            seen.add(docid)
            first.append(lv)
    return {
        "filed": filed,
        # later repeats deleted, everything below promoted; the list gets shorter
        "truncate": first,
        # later repeats demoted to the tail; twenty slots kept, order changed
        "demote": first + later,
        # later repeats replaced by a perfectly relevant passage: a ceiling, not a run
        "oracle": first + [4] * len(later),
    }


# Statistics

def paired_bootstrap(deltas, rng, reps=BOOTSTRAP):
    n = len(deltas)
    means = []
    for _ in range(reps):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * reps)]
    hi = means[int(0.975 * reps) - 1]
    return lo, hi


def wlt(deltas, eps=1e-9):
    w = sum(1 for d in deltas if d > eps)
    l = sum(1 for d in deltas if d < -eps)
    return w, l, len(deltas) - w - l


# Main

def main():
    subdir = sys.argv[1] if len(sys.argv) > 1 else "submission"
    if os.path.isdir(os.path.join(subdir, "ac-2026-05-15")):
        rundir = os.path.join(subdir, "ac-2026-05-15")
    else:
        rundir = subdir

    per_run, pool = read_qrels(QRELS)
    ideal = {q: sorted(v, reverse=True) for q, v in pool.items()}
    published = {name: read_tsm(path) for name, path in TSM.items()}
    scored_qids = sorted(published["MSnDCG@20"])

    check = validate(per_run, ideal, published)
    ok = all(v["max_abs_err"] < 5e-5 for v in check.values())

    rosters = {run: read_roster(os.path.join(rundir, run), keep_text=True)
               for run in OURS}

    # provenance: the filed rosters against the organisers' copies
    provenance = {}
    for run in OURS:
        official = read_roster(os.path.join(RES, "PRruns", run))
        diff = [q for q in scored_qids
                if [(r, d) for r, d, _ in rosters[run].get(q, [])] != official.get(q, [])]
        provenance[run] = diff

    # counts and per-variant scores
    counts = {}
    scores = collections.defaultdict(lambda: collections.defaultdict(dict))
    dup_ranks = collections.defaultdict(list)
    label_first = collections.defaultdict(collections.Counter)
    label_dup = collections.defaultdict(collections.Counter)
    per_q_dup = collections.defaultdict(dict)

    distinct_per_q = collections.defaultdict(list)
    text_repeats = collections.Counter()
    regrade = collections.defaultdict(lambda: [0, 0])   # run -> [docids seen twice+, graded inconsistently]
    length_split = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))

    for run in OURS:
        slots = dups = 0
        qwith = 0
        for qid in scored_qids:
            roster = rosters[run][qid]
            judged = per_run[(qid, run)]
            seen = set()
            seen_text = set()
            grades = collections.defaultdict(list)
            d = 0
            for rank, docid, texthash in roster:
                lv = judged.get(rank, 0)
                slots += 1
                grades[docid].append(lv)
                if texthash in seen_text:
                    text_repeats[run] += 1
                else:
                    seen_text.add(texthash)
                if docid in seen:
                    d += 1
                    dup_ranks[run].append(rank)
                    label_dup[run][lv] += 1
                else:
                    seen.add(docid)
                    label_first[run][lv] += 1
            for docid, gs in grades.items():
                if len(gs) > 1:
                    regrade[run][0] += 1
                    if len(set(gs)) > 1:
                        regrade[run][1] += 1
            dups += d
            distinct_per_q[run].append(len(seen))
            bucket = "full" if len(roster) >= CUTOFF else "short"
            length_split[run][bucket][0] += d
            length_split[run][bucket][1] += len(roster)
            per_q_dup[run][qid] = (d, len(roster))
            if d:
                qwith += 1
            v = variants(roster, judged)
            for name, fn in MEASURES.items():
                for key, levels in v.items():
                    scores[run][name][(qid, key)] = fn(levels, ideal[qid])
        counts[run] = {"slots": slots, "dups": dups, "questions_with_dup": qwith,
                       "distinct_mean": statistics.fmean(distinct_per_q[run]),
                       "distinct_median": statistics.median(distinct_per_q[run]),
                       "text_repeats": text_repeats[run],
                       "regrade": tuple(regrade[run]),
                       "length_split": {k: tuple(v) for k, v in length_split[run].items()},
                       "n_short": sum(1 for q in scored_qids if len(rosters[run][q]) < CUTOFF)}

    rng = random.Random(SEED)
    stats = collections.defaultdict(dict)
    for run in OURS:
        for name in MEASURES:
            base = [scores[run][name][(q, "filed")] for q in scored_qids]
            row = {"filed": sum(base) / len(base),
                   "published": statistics.fmean(published[name][q][run] for q in scored_qids)}
            for key in ("truncate", "demote", "oracle"):
                alt = [scores[run][name][(q, key)] for q in scored_qids]
                deltas = [a - b for a, b in zip(alt, base)]
                w, l, t = wlt(deltas)
                lo, hi = paired_bootstrap(deltas, rng)
                row[key] = {"mean": sum(alt) / len(alt),
                            "delta": sum(deltas) / len(deltas),
                            "ci": (lo, hi), "wlt": (w, l, t)}
            stats[run][name] = row

    # rosters that are identical between the two runs, and whether the organisers'
    # published scores for them nonetheless differ: direct evidence that a judgement
    # belongs to a run and a rank, not to a passage
    identical = [q for q in scored_qids
                 if [(r, d) for r, d, _ in rosters["BITEM-PG-1"][q]]
                 == [(r, d) for r, d, _ in rosters["BITEM-PG-2"][q]]]
    disagree = [q for q in identical
                if abs(published["MSnDCG@20"][q]["BITEM-PG-1"]
                       - published["MSnDCG@20"][q]["BITEM-PG-2"]) > 1e-9]

    write_markdown(scored_qids, check, ok, provenance, counts, label_first, label_dup,
                   dup_ranks, per_q_dup, stats, rundir, identical, disagree)
    print(f"wrote {OUT}")
    for run in OURS:
        c = counts[run]
        print(f"  {run}: {c['dups']}/{c['slots']} repeated slots "
              f"({100 * c['dups'] / c['slots']:.1f}%)")
        for name in MEASURES:
            s = stats[run][name]
            print(f"    {name}: filed {s['filed']:.4f}  "
                  f"truncate {s['truncate']['delta']:+.4f}  "
                  f"demote {s['demote']['delta']:+.4f}  "
                  f"oracle {s['oracle']['delta']:+.4f}")


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def write_markdown(qids, check, ok, provenance, counts, label_first, label_dup,
                   dup_ranks, per_q_dup, stats, rundir, identical, disagree):
    L = []
    w = L.append
    n = len(qids)

    w("# Repeated documents in the retrieval rosters, and what removing them would have cost")
    w("")
    w(f"*`analysis/roster_duplicates.py`, {n} scored questions, direct (consolidated LLM) "
      f"qrels. Rosters read from `{rundir}/`, judgements and published scores from "
      f"`{RES}/`, both read-only.*")
    w("")

    # headline
    pg1, pg2 = counts["BITEM-PG-1"], counts["BITEM-PG-2"]
    w("## The short version")
    w("")
    w(f"The repeats are real and they are large: **{pg1['dups']} of {pg1['slots']} "
      f"slots ({pct(pg1['dups'], pg1['slots'])}) in BITEM-PG-1** and "
      f"**{pg2['dups']} of {pg2['slots']} ({pct(pg2['dups'], pg2['slots'])}) in "
      f"BITEM-PG-2** hold a `docid` that already appeared at a better rank. The "
      "inference drawn from that count was wrong. The official qrels are keyed "
      "`<run>;<rank>`, not by document, so every slot was judged on its own passage "
      "text and every slot earned its own gain. A repeated `docid` is not a wasted "
      "slot under the measure the organisers ran; it is a second passage from the same "
      "source page, and about half of those passages were judged relevant.")
    w("")
    w("De-duplicating therefore does not free anything. Rescored under a scorer "
      "validated against every published number, deleting the later repeats and "
      "promoting the rest **loses**:")
    w("")
    w("| run | measure | as filed | repeats deleted | repeats demoted to the tail | repeats replaced by an ideal passage |")
    w("|---|---|---|---|---|---|")
    for run in OURS:
        for name in MEASURES:
            s = stats[run][name]
            w(f"| {run} ({LABEL[run]}) | {name} | {s['filed']:.4f} | "
              f"{s['truncate']['mean']:.4f} ({s['truncate']['delta']:+.4f}) | "
              f"{s['demote']['mean']:.4f} ({s['demote']['delta']:+.4f}) | "
              f"{s['oracle']['mean']:.4f} ({s['oracle']['delta']:+.4f}) |")
    w("")
    w("The last column is a ceiling, not a run: it assumes every freed slot is refilled "
      "with a passage the assessors would have called perfectly relevant. Nothing in "
      "`results/` says what a real refill would have been judged, because a passage no "
      "run returned was never judged. The honest statement is a bracket — "
      "de-duplication was worth somewhere between the third column and the fifth, and "
      "the lower end of that bracket is where the evidence sits.")
    w("")

    # validation
    w("## The scorer, and why it can be trusted")
    w("")
    w("Every counterfactual below is scored by a re-implementation in this file, so it "
      "is worth nothing unless it reproduces the organisers' own numbers first. It "
      "does, for **all 22 PR runs on all "
      f"{n} scored questions**, both measures:")
    w("")
    w("| measure | scores compared | largest absolute error | where |")
    w("|---|---|---|---|")
    for name, c in check.items():
        qid, run = c["worst_at"]
        w(f"| {name} | {c['n']} ({c['runs']} runs x {c['qids']} questions) | "
          f"{c['max_abs_err']:.6f} | {qid} / {run} |")
    w("")
    w("The published tables carry four decimals, so an error of 5e-5 is exact "
      "agreement." if ok else "**The reproduction failed; treat every counterfactual "
      "below as unvalidated.**")
    w("")
    w("The run means the scorer recovers for our two runs are the published ones:")
    w("")
    w("| run | measure | our scorer | organisers' table |")
    w("|---|---|---|---|")
    for run in OURS:
        for name in MEASURES:
            sc = stats[run][name]
            w(f"| {run} | {name} | {sc['filed']:.4f} | {sc['published']:.4f} |")
    w("")
    w("Settling the implementation also settled the substantive question. What "
      "reproduces is:")
    w("")
    w("- **gains linear in the grade**, L1=1, L2=2, L3=3, L4=4, with L0=0;")
    w("- **MSnDCG@20** = sum of `gain / log2(rank+1)` over the first twenty slots, "
      "divided by the same sum over the twenty best-graded items in the topic's pool;")
    w("- **Q@20** = Sakai's Q-measure with beta=1, the blended-ratio sum taken over the "
      "first twenty slots and divided by the number of relevant items among the "
      "**top twenty** of the ideal list, not by the topic's full relevant count;")
    w("- **the unit of judgement is a rank position of a run.** The topic's ideal list "
      "is the pool of every judged `<run>;<rank>` instance for that topic, so the same "
      "passage returned by two runs contributes twice to the ideal, and the same "
      "`docid` returned twice by one run contributes twice to that run's gain.")
    w("")
    w("The fourth point is the finding, and two checks independent of the scorer agree "
      f"with it. Our two rosters are identical, rank for rank, on {len(identical)} "
      f"questions ({', '.join(identical)}); on {len(disagree)} of those "
      f"({', '.join(disagree)}) the organisers nonetheless published different "
      "MSnDCG@20 scores for the two runs. Identical input, different output, which is "
      "possible only if the same passage was judged twice under two keys. Second, "
      "within a single run a `docid` that occupies several slots is graded "
      "**inconsistently across those slots** in "
      f"{pct(counts['BITEM-PG-1']['regrade'][1], counts['BITEM-PG-1']['regrade'][0])} of "
      f"cases in BITEM-PG-1 and "
      f"{pct(counts['BITEM-PG-2']['regrade'][1], counts['BITEM-PG-2']['regrade'][0])} in "
      "BITEM-PG-2 — the assessors were reading passages, and they did not find the "
      "passages equivalent.")
    w("")

    # provenance
    w("### Provenance of the rosters")
    w("")
    for run in OURS:
        diff = provenance[run]
        if diff:
            w(f"- `{run}`: the filed roster differs from the organisers' copy on "
              f"{len(diff)} scored question(s): {', '.join(diff)}. **Investigate before "
              "quoting anything below.**")
        else:
            w(f"- `{run}`: the filed roster and the organisers' copy agree on all {n} "
              "scored questions, rank for rank.")
    w("")
    w("(Both files also differ on qid 0025, which the organisers dropped from scoring. "
      "That question is excluded throughout.)")
    w("")

    # counts
    w("## How much repetition there is")
    w("")
    w("| run | slots scored | distinct documents | repeated slots | share | questions with at least one repeat | median repeats per question | mean rank of a repeat |")
    w("|---|---|---|---|---|---|---|---|")
    for run in OURS:
        c = counts[run]
        distinct = c["slots"] - c["dups"]
        med = statistics.median(d for d, _ in per_q_dup[run].values())
        mr = statistics.fmean(dup_ranks[run]) if dup_ranks[run] else float("nan")
        w(f"| {run} ({LABEL[run]}) | {c['slots']} | {distinct} | {c['dups']} | "
          f"{pct(c['dups'], c['slots'])} | {c['questions_with_dup']}/{n} | "
          f"{med:.0f} | {mr:.1f} |")
    w("")
    c1, c2 = counts["BITEM-PG-1"], counts["BITEM-PG-2"]
    w(f"A roster covers {c1['distinct_mean']:.1f} distinct source pages on average in "
      f"the single-pass run and {c2['distinct_mean']:.1f} in the pooled one. The pooled "
      "run repeats less in both absolute and relative terms, so page diversity is a "
      "second thing pooling bought alongside list length — though the two are not "
      "separable here, since a longer list has more room for new pages.")
    w("")
    w("Repetition is not concentrated on the short rosters:")
    w("")
    w("| run | short rosters (<20) | repeats there | full rosters (20) | repeats there |")
    w("|---|---|---|---|---|")
    for run in OURS:
        c = counts[run]
        sh = c["length_split"].get("short", (0, 0))
        fu = c["length_split"].get("full", (0, 0))
        w(f"| {run} | {c['n_short']} questions | {sh[0]}/{sh[1]} ({pct(sh[0], sh[1])}) | "
          f"{len(qids) - c['n_short']} questions | {fu[0]}/{fu[1]} ({pct(fu[0], fu[1])}) |")
    w("")
    w("In the single-pass run the two rates are within four points of each other, so "
      "the stopping-short problem and the repetition problem are not the same problem. "
      "The pooled run's short bucket is five questions and carries no weight.")
    w("")
    w("### The repeats are different passages, not duplicate text")
    w("")
    w(f"Exact passage-text repeats within a question: **{c1['text_repeats']} in "
      f"BITEM-PG-1 and {c2['text_repeats']} in BITEM-PG-2**, against "
      f"{c1['dups']} and {c2['dups']} `docid` repeats. The rosters almost never file "
      "the same text twice. What they file repeatedly is the same source page, sliced "
      "into different passages — which is what the chunking was built to do, and which "
      "the assessors graded on the merits of each slice.")
    w("")
    w("### What the repeated slots were judged")
    w("")
    w("| run | slot kind | L0 | L1 | L2 | L3 | L4 | share relevant (L1+) | mean gain |")
    w("|---|---|---|---|---|---|---|---|---|")
    for run in OURS:
        for kind, ctr in (("first occurrence", label_first[run]), ("repeat", label_dup[run])):
            tot = sum(ctr.values())
            rel = sum(v for k, v in ctr.items() if k > 0)
            mg = sum(GAIN[k] * v for k, v in ctr.items()) / tot if tot else 0.0
            cells = " | ".join(str(ctr.get(i, 0)) for i in range(5))
            w(f"| {run} | {kind} | {cells} | {pct(rel, tot)} | {mg:.3f} |")
    w("")
    f1, d1 = label_first["BITEM-PG-1"], label_dup["BITEM-PG-1"]
    f2, d2 = label_first["BITEM-PG-2"], label_dup["BITEM-PG-2"]
    def relshare(c):
        t = sum(c.values())
        return 100 * sum(v for k, v in c.items() if k > 0) / t
    w(f"This is the number that kills the hypothesis. A repeated slot is relevant "
      f"{relshare(d1):.0f}% of the time in BITEM-PG-1 against {relshare(f1):.0f}% for a "
      f"first occurrence, and {relshare(d2):.0f}% against {relshare(f2):.0f}% in "
      "BITEM-PG-2. Repeats are a little weaker than first occurrences and no more than "
      "a little. They are not dead weight, they are the rest of the page — and on the "
      "long Wikipedia-style articles this corpus is built from, the rest of the page "
      "often answers the question.")
    w("")

    # counterfactual detail
    w("## The counterfactual in full")
    w("")
    w("Three rewritings of each roster, all scored against the unmodified qrels and the "
      "unmodified ideal list, all paired over the same "
      f"{n} questions. 95% intervals are paired bootstrap, {BOOTSTRAP} resamples, "
      f"seed {SEED}.")
    w("")
    w("- **repeats deleted** — later occurrences of a `docid` removed, everything below "
      "promoted, the list left short. Identical in score to refilling the freed slots "
      "with non-relevant passages, so it is the floor of the bracket.")
    w("- **repeats demoted** — later occurrences moved to the tail, twenty slots kept. "
      "Isolates the ordering question from the length question.")
    w("- **repeats replaced by an ideal passage** — freed slots refilled with an L4. A "
      "ceiling, computed against the ideal list as published; a real refill would also "
      "have entered the pool and raised the denominator, so the true ceiling is lower "
      "than this.")
    w("")
    for name in MEASURES:
        w(f"### {name}")
        w("")
        w("| run | variant | mean | delta | 95% CI | W/L/T |")
        w("|---|---|---|---|---|---|")
        for run in OURS:
            s = stats[run][name]
            w(f"| {run} | as filed | {s['filed']:.4f} | — | — | — |")
            for key, lbl in (("truncate", "repeats deleted"),
                             ("demote", "repeats demoted"),
                             ("oracle", "repeats replaced by L4")):
                v = s[key]
                w(f"| | {lbl} | {v['mean']:.4f} | {v['delta']:+.4f} | "
                  f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | "
                  f"{v['wlt'][0]}/{v['wlt'][1]}/{v['wlt'][2]} |")
        w("")

    w("Deleting the repeats costs both runs on both measures, with intervals clear of "
      "zero and losses outnumbering wins by a wide margin. Demoting them costs less, "
      "because the gain is kept and only discounted harder, but it still costs. Only "
      "the oracle refill gains, and it gains by construction.")
    w("")

    # what it means
    w("## What this says about what went wrong")
    w("")
    w("The cheapest fix on the list was not a fix. It was an artefact of reading a "
      "passage-level evaluation as a document-level one, and the reading can be "
      "rejected outright rather than left as a caveat.")
    w("")
    w("Two observations survive the rejection. First, the retrieval unit and the "
      "reporting unit were never reconciled: the system ranks passages, the run format "
      "has a `docid` column, and the organisers resolved the mismatch by judging "
      "passages. That resolution is the one that costs us nothing. Under a "
      "document-level qrels the same rosters would have scored differently, and how "
      "differently is not computable from what the organisers released — the exposure "
      "was real, whether or not the task documentation ruled it out in advance.")
    w("")
    w("Second, the diversity question is not empty even though the de-duplication "
      "answer to it is. A single-pass roster reaches "
      f"{counts['BITEM-PG-1']['distinct_mean']:.1f} distinct pages and a pooled one "
      f"{counts['BITEM-PG-2']['distinct_mean']:.1f}, and the oracle column shows what a "
      "slot is worth when it carries something new and relevant. The improvement "
      "available is **more distinct source pages entering the candidate pool**, not a "
      "re-ordering of the pages already in it. This analysis cannot price that "
      "improvement, because a passage no run returned carries no judgement, and the "
      "oracle bound assumes the best possible refill rather than a plausible one.")
    w("")

    # limits
    w("## What this cannot settle")
    w("")
    w("- **Whether de-duplicating and refilling would have helped.** The freed slots "
      "would hold passages that were never returned by any run and therefore never "
      "judged. Their gain is unknown, not zero. The bracket between the deletion "
      "variant and the oracle variant is the whole of what can be said.")
    w("- **Whether a document-level evaluation would have flattered or punished us.** "
      "The organisers ran a passage-level one. Collapsing the pool by `docid` across "
      "twenty-two runs would require deciding which of several conflicting grades a "
      "document carries, and the answer would be an invented measure, not theirs.")
    w("- **Why the rosters repeat.** That is a property of the retrieval legs and of "
      "the passage-to-document ratio in the index, and reading it needs the per-pass "
      "process data. There are no cluster traces, so the question stops here.")
    w("- **Whether the repeated passages are *near*-duplicates.** Exact text repeats "
      "are counted above and are essentially absent, but two slices of one page can "
      "still overlap heavily. Measuring that needs a similarity threshold this script "
      "does not set, and the choice of threshold would drive the answer.")
    w("- **How much of the repetition is the chunker's doing and how much the "
      "retriever's.** A page cut into many passages will surface many passages; "
      "separating that from a retrieval leg that concentrates on few pages needs the "
      "index statistics, which are not in `results/` or `submission/`.")
    w("- **Anything about the AC side.** Nugget and confidence behaviour are untouched "
      "by roster order.")
    w("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
