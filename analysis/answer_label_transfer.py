#!/usr/bin/env python3
"""Where the labels for the unfiled runs of the paper's Additional experiments come from.

    /usr/bin/python3 analysis/answer_label_transfer.py submission

The three runs in that section were never filed, so the organisers never judged them. Each of
their 3 x 64 = 192 answers is nonetheless labelled, and this script reports how, so that the
section's claim about our own evaluation can be checked.

What is measured. For each answer, the first of these that applies:

    own        the same answer, normalised, was judged by the organisers in one of OUR filed runs
    other      it was judged in another team's run and in none of ours
    both       judged in ours and in another team's, which must agree or the conflict is reported
    hand       no judged match anywhere, and we ruled on it by hand (analysis/manual_judgements.tsv)
    rule       no judged match and no hand verdict, so the containment fallback decided it

The share of `own` against `other` is the main figure. These runs are rebuilt from our own
traces, so a high `own` share means the rebuilding mostly reshuffles answers we already filed
rather than reaching new ones, and the label transfer is then close to free. An answer credited
only by another team's judgement is one the organisers never saw from us.

The stricter normalisation. `score_ac.norm` casefolds, turns every non-word character into a
space and collapses whitespace, so "Movie1, Movie2" and "movie1,Movie2" already agree. Deleting
whitespace entirely would additionally join "AT-AT" to "ATAT". This script reports how many
further answers that would label and prints each of them beside the answer it matched, because a
looser key is only worth having if each new pair is a real equivalence rather than two different
answers colliding. The script does not adopt the looser key: the paper uses the tighter rule, and
this table is the evidence for whether that should change.

Pure read. Touches no submitted file and nothing under results/.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import score_ac  # noqa: E402

OURS = ("BITEM-AC-1", "BITEM-AC-2", "BITEM-AC-3", "BITEM-AC-4")

#: The runs of the Additional experiments section, in the order the table prints them.
SECTION_RUNS = (
    ("model as a judge (recovered)", "submission/unofficial/model-as-judge.AC"),
    ("single pass, longest deliberation", "submission/unofficial/revised-longest-deliberation.AC"),
    ("four passes, independent agreement", "submission/unofficial/revised-independent-agreement.AC"),
)


def tight(s: str) -> str:
    """score_ac.norm with the whitespace removed as well."""
    return score_ac.norm(s).replace(" ", "")


def judged_answers(collect=None):
    """{(qid, norm answer): {run name: label}} over every judged answer of every AC run."""
    runs = ROOT / "results/OfficialResults/ACruns"
    nev = ROOT / "results/OfficialResults/ACnev"
    out = collections.defaultdict(dict)
    for rf in sorted(runs.iterdir()):
        stem = re.sub(r"\.txt$", "", rf.name)
        nf = nev / f"{stem}.official.ACnev"
        if not nf.exists():
            continue
        correct = {m.group(1): int(m.group(2)) for m in
                   re.finditer(r"(\d{4})\s+Correct=\s*(-?\d+)",
                               nf.read_text(encoding="utf-8", errors="replace"))}
        for q, pair in score_ac.read_ac(rf).items():
            ans = pair[0]
            if correct.get(q, -1) >= 0:
                name = stem.replace(".official", "")
                out[(q, score_ac.norm(ans))][name] = correct[q]
                if collect is not None:
                    collect.setdefault(q, []).append((name, ans, correct[q]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", nargs="?", default="submission")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "answer_label_transfer.md"))
    a = ap.parse_args()

    all_judged = {}
    judged = judged_answers(all_judged)
    by_tight = collections.defaultdict(set)
    for (q, n), labels in judged.items():
        by_tight[(q, n.replace(" ", ""))].add((n, tuple(sorted(labels.items()))))

    L = ["# Where the labels for the unfiled runs come from", "",
         "Produced by `analysis/answer_label_transfer.py`. Every answer of the three runs in the",
         "paper's Additional experiments section, and how it came to be labelled.", ""]

    rows, extra, conflicts = [], [], []
    totals = collections.Counter()
    for name, rel in SECTION_RUNS:
        filed = score_ac.read_ac(ROOT / rel)
        c = collections.Counter()
        for q in sorted(score_ac.SCORED):
            if q not in filed:
                continue
            ans = filed[q][0]
            key = (q, score_ac.norm(ans))
            labels = judged.get(key, {})
            if labels:
                mine = {r: v for r, v in labels.items() if any(o in r for o in OURS)}
                theirs = {r: v for r, v in labels.items() if not any(o in r for o in OURS)}
                kind = "both" if mine and theirs else ("own" if mine else "other")
                if len(set(labels.values())) > 1:
                    conflicts.append((name, q, ans, labels))
            elif (q, score_ac.norm(ans)) in score_ac.MANUAL:
                kind = "hand"
            else:
                kind = "rule"
                hits = by_tight.get((q, tight(ans)), set())
                for other_norm, labs in sorted(hits):
                    extra.append((name, q, ans, other_norm, dict(labs)))
            c[kind] += 1
            totals[kind] += 1
        rows.append((name, c, sum(c.values())))

    L += ["## 1. Per run", "",
          "| run | own | other team only | both | hand | fallback | n |",
          "|---|---|---|---|---|---|---|"]
    for name, c, n in rows:
        L.append("| %s | %d | %d | %d | %d | %d | %d |"
                 % (name, c["own"], c["other"], c["both"], c["hand"], c["rule"], n))
    n_all = sum(n for _, _, n in rows)
    L.append("| **all three** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** |"
             % (totals["own"], totals["other"], totals["both"], totals["hand"],
                totals["rule"], n_all))
    L += ["",
          "`own` and `both` are answers the organisers had already judged coming from us, so the",
          "label costs nothing and the rebuilding reused a string we filed. `other team only` is an",
          "answer no run of ours ever filed that another team did, and the organisers' judgement of",
          "theirs carries over to ours.", ""]

    L += ["## 2. Would a whitespace-free key label any of the rest?", ""]
    if not extra:
        L += ["No. Every answer that the exact key misses is missed by the whitespace-free key too,",
              "so the looser rule would label nothing further and there is nothing to inspect.", ""]
    else:
        L += ["%d further answer(s) would be labelled. Each is printed against the judged answer it"
              % len(extra),
              "would be joined to, because a looser key is worth adopting only if every pair is a",
              "real equivalence.", "",
              "| run | qid | our answer | judged answer it would match | label(s) |",
              "|---|---|---|---|---|"]
        for name, q, ans, other, labs in extra:
            L.append("| %s | %s | `%s` | `%s` | %s |"
                     % (name, q, ans[:70], other[:70],
                        ", ".join("%s=%d" % kv for kv in sorted(labs.items()))[:60]))
        L.append("")

    L += ["## 4. The other-team evidence behind each hand verdict", "",
          "A hand verdict was read beside the answers the organisers had already judged on the same",
          "question, so each is shown here with them. A verdict agreeing with how they treated an",
          "equivalent answer is supported; one with nothing comparable beside it is not.", ""]
    import csv as _csv
    with (ROOT / "analysis" / "manual_judgements.tsv").open(encoding="utf-8") as fh:
        for r in _csv.DictReader(fh, delimiter="\t"):
            q = r["qid"]
            others = [(rn, aj, c) for (rn, aj, c) in all_judged.get(q, [])
                      if not rn.startswith("BITEM")]
            L.append("**%s**, gold `%s`, our verdict **%s** for `%s`"
                     % (q, score_ac.GOLD.get(q, ""), "right" if r["correct"] == "1" else "wrong",
                        r["norm_answer"][:80]))
            L.append("")
            if not others:
                L += ["No other team's answer to this question was judged.", ""]
                continue
            L += ["| run | answer | judged |", "|---|---|---|"]
            for rn, aj, c in sorted(others):
                L.append("| %s | %s | %s |" % (rn, aj[:90].replace("|", "/"),
                                               "right" if c else "wrong"))
            L.append("")

    L += ["## 3. Answers the organisers judged inconsistently", ""]
    if not conflicts:
        L += ["None among these runs' answers.", ""]
    else:
        L += ["| run | qid | answer | labels |", "|---|---|---|---|"]
        for name, q, ans, labels in conflicts:
            L.append("| %s | %s | `%s` | %s |"
                     % (name, q, ans[:60],
                        ", ".join("%s=%d" % kv for kv in sorted(labels.items()))[:70]))
        L += ["", "`score_ac._official_labels` takes the optimistic label on a conflict.", ""]

    pathlib.Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
