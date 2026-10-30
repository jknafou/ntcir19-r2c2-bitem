#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""How well three measures of the first pass separate its wrong answers from its right ones.

For the first pass on each scored question, each measure is used alone as a score, and its AUC
is the probability that a wrong answer scores higher than a right one, ties counted at half.
The answers are those of the rebuilt single-pass run, which takes each first pass's own answer.

    python3 analysis/longest_turn_auc.py

Reads analysis/reasoning_behaviour.tsv and submission/unofficial/revised-longest-deliberation.AC,
and the organisers' judgements through score_ac.
"""
import csv, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_ac import read_ac, judge, SCORED           # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
MEASURES = [("cot_max_chars", "longest reasoning turn, in characters"),
            ("n_distinct_q", "distinct search queries"),
            ("calls_used", "tool calls")]


def auc(scores, wrong):
    w = [s for s, k in zip(scores, wrong) if k]
    r = [s for s, k in zip(scores, wrong) if not k]
    return sum((a > b) + 0.5 * (a == b) for a in w for b in r) / (len(w) * len(r))


def main():
    run = read_ac(HERE.parent / "submission" / "unofficial" / "revised-longest-deliberation.AC")
    with (HERE / "reasoning_behaviour.tsv").open(encoding="utf-8") as fh:
        first = {r["qid"].zfill(4): r for r in csv.DictReader(fh, delimiter="\t") if r["idx"] == "0"}
    qs = sorted(SCORED & set(run) & set(first))
    wrong = [not judge(q, run[q][0])[0] for q in qs]
    print(f"{len(qs)} first passes, {sum(wrong)} wrong, {len(qs) - sum(wrong)} right")
    for key, label in MEASURES:
        print(f"  AUC {auc([float(first[q][key]) for q in qs], wrong):.3f}  {label}")


if __name__ == "__main__":
    main()
