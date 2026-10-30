#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Did assembling passes help *answering*?

BITEM-AC-1 is built from the first pass on each question. BITEM-AC-2 (max-conf)
and BITEM-AC-3 (conf-sum-cluster) are two aggregators over all the recorded
passes. This script compares them on the official per-question data, decomposes
HMR into R_O and R_U so it is visible which half moved, and separates the
accuracy question from the confidence question.

BITEM-AC-4 is excluded throughout. The filed BITEM-AC-4 is a byte-for-byte copy
of BITEM-AC-3 (both MD5 dc04428a47b429b5363b150627bc7fb9, in our
`submission/ac-2026-05-15/` and in the organisers' `ACruns/`), so it is not a
distinct system; it is used here only once, as a free measurement of the
nugget-assessment pipeline's own noise, since the organisers marked the same
bytes twice.

Sources, all read-only:
  analysis/results_cache.tsv                        per-question Conf/Correct/NugPrec
  results/OfficialResults/official.AC.csv           published summary, for verification
  results/OfficialResults/markedACruns/*.marked.output   per-nugget R/N/B marks
  submission/ac-2026-05-15/BITEM-AC-{1,2,3}         filed answers and confidences

Usage:  uv run python analysis/ac_ensemble.py submission
"""

from __future__ import annotations

import csv
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OFFICIAL = REPO / "results" / "OfficialResults"
CACHE = HERE / "results_cache.tsv"
OUT_MD = HERE / "ac_ensemble.md"

RUNS = ["AC-1", "AC-2", "AC-3"]
KEY = {"AC-1": "ac1", "AC-2": "ac2", "AC-3": "ac3"}
LABEL = {
    "AC-1": "BITEM-AC-1 (first pass)",
    "AC-2": "BITEM-AC-2 (all passes, max-conf)",
    "AC-3": "BITEM-AC-3 (all passes, conf-sum)",
}
BOOT = 20000
SEED = 20260822


# Measures

def rewards(pairs):
    """pairs: list of (p in [0,1], correct in {0,1}). Returns the AC measure set."""
    n = len(pairs)
    pos = [p for p, c in pairs if c == 1]
    neg = [p for p, c in pairs if c == 0]
    acc = len(pos) / n
    avconf = sum(p for p, _ in pairs) / n
    r_o = 1.0 if not neg else 1.0 - sum(neg) / len(neg)
    r_u = 1.0 if not pos else sum(pos) / len(pos)
    hmr = 0.0 if (r_o == 0 and r_u == 0) else 2 * r_o * r_u / (r_o + r_u)
    return {
        "n": n, "acc": acc, "avconf": avconf, "R_O": r_o, "R_U": r_u,
        "HMR": hmr, "OB": avconf - acc, "n_wrong": len(neg), "n_right": len(pos),
    }


def hmr_of(r_o, r_u):
    return 0.0 if (r_o == 0 and r_u == 0) else 2 * r_o * r_u / (r_o + r_u)


def auc(pairs):
    """P(conf of a random correct > conf of a random incorrect), ties at 0.5."""
    pos = [p for p, c in pairs if c == 1]
    neg = [p for p, c in pairs if c == 0]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def tie_rate(pairs):
    """Fraction of (correct, incorrect) pairs on which the two confidences are equal."""
    pos = [p for p, c in pairs if c == 1]
    neg = [p for p, c in pairs if c == 0]
    if not pos or not neg:
        return None
    return sum(a == b for a in pos for b in neg) / (len(pos) * len(neg))


# Loading

def load_cache():
    with CACHE.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    scored = [r for r in rows if r["ac1_correct"] != "NA"]
    return rows, scored


def load_official_summary():
    out = {}
    with (OFFICIAL / "official.AC.csv").open() as fh:
        for row in csv.DictReader(fh):
            name = row[""].replace(".official", "")
            if not name.startswith("BITEM-AC"):
                continue
            out[name.replace("BITEM-", "")] = {
                "n": int(row["n"]), "R_O": float(row["R_O"]), "R_U": float(row["R_U"]),
                "HMR": float(row["HMR"]), "avconf": float(row["avconf"]),
                "acc": float(row["accuracy"]), "OB": float(row["OB"]),
            }
    return out


NUG_RE = re.compile(r"^([RNB])(\d+);")

_AC4_NP = None


def _ac4_np(qid):
    """AC-4's per-question NugPrec, read from ACnev. AC-4 is otherwise excluded;
    it is used only to size the marking pipeline's own noise on identical bytes."""
    global _AC4_NP
    if _AC4_NP is None:
        _AC4_NP = {}
        path = OFFICIAL / "ACnev" / "BITEM-AC-4.official.ACnev"
        for line in path.read_text().splitlines():
            m = re.match(r"^(\d{4})\s+NugPrec=\s*([\d.]+)", line)
            if m:
                _AC4_NP[m.group(1)] = float(m.group(2))
    return _AC4_NP[qid]


def load_marks(run):
    """Per-qid list of R/N/B nugget marks, in file order, from the marked file.

    Order matters: AC-3 and AC-4 are the same bytes marked twice, so the two
    marking passes can only be compared nugget by nugget, in position.
    """
    path = OFFICIAL / "markedACruns" / f"BITEM-{run}.marked.output"
    marks, qid = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        m = re.fullmatch(r"<(\d{4})>", line)
        if m:
            qid = m.group(1)
            marks[qid] = []
            continue
        if line.startswith("</"):
            qid = None
            continue
        m = NUG_RE.match(line)
        if m and qid:
            marks[qid].append(m.group(1))
    return marks


def load_filed(subdir, run):
    """Per-qid (answer, confidence) from our filed file, for cross-checking."""
    path = subdir / f"BITEM-{run}"
    if not path.exists():
        return None
    out, qid, expect = {}, None, False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        m = re.fullmatch(r"<(\d{4})>", stripped)
        if m:
            qid, expect = m.group(1), True
            continue
        if stripped.startswith("</"):
            qid, expect = None, False
            continue
        if qid and expect:
            # the first line of a block is the answer line, whatever it looks like
            ans, _, conf = stripped.rpartition(";")
            out[qid] = (ans, conf.strip())
            expect = False
    return out


# Statistics

def boot_ci(fn_a, fn_b, scored, rng, reps=BOOT, alpha=0.05):
    """Paired bootstrap over questions of a run-level statistic difference."""
    n = len(scored)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [scored[i] for i in idx]
        try:
            diffs.append(fn_b(sample) - fn_a(sample))
        except ZeroDivisionError:
            continue
    diffs.sort()
    lo = diffs[int(alpha / 2 * len(diffs))]
    hi = diffs[int((1 - alpha / 2) * len(diffs)) - 1]
    frac_le0 = sum(d <= 0 for d in diffs) / len(diffs)
    return lo, hi, frac_le0


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired 0/1 vectors. Returns (b01, b10, p)."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b01, b10, min(1.0, 2 * tail)


def perm_test_paired(vals_a, vals_b, rng, reps=BOOT):
    """Two-sided paired sign-flip randomisation test on a mean difference."""
    d = [b - a for a, b in zip(vals_a, vals_b)]
    obs = sum(d) / len(d)
    hits = 0
    for _ in range(reps):
        s = sum(x if rng.random() < 0.5 else -x for x in d) / len(d)
        if abs(s) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (reps + 1)


# Entry point

def main():
    subdir = None
    if len(sys.argv) > 1:
        subdir = (REPO / sys.argv[1]) if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
        if subdir.name == "submission":
            subdir = subdir / "ac-2026-05-15"

    rng = random.Random(SEED)
    rows, scored = load_cache()
    official = load_official_summary()

    pairs = {r: [(int(x[KEY[r] + "_conf"]) / 100, int(x[KEY[r] + "_correct"])) for x in scored]
             for r in RUNS}
    stats = {r: rewards(pairs[r]) for r in RUNS}

    # Verification against the published table
    verif = []
    for r in RUNS:
        for k in ("R_O", "R_U", "HMR", "avconf", "OB"):
            verif.append((r, k, stats[r][k], official[r][k],
                          abs(round(stats[r][k], 4) - official[r][k]) < 5e-5))
        verif.append((r, "acc", stats[r]["acc"], official[r]["acc"],
                      abs(round(stats[r]["acc"], 4) - official[r]["acc"]) < 5e-5))
    verif_ok = all(v[-1] for v in verif)

    # AC-4 is scored in the official table with the same numbers as AC-3; confirm it
    ac4_same = official.get("AC-4") == official.get("AC-3")

    # filed-file cross-check
    filed_ok = None
    if subdir is not None:
        filed_ok = True
        for r in RUNS:
            filed = load_filed(subdir, r)
            if filed is None:
                filed_ok = None
                break
            for x in scored:
                if filed[x["qid"]][1] != x[KEY[r] + "_conf"]:
                    filed_ok = False

    # Cross-combination: whose correctness, whose confidence
    cross = {}
    for rc in RUNS:          # correctness from
        for rp in RUNS:      # confidence from
            mix = [(int(x[KEY[rp] + "_conf"]) / 100, int(x[KEY[rc] + "_correct"]))
                   for x in scored]
            cross[(rc, rp)] = rewards(mix)

    # Correctness disagreement
    patterns = Counter(tuple(int(x[KEY[r] + "_correct"]) for r in RUNS) for x in scored)
    wrongsets = {r: [x["qid"] for x in scored if int(x[KEY[r] + "_correct"]) == 0] for r in RUNS}
    uniquely_right = {r: sorted(set().union(*[set(wrongsets[o]) for o in RUNS if o != r])
                                - set(wrongsets[r])) for r in RUNS}
    uniquely_wrong = {r: sorted(set(wrongsets[r])
                                - set().union(*[set(wrongsets[o]) for o in RUNS if o != r]))
                      for r in RUNS}

    # Confidence shape
    confdist = {}
    for r in RUNS:
        cs = [int(x[KEY[r] + "_conf"]) for x in scored]
        confdist[r] = {
            "distinct": len(set(cs)),
            "at100": sum(c == 100 for c in cs),
            "ge95": sum(c >= 95 for c in cs),
            "le10": sum(c <= 10 for c in cs),
            "median": sorted(cs)[len(cs) // 2],
            "wrong_at100": sum(1 for x in scored
                               if int(x[KEY[r] + "_conf"]) == 100
                               and int(x[KEY[r] + "_correct"]) == 0),
            "mean_conf_wrong": (sum(int(x[KEY[r] + "_conf"]) for x in scored
                                    if int(x[KEY[r] + "_correct"]) == 0)
                                / max(1, stats[r]["n_wrong"])),
            "auc": auc(pairs[r]),
            "ties": tie_rate(pairs[r]),
            "top": Counter(cs).most_common(5),
        }

    # Pairwise tests, AC-1 against each ensemble run
    tests = {}
    for r in ("AC-2", "AC-3"):
        a, b = "AC-1", r
        ca = [int(x[KEY[a] + "_correct"]) for x in scored]
        cb = [int(x[KEY[b] + "_correct"]) for x in scored]
        b01, b10, p_mcn = mcnemar_exact(ca, cb)
        t = {"mcnemar": (b01, b10, p_mcn)}
        for stat in ("HMR", "R_O", "R_U", "acc", "avconf"):
            fa = lambda s, a=a, k=stat: rewards(
                [(int(x[KEY[a] + "_conf"]) / 100, int(x[KEY[a] + "_correct"])) for x in s])[k]
            fb = lambda s, b=b, k=stat: rewards(
                [(int(x[KEY[b] + "_conf"]) / 100, int(x[KEY[b] + "_correct"])) for x in s])[k]
            t[stat] = (stats[b][stat] - stats[a][stat],) + boot_ci(fa, fb, scored, rng)
        # AUC of the confidence signal, paired bootstrap
        fa = lambda s, a=a: auc(
            [(int(x[KEY[a] + "_conf"]) / 100, int(x[KEY[a] + "_correct"])) for x in s])
        fb = lambda s, b=b: auc(
            [(int(x[KEY[b] + "_conf"]) / 100, int(x[KEY[b] + "_correct"])) for x in s])
        def safe(fn):
            def g(s):
                v = fn(s)
                if v is None:
                    raise ZeroDivisionError
                return v
            return g
        t["AUC"] = ((confdist[b]["auc"] - confdist[a]["auc"],)
                    + boot_ci(safe(fa), safe(fb), scored, rng))
        # nugget precision, paired sign-flip
        na = [float(x[KEY[a] + "_nugprec"]) for x in scored]
        nb = [float(x[KEY[b] + "_nugprec"]) for x in scored]
        t["nugprec"] = perm_test_paired(na, nb, rng)
        tests[r] = t

    # Which component of HMR carries the change
    decomp = {}
    for b in ("AC-2", "AC-3"):
        a = "AC-1"
        base = hmr_of(stats[a]["R_O"], stats[a]["R_U"])
        ro_only = hmr_of(stats[b]["R_O"], stats[a]["R_U"])
        ru_only = hmr_of(stats[a]["R_O"], stats[b]["R_U"])
        full = hmr_of(stats[b]["R_O"], stats[b]["R_U"])
        decomp[b] = {"base": base, "d_ro": ro_only - base, "d_ru": ru_only - base,
                     "inter": full - base - (ro_only - base) - (ru_only - base),
                     "full": full}

    # AC-2 vs AC-3, the two aggregators against each other
    ca = [int(x["ac2_correct"]) for x in scored]
    cb = [int(x["ac3_correct"]) for x in scored]
    ac23_mcnemar = mcnemar_exact(ca, cb)

    # Per question type
    types = sorted({x["qtype"] for x in scored})
    per_type = {}
    for t in types:
        sub = [x for x in scored if x["qtype"] == t]
        per_type[t] = {"n": len(sub)}
        for r in RUNS:
            per_type[t][r] = rewards(
                [(int(x[KEY[r] + "_conf"]) / 100, int(x[KEY[r] + "_correct"])) for x in sub])

    # Nuggets
    marks = {r: load_marks(r) for r in RUNS + ["AC-4"]}
    nug = {}
    for r in RUNS:
        tot = Counter()
        for q in [x["qid"] for x in scored]:
            tot += Counter(marks[r].get(q, []))
        n_all = tot["R"] + tot["N"] + tot["B"]
        nug[r] = {
            "R": tot["R"], "N": tot["N"], "B": tot["B"], "total": n_all,
            "per_q": n_all / len(scored),
            "meanNP": sum(float(x[KEY[r] + "_nugprec"]) for x in scored) / len(scored),
            "b_rate": tot["B"] / n_all,
        }
    # NugPrec is forced to 0 on a wrong answer (all non-bogus nuggets of a NO question
    # are auto-labelled N), so restrict to the questions all three runs got right.
    common = [x for x in scored if all(int(x[KEY[r] + "_correct"]) == 1 for r in RUNS)]
    for r in RUNS:
        nug[r]["meanNP_common"] = (sum(float(x[KEY[r] + "_nugprec"]) for x in common)
                                   / len(common))
    nug["_common_n"] = len(common)

    # AC-3 vs AC-4: the same bytes marked twice, which gives the assessment noise floor
    flips, tot3, flip_qs, flip_kinds = 0, 0, 0, Counter()
    for q in [x["qid"] for x in scored]:
        m3, m4 = marks["AC-3"].get(q, []), marks["AC-4"].get(q, [])
        assert len(m3) == len(m4), f"marked files misaligned at {q}"
        tot3 += len(m3)
        d = sum(1 for a, b in zip(m3, m4) if a != b)
        flips += d
        flip_qs += d > 0
        flip_kinds += Counter(f"{a}->{b}" for a, b in zip(m3, m4) if a != b)
    np_diff = sum(1 for x in scored
                  if abs(float(x["ac3_nugprec"]) - _ac4_np(x["qid"])) > 1e-9)

    write_md(locals())
    print(f"wrote {OUT_MD}")


# Report

def d_share(run, decomp):
    d = decomp[run]
    tot = abs(d["d_ro"]) + abs(d["d_ru"])
    return abs(d["d_ro"]) / tot if tot else 0.0


def f(x, nd=4):
    return "NA" if x is None else f"{x:.{nd}f}"


def write_md(ns):
    stats, cross, RUNS_, LABEL_ = ns["stats"], ns["cross"], RUNS, LABEL
    scored, tests, per_type, nug = ns["scored"], ns["tests"], ns["per_type"], ns["nug"]
    confdist, patterns, decomp = ns["confdist"], ns["patterns"], ns["decomp"]
    L = []
    w = L.append

    w("# Did assembling passes help answering?")
    w("")
    w("*Generated by `analysis/ac_ensemble.py`. Read-only over "
      "`results/OfficialResults/` and `submission/ac-2026-05-15/`.*")
    w("")
    w("**BITEM-AC-1** is built from the first pass on each question. **BITEM-AC-2** "
      "(max-conf) and **BITEM-AC-3** (conf-sum-cluster) are two aggregators over all the "
      "recorded passes. Comparing AC-1 against AC-2 and AC-3 is the whole of the "
      "answering-side ensemble question.")
    w("")
    w("**BITEM-AC-4 is excluded.** The filed file is a byte-for-byte copy of BITEM-AC-3 "
      "(MD5 `dc04428a47b429b5363b150627bc7fb9` in our `submission/ac-2026-05-15/` and in "
      "the organisers' `ACruns/` alike), so it is not a distinct system. It was nevertheless "
      "scored as a 25th run, so the official AC table carries two identical BITEM rows and "
      "every field-wide significance test in the shipped p-value files is computed over a "
      "duplicated column. AC-4 appears once below, and only as a measurement of the "
      "organisers' own nugget-assessment noise.")
    w("")
    w(f"All numbers below are over the **{len(scored)} scored questions** "
      "(the 65 filed minus 0025, which the organisers excluded from every official table).")
    w("")

    w("## 0. The short answer")
    w("")
    w("**Not on accuracy — and the design could not have shown it if it had.** "
      f"{stats['AC-1']['n_wrong']}, {stats['AC-2']['n_wrong']} and "
      f"{stats['AC-3']['n_wrong']} wrong answers out of {stats['AC-1']['n']}, "
      "56 questions answered correctly by all three runs, and one or two discordant "
      "questions on every pairwise comparison. Nothing separable.")
    w("")
    w("**Yes on the confidence signal, and that is the interesting result.** The "
      "first-pass run's confidence does not rank its correct answers above its wrong ones "
      f"at all: AUC {f(confdist['AC-1']['auc'], 3)}, which is chance, or a hair below it. "
      f"Both aggregators over all the passes do rank them "
      f"(AUC {f(confdist['AC-2']['auc'], 3)} and {f(confdist['AC-3']['auc'], 3)}), and HMR "
      f"rises accordingly, by {stats['AC-2']['HMR'] - stats['AC-1']['HMR']:+.4f} and "
      f"{stats['AC-3']['HMR'] - stats['AC-1']['HMR']:+.4f}. Aggregating across passes "
      "produced a confidence number that carries information the single pass's did not.")
    w("")
    w("**The result rests on very little.** Everything in the paragraph above is decided by "
      "eight questions — the union of the three runs' wrong answers — and in particular by "
      "three questions the aggregators got wrong just as the first pass did, but filed at "
      "lower confidence (0019 at 84 → 70, 0039 at 41 → 0, 0049 at 78 → 60). Section 6 gives "
      "intervals; they are wide, and the honest summary is a direction, not an established "
      "effect.")
    w("")
    w("**A separate cost is visible.** The ensemble runs cite twice the nuggets and roughly "
      "twice the *relevant* nuggets, at lower nugget precision (Section 8). The official "
      "measure has no recall term, so that trade is scored as a loss.")
    w("")

    w("## 1. Verification")
    w("")
    w("Every measure is recomputed here from the per-question `ACnev` data rather than "
      "copied from the published table, and then checked against it.")
    w("")
    w(f"- recomputed vs `official.AC.csv`, all 18 values (R_O, R_U, HMR, avconf, accuracy, "
      f"OB × 3 runs): **{'match to 4 dp' if ns['verif_ok'] else 'MISMATCH'}**")
    w(f"- the official AC-4 row is identical to the official AC-3 row on every column: "
      f"**{ns['ac4_same']}**")
    if ns["filed_ok"] is not None:
        w(f"- confidences in the recomputed table agree with the filed submission files: "
          f"**{ns['filed_ok']}**")
    w("")

    w("## 2. The headline: accuracy is flat, HMR is not")
    w("")
    w("| Run | accuracy | wrong | R_O | R_U | HMR | avconf | OB |")
    w("|---|---|---|---|---|---|---|---|")
    for r in RUNS_:
        s = stats[r]
        w(f"| {LABEL_[r]} | {f(s['acc'])} | {s['n_wrong']}/{s['n']} | {f(s['R_O'])} | "
          f"{f(s['R_U'])} | {f(s['HMR'])} | {f(s['avconf'])} | {s['OB']:+.4f} |")
    w("")
    w("Read the columns separately, because they say opposite things.")
    w("")
    w(f"**Accuracy.** Three runs, {stats['AC-1']['n_wrong']}, {stats['AC-2']['n_wrong']} and "
      f"{stats['AC-3']['n_wrong']} wrong answers out of {stats['AC-1']['n']}. "
      f"The whole spread between the best and worst of our three systems is two questions. "
      "Assembling passes did not move accuracy in either direction by an amount this "
      "question set can see.")
    w("")
    w(f"**Confidence.** HMR moves by "
      f"{stats['AC-2']['HMR'] - stats['AC-1']['HMR']:+.4f} for AC-2 and "
      f"{stats['AC-3']['HMR'] - stats['AC-1']['HMR']:+.4f} for AC-3 against AC-1. "
      "Both aggregators over all the passes score better than the first-pass run on the "
      "task's primary measure.")
    w("")

    w("### 2.1 Which half of HMR moved")
    w("")
    w("| Comparison | ΔR_O | ΔR_U | ΔHMR | Δaccuracy |")
    w("|---|---|---|---|---|")
    for r in ("AC-2", "AC-3"):
        w(f"| {r} − AC-1 | {stats[r]['R_O'] - stats['AC-1']['R_O']:+.4f} | "
          f"{stats[r]['R_U'] - stats['AC-1']['R_U']:+.4f} | "
          f"{stats[r]['HMR'] - stats['AC-1']['HMR']:+.4f} | "
          f"{stats[r]['acc'] - stats['AC-1']['acc']:+.4f} |")
    w(f"| AC-3 − AC-2 | {stats['AC-3']['R_O'] - stats['AC-2']['R_O']:+.4f} | "
      f"{stats['AC-3']['R_U'] - stats['AC-2']['R_U']:+.4f} | "
      f"{stats['AC-3']['HMR'] - stats['AC-2']['HMR']:+.4f} | "
      f"{stats['AC-3']['acc'] - stats['AC-2']['acc']:+.4f} |")
    w("")
    w("R_O is one minus the mean confidence on the answers a run got wrong; R_U is the mean "
      "confidence on the ones it got right. Both aggregators raise R_O over the first-pass "
      "run, and AC-3 raises R_U almost to its ceiling as well — but HMR is a harmonic mean, "
      "so it tracks the smaller component, and R_O is the smaller component for all three "
      "runs by a wide margin.")
    w("")
    w("Which of the two components carries the HMR change is not read off those Δ columns, "
      "because HMR is nonlinear in both. Substituting one component at a time into AC-1's "
      "HMR settles it:")
    w("")
    w("| | AC-1 baseline | + the run's R_O only | + its R_U only | interaction | its HMR |")
    w("|---|---|---|---|---|---|")
    for b in ("AC-2", "AC-3"):
        d = decomp[b]
        w(f"| {b} | {d['base']:.4f} | {d['d_ro']:+.4f} | {d['d_ru']:+.4f} | "
          f"{d['inter']:+.4f} | {d['full']:.4f} |")
    w("")
    w(f"For AC-2 the R_O term carries {d_share('AC-2', decomp):.0%} of the gain and for "
      f"AC-3 {d_share('AC-3', decomp):.0%}. AC-3's very large ΔR_U — it pushes mean "
      "confidence on its correct answers to 0.99 — buys it almost nothing, because a "
      "harmonic mean is governed by its smaller term and R_O is the smaller term for all "
      "three runs by a factor of three or more. **The wrong-answer half is where our HMR "
      "is decided, and it is where the ensemble made its difference.**")
    w("")
    w("The scale to keep in view: with only "
      f"{stats['AC-1']['n_wrong']}–{stats['AC-3']['n_wrong']} wrong answers, "
      "R_O is a mean over five to seven numbers. A single wrong answer filed at confidence "
      "100 costs between 1/5 and 1/7 of R_O outright. High accuracy does not protect R_O; "
      "it concentrates it.")
    w("")

    w("### 2.2 The wrong answers, one line each")
    w("")
    allwrong = sorted(set().union(*[set(ns["wrongsets"][r]) for r in RUNS_]))
    byq = {x["qid"]: x for x in scored}
    w("| qid | type | AC-1 conf / correct | AC-2 | AC-3 |")
    w("|---|---|---|---|---|")
    for q in allwrong:
        x = byq[q]
        cells = []
        for r in RUNS_:
            c = int(x[KEY[r] + "_conf"])
            ok = int(x[KEY[r] + "_correct"])
            cells.append(f"{c} / {'right' if ok else '**wrong**'}")
        w(f"| {q} | {x['qtype']} | " + " | ".join(cells) + " |")
    w("")
    w("This table is the entire R_O story. Every question not listed here was answered "
      "correctly by all three runs, and correct answers are invisible to R_O.")
    w("")

    w("## 3. Whose correctness, whose confidence")
    w("")
    w("HMR mixes two ingredients a run supplies separately: which questions it gets right, "
      "and what confidence it prints. Pairing one run's correctness vector with another "
      "run's confidence vector separates them. The diagonal is the real runs; off-diagonal "
      "cells are counterfactual.")
    w("")
    w("| HMR — correctness from ↓ / confidence from → | AC-1 | AC-2 | AC-3 |")
    w("|---|---|---|---|")
    for rc in RUNS_:
        w(f"| {rc} | " + " | ".join(f(cross[(rc, rp)]['HMR']) for rp in RUNS_) + " |")
    w("")
    w("| R_O — correctness from ↓ / confidence from → | AC-1 | AC-2 | AC-3 |")
    w("|---|---|---|---|")
    for rc in RUNS_:
        w(f"| {rc} | " + " | ".join(f(cross[(rc, rp)]['R_O']) for rp in RUNS_) + " |")
    w("")
    row_span = max(abs(cross[(rc, RUNS_[0])]["HMR"] - cross[(rc, RUNS_[2])]["HMR"]) for rc in RUNS_)
    col_span = max(abs(cross[(RUNS_[0], rp)]["HMR"] - cross[(RUNS_[2], rp)]["HMR"]) for rp in RUNS_)
    w(f"Moving along a row — same correctness, different confidence — swings HMR by up to "
      f"{row_span:.4f}. Moving down a column — same confidence, different correctness — "
      f"swings it by up to {col_span:.4f}. Both axes matter, and the column effect is not "
      "negligible; but note what the column effect *is*. Changing the correctness vector "
      "changes which questions land in I⁻ and therefore what R_O averages over, so a cell "
      "like (AC-2 correctness, AC-1 confidence) at 0.4857 mostly records that AC-2 gets "
      "0060 right — a question AC-1 filed wrong at 99 — rather than any improvement in the "
      "confidence numbers themselves. With five to seven wrong answers per run, one "
      "question changing side moves R_O by up to a fifth.")
    w("")

    w("## 4. Which questions each run gets right that the others do not")
    w("")
    w("| AC-1 | AC-2 | AC-3 | n | questions |")
    w("|---|---|---|---|---|")
    for pat, cnt in sorted(patterns.items(), key=lambda kv: -kv[1]):
        qs = [x["qid"] for x in scored
              if tuple(int(x[KEY[r] + "_correct"]) for r in RUNS_) == pat]
        mark = ["right" if v else "wrong" for v in pat]
        listing = ", ".join(qs) if cnt <= 8 else "(not listed)"
        w(f"| {mark[0]} | {mark[1]} | {mark[2]} | {cnt} | {listing} |")
    w("")
    for r in RUNS_:
        ur, uw = ns["uniquely_right"][r], ns["uniquely_wrong"][r]
        w(f"- **{r}** is alone in getting right: {', '.join(ur) if ur else 'nothing'}. "
          f"Alone in getting wrong: {', '.join(uw) if uw else 'nothing'}.")
    w("")
    b01, b10, p = tests["AC-2"]["mcnemar"]
    w(f"Exact McNemar, AC-1 vs AC-2: {b10} question(s) only AC-1 got right, {b01} only AC-2, "
      f"p = {p:.3f}.")
    b01, b10, p = tests["AC-3"]["mcnemar"]
    w(f"Exact McNemar, AC-1 vs AC-3: {b10} only AC-1, {b01} only AC-3, p = {p:.3f}.")
    b01, b10, p = ns["ac23_mcnemar"]
    w(f"Exact McNemar, AC-2 vs AC-3: {b10} only AC-2, {b01} only AC-3, p = {p:.3f}.")
    w("")
    w("Discordant pairs in the low single digits. No accuracy claim of any kind is "
      "supportable from these counts, in either direction.")
    w("")

    w("## 5. The shape of the confidence numbers")
    w("")
    w("| Run | distinct values | at 100 | ≥95 | ≤10 | median | wrong answers at 100 | "
      "mean conf when wrong | AUC | tied pairs |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in RUNS_:
        c = confdist[r]
        w(f"| {r} | {c['distinct']} | {c['at100']} | {c['ge95']} | {c['le10']} | "
          f"{c['median']} | {c['wrong_at100']} | {c['mean_conf_wrong']:.1f} | "
          f"{f(c['auc'], 3)} | {f(c['ties'], 3)} |")
    w("")
    w("AUC is the probability that a randomly chosen correct answer carried a higher "
      "confidence than a randomly chosen wrong one, ties counted as half. It is "
      "scale-free: unlike HMR it does not care where the numbers sit, only whether they "
      "*separate* right from wrong. This is the sharpest way to ask whether assembling "
      "passes produced a better confidence signal rather than merely a differently "
      "scaled one.")
    w("")
    w(f"**AC-1's confidence is at chance.** {f(confdist['AC-1']['auc'], 3)} means that if "
      "you picked one answer AC-1 got right and one it got wrong, its confidence was as "
      "likely to be higher on the wrong one as on the right one. The first-pass run printed "
      "numbers that carried no usable information about its own correctness. Both ensemble "
      "runs are above chance, and AC-3 — the run with the *worst* confidence scale, 53 of "
      "64 questions pinned at 100 — has the nominally best ordering of the three. Qualify "
      f"that last one: {confdist['AC-3']['ties']:.0%} of AC-3's (correct, incorrect) pairs "
      "are exact ties, counted as half a win each, against "
      f"{confdist['AC-1']['ties']:.0%} for AC-1. AC-3's AUC is carried by a handful of "
      "resolvable pairs sitting on top of a large tied mass.")
    w("")
    w("| Comparison | ΔAUC | 95% bootstrap CI | share of resamples with Δ ≤ 0 |")
    w("|---|---|---|---|")
    for r in ("AC-2", "AC-3"):
        d, lo, hi, le0 = tests[r]["AUC"]
        w(f"| {r} − AC-1 | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {le0:.3f} |")
    w("")
    w("Read those intervals with the denominator in mind: an AUC here is computed over 5–7 "
      "wrong answers against 57–59 correct ones, so a resample is dominated by which wrong "
      "answers it happens to draw. This is the clearest direction in the data, not a "
      "demonstrated effect.")
    w("")
    w("The AUC/HMR split is also the cleanest statement of why AC-3 scores worse than AC-2 "
      "despite ordering its questions better. AC-3 sums confidences across passes and "
      "clamps at 100, so it prints 100 on 53 of 64 questions; the ordering underneath is "
      "good, but the scale has been crushed against the ceiling, and HMR reads the scale.")
    w("")
    for r in RUNS_:
        w(f"- {r}: {', '.join(f'{v}×{k}' for k, v in confdist[r]['top'])} "
          "(five most frequent values)")
    w("")

    w("## 6. Interval estimates")
    w("")
    w("Paired bootstrap over the 64 questions, 20 000 resamples, seed fixed in the script. "
      "The runs are resampled together, so the pairing is preserved. The bootstrap is "
      "reported because the shipped Randomised Tukey HSD p-values are corrected across the "
      "whole 22- or 25-run field and cannot address a two-run question at anything near its "
      "natural power; on that test every BITEM-vs-BITEM pair returns p ≈ 1, which is a "
      "non-rejection and not evidence of no difference.")
    w("")
    w("| Comparison | statistic | Δ | 95% bootstrap CI | share of resamples with Δ ≤ 0 |")
    w("|---|---|---|---|---|")
    for r in ("AC-2", "AC-3"):
        for stat in ("HMR", "R_O", "R_U", "acc"):
            d, lo, hi, le0 = tests[r][stat]
            w(f"| {r} − AC-1 | {stat} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {le0:.3f} |")
    w("")
    w("Every interval on HMR and R_O straddles a wide range and the R_U intervals are "
      "narrow only because R_U is a mean over ~58 questions rather than over ~6. "
      "The CIs on R_O are wide for the reason already given: R_O is an average over five "
      "to seven numbers, so its sampling distribution is dominated by whether a particular "
      "confidently-wrong question is drawn.")
    w("")

    w("## 7. Per question type")
    w("")
    w("Type counts are tiny — the largest class has 26 questions and four classes have "
      "seven. Descriptive only.")
    w("")
    w("A zero in a HMR cell is not a formatting artefact: it means every wrong answer in "
      "that class was filed at confidence 100, so R_O = 0 and the harmonic mean collapses. "
      "All three runs do this on `simple`, the largest class.")
    w("")
    w("| type | n | AC-1 acc / HMR | AC-2 acc / HMR | AC-3 acc / HMR |")
    w("|---|---|---|---|---|")
    for t, d in sorted(per_type.items(), key=lambda kv: -kv[1]["n"]):
        cells = [f"{f(d[r]['acc'], 3)} / {f(d[r]['HMR'], 3)}" for r in RUNS_]
        w(f"| {t} | {d['n']} | " + " | ".join(cells) + " |")
    w("")

    w("## 8. Nuggets: the ensemble produced more evidence, at lower precision")
    w("")
    w(f"| Run | nuggets | per question | R | N | B | B rate | mean NugPrec | "
      f"same, on the {nug['_common_n']} commonly-correct questions |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in RUNS_:
        d = nug[r]
        w(f"| {r} | {d['total']} | {d['per_q']:.1f} | {d['R']} | {d['N']} | {d['B']} | "
          f"{d['b_rate']:.3f} | {f(d['meanNP'])} | {f(d['meanNP_common'])} |")
    w("")
    w("The last column matters. Nugget precision is forced to zero on a wrong answer by "
      "construction — the organisers auto-label every non-bogus nugget of a NO question as "
      "nonrelevant — so a run with one extra wrong answer loses nugget precision "
      "mechanically. Restricting to the questions all three runs got right removes that "
      "confound, and the ordering survives it: the first-pass run is still ahead.")
    w("")
    obs, p = tests["AC-2"]["nugprec"]
    w(f"Paired sign-flip randomisation on per-question nugget precision, AC-2 − AC-1: "
      f"{obs:+.4f}, p = {p:.4f}.")
    obs, p = tests["AC-3"]["nugprec"]
    w(f"AC-3 − AC-1: {obs:+.4f}, p = {p:.4f}.")
    w("")
    w(f"The ensemble runs cite roughly twice as many nuggets as the first-pass run "
      f"({nug['AC-2']['total']} and {nug['AC-3']['total']} against {nug['AC-1']['total']}) "
      f"and land roughly twice as many *relevant* ones "
      f"({nug['AC-2']['R']} and {nug['AC-3']['R']} against {nug['AC-1']['R']}), while "
      "precision falls. Nugget precision is the only per-question quantity on which the "
      "first-pass run is consistently ahead. Note that the measure has no recall term, so "
      "'twice as much relevant evidence at lower precision' is not scored as a gain "
      "anywhere in the official tables.")
    w("")
    w(f"**A noise floor for that comparison, and the one use AC-4 has.** The organisers "
      "marked BITEM-AC-3 and BITEM-AC-4 separately even though the two files are "
      f"byte-identical. {ns['flips']} of {ns['tot3']} nugget marks differ between the two "
      f"markings ({', '.join(f'{k} ×{v}' for k, v in sorted(ns['flip_kinds'].items()))}), "
      f"moving nugget precision on {ns['np_diff']} of {len(scored)} questions and the mean "
      "from 0.7105 to 0.6989 — a gap of 0.0117 at zero true system difference. "
      f"The AC-3 − AC-1 gap of {abs(tests['AC-3']['nugprec'][0]):.4f} is six times that "
      f"floor and the AC-2 − AC-1 gap of {abs(tests['AC-2']['nugprec'][0]):.4f} is about "
      f"four times it, so neither is assessment noise. The gap between the two "
      f"aggregators, "
      f"{abs(nug['AC-2']['meanNP'] - nug['AC-3']['meanNP']):.4f}, is only twice the floor "
      "and should not be leaned on.")
    w("")

    w("## 9. What these numbers cannot settle")
    w("")
    for line in [
        "**Nothing here is per-pass.** `results/` and `submission/` contain only the "
        "aggregator output. How many passes ran on a question, which pass an answer came "
        "from, and which rung of the fallback ladder fired are all absent. The comparison "
        "above is an outcome comparison between three filed files, not a measurement of "
        "the ensemble mechanism.",
        "**The passes are not independent draws.** A fallback lets a later pass file an "
        "answer an earlier pass on the same question already committed. AC-2 (max-conf) "
        "and AC-3 (conf-sum) both aggregate as if the passes were independent, so whatever "
        "agreement they exploit is inflated by an unknown amount. AC-3 in particular sums "
        "confidences across passes; a carried-over answer contributes to that sum. The "
        "size of the inflation is not recoverable from the official results and needs a "
        "trace-reading script.",
        "**Diversity across passes is manufactured, not sampled.** Each pass retrieves "
        "from a corpus with everything its predecessors saw removed, so 'the ensemble' is "
        "not repeated sampling from one distribution and the usual self-consistency "
        "intuition does not transfer.",
        "**Whether AC-2 and AC-3 differ because the passes disagreed or because the two "
        "aggregators broke a tie differently** is not answerable from the filed files.",
        "**The LLM-as-judge strategy is unevaluated.** AC-4's real output was never filed, "
        "so the fourth aggregator over the same bundle has no score at all.",
        "**Six wrong answers is not a sample.** R_O, the component that decides our HMR, "
        "is a mean over five to seven questions. Every statement about R_O in this "
        "document is a statement about the confidence attached to a handful of specific "
        "questions, and would be overturned by a different handful.",
        "**Accuracy is at ceiling for this comparison.** With 56 of 64 questions answered "
        "correctly by all three runs, the design has almost no room to show an accuracy "
        "effect from aggregation even if one existed. This is a power problem, not a null "
        "result about aggregation and accuracy.",
        "**Nugget precision is not an independent evidence-quality signal.** A wrong answer "
        "forces every one of its nuggets to nonrelevant, so nugget precision is partly a "
        "restatement of accuracy. The commonly-correct-questions column in Section 8 is the "
        "control for that, and it is computed on 56 questions with no correctness variation "
        "left in them.",
        f"**The AUC comparison rests on a tied mass.** {confdist['AC-3']['ties']:.0%} of "
        "AC-3's (correct, incorrect) confidence pairs are exact ties at 100, scored as half "
        "a win each. Its nominal AUC advantage over AC-2 is decided by a small number of "
        "resolvable pairs.",
        "**qid 0025 is invisible.** It was filed and is excluded from every official "
        "table; nothing about it can be checked here.",
    ]:
        w(f"- {line}")
    w("")

    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
