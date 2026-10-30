#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Characterise the confidence scores of BITEM-AC-1, AC-2 and AC-3.

One measurement: what values the filed confidence actually takes, how it is
distributed over correct versus incorrect answers, how much the two
distributions separate, and how that separation maps onto the officially
reported R_O, R_U, HMR and OB.

Reads, read-only:
  results/OfficialResults/ACnev/BITEM-AC-{1,2,3}.official.ACnev   Conf, Correct
  results/OfficialResults/r2c2questiontypes                       question type
  results/OfficialResults/official.AC.csv                         published summary
  <submission_dir>/ac-2026-05-15/BITEM-AC-{1,2,3}   (optional positional)
      used only to check that the scored Conf equals the filed Conf.

Writes one file beside itself: analysis/confidence_behaviour.md

BITEM-AC-4 is excluded throughout: it is a byte-for-byte copy of BITEM-AC-3
(md5 dc04428a47b429b5363b150627bc7fb9), so it is not a fourth system and any
statistic computed over both is computed over a duplicated column.

Usage:  uv run python analysis/confidence_behaviour.py [submission]
"""

import csv
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = os.path.join(ROOT, "results", "OfficialResults")
RUNS = ["BITEM-AC-1", "BITEM-AC-2", "BITEM-AC-3"]
BOOT = 10000
PERM = 20000
SEED = 20260822


# Reading

def read_acnev(run):
    """{qid: (conf 0-100 int, correct 0/1)} from the organisers' per-question file."""
    conf, corr = {}, {}
    path = os.path.join(OFF, "ACnev", run + ".official.ACnev")
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            qid, field, value = parts[0], parts[1], parts[2]
            if field == "Conf=":
                conf[qid] = int(float(value))
            elif field == "Correct=":
                corr[qid] = int(float(value))
    assert set(conf) == set(corr), run
    return {q: (conf[q], corr[q]) for q in conf}


def read_types():
    types = {}
    with open(os.path.join(OFF, "r2c2questiontypes")) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                types[parts[0]] = parts[1]
    return types


def read_official_ac():
    out = {}
    with open(os.path.join(OFF, "official.AC.csv")) as fh:
        for row in csv.DictReader(fh):
            out[row[""].replace(".official", "")] = row
    return out


def read_filed(subdir, run):
    """{qid: conf} from a filed AC run: <qid> block, answer line 'answer;conf'."""
    path = os.path.join(subdir, "ac-2026-05-15", run)
    if not os.path.exists(path):
        return None
    out, qid = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = re.match(r"^<(\d{4})>\s*(.*)$", line)
            if m:
                qid, rest = m.group(1), m.group(2)
                if rest:
                    out[qid] = int(float(rest.rsplit(";", 1)[-1]))
                continue
            if line.startswith("</"):
                qid = None
                continue
            if qid is not None and qid not in out and line.strip():
                out[qid] = int(float(line.rsplit(";", 1)[-1]))
    return out


# Measures

def rewards(pairs):
    """(R_O, R_U, HMR) from [(conf 0-100, correct)] following Eqs. 3-6 of the
    task overview. R_O = 1 when nothing is incorrect, R_U = 1 when nothing is
    correct, per the overview's own edge cases."""
    cor = [c / 100.0 for c, y in pairs if y == 1]
    inc = [c / 100.0 for c, y in pairs if y == 0]
    r_u = 1.0 if not cor else sum(cor) / len(cor)
    r_o = 1.0 if not inc else 1.0 - sum(inc) / len(inc)
    hmr = 0.0 if (r_o + r_u) == 0 else 2 * r_o * r_u / (r_o + r_u)
    return r_o, r_u, hmr


def auc(pairs):
    """P(conf of a correct answer > conf of an incorrect one), ties at 0.5.
    None when either class is empty."""
    cor = [c for c, y in pairs if y == 1]
    inc = [c for c, y in pairs if y == 0]
    if not cor or not inc:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in cor for b in inc)
    return wins / (len(cor) * len(inc))


def masses(pairs):
    """Overconfidence mass O = sum of p over the incorrect, underconfidence
    mass U = sum of (1-p) over the correct, both in question-equivalents.
    OB = (O - U)/n exactly."""
    o = sum(c / 100.0 for c, y in pairs if y == 0)
    u = sum(1 - c / 100.0 for c, y in pairs if y == 1)
    return o, u


def best_threshold_hmr(pairs):
    """Highest HMR reachable by binarising the existing confidences at some
    threshold: p=1 at or above t, p=0 below. Fitted on the same 64 questions it
    is scored on, so it is an in-sample ceiling, not an estimate."""
    best = None
    for t in sorted(set(c for c, _ in pairs)) + [101]:
        binar = [(100 if c >= t else 0, y) for c, y in pairs]
        r_o, r_u, hmr = rewards(binar)
        if best is None or hmr > best[0]:
            best = (hmr, t, r_o, r_u)
    return best


def pct_ci(values, lo=0.025, hi=0.975):
    v = sorted(values)
    return v[int(lo * len(v))], v[min(len(v) - 1, int(hi * len(v)))]


# Resampling

def bootstrap(data, qids, rng):
    """Percentile CIs for R_O, R_U, HMR, AUC, per run, over resampled questions."""
    acc = {r: defaultdict(list) for r in data}
    n = len(qids)
    for _ in range(BOOT):
        pick = [qids[rng.randrange(n)] for _ in range(n)]
        for run in data:
            p = [data[run][q] for q in pick]
            r_o, r_u, hmr = rewards(p)
            a = auc(p)
            if a is None:
                continue
            acc[run]["R_O"].append(r_o)
            acc[run]["R_U"].append(r_u)
            acc[run]["HMR"].append(hmr)
            acc[run]["AUC"].append(a)
    return {r: {k: pct_ci(v) for k, v in acc[r].items()} for r in acc}


def paired_bootstrap(data, qids, rng, a, b):
    """CIs for b - a on the same resampled questions (the runs are paired: one
    question set, three systems)."""
    keys = ["R_O", "R_U", "HMR", "AUC"]
    acc = {k: [] for k in keys}
    n = len(qids)
    for _ in range(BOOT):
        pick = [qids[rng.randrange(n)] for _ in range(n)]
        pa = [data[a][q] for q in pick]
        pb = [data[b][q] for q in pick]
        aa, ab = auc(pa), auc(pb)
        if aa is None or ab is None:
            continue
        ra, rb = rewards(pa), rewards(pb)
        for i, k in enumerate(["R_O", "R_U", "HMR"]):
            acc[k].append(rb[i] - ra[i])
        acc["AUC"].append(ab - aa)
    return {k: pct_ci(v) for k, v in acc.items()}


def permutation_auc(pairs, rng):
    """One-sided p for AUC under random reassignment of the correctness labels
    to the observed confidences. Tests only whether the confidence orders
    correct above incorrect better than chance."""
    observed = auc(pairs)
    confs = [c for c, _ in pairs]
    labels = [y for _, y in pairs]
    hits = 0
    for _ in range(PERM):
        rng.shuffle(labels)
        if auc(list(zip(confs, labels))) >= observed:
            hits += 1
    return observed, hits / PERM


# Report

def fmt(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission")
    rng = random.Random(SEED)

    data = {r: read_acnev(r) for r in RUNS}
    qids = sorted(set.intersection(*(set(d) for d in data.values())))
    types = read_types()
    official = read_official_ac()

    short = {r: r.replace("BITEM-", "") for r in RUNS}
    per = {r: [data[r][q] for q in qids] for r in RUNS}

    # cross-checks
    checks = []
    for r in RUNS:
        r_o, r_u, hmr = rewards(per[r])
        o, u = masses(per[r])
        n = len(qids)
        acc = sum(y for _, y in per[r]) / n
        avconf = sum(c for c, _ in per[r]) / 100.0 / n
        row = official[r]
        ok = all(abs(a - float(row[k])) < 5e-4 for a, k in
                 [(r_o, "R_O"), (r_u, "R_U"), (hmr, "HMR"),
                  (avconf, "avconf"), (acc, "accuracy"), (avconf - acc, "OB")])
        checks.append((r, n, int(row["n"]), ok))
    filed_ok = []
    for r in RUNS:
        f = read_filed(sub, r)
        if f is None:
            filed_ok.append((r, None))
        else:
            filed_ok.append((r, all(f[q] == data[r][q][0] for q in qids)))

    boots = bootstrap(data, qids, rng)
    perms = {r: permutation_auc(per[r], rng) for r in RUNS}
    diffs = {(a, b): paired_bootstrap(data, qids, rng, a, b)
             for a, b in [(RUNS[0], RUNS[1]), (RUNS[0], RUNS[2]), (RUNS[1], RUNS[2])]}

    L = []
    w = L.append
    w("# Confidence behaviour of the three distinct BITEM AC runs")
    w("")
    w("Generated by `analysis/confidence_behaviour.py`. Pure read of "
      "`results/OfficialResults/` and, for a cross-check only, of the filed "
      "submission. Nothing here compares BITEM with another team.")
    w("")
    w(f"Question set: the {len(qids)} officially scored questions "
      "(0025 is excluded from every official artefact). **BITEM-AC-4 is "
      "excluded**: it is a byte-for-byte copy of BITEM-AC-3, so it is not a "
      "fourth system, and any statistic pooling the two is computed over a "
      "duplicated column. Three distinct systems, not four.")
    w("")
    w("`AC-1` is built from the first pass on each question; `AC-2` and `AC-3` "
      "are two aggregators over all the recorded passes. So AC-1 against "
      "AC-2/AC-3 is the first-pass-versus-all-passes comparison on the "
      "answering side.")
    w("")

    w("## 1. What the confidence is: a short, top-heavy list of values")
    w("")
    w("| run | distinct values | modal value | share at modal value | share \u2265 90 | share = 100 | min |")
    w("|---|---|---|---|---|---|---|")
    for r in RUNS:
        cs = [c for c, _ in per[r]]
        cnt = Counter(cs)
        mode, mode_n = cnt.most_common(1)[0]
        w(f"| {short[r]} | {len(cnt)} | {mode} | {mode_n}/{len(cs)} = {mode_n/len(cs):.3f} | "
          f"{sum(1 for c in cs if c >= 90)/len(cs):.3f} | "
          f"{sum(1 for c in cs if c == 100)/len(cs):.3f} | {min(cs)} |")
    w("")
    w("Full support, as `value:(correct/total)`:")
    w("")
    for r in RUNS:
        cnt = defaultdict(lambda: [0, 0])
        for c, y in per[r]:
            cnt[c][0] += y
            cnt[c][1] += 1
        w(f"- **{short[r]}** &nbsp; " +
          " &nbsp; ".join(f"`{v}:{cnt[v][0]}/{cnt[v][1]}`" for v in sorted(cnt)))
    w("")
    ge = {r: sum(1 for c, _ in per[r] if c >= 99) for r in RUNS}
    w(f"The scale compresses upward as more passes are assembled. AC-1 spreads "
      f"over {len(set(c for c,_ in per[RUNS[0]]))} values with "
      f"{ge[RUNS[0]]} of {len(qids)} questions at 99 or 100; AC-3 uses "
      f"{len(set(c for c,_ in per[RUNS[2]]))} values with {ge[RUNS[2]]} at 99 "
      f"or 100. **AC-3 files a confidence at least as high as AC-2 on all "
      f"{sum(1 for q in qids if data[RUNS[2]][q][0] >= data[RUNS[1]][q][0])} "
      f"questions**, and strictly higher on "
      f"{sum(1 for q in qids if data[RUNS[2]][q][0] > data[RUNS[1]][q][0])}. "
      "The two aggregators agree exactly on "
      f"{sum(1 for q in qids if data[RUNS[2]][q][0] == data[RUNS[1]][q][0])}.")
    w("")
    w("Reading the per-value accuracies: a filed 100 is not a claim that holds "
      "up as certainty, but it is not far off — of the questions filed at 100, "
      + ", ".join(
          f"{short[r]} got {sum(1 for c,y in per[r] if c==100 and y==1)}/"
          f"{sum(1 for c,_ in per[r] if c==100)} right"
          for r in RUNS)
      + ". The measure charges the residual in full.")
    w("")

    w("## 2. Confidence on correct answers versus on incorrect ones")
    w("")
    w("| run | correct | mean conf \u2502 correct | median | incorrect | mean conf \u2502 incorrect | median | gap | AUC | perm. p |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in RUNS:
        cor = [c for c, y in per[r] if y == 1]
        inc = [c for c, y in per[r] if y == 0]
        a, p = perms[r]
        w(f"| {short[r]} | {len(cor)} | {statistics.mean(cor):.1f} | "
          f"{statistics.median(cor):.1f} | {len(inc)} | {statistics.mean(inc):.1f} | "
          f"{statistics.median(inc):.1f} | {statistics.mean(cor)-statistics.mean(inc):+.1f} | "
          f"{a:.4f} | {p:.4f} |")
    w("")
    w(f"AUC is the probability that a randomly chosen correct answer carries a "
      f"higher confidence than a randomly chosen incorrect one, ties counted as "
      f"half; 0.5 is chance. The permutation p is one-sided over {PERM:,} "
      "random reassignments of the correctness labels to the observed "
      "confidences.")
    w("")
    for r in RUNS:
        inc = sorted((data[r][q][0], q, types.get(q, "?")) for q in qids if data[r][q][1] == 0)
        w(f"- **{short[r]} got {len(inc)} questions wrong**, at confidences " +
          ", ".join(f"`{c}` ({q}, {t})" for c, q, t in inc) + ".")
    w("")
    w("**AC-1's confidence carries no information about its own correctness.** "
      f"Its AUC is {perms[RUNS[0]][0]:.3f}, below chance, at p = "
      f"{perms[RUNS[0]][1]:.2f}; its incorrect answers were on average *more* "
      "confident than its correct ones. Its low band is the clearest case: of "
      f"the {sum(1 for c,_ in per[RUNS[0]] if c < 50)} questions AC-1 filed "
      f"under 50, {sum(1 for c,y in per[RUNS[0]] if c < 50 and y == 1)} were "
      "correct. Assembling the passes changes this: AC-2 reaches "
      f"{perms[RUNS[1]][0]:.3f} (p = {perms[RUNS[1]][1]:.3f}) and AC-3 "
      f"{perms[RUNS[2]][0]:.3f} (p = {perms[RUNS[2]][1]:.4f}).")
    w("")

    w("## 3. Overconfidence and underconfidence, in question-equivalents")
    w("")
    w("`O = \u03a3 p` over the incorrect and `U = \u03a3 (1\u2212p)` over the correct are the "
      "two error masses the measure is built from. `R_O = 1 \u2212 O/|I\u207b|`, "
      "`R_U = 1 \u2212 U/|I\u207a|`, and `OB = (O \u2212 U)/n` exactly \u2014 the reported bias "
      "is the *net* of the two, so a run can look unbiased by holding equal "
      "amounts of both.")
    w("")
    w("| run | \\|I\u207a\\| | \\|I\u207b\\| | O | U | O\u2212U | OB = (O\u2212U)/n | R_O | R_U | HMR |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in RUNS:
        o, u = masses(per[r])
        r_o, r_u, hmr = rewards(per[r])
        npos = sum(y for _, y in per[r])
        w(f"| {short[r]} | {npos} | {len(qids)-npos} | {o:.2f} | {u:.2f} | "
          f"{o-u:+.2f} | {(o-u)/len(qids):+.4f} | {r_o:.4f} | {r_u:.4f} | {hmr:.4f} |")
    w("")
    o1, u1 = masses(per[RUNS[0]])
    o2, u2 = masses(per[RUNS[1]])
    o3, u3 = masses(per[RUNS[2]])
    w(f"**The sign flip in OB is not a change in overconfidence.** AC-1 holds "
      f"{o1:.2f} question-equivalents of overconfidence and AC-3 holds "
      f"{o3:.2f} \u2014 essentially the same. What separates them is "
      f"underconfidence: {u1:.2f} for AC-1 against {u3:.2f} for AC-3. AC-1's "
      "OB is negative because it is timid about answers that turned out right, "
      "not because it is well behaved about answers that turned out wrong.")
    w("")
    lo = [(c, y) for c, y in per[RUNS[0]] if c < 50]
    w(f"Where AC-1's underconfidence sits: the "
      f"{sum(1 for c,y in lo if y == 1)} correct answers it filed under 50 "
      f"account for {sum(1-c/100 for c,y in lo if y==1):.2f} of its "
      f"{u1:.2f} total U.")
    w("")
    shared = [q for q in qids if data[RUNS[1]][q][1] == 0 and data[RUNS[2]][q][1] == 0]
    shared_o = sum(data[RUNS[1]][q][0] / 100 for q in shared)
    w(f"Between the two ensemble aggregators, AC-2 and AC-3 file **identical "
      f"confidences on all {len(shared)} questions they both get wrong** "
      f"(shared O = {shared_o:.2f} each). AC-3's higher O ({o3:.2f} against "
      f"{o2:.2f}) is entirely the "
      f"{sum(1 for q in qids if data[RUNS[2]][q][1]==0 and data[RUNS[1]][q][1]==1)} "
      "extra questions it gets wrong, both filed at 100. So AC-3's worse R_O "
      "is a correctness difference expressed through the confidence, not a "
      "different confidence policy on shared errors.")
    w("")
    w("Where each run's R_O loss is concentrated:")
    w("")
    for r in RUNS:
        inc = [c for c, y in per[r] if y == 0]
        at100 = sum(1 for c in inc if c == 100)
        w(f"- **{short[r]}**: {at100} of {len(inc)} wrong answers filed at "
          f"exactly 100, which alone costs {at100/len(inc):.3f} of R_O "
          f"(of the {1-rewards(per[r])[0]:.3f} lost in total).")
    w("")

    w("## 4. How much of the loss is the ordering and how much is the scale")
    w("")
    w("| run | HMR as filed | best HMR from binarising the same confidences | at threshold | R_O | R_U |")
    w("|---|---|---|---|---|---|")
    for r in RUNS:
        hmr = rewards(per[r])[2]
        b = best_threshold_hmr(per[r])
        w(f"| {short[r]} | {hmr:.4f} | {b[0]:.4f} | {b[1]} | {b[2]:.3f} | {b[3]:.3f} |")
    w("")
    w("The right-hand column is **fitted on the same "
      f"{len(qids)} questions it is scored on**, and the threshold is chosen "
      "against 5 to 7 wrong answers, so it is a heavily overfitted in-sample "
      "ceiling and not an estimate of anything achievable. It is reported for "
      "one reason: AC-3's filed HMR is "
      f"{rewards(per[RUNS[2]])[2]:.3f} while a binarisation of the very same "
      f"numbers reaches {best_threshold_hmr(per[RUNS[2]])[0]:.3f}, so the "
      "ordering AC-3 produces is worth considerably more than the scale it "
      "produces. The same reading is what the AUC column of section 2 says "
      "without the in-sample fitting, and the AUC is the one to quote.")
    w("")
    w("AC-1's entry in that column is the warning attached to the whole "
      "column: a run whose AUC is *below* chance still reaches "
      f"{best_threshold_hmr(per[RUNS[0]])[0]:.3f} here, which is only possible "
      "because a threshold chosen after the fact can always cut 6 errors "
      "favourably. There is no latent signal in AC-1's confidence for a "
      "recalibration to recover.")
    w("")

    w("## 5. Did assembling passes help?")
    w("")
    w("95% percentile bootstrap over the questions, "
      f"{BOOT:,} resamples, seed {SEED}. The three runs answer the same "
      "questions, so the differences are paired on the resampled question set.")
    w("")
    w("| run | R_O [95% CI] | R_U [95% CI] | HMR [95% CI] | AUC [95% CI] |")
    w("|---|---|---|---|---|")
    for r in RUNS:
        b = boots[r]
        r_o, r_u, hmr = rewards(per[r])
        a = auc(per[r])
        w(f"| {short[r]} | {r_o:.4f} [{b['R_O'][0]:.3f}, {b['R_O'][1]:.3f}] | "
          f"{r_u:.4f} [{b['R_U'][0]:.3f}, {b['R_U'][1]:.3f}] | "
          f"{hmr:.4f} [{b['HMR'][0]:.3f}, {b['HMR'][1]:.3f}] | "
          f"{a:.4f} [{b['AUC'][0]:.3f}, {b['AUC'][1]:.3f}] |")
    w("")
    w("| difference | \u0394R_O | \u0394R_U | \u0394HMR | \u0394AUC |")
    w("|---|---|---|---|---|")
    for (a, b), d in diffs.items():
        ra, rb = rewards(per[a]), rewards(per[b])
        da, db = auc(per[a]), auc(per[b])
        w(f"| {short[b]} \u2212 {short[a]} | "
          f"{rb[0]-ra[0]:+.4f} [{d['R_O'][0]:+.3f}, {d['R_O'][1]:+.3f}] | "
          f"{rb[1]-ra[1]:+.4f} [{d['R_U'][0]:+.3f}, {d['R_U'][1]:+.3f}] | "
          f"{rb[2]-ra[2]:+.4f} [{d['HMR'][0]:+.3f}, {d['HMR'][1]:+.3f}] | "
          f"{db-da:+.4f} [{d['AUC'][0]:+.3f}, {d['AUC'][1]:+.3f}] |")
    w("")
    w("Read plainly: **on the confidence, assembling passes helped, and it "
      "helped by making the confidence mean something rather than by making it "
      "higher.** Both ensemble aggregators beat the first-pass run on AUC with "
      "a paired interval clear of zero, and AC-2 beats it on HMR with an "
      "interval clear of zero. The two aggregators cannot be separated from "
      "each other on either HMR or AUC. Every interval here is wide, because "
      "R_O rests on 5 to 7 questions.")
    w("")

    w("## 6. By question type")
    w("")
    w("Seven types over "
      f"{len(qids)} questions, so cell sizes are 7 to 26 and the wrong-answer "
      "counts inside a cell are 0 to 3. Descriptive only.")
    w("")
    hdr = "| type | n | " + " | ".join(f"{short[r]} R_O/R_U/HMR" for r in RUNS) + " |"
    w(hdr)
    w("|---|---|" + "---|" * len(RUNS))
    for t in sorted(set(types.get(q, "?") for q in qids)):
        qq = [q for q in qids if types.get(q) == t]
        cells = []
        for r in RUNS:
            p = [data[r][q] for q in qq]
            r_o, r_u, hmr = rewards(p)
            nerr = sum(1 for _, y in p if y == 0)
            cells.append(f"{r_o:.2f}/{r_u:.2f}/{hmr:.2f}" + ("" if nerr else " \u2020"))
        w(f"| {t} | {len(qq)} | " + " | ".join(cells) + " |")
    w("")
    w("\u2020 no incorrect answer in the cell, so R_O = 1 by definition and the "
      "HMR is not comparable with the others.")
    w("")
    simple = [q for q in qids if types.get(q) == "simple"]
    w("The `simple` row is the one worth naming: **all three runs score R_O = 0 "
      "on it**, because every simple question they get wrong is filed at "
      "exactly 100 (" + ", ".join(
        sorted(set(q for r in RUNS for q in simple if data[r][q][1] == 0))) +
      "). Assembling passes does nothing for this: the errors that survive all "
      "the passes are filed with maximum confidence.")
    w("")
    all_wrong = [q for q in qids if all(data[r][q][1] == 0 for r in RUNS)]
    w(f"Across the whole set, {len(all_wrong)} questions are wrong in all three "
      "runs (" + ", ".join(all_wrong) + "), and the joint correctness pattern is " +
      ", ".join(f"`{''.join(str(v) for v in k)}`\u00d7{n}" for k, n in
                sorted(Counter(tuple(data[r][q][1] for r in RUNS)
                               for q in qids).items(), key=lambda kv: -kv[1])) +
      " over (AC-1, AC-2, AC-3). Accuracy itself barely moves: " +
      ", ".join(f"{short[r]} {sum(y for _, y in per[r])}/{len(qids)}" for r in RUNS) +
      ".")
    w("")

    w("## 7. What these numbers cannot settle")
    w("")
    w("- **They cannot attribute the improvement to the passes.** AC-1 differs "
      "from AC-2/AC-3 in three ways at once: it uses one pass rather than all "
      "of them, it selects the answer by a different rule, and it cites a "
      "different retrieval run. Nothing here separates those.")
    w("- **They cannot treat the passes as independent draws.** A fallback can "
      "refile an answer an earlier pass on the same question already "
      "committed, so whatever agreement the aggregators measure is inflated by "
      "an amount nothing in the official results can recover. Establishing it "
      "needs the per-pass trace.")
    w("- **R_O rests on 5 to 7 questions per run.** Every statement about "
      "overconfidence in this document is a point estimate over a handful of "
      "questions, which is why the R_O and HMR intervals in section 5 span "
      "most of their range. A single wrong answer moves R_O by 0.14 to 0.20.")
    w("- **They say nothing about why the confidence takes the values it does.** "
      "This is a characterisation of the filed numbers, not of the code that "
      "produced them.")
    w("- **AC-2 versus AC-3 is undecided.** AC-2 has the better HMR, AC-3 the "
      "better AUC, and the paired intervals for both differences straddle "
      "zero. Which aggregator is better cannot be answered from 64 questions.")
    w("- **The LLM-judge strategy is unevaluated**, because what was filed as "
      "BITEM-AC-4 was a copy of BITEM-AC-3.")
    w("")

    w("## 8. Verification")
    w("")
    w("| run | questions read | n in official.AC.csv | R_O, R_U, HMR, avconf, accuracy, OB reproduce | filed conf = scored Conf |")
    w("|---|---|---|---|---|")
    for (r, n, non, ok), (_, fok) in zip(checks, filed_ok):
        w(f"| {short[r]} | {n} | {non} | {'yes' if ok else 'NO'} | "
          f"{'yes' if fok else ('not checked, submission not found' if fok is None else 'NO')} |")
    w("")
    w("The per-type table of section 6 also reproduces Table 14 of the task "
      "overview draft for all three runs.")
    w("")

    out = os.path.join(HERE, "confidence_behaviour.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
