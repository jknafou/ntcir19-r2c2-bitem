# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Per-question confidence of the three answer runs, drawn the way the organisers draw it.

One measurement, rendered rather than tabulated: for each of BITEM-AC-1, -2 and -3, the
confidence filed on every scored question, coloured by whether that run's answer was judged
correct. It is the organisers' Figure 2 idiom (Overview, Fig. 2) applied to our own runs and
stacked so that a column is one question across all three, which is what makes the aggregators'
effect on the confidence scale legible.

Reads analysis/results_cache.tsv and writes exactly one figure source,
docs/figures/confidence.tex, plus analysis/confidence_figure.md. Nothing under results/ or
submission/ is touched. Build the figure afterwards with:

    bash docs/figures/build.sh confidence

Usage:  uv run python analysis/confidence_figure.py
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "analysis" / "results_cache.tsv"
# --with-ac4 appends the recovered model-as-judge run, which the organisers never scored, and
# writes a separate figure so the paper's own figure keeps only judged runs.
WITH_AC4 = "--with-ac4" in sys.argv
# --with-revised appends the two runs rebuilt from the recorded passes -- one reading only the
# first pass, one reading the first four -- which were never filed and are marked as such.
WITH_REVISED = "--with-revised" in sys.argv
# --panels=k1,k2 keeps only those panels, by key (ac1 ac2 ac3 ac4 rev1 rev4), and names the
# figure after them. The paper splits the filed runs from the rebuilt ones into two figures, so
# neither is read as a continuation of the other.
PANELS = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--panels=")), None)
TEX = ROOT / "docs" / "figures" / (
    ("confidence-" + "-".join(PANELS) + ".tex") if PANELS else
    "confidence-full.tex" if WITH_REVISED else
    "confidence-ac4.tex" if WITH_AC4 else "confidence.tex")
MD = ROOT / "analysis" / ("confidence_figure" + TEX.stem[len("confidence"):] + ".md")

RUNS = [("ac1", "BITEM-AC-1", "single pass"),
        ("ac2", "BITEM-AC-2", "most confident"),
        ("ac3", "BITEM-AC-3", "summed agreement")]
if WITH_AC4:
    # The dagger matches Table 3: this run was recovered, never filed, and never scored by the
    # organisers, so its panel must not be read as a result.
    RUNS.append(("ac4", r"BITEM-AC-4$^{\dagger}$", "model as judge"))
if WITH_REVISED:
    # Neither was filed: both are rebuilt from the recorded passes, so their panels are marked
    # and must not be read as submitted results. The repaired aggregator is deliberately not a
    # panel: it changes six answers and no confidence rule, so its profile is AC-3's and a reader
    # comparing shapes would learn nothing from it. Its numbers stay in Table 3.
    RUNS.append(("rev1", r"revised$^{\ddagger}$", "single pass, longest deliberation"))
    RUNS.append(("rev4", r"revised$^{\ddagger}$", "four passes, independent agreement"))

if PANELS:
    RUNS = [r for r in RUNS if r[0] in PANELS]
    if not RUNS:
        raise SystemExit(f"--panels matched nothing; keys are ac1 ac2 ac3 ac4 rev1 rev4")

#: Which panels are runs we actually filed. The figure rules a line between the two bands, because
#: the whole point of the lower ones is that they were never submitted and must not be read as
#: results. A run's band, not its position, decides which side it falls on.
FILED = {"ac1", "ac2", "ac3"}
#: A modest amount of extra space at the filed/rebuilt boundary, plus a rule. The rule must clear
#: the first rebuilt panel's run label, which sits 0.155 above its panel and carries a white
#: backing roughly 0.13 high, so anything below ~1.11 above the panel bottom cuts through that
#: backing rather than sitting over it.
BAND_GAP = 0.18
LABEL_CLEAR = 0.40     # rule offset above the first rebuilt panel's top gridline

# The organisers' order, from their Tables 12 and 13, with the short forms the column width allows.
TYPES = [("simple", "simple"), ("simple-w-condition", "simple w/c"), ("comparison", "comparison"),
         ("aggregation", "aggregation"), ("multi-hop", "multi-hop"),
         ("post-processing-heavy", "post-proc.")]

# Geometry, in cm. The whole drawing has to sit in one ACM sigconf column (8.5cm), so the bars
# are packed at a pitch a hair over a millimetre and the panels are kept short: the reader has
# to see high against low, not read a value off the axis.
PLOT_W = 7.55
PANEL_H, PANEL_GAP = 0.80, 0.34
METRIC_H = 0.02        # band under each panel holding acc, R_O, R_U and HMR per question type
GROUP_GAP = 0.90        # in units of one bar pitch, between question-type groups


def reward(conf, corr):
    """The organisers' per-question-set reward terms, on one run's slice of one question type.

    Definitions are the task design paper's, eq. (1)-(4): with I- and I+ the incorrect and correct
    question sets, R_O = 1 - mean over I- of the confidence, R_U = mean over I+ of it, and HMR
    their harmonic mean, zero when both are zero. Where a set is empty the reward is 1 by
    definition, the paper's reasoning being that a run answering everything correctly had no
    opportunity to be overconfident. That case arises here per question type, never for a whole
    run: none of the 25 official runs has accuracy 1.

    On a seven-question type each reward still rests on a handful of questions, which is why the
    figure caption calls these cells descriptive.
    """
    neg = [c / 100 for c, k in zip(conf, corr) if k == 0]
    pos = [c / 100 for c, k in zip(conf, corr) if k == 1]
    ro = 1.0 if not neg else 1 - sum(neg) / len(neg)
    ru = 1.0 if not pos else sum(pos) / len(pos)
    hmr = 0.0 if ro + ru == 0 else 2 * ro * ru / (ro + ru)
    return {"acc": sum(corr) / len(corr), "$R_O$": ro, "$R_U$": ru, "HMR": hmr}


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l == 1]
    neg = [s for s, l in zip(score, label) if l == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main() -> int:
    rows = [r for r in csv.DictReader(CACHE.open(encoding="utf-8"), delimiter="\t")
            if re.fullmatch(r"-?[0-9.]+", (r.get("ac1_correct") or "").strip())]
    if WITH_AC4:
        spec = importlib.util.spec_from_file_location("lost_ac4", ROOT / "analysis" / "lost_ac4.py")
        la = importlib.util.module_from_spec(spec); spec.loader.exec_module(la)
        rec = la.parse(ROOT / "submission/unofficial/model-as-judge.AC")
        filed = {k: la.parse(ROOT / f"submission/ac-2026-05-15/BITEM-AC-{k[-1]}")
                 for k in ("ac1", "ac2", "ac3")}
        for r in rows:
            q = r["qid"]; ans = rec[q][0]
            hit = next((k for k in filed if filed[k][q][0] == ans), None)
            r["ac4_conf"] = str(rec[q][1])
            r["ac4_correct"] = str(int(r[hit + "_correct"]) if hit else int(la.ADJUDICATED[q][0]))
    if WITH_REVISED:
        sys.path.insert(0, str(ROOT / "analysis"))
        from score_ac import read_ac, judge          # noqa: E402
        for key, fn in (("rev4", "revised-independent-agreement.AC"),
                        ("rev1", "revised-longest-deliberation.AC")):
            run = read_ac(ROOT / "submission" / "unofficial" / fn)
            for r in rows:
                q = r["qid"]
                r[key + "_conf"] = str(run[q][1])
                r[key + "_correct"] = str(int(judge(q, run[q][0])[0]))
    n = len(rows)
    # Group by question type, organisers' order, qid order within a group. Every panel shares
    # these x positions, so a column still reads down as one question.
    groups, seen = [], 0
    for key, label in TYPES:
        g = sorted((r for r in rows if r["qtype"] == key), key=lambda r: r["qid"])
        groups.append((label, g)); seen += len(g)
    assert seen == n, f"{seen} of {n} questions matched a type"
    pitch = PLOT_W / (n + GROUP_GAP * (len(TYPES) - 1))
    bw = pitch * 0.72
    # x of each question, in the grouped order, plus each group's span for its label and rule
    xs, spans, cur = [], [], 0.0
    for _, g in groups:
        start = cur
        for _ in g:
            xs.append(cur); cur += pitch
        spans.append((start, cur - pitch + bw))
        cur += GROUP_GAP * pitch
    order = [r for _, g in groups for r in g]

    L = []
    A = L.append
    A(r"% confidence -- GENERATED by analysis/confidence_figure.py; do not edit by hand.")
    A(r"% Build:  bash docs/figures/build.sh confidence")
    A(r"%")
    A(r"% The organisers' Figure 2 idiom applied to our three answer runs: one bar per question,")
    A(r"% height the filed confidence, green where that run's answer was judged correct and red")
    A(r"% where it was not. Bars are grouped by question type in the organisers' order, and each")
    A(r"% is labelled with its question number, so the prose can point at one.")
    A(r"\documentclass[border=2pt]{standalone}")
    A(r"\usepackage{tikz}")
    A(r"\definecolor{okgreen}{HTML}{1B7A3E}")
    A(r"\definecolor{badred}{HTML}{C62828}")
    A(r"\definecolor{gridgrey}{HTML}{C8CED6}")
    A(r"\definecolor{rulegrey}{HTML}{8A929C}")
    A(r"\begin{document}")
    A(r"\begin{tikzpicture}[x=1cm,y=1cm,line width=0.3pt]")

    block = PANEL_H + METRIC_H + PANEL_GAP
    # Panels are laid out top-down. bidx is the first rebuilt panel; every panel above it is
    # lifted by BAND_GAP so the two bands read as two blocks rather than one list of seven.
    bidx = next((i for i, (k, _, _) in enumerate(RUNS) if k not in FILED), len(RUNS))
    # Both bands must be present for a divider to mean anything: with only filed panels, or
    # only rebuilt ones, there is nothing to divide and the rule is just a line at the top.
    split = 0 < bidx < len(RUNS)
    lift = (lambda k: BAND_GAP if k < bidx else 0.0) if split else (lambda k: 0.0)
    top = (len(RUNS) - 1) * block + METRIC_H + PANEL_H + lift(0)
    metrics = {}   # (run, type) -> the four terms, for the bolding pass
    for key, _, _ in RUNS:
        for label, g in groups:
            metrics[(key, label)] = reward([float(r[f'{key}_conf']) for r in g],
                                          [int(r[f'{key}_correct']) for r in g])
    ROWS = ['acc', '$R_O$', '$R_U$', 'HMR']
    # The group rules go down first: they must sit under the bars and under the white backing of
    # the run labels, which straddle the gaps between panels.
    for i in range(1, len(groups)):
        xm = (spans[i - 1][1] + spans[i][0]) / 2
        L.append(rf"  \draw[rulegrey,line width=0.2pt] ({xm:.4f},-0.26) -- ({xm:.4f},{top:.3f});")
    if split:
        # Above the run label's white backing, and below the last filed panel's zero gridline,
        # which BAND_GAP has pushed to PANEL_H + METRIC_H + PANEL_GAP + BAND_GAP.
        ydiv = (len(RUNS) - 1 - bidx) * block + METRIC_H + PANEL_H + LABEL_CLEAR
        L.append(rf"  \draw[rulegrey,line width=0.45pt] "
                 rf"(-0.30,{ydiv:.3f}) -- ({PLOT_W:.3f},{ydiv:.3f});")
    best = {}
    for label, _ in groups:
        for m in ROWS:
            v = [metrics[(k, label)][m] for k, _, _ in RUNS]
            v = [x for x in v if x is not None]
            best[(label, m)] = max(v) if v else None
    for k, (key, run, nick) in enumerate(RUNS):
        bottom = (len(RUNS) - 1 - k) * block + lift(k)
        base = bottom + METRIC_H
        conf = [float(r[f"{key}_conf"]) for r in order]
        corr = [int(r[f"{key}_correct"]) for r in order]
        for frac, lab in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
            y = base + frac * PANEL_H
            A(rf"  \draw[gridgrey] (0,{y:.3f}) -- ({PLOT_W:.3f},{y:.3f});")
            A(rf"  \node[anchor=east,font=\tiny,inner sep=1pt] at (-0.02,{y:.3f}) {{{lab}}};")
        # the group rules run the full height, so the run label carries a translucent white
        # backing rather than letting a rule cut through the text
        A(rf"  \node[anchor=west,font=\tiny,inner sep=1.2pt,fill=white,fill opacity=0.82,"
          rf"text opacity=1] at (-0.04,{base + PANEL_H + 0.155:.3f}) "
          rf"{{\textbf{{{run}}} \emph{{{nick}}}}};")
        # A confidence of exactly zero still has to show its colour: question 0039 is filed at
        # 0 by two of the three runs and wrong in both, which is the behaviour the reward term
        # is built to pay for. Floor the drawn height rather than drop the bar.
        for x, c, ok in zip(xs, conf, corr):
            col = "okgreen" if ok else "badred"
            h = max(c / 100 * PANEL_H, 0.022 * PANEL_H)
            A(rf"  \fill[{col}] ({x:.4f},{base:.4f}) rectangle ({x + bw:.4f},{base + h:.4f});")
        # The per-type reward terms are not drawn. They are computed above and written to
        # confidence_figure.md instead: R_O averages over the questions a run got wrong, which per
        # type is between none and three here, so they cannot carry a claim, and four numbers in a
        # seven-question group would need 2.7pt type.
    # question number under every bar, reading upwards so 64 of them fit in one column
    for x, r in zip(xs, order):
        A(rf"  \node[anchor=east,rotate=90,font=\fontsize{{3}}{{3.4}}\selectfont,inner sep=1.2pt] "
          rf"at ({x + bw / 2:.4f},-0.01) {{{int(r['qid'])}}};")
    # type bands: a rule spanning the panels at each boundary, and a label under the numbers
    for i, ((label, g), (x0, x1)) in enumerate(zip(groups, spans)):
        A(rf"  \node[anchor=north,font=\fontsize{{3.2}}{{3.6}}\selectfont,inner sep=1pt] at "
          rf"({(x0 + x1) / 2:.4f},-0.28) {{{label}}};")
    ly = top + 0.145
    A(rf"  \fill[okgreen] ({PLOT_W - 1.98:.3f},{ly - 0.05:.3f}) rectangle ({PLOT_W - 1.86:.3f},{ly + 0.07:.3f});")
    A(rf"  \node[anchor=west,font=\tiny,inner sep=1pt] at ({PLOT_W - 1.84:.3f},{ly:.3f}) {{correct}};")
    A(rf"  \fill[badred] ({PLOT_W - 1.00:.3f},{ly - 0.05:.3f}) rectangle ({PLOT_W - 0.88:.3f},{ly + 0.07:.3f});")
    A(rf"  \node[anchor=west,font=\tiny,inner sep=1pt] at ({PLOT_W - 0.86:.3f},{ly:.3f}) {{incorrect}};")
    A(r"\end{tikzpicture}")
    A(r"\end{document}")
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8")

    lines = [f"# Per-question confidence, {n} scored questions", "",
             f"Source for `docs/figures/{TEX.stem}.tex`, the organisers' Figure 2 idiom applied to",
             f"{len(RUNS)} answer runs. One bar per question, height the filed confidence, green",
             "where the answer was judged correct.", "", "| run | AUC | mean conf | filed at 100 |",
             "|---|---|---|---|"]
    for key, run, nick in RUNS:
        conf = [float(r[f"{key}_conf"]) for r in rows]
        corr = [int(r[f"{key}_correct"]) for r in rows]
        lines.append(f"| {run} *{nick}* | {auc(conf, corr):.3f} | {sum(conf)/n:.1f} | "
                     f"{sum(1 for c in conf if c >= 100)} |")
    lines += ["", "## Reward terms per question type", "",
              "`R_O` averages over the questions a run got wrong. Per type that is between none and",
              "three questions here, so these are descriptive only; where a run has no wrong answer",
              "in a type the organisers' definition leaves `R_O` at 1.", "",
              "| type | run | acc | R_O | R_U | HMR | errors |", "|---|---|---|---|---|---|---|"]
    for label, g in groups:
        for key, run, nick in RUNS:
            m = metrics[(key, label)]
            e = sum(1 for r in g if int(r[f"{key}_correct"]) == 0)
            f = lambda v: "n/a" if v is None else f"{v:.2f}"
            lines.append(f"| {label} | {run} *{nick}* | {f(m['acc'])} | {f(m['$R_O$'])} | "
                         f"{f(m['$R_U$'])} | {f(m['HMR'])} | {e} |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TEX}")
    print(f"wrote {MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
