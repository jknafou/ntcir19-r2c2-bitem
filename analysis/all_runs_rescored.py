#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Every official AC run rescored under accHMR, for the paper's appendix table.

Reads only `results/OfficialResults/` -- the per-question ACnev files give the filed confidence
and the organisers' correctness judgement, which is everything accHMR needs. Reproduces the
organisers' own n, accuracy, R_O, R_U and HMR from the same files as a check, to 5e-5.

A question carrying `Conf= -1` was not answered by that run and is excluded from Q', exactly as
the organisers exclude it. The table follows the official convention so the two orderings are
comparable.

    /usr/bin/python3 analysis/all_runs_rescored.py

Writes analysis/all_runs_rescored.md, containing the LaTeX table body for the paper's appendix.
"""
from __future__ import annotations
import csv, glob, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OFF = ROOT / "results" / "OfficialResults"


def load():
    official = {r[""].replace(".official", ""): r
                for r in csv.DictReader((OFF / "official.AC.csv").open(encoding="utf-8"))}
    rows, maxerr = [], 0.0
    for p in sorted(glob.glob(str(OFF / "ACnev" / "*.ACnev"))):
        name = os.path.basename(p).replace(".official.ACnev", "")
        conf, corr = {}, {}
        for line in open(p, encoding="utf-8"):
            t = line.split()
            if len(t) < 2:
                continue
            if t[1].startswith("Conf="):
                conf[t[0]] = float(line.split("=")[1]) / 100.0
            elif t[1].startswith("Correct="):
                corr[t[0]] = int(float(line.split("=")[1]))
        qs = sorted(q for q in set(conf) & set(corr) if conf[q] >= 0)
        n = len(qs)
        acc = sum(corr[q] for q in qs) / n
        av = sum(conf[q] for q in qs) / n
        bs = sum((conf[q] - corr[q]) ** 2 for q in qs) / n
        neg = [conf[q] for q in qs if not corr[q]]
        pos = [conf[q] for q in qs if corr[q]]
        ro = 1.0 if not neg else 1 - sum(neg) / len(neg)
        ru = 1.0 if not pos else sum(pos) / len(pos)
        hmr = 0.0 if ro + ru == 0 else 2 * ro * ru / (ro + ru)
        if name in official:
            for key, val in (("n", n), ("accuracy", acc), ("R_O", ro), ("R_U", ru), ("HMR", hmr)):
                maxerr = max(maxerr, abs(float(official[name][key]) - val))
        rows.append({"run": name, "n": n, "acc": acc, "R_O": ro, "R_U": ru, "HMR": hmr,
                     "avconf": av, "BS": bs, "accBS": acc * (1 - bs), "accHMR": acc * hmr})
    return rows, maxerr


def main() -> int:
    rows, maxerr = load()
    # The official HMR rank is taken over all 25 filed runs, before BITEM-AC-4 is dropped from
    # the display. Ranking the remaining runs instead would shift every row below AC-4 by one and
    # disagree with the overview's table.
    # Competition ranking: tied HMRs share a rank. BITEM-AC-3 and BITEM-AC-4 have identical HMR
    # (they are the same file), so a strict ordering between them would be arbitrary.
    hmr_rank = {r["run"]: 1 + sum(1 for o in rows if o["HMR"] > r["HMR"] + 1e-12) for r in rows}
    rows = [r for r in rows if r["run"] != "BITEM-AC-4"]      # byte-copy of AC-3, filed in error
    full = sorted([r for r in rows if r["n"] == 64], key=lambda r: -r["accHMR"])
    part = sorted([r for r in rows if r["n"] != 64], key=lambda r: -r["accHMR"])

    def body(rs, start=1):
        out = []
        for i, r in enumerate(rs, start):
            nm = r["run"]
            nm = f"\\textbf{{{nm}}}" if nm.startswith("BITEM") else nm
            out.append(f"{i} & {nm} & {r['n']} & {r['acc']:.4f} & {r['R_O']:.4f} & "
                       f"{r['R_U']:.4f} & {r['HMR']:.4f} & {r['avconf']:.4f} & "
                       f"{r['accHMR']:.4f} & {hmr_rank[r['run']]} \\\\")
        return out

    L = ["# Every answer run rescored under accBS", "",
         f"Reproduces the organisers' `n`, accuracy, $R_O$, $R_U$ and HMR from the ACnev files to "
         f"**{maxerr:.1e}**. Ranked by accHMR $= \\mathrm{{acc}}\\cdot\\mathrm{{HMR}}$.", "",
         "Rows for the LaTeX table in `docs/appendix-a.md`:", "", "```latex"]
    L += body(full)
    L += ["\\midrule"] + body(part, len(full) + 1) + ["```", ""]
    b = [r for r in full if r["run"].startswith("BITEM")]
    L += ["## Where our runs land", "",
          "| run | accHMR rank | HMR rank |", "|---|---:|---:|"]
    for r in b:
        L.append(f"| {r['run']} | {full.index(r) + 1} | {hmr_rank[r['run']]} |")
    (ROOT / "analysis" / "all_runs_rescored.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"reproduction max abs error {maxerr:.1e}   full-length {len(full)}   partial {len(part)}")
    for r in b:
        print(f"  {r['run']}: accHMR rank {full.index(r) + 1}, HMR rank {hmr_rank[r['run']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
