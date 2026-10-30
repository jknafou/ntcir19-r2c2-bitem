# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Per-question-type cut of the BITEM NTCIR-19 R2C2 runs.

One measurement: does the first-pass -> all-passes gain hold within each of the
six organiser question types, in the PR subtask (BITEM-PG-1 vs BITEM-PG-2) and
in the AC subtask (BITEM-AC-1 vs BITEM-AC-2 / BITEM-AC-3)?

Pure read.  Reads results/ and submission/ and writes exactly one file,
analysis/by_question_type.md.  Nothing under results/ or submission/ is touched.

Usage:  uv run python analysis/by_question_type.py submission
"""

from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "OfficialResults"
OUT = Path(__file__).resolve().with_suffix(".md")

PR_RUNS = ["BITEM-PG-1", "BITEM-PG-2"]
AC_RUNS = ["BITEM-AC-1", "BITEM-AC-2", "BITEM-AC-3"]  # AC-4 excluded: byte-identical to AC-3

# Order types by the organisers' own table order, not by count.
TYPE_ORDER = [
    "simple",
    "simple-w-condition",
    "comparison",
    "aggregation",
    "multi-hop",
    "post-processing-heavy",
]

# Values transcribed from the task overview draft, used only as a check on our
# own extraction.  Tables 12 and 13 (PR by type, direct qrels) and Table 14
# (AC HMR / R_O / R_U by type).
OVERVIEW_T12 = {  # MSnDCG@20
    "BITEM-PG-1": [0.3034, 0.2888, 0.4168, 0.4498, 0.3749, 0.3900],
    "BITEM-PG-2": [0.3608, 0.3403, 0.4392, 0.4859, 0.4747, 0.5725],
}
OVERVIEW_T13 = {  # Q@20
    "BITEM-PG-1": [0.1875, 0.2394, 0.3703, 0.3876, 0.3231, 0.2926],
    "BITEM-PG-2": [0.2530, 0.3125, 0.3869, 0.4721, 0.4321, 0.5461],
}
OVERVIEW_T14 = {  # (HMR, R_O, R_U) per type
    "BITEM-AC-1": [(0.0000, 0.0000, 0.8804), (0.2669, 0.1600, 0.8033),
                   (0.7676, 1.0000, 0.6229), (0.3417, 0.2200, 0.7650),
                   (0.0198, 0.0100, 0.7911), (0.6891, 0.5900, 0.8283)],
    "BITEM-AC-2": [(0.0000, 0.0000, 0.9376), (0.4402, 0.3000, 0.8267),
                   (0.6616, 1.0000, 0.4943), (0.5233, 0.4000, 0.7567),
                   (0.0582, 0.0300, 0.9967), (0.8764, 1.0000, 0.7800)],
    "BITEM-AC-3": [(0.0000, 0.0000, 0.9913), (0.4570, 0.3000, 0.9583),
                   (0.9928, 1.0000, 0.9857), (0.5714, 0.4000, 1.0000),
                   (0.0583, 0.0300, 1.0000), (1.0000, 1.0000, 1.0000)],
}


# Loading

def load_types() -> dict[str, str]:
    types = {}
    for line in (RES / "r2c2questiontypes").read_text().splitlines():
        if not line.strip():
            continue
        qid, qtype = line.split()
        types[qid] = qtype
    return types


def relevant_per_question(types: dict[str, str]) -> dict[str, tuple[float, float, int]]:
    """Mean number of passages judged L1 or above per question, by type, for PG-1 and PG-2,
    under the direct qrels, whose keys are <run>;<rank> so every listed passage is judged."""
    n: dict[tuple[str, str], int] = {}
    for line in (RES / "official.consolidated.directL0-L4.qrels").read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or ";" not in parts[1]:
            continue
        qid, key, level = parts
        run = key.split(";")[0]
        if run in ("BITEM-PG-1", "BITEM-PG-2") and level != "L0":
            n[(qid, run)] = n.get((qid, run), 0) + 1
    judged = {q for q, _ in n}
    out = {}
    for t in sorted(set(types.values())):
        qs = [q for q in types if types[q] == t and q in judged]
        if qs:
            out[t] = (mean([n.get((q, "BITEM-PG-1"), 0) for q in qs]),
                      mean([n.get((q, "BITEM-PG-2"), 0) for q in qs]), len(qs))
    return out


def load_matrix(name: str) -> dict[str, dict[str, float]]:
    """A *.tsmdressed.csv -> {qid: {run: score}}."""
    out = {}
    with (RES / name).open(newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        runs = header[1:]
        for row in rdr:
            if not row or not row[0].strip():
                continue
            out[row[0]] = {r: float(v) for r, v in zip(runs, row[1:])}
    return out


def load_acnev(run: str) -> dict[str, dict[str, float]]:
    """ACnev/<run>.official.ACnev -> {qid: {Conf, Correct, NugPrec}}."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    pat = re.compile(r"^(\d{4})\s+(Conf|Correct|NugPrec)=\s*([-\d.]+)\s*$")
    for line in (RES / "ACnev" / f"{run}.official.ACnev").read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)][m.group(2)] = float(m.group(3))
    return dict(out)


def load_gold() -> dict[str, tuple[str, str]]:
    """The topics workbook -> {qid: (question text, gold answer)}.

    The sheet has no header row; it is a block per question, the qid and question
    in columns A and B of the first row and the gold answer in column C of the
    next.  Parsed with zipfile + ElementTree so analysis/ stays dependency-free.
    """
    path = REPO / "results" / "r2c2topics-and-goldanswers.xlsx"
    if not path.exists():
        return {}
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared = ["".join(t.text or "" for t in si.iter(ns + "t"))
                  for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
            cells = {}
            for c in row.iter(ns + "c"):
                col = "".join(ch for ch in c.get("r") if ch.isalpha())
                v = c.find(ns + "v")
                if v is None:
                    continue
                cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(cells)
    out = {}
    for i, r in enumerate(rows):
        qid = r.get("A", "")
        if re.fullmatch(r"\d{4}", qid) and i + 1 < len(rows):
            out[qid] = (r.get("B", ""), rows[i + 1].get("C", ""))
    return out


def load_filed_answer(subdir: Path, run: str) -> dict[str, tuple[str, int]]:
    """The filed AC run -> {qid: (answer string, confidence)}."""
    out = {}
    qid = None
    for line in (subdir / run).read_text().splitlines():
        m = re.match(r"^<(\d{4})>\s*$", line)
        if m:
            qid = m.group(1)
            continue
        if qid and qid not in out and not line.startswith("</") and ";" in line \
                and not re.match(r"^\d+;", line):
            ans, conf = line.rsplit(";", 1)
            out[qid] = (ans, int(conf))
    return out


def load_filed_conf(subdir: Path, run: str) -> dict[str, int]:
    """Confidences as we filed them, for a cross-check against ACnev."""
    txt = (subdir / run).read_text()
    out = {}
    qid = None
    for line in txt.splitlines():
        m = re.match(r"^<(\d{4})>\s*$", line)
        if m:
            qid, seen = m.group(1), False
            continue
        if qid and not line.startswith("</") and ";" in line and not re.match(r"^\d+;", line):
            if qid not in out:
                out[qid] = int(line.rsplit(";", 1)[1])
    return out


# Statistics

def hmr_parts(pairs: list[tuple[float, int]]) -> tuple[float, float, float, float, float]:
    """pairs = [(confidence 0-100, correct 0/1)].  -> acc, R_O, R_U, HMR, avconf."""
    if not pairs:
        return (float("nan"),) * 5
    ps = [c / 100.0 for c, _ in pairs]
    wrong = [p for p, (_, k) in zip(ps, pairs) if k == 0]
    right = [p for p, (_, k) in zip(ps, pairs) if k == 1]
    acc = len(right) / len(pairs)
    r_o = 1.0 if not wrong else 1.0 - sum(wrong) / len(wrong)
    r_u = 1.0 if not right else sum(right) / len(right)
    hmr = 0.0 if (r_o == 0 and r_u == 0) else (2 * r_o * r_u / (r_o + r_u) if (r_o + r_u) else 0.0)
    return acc, r_o, r_u, hmr, sum(ps) / len(ps)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def f(x, nd=4):
    return "—" if x != x else f"{x:.{nd}f}"


# Main

def main() -> int:
    subdir_arg = sys.argv[1] if len(sys.argv) > 1 else "submission"
    subdir = Path(subdir_arg)
    if not subdir.is_absolute():
        subdir = REPO / subdir
    ac_subdir = subdir / "ac-2026-05-15"

    types = load_types()
    checks: list[str] = []

    msndcg_d = load_matrix("official.PRruns.cons.directL0-L4.nev.MSnDCG@0020.tsmdressed.csv")
    q_d = load_matrix("official.PRruns.cons.directL0-L4.nev.Q@0020.tsmdressed.csv")
    msndcg_r = load_matrix("derivedL0-L4.PRruns.derivedL0-L4.nev.MSnDCG@0020.tsmdressed.csv")
    q_r = load_matrix("derivedL0-L4.PRruns.derivedL0-L4.nev.Q@0020.tsmdressed.csv")
    ac = {r: load_acnev(r) for r in AC_RUNS}

    scored = sorted(msndcg_d)                      # 64 qids, direct regime
    scored_ac = sorted(next(iter(ac.values())))
    derived_qids = sorted(msndcg_r)                # 63 qids
    filed = sorted(types)                          # 65 qids

    checks.append(f"question types file: {len(filed)} qids; PR direct matrices: {len(scored)}; "
                  f"AC ACnev: {len(scored_ac)}; PR derived matrices: {len(derived_qids)}")
    assert scored == scored_ac, "PR-direct and AC question sets disagree"
    missing = sorted(set(filed) - set(scored))
    checks.append(f"filed but not scored: {missing} (type: "
                  f"{', '.join(types[q] for q in missing)})")
    checks.append(f"scored under direct but not derived: "
                  f"{sorted(set(scored) - set(derived_qids))}")

    # cross-check filed confidences against ACnev
    if ac_subdir.is_dir():
        bad = []
        for run in AC_RUNS:
            fc = load_filed_conf(ac_subdir, run)
            for qid in scored_ac:
                if qid in fc and fc[qid] != int(ac[run][qid]["Conf"]):
                    bad.append((run, qid, fc[qid], ac[run][qid]["Conf"]))
        checks.append(f"filed confidence vs ACnev Conf, {len(AC_RUNS)}x{len(scored_ac)} pairs: "
                      f"{'all agree' if not bad else f'{len(bad)} MISMATCH {bad[:5]}'}")
    else:
        checks.append(f"submission dir {ac_subdir} not found — filed-confidence cross-check skipped")

    # Counts per type
    filed_counts = {t: sum(1 for q in filed if types[q] == t) for t in TYPE_ORDER}
    scored_counts = {t: sum(1 for q in scored if types[q] == t) for t in TYPE_ORDER}
    derived_counts = {t: sum(1 for q in derived_qids if types[q] == t) for t in TYPE_ORDER}

    by_type = {t: [q for q in scored if types[q] == t] for t in TYPE_ORDER}
    by_type_derived = {t: [q for q in derived_qids if types[q] == t] for t in TYPE_ORDER}

    # PR
    def pr_block(mat, groups):
        rows = {}
        for t in TYPE_ORDER:
            qs = groups[t]
            m1 = mean(mat[q]["BITEM-PG-1"] for q in qs)
            m2 = mean(mat[q]["BITEM-PG-2"] for q in qs)
            diffs = [mat[q]["BITEM-PG-2"] - mat[q]["BITEM-PG-1"] for q in qs]
            wins = sum(1 for d in diffs if d > 1e-9)
            losses = sum(1 for d in diffs if d < -1e-9)
            rows[t] = (len(qs), m1, m2, m2 - m1, wins, losses, len(qs) - wins - losses)
        allq = [q for t in TYPE_ORDER for q in groups[t]]
        m1 = mean(mat[q]["BITEM-PG-1"] for q in allq)
        m2 = mean(mat[q]["BITEM-PG-2"] for q in allq)
        diffs = [mat[q]["BITEM-PG-2"] - mat[q]["BITEM-PG-1"] for q in allq]
        wins = sum(1 for d in diffs if d > 1e-9)
        losses = sum(1 for d in diffs if d < -1e-9)
        rows["ALL"] = (len(allq), m1, m2, m2 - m1, wins, losses, len(allq) - wins - losses)
        return rows

    pr_msndcg = pr_block(msndcg_d, by_type)
    pr_q = pr_block(q_d, by_type)
    pr_msndcg_der = pr_block(msndcg_r, by_type_derived)
    pr_q_der = pr_block(q_r, by_type_derived)

    # verify against overview tables 12/13
    for label, rows, ref in (("Table 12 MSnDCG", pr_msndcg, OVERVIEW_T12),
                             ("Table 13 Q", pr_q, OVERVIEW_T13)):
        errs = []
        for run, idx in (("BITEM-PG-1", 1), ("BITEM-PG-2", 2)):
            for i, t in enumerate(TYPE_ORDER):
                got, want = rows[t][idx], ref[run][i]
                if abs(got - want) > 5e-5:
                    errs.append(f"{run}/{t}: {got:.4f} vs {want:.4f}")
        checks.append(f"overview {label} reproduced: {'yes' if not errs else 'NO — ' + '; '.join(errs)}")

    # AC
    ac_rows = {}   # run -> type -> tuple
    for run in AC_RUNS:
        ac_rows[run] = {}
        for t in TYPE_ORDER + ["ALL"]:
            qs = scored if t == "ALL" else by_type[t]
            pairs = [(ac[run][q]["Conf"], int(ac[run][q]["Correct"])) for q in qs]
            npv = mean(ac[run][q]["NugPrec"] for q in qs)
            ac_rows[run][t] = hmr_parts(pairs) + (npv, len(qs))

    errs = []
    for run in AC_RUNS:
        for i, t in enumerate(TYPE_ORDER):
            acc, r_o, r_u, hmr, _, _, _ = ac_rows[run][t]
            w_hmr, w_ro, w_ru = OVERVIEW_T14[run][i]
            for got, want, nm in ((hmr, w_hmr, "HMR"), (r_o, w_ro, "R_O"), (r_u, w_ru, "R_U")):
                if abs(got - want) > 5e-5:
                    errs.append(f"{run}/{t}/{nm}: {got:.4f} vs {want:.4f}")
    checks.append(f"overview Table 14 reproduced: {'yes' if not errs else 'NO — ' + '; '.join(errs)}")

    # per-question correctness disagreements between the first-pass run and the two ensembles
    disagree = []
    for q in scored:
        ks = {r: int(ac[r][q]["Correct"]) for r in AC_RUNS}
        if len(set(ks.values())) > 1:
            disagree.append((q, types[q], ks,
                             {r: int(ac[r][q]["Conf"]) for r in AC_RUNS}))

    # confidence shape per type: how many questions filed at exactly 100
    ceil_counts = {run: {t: sum(1 for q in by_type[t] if ac[run][q]["Conf"] >= 100)
                         for t in TYPE_ORDER} for run in AC_RUNS}

    # leave-one-out fragility: how far does a cell move if one question is removed?
    def jack_hmr(run, t):
        qs = by_type[t]
        vals = []
        for drop in qs:
            pairs = [(ac[run][q]["Conf"], int(ac[run][q]["Correct"]))
                     for q in qs if q != drop]
            vals.append(hmr_parts(pairs)[3])
        return min(vals), max(vals)

    def jack_pr(mat, t):
        qs = by_type[t]
        vals = []
        for drop in qs:
            keep = [q for q in qs if q != drop]
            vals.append(mean(mat[q]["BITEM-PG-2"] for q in keep)
                        - mean(mat[q]["BITEM-PG-1"] for q in keep))
        return min(vals), max(vals)

    jack_ac = {run: {t: jack_hmr(run, t) for t in TYPE_ORDER} for run in AC_RUNS}
    jack_msndcg = {t: jack_pr(msndcg_d, t) for t in TYPE_ORDER}

    gold = load_gold()
    filed_ans = ({run: load_filed_answer(ac_subdir, run) for run in AC_RUNS}
                 if ac_subdir.is_dir() else {})
    checks.append(f"gold answers loaded from the topics workbook: {len(gold)} questions")

    # wrong answers per type with their confidence — this is what R_O is made of
    wrongs = {run: defaultdict(list) for run in AC_RUNS}
    for run in AC_RUNS:
        for q in scored:
            if int(ac[run][q]["Correct"]) == 0:
                wrongs[run][types[q]].append((q, int(ac[run][q]["Conf"])))

    # Write
    L: list[str] = []
    w = L.append

    w("# The per-question-type cut")
    w("")
    w("One measurement: whether the first-pass-versus-all-passes design shows up inside each")
    w("question type, in retrieval and in answering. Generated by `analysis/by_question_type.py`;")
    w("reruns from the repository alone.")
    w("")
    w("## 0. How small the cells are")
    w("")
    w("This is the first thing to read and it governs everything after it. The organisers' six")
    w("types split 64 scored questions into cells of 26, 7, 7, 7, 10 and 7. Four of the six types")
    w("hold seven questions. **Nothing below is a significance claim, and no test is run on any of")
    w("it.** One question changing its label moves a seven-question accuracy cell by 14 points and")
    w("a seven-question `R_O` cell by as much as 0.14. Read the whole document as description.")
    w("")
    w("| Question type | filed | scored (direct qrels, AC) | scored (derived qrels) |")
    w("|---|---:|---:|---:|")
    for t in TYPE_ORDER:
        w(f"| {t} | {filed_counts[t]} | {scored_counts[t]} | {derived_counts[t]} |")
    w(f"| **total** | **{sum(filed_counts.values())}** | **{sum(scored_counts.values())}** "
      f"| **{sum(derived_counts.values())}** |")
    w("")
    w(f"We filed 65 questions; the official tables score 64. The missing one is "
      f"{', '.join(missing)} ({', '.join(types[q] for q in missing)}), excluded by the organisers")
    w("because the topic was fixed after release. The derived qrels drop one more, 0039")
    w("(post-processing-heavy), for lack of relevant documents — so the derived-qrels")
    w("post-processing-heavy cell holds six questions, not seven, and is not row-aligned with the")
    w("direct one.")
    w("")

    w("## 1. Retrieval: does the PG-1 → PG-2 gain hold within each type?")
    w("")
    w("Direct (consolidated LLM) qrels, the honest regime. `Δ` is PG-2 minus PG-1 on the mean;")
    w("`W/L/T` counts the questions inside the cell where PG-2 scored above, below, or level with")
    w("PG-1.")
    w("")
    w("### MSnDCG@20")
    w("")
    w("| Question type | n | PG-1 | PG-2 | Δ | W/L/T |")
    w("|---|---:|---:|---:|---:|---:|")
    for t in TYPE_ORDER + ["ALL"]:
        n, m1, m2, d, wi, lo, ti = pr_msndcg[t]
        nm = "**all**" if t == "ALL" else t
        w(f"| {nm} | {n} | {f(m1)} | {f(m2)} | {d:+.4f} | {wi}/{lo}/{ti} |")
    w("")
    w("### Q@20")
    w("")
    w("| Question type | n | PG-1 | PG-2 | Δ | W/L/T |")
    w("|---|---:|---:|---:|---:|---:|")
    for t in TYPE_ORDER + ["ALL"]:
        n, m1, m2, d, wi, lo, ti = pr_q[t]
        nm = "**all**" if t == "ALL" else t
        w(f"| {nm} | {n} | {f(m1)} | {f(m2)} | {d:+.4f} | {wi}/{lo}/{ti} |")
    w("")

    w("### The same cut under the derived qrels — reported only as a caution")
    w("")
    w("The derived qrels are built from the AC evaluation, and a large share of the AC runs")
    w("consumed a BITEM PR run, so our passages are over-represented among what got judged. These")
    w("numbers are pool-biased and are **not** a result. They are here because they show the")
    w("direction of the PG-1 → PG-2 difference is not stable across regimes.")
    w("")
    w("| Question type | n | MSnDCG Δ | Q Δ |")
    w("|---|---:|---:|---:|")
    for t in TYPE_ORDER + ["ALL"]:
        n = pr_msndcg_der[t][0]
        nm = "**all**" if t == "ALL" else t
        w(f"| {nm} | {n} | {pr_msndcg_der[t][3]:+.4f} | {pr_q_der[t][3]:+.4f} |")
    w("")

    w("## 2. Answering: accuracy and HMR by type")
    w("")
    w("BITEM-AC-1 is the first-pass run; BITEM-AC-2 (highest-confidence answer) and BITEM-AC-3")
    w("(highest confidence-sum cluster) aggregate all the recorded passes. **BITEM-AC-4 is excluded")
    w("throughout**: it is a byte-for-byte copy of BITEM-AC-3 (MD5")
    w("`dc04428a47b429b5363b150627bc7fb9`), filed by mistake, so it is not a fourth system and")
    w("reporting it would double-count AC-3. `R_O` is one minus the mean confidence on the")
    w("questions the run got wrong; `R_U` is the mean confidence on the ones it got right; HMR is")
    w("their harmonic mean. All three are computed inside the cell, over that cell's questions only.")
    w("")
    for run in AC_RUNS:
        tag = "first pass only" if run == "BITEM-AC-1" else "all recorded passes"
        w(f"### {run} ({tag})")
        w("")
        w("| Question type | n | accuracy | R_O | R_U | HMR | avconf | mean NugPrec |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in TYPE_ORDER + ["ALL"]:
            acc, r_o, r_u, hmr, avc, npv, n = ac_rows[run][t]
            nm = "**all**" if t == "ALL" else t
            w(f"| {nm} | {n} | {f(acc)} | {f(r_o)} | {f(r_u)} | {f(hmr)} | {f(avc)} | {f(npv)} |")
        w("")

    w("### Side by side: HMR, and accuracy, first pass against the ensembles")
    w("")
    w("| Question type | n | HMR AC-1 | HMR AC-2 | HMR AC-3 | acc AC-1 | acc AC-2 | acc AC-3 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in TYPE_ORDER + ["ALL"]:
        nm = "**all**" if t == "ALL" else t
        n = ac_rows["BITEM-AC-1"][t][6]
        h = [ac_rows[r][t][3] for r in AC_RUNS]
        a = [ac_rows[r][t][0] for r in AC_RUNS]
        w(f"| {nm} | {n} | {f(h[0])} | {f(h[1])} | {f(h[2])} | {f(a[0])} | {f(a[1])} | {f(a[2])} |")
    w("")

    w("### What each `R_O` cell is actually made of")
    w("")
    w("`R_O` is a mean over the wrong answers alone, so in a seven-question cell with one mistake")
    w("it is one number read off one question. The questions each run got wrong, with the")
    w("confidence it filed them at:")
    w("")
    w("| Question type | BITEM-AC-1 | BITEM-AC-2 | BITEM-AC-3 |")
    w("|---|---|---|---|")
    for t in TYPE_ORDER:
        cells = []
        for run in AC_RUNS:
            ws = wrongs[run].get(t, [])
            cells.append(", ".join(f"{q}@{c}" for q, c in ws) if ws else "—")
        w(f"| {t} | {cells[0]} | {cells[1]} | {cells[2]} |")
    w("")
    w("Questions filed at exactly confidence 100, per cell:")
    w("")
    w("| Question type | n | AC-1 | AC-2 | AC-3 |")
    w("|---|---:|---:|---:|---:|")
    for t in TYPE_ORDER:
        w(f"| {t} | {scored_counts[t]} | {ceil_counts['BITEM-AC-1'][t]} | "
          f"{ceil_counts['BITEM-AC-2'][t]} | {ceil_counts['BITEM-AC-3'][t]} |")
    w("")

    w("### Where the three runs disagree on correctness")
    w("")
    if not disagree:
        w("Nowhere.")
    else:
        w("| qid | type | AC-1 | AC-2 | AC-3 |")
        w("|---|---|---|---|---|")
        for q, t, ks, cs in disagree:
            cells = [f"{'correct' if ks[r] else 'wrong'} @ {cs[r]}" for r in AC_RUNS]
            w(f"| {q} | {t} | {cells[0]} | {cells[1]} | {cells[2]} |")
        w("")
        w(f"That is {len(disagree)} question(s) out of 64. Everything separating the accuracy of")
        w("the first-pass run from the accuracy of the two ensembles lives in these rows.")
    w("")

    w("### The four swaps, in full")
    w("")
    w("Because only four questions separate the three runs, they can simply be shown. This is the")
    w("entire accuracy difference between the first-pass run and the two ensembles.")
    w("")
    if filed_ans and gold:
        w("| qid | type | gold | AC-1 (first pass) | AC-2 (max-conf) | AC-3 (conf-sum) |")
        w("|---|---|---|---|---|---|")
        for q, t, ks, _ in disagree:
            g = gold.get(q, ("", "?"))[1]
            cells = []
            for run in AC_RUNS:
                a, c = filed_ans[run][q]
                a = (a[:60] + "…") if len(a) > 60 else a
                mark = "✓" if ks[run] else "✗"
                cells.append(f"{mark} {a} @{c}")
            w(f"| {q} | {t} | {g} | {cells[0]} | {cells[1]} | {cells[2]} |")
        w("")
        w("Read as a ledger, assembling the passes fixed 0029 and 0060 and broke 0055 and 0065.")
        w("0065 is the costly one: the first pass answered it correctly and filed it at 25, and both")
        w("ensembles replaced that with a wrong answer at 97. A correct answer held modestly became")
        w("a wrong answer held confidently, which is the exact failure the HMR measure is built to")
        w("punish.")
        w("")

    w("### How much of this survives dropping one question")
    w("")
    w("A leave-one-out sweep, run inside each cell: the range each number takes over the seven (or")
    w("ten, or twenty-six) versions of the cell with one question removed. It is not an interval")
    w("estimate and carries no coverage guarantee — it is a direct display of how far one question")
    w("moves the number.")
    w("")
    w("| Question type | n | HMR AC-1 [min, max] | HMR AC-2 | HMR AC-3 | MSnDCG Δ [min, max] |")
    w("|---|---:|---|---|---|---|")
    for t in TYPE_ORDER:
        lo1, hi1 = jack_ac["BITEM-AC-1"][t]
        lo2, hi2 = jack_ac["BITEM-AC-2"][t]
        lo3, hi3 = jack_ac["BITEM-AC-3"][t]
        plo, phi = jack_msndcg[t]
        w(f"| {t} | {scored_counts[t]} | {f(lo1,3)}–{f(hi1,3)} | {f(lo2,3)}–{f(hi2,3)} | "
          f"{f(lo3,3)}–{f(hi3,3)} | {plo:+.3f}–{phi:+.3f} |")
    w("")
    w("The retrieval column is the stable one. Five of the six types keep a clearly positive Δ")
    w("however the cell is trimmed. The exception is comparison, whose whole +0.0223 rests on one")
    w("question — 0021, where PG-2 gains 0.158 — and collapses to −0.0003 without it. The two types")
    w("the overview singles out are the two most robust: post-processing-heavy never falls below")
    w("+0.132 and multi-hop never below +0.072.")
    w("")
    w("The HMR columns are not stable, and the sweep understates their instability rather than")
    w("overstating it. When the dropped question is a cell's only wrong answer, `I⁻` becomes empty")
    w("and the measure is defined to return `R_O = 1`, so the cell jumps to near-perfect for a")
    w("definitional reason rather than a behavioural one. That is what produces the 0.000–0.968")
    w("range on AC-2's simple cell and the flat 1.000–1.000 on AC-3's post-processing-heavy cell —")
    w("the latter looks immovable but is a cell with one mistake in it, and dropping that mistake")
    w("leaves perfection by definition.")
    w("")

    w("## 3. What the cut shows, and what it cannot")
    w("")
    w("**Retrieval.** The first-pass-to-all-passes gain holds in every one of the six types, on")
    w("both measures, with no sign flip anywhere: twelve cells, twelve positive Δ. It is not a gain")
    w("that lives in one kind of question. It is, however, very unevenly sized — roughly +0.02 on")
    w("comparison against +0.18 on post-processing-heavy under MSnDCG — and it is largest exactly")
    w("where the overview says BITEM-PG-2 stands out, on post-processing-heavy and multi-hop. Our")
    w("numbers reproduce the overview's per-type tables to four decimals, so we are looking at the")
    w("same thing they were; what we can add is that within our own pair the same two types are")
    w("where assembling the passes bought the most, and that this survives dropping any single")
    w("question in those cells. We deliberately make no statement about how PG-2 compares with any")
    w("other team's run.")
    w("")
    w("**Answering.** Nothing comparable happens. Accuracy is flat: identical across all three runs")
    w("in four of the six types, and the entire spread across the subtask is four questions of 64,")
    w("two of which the ensembles fixed and two of which they broke. Aggregating the passes did not")
    w("make the system answer more questions correctly in any type.")
    w("")
    w("What aggregation moved was confidence, and per type it moved it in both directions. AC-3's")
    w("confidences pile up at the ceiling — 22 of 26 simple questions and 6 of 7 in three other")
    w("types at exactly 100 — which is rewarded in the two cells with no mistakes in them, where")
    w("`R_O` is 1 by definition and HMR reduces to a prize for high confidence, and punished")
    w("everywhere else. That is the whole of AC-3's per-type profile: 0.9928 on comparison and")
    w("1.0000 on post-processing-heavy, against 0.0000 on simple and 0.0583 on multi-hop.")
    w("")
    w("The single best per-type number we recorded is the most fragile thing in this document.")
    w("AC-3's post-processing-heavy HMR of 1.0000 is built from six correct answers at confidence")
    w("100 and one wrong answer, 0039, filed at confidence exactly 0. The system was asked how many")
    w("days separate two premieres, answered 8604 against a gold 8612, and reported no confidence")
    w("at all. Seven questions, one of them carrying the entire `R_O` term. It should not be quoted")
    w("as a strength of the method.")
    w("")
    w("**What this cannot settle.** Four things, plainly.")
    w("")
    w("1. *Nothing here is a significance claim.* No test is run and none would pass; the")
    w("   organisers' own Randomised Tukey HSD, computed over the full 22- and 25-run fields, has a")
    w("   critical difference near 0.09 on PR MSnDCG and near 0.22 on AC correctness, and every")
    w("   BITEM-vs-BITEM difference in the whole task — let alone inside a seven-question cell — is")
    w("   below it. Per-type cells of seven are for describing, not for deciding.")
    w("2. *The passes are not independent draws.* A fallback can refile an answer an earlier pass")
    w("   already committed, so whatever agreement AC-2 and AC-3 aggregate over is inflated by an")
    w("   unknown amount. A per-type reading of the ensemble inherits that, and nothing in")
    w("   `results/` measures it. It needs the cluster traces.")
    w("3. *We cannot say why a type behaves as it does.* The retrieval gain being largest on")
    w("   post-processing-heavy is consistent with the shrinking corpus forcing later passes toward")
    w("   material the first pass missed, and it is equally consistent with those questions simply")
    w("   having more relevant passages to find. Nothing in the official results distinguishes the")
    w("   two, and the per-pass records that could are not in this material.")
    w("4. *The AC per-type numbers are not comparable to the PR ones.* HMR has no per-question form")
    w("   — it is a function of a whole `I⁺`/`I⁻` split — so a per-type HMR is a small-sample")
    w("   recomputation of the measure, not an average of per-question scores. Accuracy and nugget")
    w("   precision do average per question; HMR, `R_O` and `R_U` do not.")
    w("")
    w("## 4. Checks run while producing this")
    w("")
    for c in checks:
        w(f"- {c}")
    w("")
    w("The overview's own per-type tables are recomputed here from the shipped per-question")
    w("matrices rather than transcribed, and the check above states whether they match.")
    w("")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    for c in checks:
        print(" *", c)
    print("relevant passages per question (L1 or above, direct qrels): PG-1, PG-2, difference")
    for t, (one, two, k) in relevant_per_question(types).items():
        print(f"  {t:28s} {one:6.2f} {two:6.2f} {two - one:+6.2f}   ({k} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
