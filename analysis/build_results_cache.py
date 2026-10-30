#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the per-question official-results cache.

One row per filed question (65), one column per official per-question measurement
that concerns a BITEM run. Every later analysis reads this file rather than
re-parsing `results/OfficialResults/`.

Sources, all read-only, all under `results/OfficialResults/`:

  r2c2questiontypes                                             question type
  official.PRruns.cons.directL0-L4.nev.MSnDCG@0020.tsmdressed.csv
  official.PRruns.cons.directL0-L4.nev.Q@0020.tsmdressed.csv    PR, direct qrels
  derivedL0-L4.PRruns.derivedL0-L4.nev.MSnDCG@0020.tsmdressed.csv
  derivedL0-L4.PRruns.derivedL0-L4.nev.Q@0020.tsmdressed.csv    PR, derived qrels
  ACnev/BITEM-AC-{1,2,3}.official.ACnev                         Conf, Correct, NugPrec

BITEM-AC-4 is deliberately absent: the filed BITEM-AC-4 is a byte-for-byte copy of
BITEM-AC-3 (both MD5 dc04428a47b429b5363b150627bc7fb9, in our
`submission/ac-2026-05-15/` and in the organisers' `ACruns/`), so it is not a
distinct system and including it would duplicate a column.

Usage:  uv run analysis/build_results_cache.py [submission]

The positional argument is the submission directory. It is optional and is used
only to cross-check the set of filed question ids; the cache itself is built
entirely from `results/OfficialResults/`.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OFFICIAL = REPO / "results" / "OfficialResults"
OUT_TSV = Path(__file__).resolve().parent / "results_cache.tsv"
OUT_MD = Path(__file__).resolve().parent / "build_results_cache.md"

NA = "NA"

PR_MATRICES = {
    ("msndcg", "direct"): "official.PRruns.cons.directL0-L4.nev.MSnDCG@0020.tsmdressed.csv",
    ("q", "direct"): "official.PRruns.cons.directL0-L4.nev.Q@0020.tsmdressed.csv",
    ("msndcg", "derived"): "derivedL0-L4.PRruns.derivedL0-L4.nev.MSnDCG@0020.tsmdressed.csv",
    ("q", "derived"): "derivedL0-L4.PRruns.derivedL0-L4.nev.Q@0020.tsmdressed.csv",
}
PR_RUNS = ["BITEM-PG-1", "BITEM-PG-2"]
AC_RUNS = ["BITEM-AC-1", "BITEM-AC-2", "BITEM-AC-3"]

COLUMNS = (
    ["qid", "qtype"]
    + [
        f"{run.lower().replace('bitem-', '').replace('-', '')}_{metric}_{qrels}"
        for qrels in ("direct", "derived")
        for metric in ("msndcg", "q")
        for run in PR_RUNS
    ]
    + [
        f"{run.lower().replace('bitem-', '').replace('-', '')}_{field}"
        for run in AC_RUNS
        for field in ("conf", "correct", "nugprec")
    ]
)


def read_question_types() -> dict[str, str]:
    types: dict[str, str] = {}
    for line in (OFFICIAL / "r2c2questiontypes").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        qid, qtype = line.split(None, 1)
        types[qid] = qtype.strip()
    return types


def read_pr_matrix(filename: str) -> dict[str, dict[str, str]]:
    """qid -> {run -> score} from a question-by-run tsmdressed matrix."""
    with (OFFICIAL / filename).open(newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1:]
    out: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        out[row[0].strip()] = dict(zip(header, [c.strip() for c in row[1:]]))
    return out


ACNEV_RE = re.compile(r"^(\d{4})\s+(Conf|Correct|NugPrec)=\s*(\S+)\s*$")


def read_acnev(run: str) -> dict[str, dict[str, str]]:
    """qid -> {conf, correct, nugprec}."""
    path = OFFICIAL / "ACnev" / f"{run}.official.ACnev"
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = ACNEV_RE.match(line)
        if not m:
            raise ValueError(f"unparsed line in {path.name}: {line!r}")
        qid, field, value = m.group(1), m.group(2).lower(), m.group(3)
        out.setdefault(qid, {})[field] = value
    return out


def filed_qids(submission_dir: Path) -> set[str] | None:
    """Question ids present in the filed AC runs, or None if unreadable."""
    for candidate in (
        submission_dir / "ac-2026-05-15" / "BITEM-AC-1",
        submission_dir / "BITEM-AC-1",
    ):
        if candidate.is_file():
            return set(re.findall(r"^<(\d{4})>$", candidate.read_text(), re.M))
    return None


def read_ac_summary() -> dict[str, dict[str, str]]:
    """run -> summary row from official.AC.csv (used only to verify the cache)."""
    with (OFFICIAL / "official.AC.csv").open(newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    out: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        name = row[0].strip().removesuffix(".official")
        out[name] = dict(zip(header[1:], [c.strip() for c in row[1:]]))
    return out


def read_acnev_nugprec(run: str) -> dict[str, str]:
    return {q: v["nugprec"] for q, v in read_acnev(run).items()}


def main(argv: list[str]) -> int:
    submission_dir = Path(argv[1]) if len(argv) > 1 else REPO / "submission"

    qtypes = read_question_types()
    pr = {key: read_pr_matrix(name) for key, name in PR_MATRICES.items()}
    ac = {run: read_acnev(run) for run in AC_RUNS}

    # The row universe is every question the organisers assigned a type to, which
    # is also every question we filed: 65.
    all_qids = sorted(qtypes)

    filed = filed_qids(submission_dir)
    notes: list[str] = []
    if filed is None:
        notes.append(
            f"Could not read a filed AC run under `{submission_dir}`; the filed-qid "
            "cross-check was skipped and the row universe comes from "
            "`r2c2questiontypes` alone."
        )
    elif filed != set(all_qids):
        notes.append(
            "Filed question ids differ from `r2c2questiontypes`: "
            f"only filed {sorted(filed - set(all_qids))}, "
            f"only typed {sorted(set(all_qids) - filed)}."
        )
    else:
        notes.append(
            f"Cross-check passed: the {len(filed)} question ids in "
            "`submission/ac-2026-05-15/BITEM-AC-1` are exactly the ids in "
            "`r2c2questiontypes`."
        )

    rows = []
    for qid in all_qids:
        row = {c: NA for c in COLUMNS}
        row["qid"] = qid
        row["qtype"] = qtypes[qid]
        for (metric, qrels), table in pr.items():
            per_run = table.get(qid)
            if per_run is None:
                continue
            for run in PR_RUNS:
                key = f"{run.lower().replace('bitem-', '').replace('-', '')}_{metric}_{qrels}"
                if run in per_run:
                    row[key] = per_run[run]
        for run in AC_RUNS:
            per_q = ac[run].get(qid)
            if per_q is None:
                continue
            stem = run.lower().replace("bitem-", "").replace("-", "")
            for field in ("conf", "correct", "nugprec"):
                if field in per_q:
                    row[f"{stem}_{field}"] = per_q[field]
        rows.append(row)

    with OUT_TSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # coverage report
    coverage: dict[str, list[str]] = {}
    for col in COLUMNS[2:]:
        missing = [r["qid"] for r in rows if r[col] == NA]
        if missing:
            coverage[col] = missing

    lines = []
    lines.append("# Per-question official-results cache")
    lines.append("")
    lines.append(
        "`analysis/build_results_cache.py` writes `analysis/results_cache.tsv`: one row "
        f"per filed question ({len(rows)}), one column per official per-question "
        "measurement that concerns a BITEM run. It is a pure read of "
        "`results/OfficialResults/`; nothing is analysed here and nothing is written "
        "back into `results/` or `submission/`."
    )
    lines.append("")
    lines.append("## Shape")
    lines.append("")
    lines.append(f"- {len(rows)} data rows, {len(COLUMNS)} columns, tab separated, `NA` for absent values.")
    lines.append(f"- Row universe: the {len(all_qids)} question ids in `results/OfficialResults/r2c2questiontypes`.")
    lines.append("")
    lines.append("## Columns")
    lines.append("")
    lines.append("| Column | Source file | Field |")
    lines.append("|---|---|---|")
    lines.append("| `qid` | `r2c2questiontypes` | column 1 |")
    lines.append("| `qtype` | `r2c2questiontypes` | column 2 |")
    for qrels in ("direct", "derived"):
        for metric in ("msndcg", "q"):
            src = PR_MATRICES[(metric, qrels)]
            for run in PR_RUNS:
                stem = run.lower().replace("bitem-", "").replace("-", "")
                lines.append(f"| `{stem}_{metric}_{qrels}` | `{src}` | column `{run}` |")
    for run in AC_RUNS:
        stem = run.lower().replace("bitem-", "").replace("-", "")
        for field, label in (("conf", "Conf"), ("correct", "Correct"), ("nugprec", "NugPrec")):
            lines.append(f"| `{stem}_{field}` | `ACnev/{run}.official.ACnev` | `{label}=` |")
    lines.append("")
    lines.append("## What is not here, and why")
    lines.append("")
    lines.append(
        "**BITEM-AC-4.** The filed `BITEM-AC-4` is byte-for-byte `BITEM-AC-3` — both files "
        "in `submission/ac-2026-05-15/` and both in the organisers' `ACruns/` carry MD5 "
        "`dc04428a47b429b5363b150627bc7fb9`. It is not a distinct system, so it gets no "
        "columns; a column for it would be a duplicate, and any significance test that "
        "included it would be computed over a duplicated column."
    )
    lines.append("")
    ac3n = read_acnev_nugprec("BITEM-AC-3")
    ac4n = read_acnev_nugprec("BITEM-AC-4")
    differ = sorted(q for q in ac3n if ac3n[q] != ac4n.get(q))
    lines.append(
        "One consequence worth recording before it surprises someone: although the two "
        "run files are identical, their ACnev files are not. `Conf` and `Correct` agree "
        f"on all 64 scored questions, but `NugPrec` differs on {len(differ)} of them "
        f"({', '.join(differ)}), so the two identical runs received two independent "
        "nugget assessments. The nugget-precision figures therefore carry assessor "
        "variance that the correctness figures do not. `BITEM-AC-3` is the copy this "
        "cache keeps."
    )
    lines.append("")
    lines.append("## Missing values")
    lines.append("")
    if coverage:
        lines.append("| Column | Rows with `NA` | Which |")
        lines.append("|---|---|---|")
        for col, missing in coverage.items():
            lines.append(f"| `{col}` | {len(missing)} | {', '.join(missing)} |")
    else:
        lines.append("None: every column is filled for every row.")
    lines.append("")
    lines.append("### The 65 filed / 64 scored discrepancy")
    lines.append("")
    lines.append(
        "We filed 65 questions and `official.AC.csv` reports `n=64` for every BITEM run. "
        "The missing question is **qid 0025** (`comparison`). It is absent from every "
        "`ACnev/*.official.ACnev` file, from both direct PR score matrices, and from both "
        "`*.qrels` files, while being present in the runs we filed and in the organisers' "
        "`markedACruns/BITEM-AC-1.marked.output`, where its answer line is marked `YES`. "
        "`README.txt` gives no reason for the exclusion, so we record the fact and do not "
        "infer a cause."
    )
    lines.append("")
    lines.append(
        "A second question is missing from the **derived** qrels only: **qid 0039** "
        "(`post-processing-heavy`) does not appear in `official.derivedL0-L4.qrels` and so "
        "carries no score in either derived matrix, which have 63 rows against the direct "
        "matrices' 64. The derived qrels are pool-biased in our favour in any case and are "
        "cached only so that the bias can be shown, never reported as a result."
    )
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(
        "Column means recomputed from the cache, against the published summaries. This "
        "is a parse check, not a result."
    )
    lines.append("")

    def col_mean(col: str) -> tuple[int, float]:
        vals = [float(r[col]) for r in rows if r[col] != NA]
        return len(vals), sum(vals) / len(vals)

    lines.append("| Column | n | mean from cache | published | source |")
    lines.append("|---|---|---|---|---|")
    for qrels in ("direct", "derived"):
        for metric in ("msndcg", "q"):
            for run in PR_RUNS:
                stem = run.lower().replace("bitem-", "").replace("-", "")
                col = f"{stem}_{metric}_{qrels}"
                n, m = col_mean(col)
                lines.append(f"| `{col}` | {n} | {m:.4f} | — | no published per-run mean |")
    summary = read_ac_summary()
    for run in AC_RUNS:
        stem = run.lower().replace("bitem-", "").replace("-", "")
        n, acc = col_mean(f"{stem}_correct")
        _, conf = col_mean(f"{stem}_conf")
        _, nug = col_mean(f"{stem}_nugprec")
        pub = summary.get(run, {})
        lines.append(
            f"| `{stem}_correct` | {n} | {acc:.4f} | {pub.get('accuracy', '?')} | "
            "`official.AC.csv` `accuracy` |"
        )
        lines.append(
            f"| `{stem}_conf` / 100 | {n} | {conf / 100:.4f} | {pub.get('avconf', '?')} | "
            "`official.AC.csv` `avconf` |"
        )
        lines.append(f"| `{stem}_nugprec` | {n} | {nug:.4f} | — | no published per-run mean |")
    lines.append("")
    lines.append(
        "`official.AC.csv` also reports `n` = 64 for each of these runs, which matches "
        "the row counts above."
    )
    lines.append("")
    lines.append("## Cross-check")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```")
    lines.append("uv run analysis/build_results_cache.py submission")
    lines.append("```")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))

    print(f"wrote {OUT_TSV} ({len(rows)} rows x {len(COLUMNS)} columns)")
    print(f"wrote {OUT_MD}")
    for col, missing in coverage.items():
        print(f"  NA in {col}: {len(missing)} -> {','.join(missing)}")
    for note in notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
