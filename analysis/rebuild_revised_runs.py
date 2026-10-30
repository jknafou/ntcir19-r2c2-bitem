# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Rebuild the two revised runs from the committed caches, so they can be regenerated here.

analysis/make_revised_runs.py is the canonical generator and reads the cluster trace tree, which
is not in the repository. This script reconstructs the same pass records from the committed
caches -- ac_harness_candidates.tsv (the submit_answer calls), ac_harness_passes.tsv (the filed
answer and the ladder rung), ac_harness_nuggets.tsv (the entailment) and reasoning_behaviour.tsv
(the longest reasoning turn) -- and calls the same rule functions, so the rules have one
implementation and this file only supplies their input.

Run it against a reference pair of .AC files to prove the reconstruction is faithful before
trusting anything it emits:

    /usr/bin/python3 analysis/rebuild_revised_runs.py --check <DIR>
    /usr/bin/python3 analysis/rebuild_revised_runs.py --emit

--emit rewrites the answer line of each question whose selection has moved, leaving every nugget
line in place, and scores the result.
"""
import argparse, collections, csv, importlib.util, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R = _load("make_revised_runs")
from score_ac import read_ac, score                       # noqa: E402

RUNS = {"revised-longest-deliberation": "single", "revised-independent-agreement": "four"}


def rows(fname):
    with (ROOT / fname).open(encoding="utf-8") as fh:
        yield from csv.DictReader(fh, delimiter="\t")


def build_passes():
    """The generator's pass records, rebuilt from the committed caches.

    cands  <- candidate rows of kind 'submit', in seq order: these are exactly the submit_answer
              tool calls the generator collects from the messages.
    nuggets<- the nugget cache. It carries no is_synthetic column, so every nugget is treated as
              genuine.
    """
    qtext, cands, nugs = {}, collections.defaultdict(list), collections.defaultdict(list)
    for r in rows("error_signals.tsv"):
        qtext[r["qid"]] = r["question_text"]
    for r in sorted(rows("ac_harness_candidates.tsv"), key=lambda r: (r["qid"], int(r["pss"]), int(r["seq"]))):
        if r["kind"] == "submit" and r["text"].strip():
            cands[(r["qid"], int(r["pss"]))].append({"answer": r["text"].strip(),
                                                     "norm": R.norm(r["text"].strip()),
                                                     "committed": r["committed"] == "1"})
    for r in rows("ac_harness_nuggets.tsv"):
        nugs[(r["qid"], int(r["pss"]))].append({"text": r["text"], "status": r["status"],
                                                "nli": float(r["nli"] or 0.0), "synthetic": False})
    out = collections.defaultdict(list)
    for r in rows("ac_harness_passes.tsv"):
        q, i = r["qid"], int(r["pss"])
        fin = (r["final_answer"] or "").strip()
        out[q].append({"qid": q, "idx": i, "answer": fin, "norm": R.norm(fin),
                       "conf": float(r["conf"] or 0), "ladder_path": r["ladder_path"] or "",
                       "cot_max": 0, "cands": cands[(q, i)],
                       "question": qtext.get(q, ""), "nuggets": nugs[(q, i)]})
    for q in out:
        out[q].sort(key=lambda p: p["idx"])
    return out


def passes():
    P = build_passes()
    cot = {}
    with (ROOT / "reasoning_behaviour.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            cot[(r["qid"].zfill(4), int(r["idx"]))] = int(r["cot_max_chars"] or 0)
    for q, ps in P.items():
        for p in ps:
            p["cot_max"] = cot.get((q, p["idx"]), 0)
    return P


def selections(P):
    ref = sorted(p["cot_max"] for ps in P.values() for p in ps if p["idx"] == 0)
    out = {}
    for q, ps in P.items():
        a1, c1, _ = R.longest_deliberation(ps, ref)
        a2, c2, _ = R.independent_agreement(ps)
        out[q] = {"single": (a1, c1), "four": (a2, c2)}
    return out


def rewrite(src, dst, sel):
    """Copy an .AC file, replacing only the answer line of each question whose selection moved."""
    keep, cur, nxt = [], None, False
    for line in src.read_text(encoding="utf-8").split("\n"):
        m = re.fullmatch(r"<(\d+)>", line)
        if m:
            cur, nxt = m.group(1), True; keep.append(line); continue
        if cur and nxt and ";" in line:
            nxt = False
            if cur in sel:
                keep.append(f"{sel[cur][0]};{sel[cur][1]:g}"); continue
        keep.append(line)
    dst.write_text("\n".join(keep), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="DIR")
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()
    sel = selections(passes())

    if a.check:
        ok = True
        for name, key in RUNS.items():
            ref = read_ac(pathlib.Path(a.check) / f"{name}.AC")
            bad = [q for q in sorted(ref)
                   if R.norm(sel[q][key][0]) != R.norm(ref[q][0])
                   or abs(sel[q][key][1] - ref[q][1]) > 1e-6]
            ok &= not bad
            print(f"{name}: {len(ref)-len(bad)}/{len(ref)} reproduced"
                  + (f"  MISMATCH {bad}" if bad else ""))
        return 0 if ok else 1

    if a.emit:
        L = ["# The revised runs, rebuilt", "",
             "Regenerated from the committed caches with the same rule functions the trace-only "
             "generator uses.", "",
             "| run | accuracy | $R_O$ | $R_U$ | HMR | avconf | accHMR |", "|---|---|---|---|---|---|---|"]
        for name, key in RUNS.items():
            src = ROOT.parent / "submission" / "unofficial" / f"{name}.AC"
            dst = ROOT / "counterfactual_runs" / f"{name}.AC"
            dst.parent.mkdir(exist_ok=True)
            rewrite(src, dst, {q: sel[q][key] for q in sel})
            s = score(dst)
            L.append(f"| {name} | {s['accuracy']:.4f} | {s['R_O']:.4f} | {s['R_U']:.4f} | "
                     f"**{s['HMR']:.4f}** | {s['avconf']:.4f} | **{s['accHMR']:.4f}** |")
            L.append(f"| | errors: {', '.join(s['errors'])} | | | | | |")
        (ROOT / "rebuild_revised_runs.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
