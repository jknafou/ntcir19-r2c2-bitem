# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The recovered BITEM-AC-4, the llm-judge run whose output was lost before filing.

submission/unofficial/model-as-judge.AC is byte-identical to git object abc2342, the real model-as-judge
output. What was filed under that name on 15 May is a byte-copy of BITEM-AC-3, so the organisers
never scored this run. Nothing here is an official result.

51 of the 64 scored questions carry an answer string identical to a run the organisers did judge,
so those labels transfer unchanged. The remaining 13 have no judgement. This script adjudicates
those 13 against the gold answers itself, records each call so it can be checked, and reports the
metric line as provisional together with the bounds that hold whatever the adjudication.

Read-only. Run:  uv run python analysis/lost_ac4.py
"""
from __future__ import annotations
import csv, pathlib, re, xml.etree.ElementTree as ET, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "lost_ac4.md"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# Our adjudication of the 13 answers the organisers never saw, judged against the gold string.
# True where the answer contains the gold answer, however verbosely; the one False is a regression.
ADJUDICATED = {
    "0014": (True,  "names The Last Jedi; AC-3's shorter 'The Last Jedi' was judged correct"),
    "0019": (True,  "answers 8, the gold count, and lists eight titles"),
    "0020": (False, "answers 8 where gold is 18; AC-3 answered 18 and was judged correct"),
    "0022": (True,  "names Katharine Hepburn, with the counts behind it"),
    "0023": (True,  "names Ingrid Bergman, with the counts behind it"),
    "0027": (True,  "gives liver and Chianti, the two elements of the gold answer"),
    "0042": (True,  "gives Castle in the Sky, plus the alternative title"),
    "0049": (True,  "answers 2, the gold count"),
    "0050": (True,  "names F9, with the box office behind it"),
    "0053": (True,  "answers 10 years old for a gold of 10"),
    "0058": (True,  "the gold quotation, missing only a comma"),
    "0063": (True,  "names the AT-AT, expanded"),
    "0064": (True,  "names Rogue One, with the box office behind it"),
}

def parse(p):
    d, cur, nxt = {}, None, False
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.fullmatch(r"<(\d+)>", line)
        if m:
            cur, nxt = m.group(1), True
            continue
        if cur and nxt and ";" in line:
            a, c = line.rsplit(";", 1)
            d[cur] = (a, float(c))
            nxt = False
    return d

def gold_answers():
    z = zipfile.ZipFile(ROOT / "results" / "r2c2topics-and-goldanswers.xlsx")
    ss = [ "".join(t.text or "" for t in si.iter(NS + "t"))
           for si in ET.fromstring(z.read("xl/sharedStrings.xml")) ]
    out, cur = {}, None
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        c = {}
        for x in r.iter(NS + "c"):
            v = x.find(NS + "v")
            c[re.match(r"[A-Z]+", x.get("r")).group()] = (
                "" if v is None else (ss[int(v.text)] if x.get("t") == "s" else v.text))
        a = c.get("A", "").strip()
        if re.fullmatch(r"\d{4}", a):
            cur = a
        elif cur and c.get("B", "").strip().lower() == "answer":
            out[cur] = c.get("C", "").strip()
    return out

def reward(conf, corr):
    neg = [c / 100 for c, k in zip(conf, corr) if k == 0]
    pos = [c / 100 for c, k in zip(conf, corr) if k == 1]
    ro = 1.0 if not neg else 1 - sum(neg) / len(neg)
    ru = 1.0 if not pos else sum(pos) / len(pos)
    hmr = 0.0 if ro + ru == 0 else 2 * ro * ru / (ro + ru)
    n = len(corr)
    acc, av = sum(corr) / n, sum(conf) / 100 / n
    # accBS = acc * (1 - BS): the paper's composite, on this run's own adjudication rather than
    # score_ac's containment fallback, so it matches the accuracy reported beside it.
    bs = sum((c / 100 - k) ** 2 for c, k in zip(conf, corr)) / n
    so = 1.0 if not neg else 1 - sum(c * c for c in neg) / len(neg)
    su = 1.0 if not pos else 1 - sum((1 - c) ** 2 for c in pos) / len(pos)
    return dict(acc=acc, RO=ro, RU=ru, HMR=hmr, avconf=av, OB=av - acc,
                BS=bs, S_O=so, S_U=su, accBS=acc * (1 - bs))

def auc(s, l):
    p = [a for a, b in zip(s, l) if b == 1]
    q = [a for a, b in zip(s, l) if b == 0]
    return sum((x > y) + 0.5 * (x == y) for x in p for y in q) / (len(p) * len(q))

def main() -> int:
    new = parse(ROOT / "submission/unofficial/model-as-judge.AC")
    filed = {k: parse(ROOT / f"submission/ac-2026-05-15/BITEM-AC-{k[-1]}") for k in ("ac1", "ac2", "ac3")}
    C = {r["qid"]: r for r in csv.DictReader(open(ROOT / "analysis/results_cache.tsv", encoding="utf-8"),
                                             delimiter="\t")
         if re.fullmatch(r"-?[0-9.]+", (r["ac1_correct"] or "").strip())}
    gold = gold_answers()

    qids = sorted(C)
    corr, src = [], {}
    for q in qids:
        ans = new[q][0]
        hit = next((k for k in ("ac1", "ac2", "ac3") if filed[k][q][0] == ans), None)
        if hit:
            corr.append(int(C[q][hit + "_correct"])); src[q] = f"transferred from {hit.upper()}"
        else:
            corr.append(int(ADJUDICATED[q][0])); src[q] = "ours: " + ADJUDICATED[q][1]
    conf = [new[q][1] for q in qids]
    m = reward(conf, corr); m["AUC"] = auc(conf, corr)

    nt = sum(1 for q in qids if src[q].startswith("transferred"))
    base = sum(c for q, c in zip(qids, corr) if src[q].startswith("transferred"))
    lo, hi = base / 64, (base + (64 - nt)) / 64

    L = ["# The recovered BITEM-AC-4 (model as judge), never scored", "",
         "`submission/unofficial/model-as-judge.AC` is byte-identical to git object `abc2342`, the real",
         "model-as-judge output. The file of that name filed on 15 May is a byte-copy of BITEM-AC-3,",
         "so the organisers never evaluated this run. **No number below is an official result.**", "",
         f"It differs from BITEM-AC-3 in {sum(1 for q in new if new[q][0] != filed['ac3'][q][0])} answer",
         f"strings and {sum(1 for q in new if new[q][1] != filed['ac3'][q][1])} confidences.", "",
         f"Of the 64 scored questions, {nt} carry an answer identical to a judged run and inherit its",
         f"label ({base} correct). The other {64 - nt} are adjudicated here against gold.", "",
         "| metric | provisional | ", "|---|---|"]
    for k in ("acc", "RO", "RU", "HMR", "avconf", "BS", "accBS", "OB", "AUC"):
        L.append(f"| {k} | {m[k]:.4f} |")
    L += ["", f"Accuracy is bounded by **{lo:.4f}** (every unjudged answer wrong) and **{hi:.4f}**",
          "(every one right) whatever the adjudication. For comparison the filed runs score 0.9062,",
          "0.9219 and 0.8906.", "", "## The 13 answers with no organiser judgement", "",
          "| q | gold | recovered AC-4 | our call |", "|---|---|---|---|"]
    for q in qids:
        if not src[q].startswith("transferred"):
            ok = "correct" if ADJUDICATED[q][0] else "**WRONG**"
            L.append(f"| {int(q)} | {gold[q]} | {new[q][0][:90]} | {ok} — {ADJUDICATED[q][1]} |")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("  provisional: " + "  ".join(f"{k}={m[k]:.4f}" for k in ("acc","RO","RU","HMR","avconf","BS","accBS","OB","AUC")))
    print(f"  accuracy bounds {lo:.4f} .. {hi:.4f}   ({nt} transferred, {64-nt} adjudicated)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
