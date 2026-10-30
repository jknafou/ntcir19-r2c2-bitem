# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Score any answer-construction submission file, filed or counterfactual.

The shared contract for every counterfactual strategy in analysis/: a strategy writes an .AC file,
this module scores it. One judge, one arithmetic, so two strategies are always comparable.

    from score_ac import re, score, judge_report
    score("analysis/counterfactual_runs/mine.AC")   -> dict of metrics

The metrics are the organisers' (design paper eq. 1-4, overview eq. 7): R_O = 1 - the mean
confidence over the questions the run got wrong, R_U = the mean confidence over the ones it got
right, HMR their harmonic mean, an empty set scoring 1, OB = avconf - accuracy.

accHMR = accuracy * HMR is what the paper reports. It is preferred to accBS because the harmonic
mean weights the two rewards equally and so cannot be satisfied by a constant: filing one value v
everywhere gives R_O = 1-v and R_U = v, hence HMR = 2v(1-v) <= 0.5 and accHMR <= a/2, half the
ceiling. accBS and the linear wHMRacc are both maximised by a constant when no signal exists.
accHMR reaches its ceiling, a, only for a confidence that is low on the errors and high on the
correct answers, and it is zero for a run that answers nothing, so abstaining, which maximises
HMR, earns nothing under it.

accBS is computed as well, for comparison between the candidate metrics:

    accBS = accuracy * (1 - BS) = accuracy * [ (1-accuracy)*S_O + accuracy*S_U ]
            S_O = 1 - mean over wrong answers of p^2         (cf. R_O = 1 - mean p)
            S_U = 1 - mean over right answers of (1-p)^2     (cf. R_U = 1 - mean (1-p))

i.e. the organisers' two rewards with the confidence deviation squared, weighted by the number of
questions each is averaged over, and scaled by accuracy. The square matters: the linear version
below is improved by rounding an already-calibrated run's confidences to 0 and 100 (22 of 25
official runs), while accBS is damaged by it (22 of 25). accBS has 0 dominance violations over the
600 ordered pairs of official runs against HMR's 57. It does not close the withholding channel
(omitting a question removes it from Q'), which no measure over Q' can.

wHMR and wHMRacc are not reported in the paper. They are computed so that earlier analysis
reports reproduce:

    wHMR    = (1-a)*R_O + a*R_U, the organisers' two rewards weighted by the number of questions
              each is averaged over rather than equally. Identically the mean per-question reward,
              p where the answer is right and 1-p where it is wrong, so every question counts once
              and the asymmetry HMR creates at high accuracy disappears.
    wHMRacc = accuracy * wHMR. Zero for a run that answers nothing correctly however well it is
              calibrated, one only for all-correct-and-confident.

Both are linear in each filed confidence and therefore not proper scoring rules: a run improves by
binarising its own confidences at the best cut point, adding no information. HMR has the same
limitation.

The binary Brier score in its CORP decomposition is also computed, which is what this paper
recommends reporting instead (Brier 1950; Dimitriadis, Gneiting and Jordan, PNAS 2021):

    BS = MCB - DSC + UNC

    BS   mean over questions of (p - y)^2, p the filed confidence, y 1 if the answer is correct.
         Strictly proper: the report that minimises it is the one the system actually believes.
    UNC  a(1-a), the score of filing the base rate everywhere -- the no-estimator landmark.
    MCB  miscalibration, BS minus the score of the isotonically recalibrated forecasts. How much
         is lost purely to the scale.
    DSC  discrimination, UNC minus that same recalibrated score. How much the ordering is worth.
    BSS  1 - BS/UNC = (DSC - MCB)/UNC. Negative means worse than filing one constant.

The recalibration is the pool-adjacent-violators algorithm, which is what makes MCB and DSC exact
rather than binning-dependent.

Judging. The organisers labelled only the answers that were filed. So:
  * an answer string they judged always keeps their label, and it is checked first;
  * anything else is judged here against the gold string by containment after case, punctuation
    and whitespace normalisation, because the organisers accepted verbose answers that contained
    the gold answer.
`judge_report()` calibrates the fallback against all 192 organiser judgements of the three filed
runs and reports its error and false-positive rates. Any accuracy over answers we never filed is
our judgement, not the organisers', and must be reported as such.

Read-only. Run:  /usr/bin/python3 analysis/score_ac.py [file.AC ...]
With no argument it self-tests against the three filed runs.
"""
from __future__ import annotations
import csv, pathlib, re, sys, zipfile
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def norm(s: str) -> str:
    s = (s or "").casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def _gold():
    z = zipfile.ZipFile(ROOT / "results" / "r2c2topics-and-goldanswers.xlsx")
    ss = ["".join(t.text or "" for t in si.iter(NS + "t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
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

def read_ac(path):
    """{qid: (answer, confidence)} from an .AC file."""
    d, cur, nxt = {}, None, False
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.fullmatch(r"<(\d+)>", line)
        if m:
            cur, nxt = m.group(1), True
            continue
        if cur and nxt and ";" in line:
            a, c = line.rsplit(";", 1)
            try:
                d[cur] = (a, float(c))
            except ValueError:
                pass
            nxt = False
    return d

GOLD = _gold()
_CACHE = [r for r in csv.DictReader((ROOT / "analysis/results_cache.tsv").open(encoding="utf-8"),
                                    delimiter="\t")
          if re.fullmatch(r"-?[0-9.]+", (r.get("ac1_correct") or "").strip())]
SCORED = {r["qid"] for r in _CACHE}

def _official_labels():
    """(qid, normalised answer) -> organiser label, from every judged run in the task.

    The organisers judged 25 answer runs, not only ours, and an answer string another team filed
    carries their verdict just as ours does. Pairing `ACruns/` with `ACnev/` lifts the table from
    the 80 distinct pairs our own three runs produce to 484, which is what lets a rebuilt run be
    scored against the organisers on answers we never filed ourselves.

    A run whose ACnev says `Correct= -1` did not answer that question, so it contributes no label.
    Exactly one pair is judged inconsistently across runs: qid 0051, "murph", labelled both 0 and
    1, which is the LLM assessor disagreeing with itself. The optimistic label is taken there,
    since one pair cannot move any figure we quote.
    """
    import collections
    runs, nev = ROOT / "results/OfficialResults/ACruns", ROOT / "results/OfficialResults/ACnev"
    seen = collections.defaultdict(set)
    for rf in sorted(runs.iterdir()):
        stem = re.sub(r"\.txt$", "", rf.name)          # some runs carry a .txt suffix
        nf = nev / f"{stem}.official.ACnev"
        if not nf.exists():
            continue
        correct = {m.group(1): int(m.group(2)) for m in
                   re.finditer(r"(\d{4})\s+Correct=\s*(-?\d+)",
                               nf.read_text(encoding="utf-8", errors="replace"))}
        for q, (ans, _conf) in read_ac(rf).items():
            if correct.get(q, -1) >= 0:
                seen[(q, norm(ans))].add(correct[q])
    return {k: max(v) for k, v in seen.items()}

OFFICIAL = _official_labels()

#: Answers we have read and ruled on by hand, keyed by (qid, normalised answer) so a verdict
#: attaches to one exact string and never leaks to a different answer for the same question.
#: They exist because the containment fallback below is position-sensitive: it credits a long
#: answer that happens to contain the gold contiguously and rejects an equally correct one whose
#: words are separated, e.g. "liver with fava beans and a nice Chianti" against a gold of "Liver
#: with Chianti", which the organisers themselves accepted from another team in almost the same
#: words. Each row records why. Curated rows override the fallback but never an organiser label.
def _manual():
    f = ROOT / "analysis" / "manual_judgements.tsv"
    out = {}
    if f.exists():
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out[(r["qid"], r["norm_answer"])] = int(r["correct"])
    return out

MANUAL = _manual()


def judge(qid, answer):
    """(correct, source) -- organiser label, else a curated verdict, else our containment rule."""
    k = (qid, norm(answer))
    if k in OFFICIAL:
        return OFFICIAL[k], "organiser"
    if k in MANUAL:
        return MANUAL[k], "curated"
    g = norm(GOLD.get(qid, ""))
    if not g:
        return 0, "no-gold"
    return int(g in norm(answer)), "ours"

def _pav(p, y):
    """Isotonic regression of y on p, pool-adjacent-violators.

    Ties in p are pooled first: the recalibration is a function of the forecast, so two
    questions filed at the same confidence have to receive the same fitted value. Without that
    pooling MCB and DSC come out too small on tie-heavy runs, and every run here is tie-heavy.
    """
    groups = {}
    for pi, yi in zip(p, y):
        g = groups.setdefault(pi, [0, 0.0])
        g[0] += 1
        g[1] += yi
    blocks = [[w, tot / w] for pi, (w, tot) in sorted(groups.items())]
    keys = [pi for pi, _ in sorted(groups.items())]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] <= blocks[i + 1][1] + 1e-15:
            i += 1
            continue
        w = blocks[i][0] + blocks[i + 1][0]
        m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w
        blocks[i:i + 2] = [[w, m]]
        keys[i:i + 2] = [keys[i]]
        i = max(i - 1, 0)
    fitted, k = {}, 0
    for (w, m), key in zip(blocks, keys):
        run = []
        seen = 0
        for pi in sorted(groups):
            if pi in fitted:
                continue
            run.append(pi)
            seen += groups[pi][0]
            if seen >= w:
                break
        for pi in run:
            fitted[pi] = m
    return [fitted[pi] for pi in p]

def corp(p, y):
    """BS, MCB, DSC, UNC, BSS. BS = MCB - DSC + UNC exactly."""
    n = len(p)
    a = sum(y) / n
    bs = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    rec = _pav(p, y)
    bs_rec = sum((r - yi) ** 2 for r, yi in zip(rec, y)) / n
    unc = a * (1 - a)
    mcb, dsc = bs - bs_rec, unc - bs_rec
    return {"BS": bs, "MCB": mcb, "DSC": dsc, "UNC": unc,
            "BSS": (dsc - mcb) / unc if unc > 0 else float("nan")}

def score(path, k=None):
    sub = read_ac(path)
    qs = sorted(SCORED & set(sub))
    corr, conf, src = [], [], []
    for q in qs:
        a, c = sub[q]
        ok, s = judge(q, a)
        corr.append(ok); conf.append(c / 100.0); src.append(s)
    n = len(qs)
    neg = [c for c, k2 in zip(conf, corr) if k2 == 0]
    pos = [c for c, k2 in zip(conf, corr) if k2 == 1]
    ro = 1.0 if not neg else 1 - sum(neg) / len(neg)
    ru = 1.0 if not pos else sum(pos) / len(pos)
    hmr = 0.0 if ro + ru == 0 else 2 * ro * ru / (ro + ru)
    acc, av = sum(corr) / n, sum(conf) / n
    out = {"n": n, "accuracy": acc, "R_O": ro, "R_U": ru, "HMR": hmr,
           "avconf": av, "OB": av - acc, "judged_by_us": src.count("ours"),
           "errors": [q for q, c in zip(qs, corr) if not c]}
    # accBS: the organisers' two rewards with the confidence deviation squared, weighted by the
    # number of questions each is averaged over, scaled by accuracy.
    #     S_O = 1 - mean_{wrong} p^2        (cf. R_O = 1 - mean_{wrong} p)
    #     S_U = 1 - mean_{right} (1-p)^2    (cf. R_U = 1 - mean_{right} (1-p))
    #     accBS = acc * ((1-acc)*S_O + acc*S_U) = acc * (1 - BS), exactly.
    # The square is what makes it proper: unlike the linear wHMR (= 1 - MAE) it is not improved
    # by rounding an already-calibrated run's confidences to 0 and 100.
    so = 1.0 if not neg else 1 - sum(c * c for c in neg) / len(neg)
    su = 1.0 if not pos else 1 - sum((1 - c) ** 2 for c in pos) / len(pos)
    out["S_O"], out["S_U"] = so, su
    out.update(corp(conf, corr))
    out["accBS"] = acc * (1 - out["BS"])
    # accHMR = accuracy x HMR, the paper's composite.
    # The harmonic mean forces both rewards up, which a constant confidence cannot do: filing v
    # everywhere gives R_O = 1-v and R_U = v, so HMR = 2v(1-v) <= 0.5 and accHMR <= a/2. The
    # ceiling, a, needs a confidence that is low on the errors and high on the rest.
    # Multiplying by accuracy also zeroes a run that abstains, which HMR alone rewards most.
    out["accHMR"] = acc * hmr
    # wHMR and wHMRacc are not reported in the paper; they are computed so that earlier
    # analysis reports reproduce.
    whmr = (1 - acc) * ro + acc * ru
    out["wHMR"], out["wHMRacc"] = whmr, acc * whmr
    return out

def judge_report():
    """Calibrate the containment judge against all organiser judgements of the filed runs."""
    agree = fp = fn = tot = 0
    for k, run in (("ac1", "BITEM-AC-1"), ("ac2", "BITEM-AC-2"), ("ac3", "BITEM-AC-3")):
        filed = read_ac(ROOT / f"submission/ac-2026-05-15/{run}")
        for r in _CACHE:
            q = r["qid"]
            if q not in filed:
                continue
            truth = int(r[k + "_correct"])
            g = norm(GOLD.get(q, ""))
            mine = int(bool(g) and g in norm(filed[q][0]))
            tot += 1
            agree += mine == truth
            fp += mine == 1 and truth == 0
            fn += mine == 0 and truth == 1
    return {"judgements": tot, "agree": agree, "false_positive": fp, "false_negative": fn,
            "accuracy": agree / tot}

if __name__ == "__main__":
    jr = judge_report()
    print(f"containment judge vs {jr['judgements']} organiser judgements: "
          f"agree {jr['agree']} ({jr['accuracy']*100:.1f}%), "
          f"false positives {jr['false_positive']}, false negatives {jr['false_negative']}")
    args = sys.argv[1:] or [f"submission/ac-2026-05-15/BITEM-AC-{i}" for i in (1, 2, 3)]
    want = {"BITEM-AC-1": 0.2719, "BITEM-AC-2": 0.4915, "BITEM-AC-3": 0.3956}
    print(f"\n{'file':<30}{'n':>4}{'acc':>8}{'HMR':>8}{'BS':>8}{'accBS':>8}{'accHMR':>9}{'ours':>6}")
    for p in args:
        m = score(p)
        nm = pathlib.Path(p).name
        flag = ""
        if nm in want:
            flag = "  OK" if abs(m["HMR"] - want[nm]) < 5e-4 else f"  MISMATCH want {want[nm]}"
        print(f"{nm:<30}{m['n']:>4}{m['accuracy']:>8.4f}{m['HMR']:>8.4f}{m['BS']:>8.4f}"
              f"{m['accBS']:>8.4f}{m['accHMR']:>9.4f}{m['judged_by_us']:>6}{flag}")
