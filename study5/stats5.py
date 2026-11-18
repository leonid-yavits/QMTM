"""Study 5 statistics (PROTOCOL.md). Test accuracy. Welch per cell, Holm within each contrast.
E1 (seeds 101-105):
  study 1: TMR_ext, QM_ext against TM_ext; QMR_ext against TMR_ext; best QM-family arm (validation choice among
           QM_ext, QMR_ext, QP, QPR) against the best of {TM_ext, TMR_ext} (validation choice); TMR_ext against TMR,
           QM_ext against QM.
  study 2 C: rule arm (TMR_ext below 3 % density, QM_ext otherwise), TMR_ext and QM_ext against TM_ext; sign test.
E2 (seeds 101-103): no-L arm against the budgeted E1 arm and against TM_ext; included literals per clause.
Cells with incomplete runs are skipped and counted. Writes logs/stats5.json."""
import json, collections, re, numpy as np
from scipy import stats
S1, S2, S4, S5 = (f"/home/claude/qmtm/{d}" for d in ("study1", "study2", "study4", "study5"))
LOGS = [f"{S1}/logs/confirm.jsonl", f"{S2}/logs/confirmC.jsonl", f"{S4}/logs/confirmD1.jsonl",
        f"{S5}/logs/confirmE1.jsonl", f"{S5}/logs/E2.jsonl"]
A, CMP = collections.defaultdict(dict), collections.defaultdict(dict)
for p in LOGS:
    try:
        for l in open(p):
            d = json.loads(l)
            if d.get("core", "fast") != "fast":
                continue
            k = (d["ds"], int(d["C"]), d["cfg"], float(d["s"])); A[k][d["seed"]] = d["test"]; CMP[k][d["seed"]] = d["comp"]
    except FileNotFoundError:
        pass


def vec(ds, C, cfg, s, seeds):
    r = A.get((ds, int(C), cfg, float(s)), {})
    return np.array([r[sd] for sd in seeds if sd in r])


def inc(ds, C, cfg, s, seeds):
    r = CMP.get((ds, int(C), cfg, float(s)), {})
    v = [r[sd]["inc_per_clause"] for sd in seeds if sd in r]
    return float(np.mean(v)) if v else None


def welch(x, y):
    return 1.0 if np.allclose(x, y) else float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def holm(rows):
    m = len(rows); run = 0.0
    for r_, i in enumerate(sorted(range(m), key=lambda i: rows[i]["p"])):
        run = max(run, min(1.0, (m - r_) * rows[i]["p"])); rows[i]["p_holm"] = run


def summarize(rows, skipped):
    holm(rows)
    npos = sum(r["diff"] > 0 for r in rows); nneg = sum(r["diff"] < 0 for r in rows)
    d = [r["diff"] for r in rows]
    return dict(n=len(rows), skipped=skipped, wins=int(sum(r["diff"] > 0 and r["p_holm"] < 0.05 for r in rows)),
                losses=int(sum(r["diff"] < 0 and r["p_holm"] < 0.05 for r in rows)), positive=int(npos), negative=int(nneg),
                mean_diff=float(np.mean(d)) if d else None, median_diff=float(np.median(d)) if d else None,
                within1=int(sum(abs(x) <= 1.0 for x in d)),
                sign_p=float(stats.binomtest(npos, npos + nneg, 0.5, alternative="greater").pvalue) if npos + nneg else None,
                rows=rows)


def row(cell, x, y, **kw):
    return dict(cell=cell, ma=100 * x.mean(), sa=100 * x.std(ddof=1), mb=100 * y.mean(), sb=100 * y.std(ddof=1),
                diff=100 * (x.mean() - y.mean()), p=welch(x, y), **kw)


SEL5 = json.load(open(f"{S5}/logs/selection5.json"))
SEL4 = json.load(open(f"{S4}/logs/selection4.json"))
SEL1 = json.load(open(f"{S1}/logs/selection.json"))
SELC = json.load(open(f"{S2}/logs/selection2.json"))["C"]
OUT = {}
SD5, SD3 = range(101, 106), range(101, 104)


def contrast(name, cells, get_a, get_b, seeds, extra=lambda k: {}):
    rows, sk = [], 0
    for k in cells:
        a, b = get_a(k), get_b(k)
        if a is None or b is None:
            sk += 1; continue
        ds, C = k.split("|")
        x, y = vec(ds, C, a[0], a[1], seeds), vec(ds, C, b[0], b[1], seeds)
        if len(x) < len(seeds) or len(y) < len(seeds):
            sk += 1; continue
        rows.append(row(k, x, y, a=list(a[:2]), b=list(b[:2]), **extra(k)))
    OUT[name] = summarize(rows, sk)


def e5(tag, k, arm):
    x = SEL5[tag].get(k, {}).get(arm)
    return (x["cfg"], x["s"], x["val"]) if x else None


def tmext(tag4, k):
    x = SEL4[tag4].get(k)
    return (x["cfg"], x["s"], x["val"]) if x else None


def orig(sel, k, arm):
    return (sel[k][arm]["cfg"], float(sel[k][arm]["s"]), sel[k][arm]["val"])


def best_qm(k):
    c = [e5("s1", k, "QM"), e5("s1", k, "QMR"), orig(SEL1, k, "QP"), orig(SEL1, k, "QPR")]
    if c[0] is None or c[1] is None:
        return None
    return max(c, key=lambda t: t[2])        # first listed wins ties (QM_ext, QMR_ext, QP, QPR)


def best_nonqm(k):
    a, b = tmext("D1_s1", k), e5("s1", k, "TMR")
    if a is None or b is None:
        return None
    return b if b[2] > a[2] else a


C1, CC = list(SEL1), list(SELC)
dens = lambda k: {"density": SELC[k]["density"]}
# E1, study 1
contrast("E1:S1:TMRext-TMext", C1, lambda k: e5("s1", k, "TMR"), lambda k: tmext("D1_s1", k), SD5)
contrast("E1:S1:QMext-TMext", C1, lambda k: e5("s1", k, "QM"), lambda k: tmext("D1_s1", k), SD5)
contrast("E1:S1:QMRext-TMRext", C1, lambda k: e5("s1", k, "QMR"), lambda k: e5("s1", k, "TMR"), SD5)
contrast("E1:S1:bestQM-bestNonQM", C1, best_qm, best_nonqm, SD5)
contrast("E1:S1:TMRext-TMR", C1, lambda k: e5("s1", k, "TMR"), lambda k: orig(SEL1, k, "TMR"), SD5)
contrast("E1:S1:QMext-QM", C1, lambda k: e5("s1", k, "QM"), lambda k: orig(SEL1, k, "QM"), SD5)
# E1, study 2 part C
rule = lambda k: e5("s2", k, "TMR" if SELC[k]["density"] < 0.03 else "QM")
contrast("E1:S2C:rule-TMext", CC, rule, lambda k: tmext("D1_s2", k), SD5, dens)
contrast("E1:S2C:TMRext-TMext", CC, lambda k: e5("s2", k, "TMR"), lambda k: tmext("D1_s2", k), SD5, dens)
contrast("E1:S2C:QMext-TMext", CC, lambda k: e5("s2", k, "QM"), lambda k: tmext("D1_s2", k), SD5, dens)
# E2
noL = lambda t: None if t is None else (re.sub(r"_L\d+", "", t[0]), t[1], t[2])
for tag, cells, tag4 in (("s1", C1, "D1_s1"), ("s2", CC, "D1_s2")):
    S = "S1" if tag == "s1" else "S2C"
    for arm in ("TMR", "QM"):
        lit = lambda k, arm=arm, tag=tag: (lambda a: {} if a is None else {
            "inc_L": inc(*k.split("|"), a[0], a[1], SD3), "inc_noL": inc(*k.split("|"), noL(a)[0], a[1], SD3)})(e5(tag, k, arm))
        contrast(f"E2:{S}:{arm}noL-{arm}L", cells, lambda k, arm=arm, tag=tag: noL(e5(tag, k, arm)),
                 lambda k, arm=arm, tag=tag: e5(tag, k, arm), SD3, lit)
        contrast(f"E2:{S}:{arm}noL-TMext", cells, lambda k, arm=arm, tag=tag: noL(e5(tag, k, arm)),
                 lambda k, tag4=tag4: tmext(tag4, k), SD3)
json.dump(OUT, open(f"{S5}/logs/stats5.json", "w"), indent=1)
for key, r in OUT.items():
    md = "-" if r["median_diff"] is None else f"{r['median_diff']:+.2f}"
    print(f"{key:26s} n={r['n']:2d} (skip {r['skipped']:2d}) W/L {r['wins']:2d}/{r['losses']:2d} pos {r['positive']:2d} "
          f"median {md} within1 {r['within1']:2d} sign p={r['sign_p'] if r['sign_p'] is None else round(r['sign_p'], 4)}")
