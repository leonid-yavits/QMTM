"""Study 2 statistics (PROTOCOL.md). Writes logs/stats2.json and prints tables. Any part with missing runs is reported as such.
python3 stats2.py"""
import json, collections, numpy as np
from scipy import stats
S1, S2 = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2"
OUT = {}


def holm(rows):
    m = len(rows); run = 0.0
    for r_, i in enumerate(sorted(range(m), key=lambda i: rows[i]["p"])):
        run = max(run, min(1.0, (m - r_) * rows[i]["p"])); rows[i]["p_holm"] = run
    return rows


def welch(x, y):
    return 1.0 if np.allclose(x, y) else float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def load(path):
    A = collections.defaultdict(dict)
    try:
        for l in open(path):
            d = json.loads(l); A[(d["ds"], d["C"], d["cfg"], float(d["s"]))][d["seed"]] = d["test"]
    except FileNotFoundError:
        pass
    return A


def vec(A, ds, C, cfg, s, seeds=range(101, 106)):
    r = A.get((ds, int(C), cfg, float(s)), {})
    return np.array([r[k] for k in seeds if k in r])


sel2 = json.load(open(f"{S2}/logs/selection2.json")) if __import__("os").path.exists(f"{S2}/logs/selection2.json") else {"C": {}, "A": {}}
sel1 = json.load(open(f"{S1}/logs/selection.json"))
A1, AC, AF, AB = load(f"{S1}/logs/confirm.jsonl"), load(f"{S2}/logs/confirmC.jsonl"), load(f"{S2}/logs/fptm_confirm.jsonl"), load(f"{S2}/logs/sweep.jsonl")

# ---- C: density rule on new datasets
for con in ["rule-TM", "TMR-TM", "QM-TM"]:
    rows = []
    for k, o in sel2["C"].items():
        ds, C = k.split("|"); a = o["rule"] if con == "rule-TM" else con.split("-")[0]
        x, y = vec(AC, ds, C, o[a]["cfg"], o[a]["s"]), vec(AC, ds, C, o["TM"]["cfg"], o["TM"]["s"])
        if len(x) == 5 and len(y) == 5:
            rows.append(dict(cell=k, arm=a, density=o["density"], ma=100 * x.mean(), sa=100 * x.std(ddof=1), mb=100 * y.mean(), sb=100 * y.std(ddof=1),
                             diff=100 * (x.mean() - y.mean()), p=welch(x, y)))
    holm(rows)
    npos = sum(r["diff"] > 0 for r in rows); nneg = sum(r["diff"] < 0 for r in rows)
    sign_p = float(stats.binomtest(npos, npos + nneg, 0.5, alternative="greater").pvalue) if npos + nneg else None
    OUT[f"C:{con}"] = dict(rows=rows, n=len(rows), wins=int(sum(r["diff"] > 0 and r["p_holm"] < 0.05 for r in rows)),
                          losses=int(sum(r["diff"] < 0 and r["p_holm"] < 0.05 for r in rows)), positive=int(npos), sign_p=sign_p)
    print(f"\n== C {con}: {len(rows)} cells, wins {OUT[f'C:{con}']['wins']}, losses {OUT[f'C:{con}']['losses']}, positive {npos}/{len(rows)}, sign test p={sign_p}")
    for r in sorted(rows, key=lambda r: r["density"]):
        print(f"   {r['cell']:14s} {r['arm']:4s} {r['ma']:6.2f}±{r['sa']:4.2f}  TM {r['mb']:6.2f}±{r['sb']:4.2f}  d={r['diff']:+6.2f}  p={r['p']:.3g} pH={r['p_holm']:.3g}  dens={100 * r['density']:.3f}%")

# ---- A: FPTM against study-1 arms
ARMS = ["TM", "TMR", "QM", "QMR", "QP", "QPR"]
for con in ["FPTM-TM", "FPTM-bestQM", "FPTM-bestArm"]:
    rows = []
    for k, o in sel2["A"].items():
        ds, C = k.split("|"); s1 = sel1[k]
        if con == "FPTM-TM":
            b = "TM"
        elif con == "FPTM-bestQM":
            b = s1["bestQM"]
        else:
            b = max(ARMS, key=lambda a: (s1[a]["val"], -ARMS.index(a)))
        x, y = vec(AF, ds, C, o["cfg"], o["S"]), vec(A1, ds, C, s1[b]["cfg"], s1[b]["s"])
        if len(x) == 5 and len(y) == 5:
            rows.append(dict(cell=k, b=b, ma=100 * x.mean(), sa=100 * x.std(ddof=1), mb=100 * y.mean(), sb=100 * y.std(ddof=1), diff=100 * (x.mean() - y.mean()), p=welch(x, y)))
    holm(rows)
    OUT[f"A:{con}"] = dict(rows=rows, n=len(rows), wins=int(sum(r["diff"] > 0 and r["p_holm"] < 0.05 for r in rows)),
                          losses=int(sum(r["diff"] < 0 and r["p_holm"] < 0.05 for r in rows)))
    print(f"\n== A {con}: {len(rows)} cells, FPTM significantly better {OUT[f'A:{con}']['wins']}, worse {OUT[f'A:{con}']['losses']}")
    for r in rows:
        print(f"   {r['cell']:18s} FPTM {r['ma']:6.2f}±{r['sa']:4.2f}  {r['b']:4s} {r['mb']:6.2f}±{r['sb']:4.2f}  d={r['diff']:+6.2f}  p={r['p']:.3g} pH={r['p_holm']:.3g}")

# ---- B: budget sweep (seeds 101-103; study-1 confirmations for C = 5, 10, 20)
B = {}
print("\n== B budget sweep (mean test %, seeds 101-103)")
for ds in ["adult", "mnist", "fmnist", "imdb", "r8_V5000", "r52_V1000", "r52_V5000", "r52_V0"]:
    B[ds] = {}
    for arm in ["TM", "TMR", "QM"]:
        B[ds][arm] = {}
        for C in [5, 10, 20, 40, 80, 160]:
            src = A1 if C <= 20 else AB; a = sel1[f"{ds}|{min(C, 20)}"][arm]
            v = vec(src, ds, C, a["cfg"], a["s"], seeds=[101, 102, 103])
            B[ds][arm][C] = float(100 * v.mean()) if len(v) == 3 else None
    def first_C(target):
        for C in [5, 10, 20, 40, 80, 160]:
            v = B[ds]["TM"][C]
            if v is not None and target is not None and v >= target:
                return C
        return ">160"
    B[ds]["TM_reaches"] = {f"{arm}@{C}": first_C(B[ds][arm][C]) for arm in ["TMR", "QM"] for C in [5, 10]}
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"
    for arm in ["TM", "TMR", "QM"]:
        print(f"   {ds:10s} {arm:4s} " + " ".join(f(B[ds][arm][C]) for C in [5, 10, 20, 40, 80, 160]))
    print(f"   {ds:10s} TM reaches: {B[ds]['TM_reaches']}")
OUT["B"] = B
json.dump(OUT, open(f"{S2}/logs/stats2.json", "w"), indent=1)
