"""Study 6 summary numbers for the paper (descriptive; written after internal review round 7). Writes logs/summary6.json.
Binary reference: the most accurate binary configuration (any arm, original or re-tuned, any log) by mean test accuracy over
seeds 101 and above. It is selected on test means, so it favors the binary side.
FPTM reference: E1 (seeds 101-105) for 6 and 10 clauses per class, E2 (seeds 101-103) for other L_F, E3 for other clause counts."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "hw"))
from binary_vs_graded import rows_for, CASES
BIN = ["study1/logs/confirm.jsonl", "study2/logs/sweep.jsonl", "study4/logs/confirmD1.jsonl", "study4/logs/confirmD2.jsonl",
       "study5/logs/confirmE1.jsonl", "study5/logs/confirmE1b.jsonl", "study6/logs/E4_binary.jsonl"]
def arm(cfg):
    base = {"AND": "TM", "QM95": "QM", "QM90": "QM", "QP10": "QP", "QP1": "QP"}[cfg.split("_")[0]]
    return base + ("R" if "fb1" in cfg else "")
def load(f):
    p = os.path.join(HERE, "logs", f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
G = {}
for f in BIN:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p): continue
    for l in open(p):
        d = json.loads(l)
        if d["seed"] >= 101: G.setdefault((d["ds"], d["C"], d["cfg"], d["s"], d["T"]), []).append(100 * d["test"])
best = {}
for (ds, C, cfg, s, T), v in G.items():
    m = st.mean(v)
    if m > best.get((ds, C), {"acc": -1})["acc"]:
        best[(ds, C)] = dict(acc=m, arm=arm(cfg), cfg=cfg, s=s, T=T, n=len(v))
def fp(records, key):
    out = {}
    for d in records: out.setdefault(key(d), []).append(d)
    return out
E1 = fp(load("E1_confirm.jsonl"), lambda d: (d["ds"], d["clauses_fptm"]))
E2 = fp(load("E2_levels.jsonl"), lambda d: (d["ds"], d["LF"]))
E2b = fp(load("E2b_control.jsonl"), lambda d: (d["ds"], d["T"]))
E3 = fp(load("E3_clauses.jsonl"), lambda d: (d["ds"], d["clauses_fptm"]))
acc = lambda rs: st.mean(100 * r["test"] for r in rs)
S = {}
# 1. FPTM (E1) against the best binary arm, 18 cells
rows = []
for (ds, cf), rs in sorted(E1.items()):
    C = {6: 5, 10: 10}[cf]; b = best[(ds, C)]
    rows.append(dict(ds=ds, C=C, fptm=acc(rs), best=b["acc"], arm=b["arm"], lead=acc(rs) - b["acc"]))
L = [r["lead"] for r in rows]
S["E1_vs_best"] = dict(rows=rows, median=st.median(L), min=min(L), max=max(L), ahead=sum(x > 0 for x in L))
# 2. E2 levels against the best binary arm with 5 clauses per class; E2b control
S["E2"] = {}
for ds in sorted({k[0] for k in E2}):
    lv = {LF: acc(E2[(ds, LF)]) for (x, LF) in E2 if x == ds}
    S["E2"][ds] = dict(levels=lv, best5=best[(ds, 5)]["acc"], best5_arm=best[(ds, 5)]["arm"], e1=acc(E1[(ds, 6)]),
                       e2b={T: acc(E2b[(ds, T)]) for (x, T) in E2b if x == ds})
# 3. clipped vote, models with L_F = 64
q16 = [acc(rs) - st.mean(100 * dict(zip(r["stats"]["q"], r["stats"]["q_test"]))[16] for r in rs) for (ds, cf), rs in E1.items() if rs[0]["LF"] == 64]
S["Q16_LF64"] = dict(n=len(q16), median_loss=st.median(q16), max_loss=max(q16))
# 4. equal-accuracy budgets (FPTM reference E1 with 6 clauses per class, or E2 with L_F = 4)
MAP = {"R52 (V=1k)": "r52_V1000", "R8 (V=5k)": "r8_V5000", "UCI Adult": "adult", "MNIST": "mnist", "Fashion-MNIST": "fmnist", "IMDb": "imdb"}
BUD = [5, 10, 20, 40, 80, 160]
def first_reach(ds, a):
    for C in BUD:
        if (ds, C) in best and best[(ds, C)]["acc"] >= a: return C, best[(ds, C)]
    return None, best[(ds, 160)]
S["cost"] = []
for name, n, K, _a, _C, _eq, _q in CASES:
    ds = MAP[name]
    for ref, rs, LF in (("E1", E1[(ds, 6)], E1[(ds, 6)][0]["LF"]), ("E2_LF4", E2[(ds, 4)], 4)):
        a = acc(rs); C, b = first_reach(ds, a); isqm = b["arm"].startswith("Q")
        nu = round(st.mean(r["stats"]["used_features"] for r in rs))
        Q = min(LF, 16)
        qa = st.mean(100 * dict(zip(r["stats"]["q"], r["stats"]["q_test"]))[Q] for r in rs)
        case = (name, n, K, b["arm"], C or 160, C is not None, isqm)
        r1 = {r["sensing"]: r for r in rows_for(*case, n_used=nu, LFs=(LF,))}
        rq = {r["sensing"]: r for r in rows_for(*case, n_used=nu, LFs=(Q,))}
        S["cost"].append(dict(encoding=name, ref=ref, fptm_acc=a, LF=LF, used_frac=nu / n, binary_C=C or 160, equal=C is not None,
                              binary_arm=b["arm"], binary_acc=b["acc"], Q=Q, dQ=qa - a, full=r1, clipped=rq))
# 5. E3 points (4 FPTM clauses per class) against the best binary arm with 160 clauses per class
S["E3_4"] = {}
for ds in ("mnist", "fmnist", "imdb"):
    rs = E3[(ds, 4)]; b = best[(ds, 160)]
    S["E3_4"][ds] = dict(fptm=acc(rs), sd=st.stdev(100 * r["test"] for r in rs), best160=b["acc"], arm=b["arm"],
                         sd160=st.stdev(G[(ds, 160, b["cfg"], b["s"], b["T"])]))
# 6. E2b collapse statistics at the best margin (chosen on test means)
S["E2b_best"] = {}
for ds in sorted({k[0] for k in E2b}):
    T = max((T for (x, T) in E2b if x == ds), key=lambda T: acc(E2b[(ds, T)])); rs = E2b[(ds, T)]
    S["E2b_best"][ds] = dict(T=T, acc=acc(rs), clause_len=st.mean(r["stats"]["clause_len_mean"] for r in rs),
                             fire=st.mean(100 * r["stats"]["score_hist"][1] / sum(r["stats"]["score_hist"]) for r in rs))
    print("E2b best", ds, S["E2b_best"][ds])
S["best"] = {f"{k[0]}|{k[1]}": v for k, v in best.items() if k[1] in BUD}
json.dump(S, open(os.path.join(HERE, "logs", "summary6.json"), "w"), indent=1, default=str)
print("E1 vs best binary: median lead", round(S["E1_vs_best"]["median"], 2), "range", round(S["E1_vs_best"]["min"], 2), round(S["E1_vs_best"]["max"], 2), "ahead", S["E1_vs_best"]["ahead"])
for r in rows: print("  ", r["ds"], r["C"], round(r["fptm"], 2), round(r["best"], 2), r["arm"], round(r["lead"], 2))
for ds, v in S["E2"].items():
    print("E2", ds, "LF1", round(v["levels"][1], 2), "LF4", round(v["levels"][4], 2), "best5", round(v["best5"], 2), v["best5_arm"], "E1", round(v["e1"], 2),
          "E2b", {T: round(a, 2) for T, a in sorted(v["e2b"].items())})
print("Q16 over LF=64 cells", S["Q16_LF64"])
for c in S["cost"]:
    f, q = c["full"], c["clipped"]
    print("COST", c["encoding"], c["ref"], "acc", round(c["fptm_acc"], 2), "LF", c["LF"], "used", round(100 * c["used_frac"], 1), "bin", c["binary_arm"], c["binary_C"], round(c["binary_acc"], 2), "eq", c["equal"], "dQ", round(c["dQ"], 2))
    for sens in ("binary", "threshold"):
        if sens in f:
            print("     ", sens, "full", {k: round(f[sens][k], 2) for k in ("latency", "thr_area", "eff", "edp")}, "be", f[sens]["breakeven"] and round(f[sens]["breakeven"], 3),
                  "| Q", {k: round(q[sens][k], 2) for k in ("latency", "thr_area", "eff", "edp")})
for ds, v in S["E3_4"].items(): print("E3_4", ds, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})
