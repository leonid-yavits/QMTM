"""Study 6 summaries. Writes logs/stats6.json and prints a report. Run from anywhere.
E1: FPTM per cell (seeds 101-105): selected L_F, test accuracy, used features, clause lengths, and test accuracy of the vote
    with the clause score clipped at Q (Q threshold searches per clause).
E2: test accuracy against L_F (6 clauses per class, seeds 101-103).
E3/E4: smallest FPTM clause count that reaches the best binary arm at each binary budget, and the binary budget (any arm,
    any log) that first reaches FPTM with 6 clauses per class.
HW: Table 9 cost ratios recomputed with the measured used-feature counts of E1 (C = 5 cells)."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "hw"))
from binary_vs_graded import rows_for, CASES


def load(f):
    p = os.path.join(HERE, "logs", f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def ms(v): return (round(100 * st.mean(v), 2), round(100 * st.stdev(v), 2) if len(v) > 1 else None, len(v))


S = {}
# ---- E1 ----
sel = json.load(open(os.path.join(HERE, "logs", "selection6.json"))) if os.path.exists(os.path.join(HERE, "logs", "selection6.json")) else {}
E1 = {}
for d in load("E1_confirm.jsonl"):
    E1.setdefault(f"{d['ds']}|{ {6: 5, 10: 10}[d['clauses_fptm']] }", []).append(d)
S["E1"] = {}
for k, rs in sorted(E1.items()):
    q = rs[0]["stats"]["q"]
    S["E1"][k] = dict(LF=rs[0]["LF"], test=ms([r["test"] for r in rs]),
                      used_features=st.mean(r["stats"]["used_features"] for r in rs), nf=rs[0]["stats"]["nf"],
                      used_frac=st.mean(r["stats"]["used_features"] / r["stats"]["nf"] for r in rs),
                      clause_len_mean=st.mean(r["stats"]["clause_len_mean"] for r in rs),
                      clause_len_max=max(r["stats"]["clause_len_max"] for r in rs),
                      quantized={str(Q): ms([r["stats"]["q_test"][i] for r in rs]) for i, Q in enumerate(q)})
# ---- E2 ----
S["E2"] = {}
for d in load("E2_levels.jsonl"):
    S["E2"].setdefault(d["ds"], {}).setdefault(str(d["LF"]), []).append(d["test"])
S["E2"] = {ds: {lf: ms(v) for lf, v in sorted(x.items(), key=lambda t: int(t[0]))} for ds, x in S["E2"].items()}
# ---- E3 / E4: equal-accuracy points ----
bin_logs = ["study1/logs/confirm.jsonl", "study2/logs/sweep.jsonl", "study4/logs/confirmD1.jsonl", "study4/logs/confirmD2.jsonl",
            "study5/logs/confirmE1.jsonl", "study5/logs/confirmE1b.jsonl", "study6/logs/E4_binary.jsonl"]
B = {}
for f in bin_logs:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p): continue
    for l in open(p):
        d = json.loads(l)
        if d["seed"] >= 101:
            B.setdefault((d["ds"], d["C"], d["cfg"], d["s"], d["T"]), []).append(d["test"])
best_bin = {}
for (ds, C, *_), v in B.items():
    m = st.mean(v)
    if m > best_bin.get((ds, C), (-1,))[0]: best_bin[(ds, C)] = (m, len(v))
F = {}
for d in load("E3_clauses.jsonl"):
    F.setdefault((d["ds"], d["clauses_fptm"]), []).append(d["test"])
S["E3"] = {}
for ds in sorted({k[0] for k in F}):
    fp = {cl: st.mean(F[(ds, cl)]) for (x, cl) in F if x == ds}
    bb = {C: m for (x, C), (m, n) in best_bin.items() if x == ds}
    fp6 = fp.get(6)
    S["E3"][ds] = dict(fptm={str(c): round(100 * a, 2) for c, a in sorted(fp.items())},
                       best_binary={str(C): round(100 * a, 2) for C, a in sorted(bb.items())},
                       binary_C_reaching_fptm6=(min([C for C, a in bb.items() if fp6 is not None and a >= fp6], default=None)),
                       fptm_clauses_reaching_binary={str(C): min([c for c, a in fp.items() if a >= b], default=None) for C, b in sorted(bb.items())})
# ---- HW with measured used features ----
MAP = {"R52 (V=1k)": "r52_V1000", "R8 (V=5k)": "r8_V5000", "UCI Adult": "adult", "MNIST": "mnist", "Fashion-MNIST": "fmnist", "IMDb": "imdb"}
S["HW"] = []
for case in CASES:
    k = f"{MAP[case[0]]}|5"
    if k not in S["E1"]: continue
    e = S["E1"][k]
    for r in rows_for(*case, n_used=round(e["used_features"]), LFs=(e["LF"],)):
        r["used_frac"] = e["used_frac"]; S["HW"].append(r)
json.dump(S, open(os.path.join(HERE, "logs", "stats6.json"), "w"), indent=1)
for k, v in S["E1"].items():
    print("E1", k, "LF", v["LF"], "test", v["test"], "used", round(v["used_features"]), "/", v["nf"], "quantized", v["quantized"])
for ds, v in S["E2"].items(): print("E2", ds, v)
for ds, v in S["E3"].items(): print("E3", ds, v)
for r in S["HW"]:
    print("HW", r["encoding"], r["arm"], r["C"], "LF", r["LF"], r["sensing"], "used", r["n_used"],
          {x: round(r[x], 2) for x in ("latency", "thr_area", "eff", "edp")}, "breakeven", r["breakeven"], "used_frac", round(r["used_frac"], 3))
