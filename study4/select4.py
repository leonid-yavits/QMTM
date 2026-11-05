"""Study 4 selection (PROTOCOL.md). D1: extended TM family per study-1 and study-2 C cell. D2: re-tuned TM per sweep cell.
Writes logs/selection4.json and jobs/confirmD1.txt (seeds 101-105, changed cells only), jobs/confirmD2.txt (seeds 101-103).
Only cells whose study-4 tuning is complete are selected."""
import json, collections
S1, S2, S4 = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2", "/home/claude/qmtm/study4"
TR = "cd /home/claude/qmtm/study1 && python3 train.py"
ORDER = ["AND", "AND_L64", "AND_L32"]


def load(path, filt=lambda d: d["seed"] == 1 and d["epochs"] == 40):
    R = collections.defaultdict(dict)
    try:
        for l in open(path):
            d = json.loads(l)
            if filt(d):
                R[(d["ds"], d["C"])][(d["cfg"], float(d["s"]))] = d
    except FileNotFoundError:
        pass
    return R


def expected(jobfile):
    E = collections.defaultdict(set)
    for l in open(jobfile):
        a = l.split("train.py")[1].split(); E[(a[0], int(a[1]), a[-1])].add((a[4], float(a[3])))
    return E


out, J1, J2 = {"D1_s1": {}, "D1_s2": {}, "D2": {}}, [], []
E1 = expected(f"{S4}/jobs/tuneD1.txt")
for tag, orig_tune, orig_sel, new_log, conf_log in (
        ("D1_s1", f"{S1}/logs/tune.jsonl", json.load(open(f"{S1}/logs/selection.json")), "tuneD1_s1.jsonl", f"{S1}/logs/confirm.jsonl"),
        ("D1_s2", f"{S2}/logs/tune.jsonl", json.load(open(f"{S2}/logs/selection2.json"))["C"], "tuneD1_s2.jsonl", f"{S2}/logs/confirmC.jsonl")):
    O, N = load(orig_tune), load(f"{S4}/logs/{new_log}")
    for key, o in orig_sel.items():
        ds, C = key.split("|"); C = int(C)
        need = E1[(ds, C, f"{S4}/logs/{new_log}")]
        if not need or not need.issubset(N[(ds, C)].keys()):
            continue
        pool = {k: v for k, v in O[(ds, C)].items() if k[0] in ORDER}
        pool.update(N[(ds, C)])
        v, _, _, c, s = max((d["val"], -s, -ORDER.index(c), c, s) for (c, s), d in pool.items())
        T = next(iter(pool.values()))["T"]
        changed = not (c == o["TM"]["cfg"] and s == float(o["TM"]["s"]))
        out[tag][key] = dict(T=T, cfg=c, s=s, val=v, orig_cfg=o["TM"]["cfg"], orig_s=float(o["TM"]["s"]), orig_val=o["TM"]["val"], changed=changed,
                             conf_log=conf_log if not changed else f"{S4}/logs/confirmD1.jsonl")
        if changed:
            for sd in range(101, 106):
                J1.append(f"{TR} {ds} {C} {T} {s} {c} {sd} 40 {S4}/logs/confirmD1.jsonl")
# D2
sel1 = json.load(open(f"{S1}/logs/selection.json"))
N2 = load(f"{S4}/logs/tuneD2.jsonl"); E2 = expected(f"{S4}/jobs/tuneD2.txt")
for (ds, C, _), need in sorted(E2.items()):
    if not need.issubset(N2[(ds, C)].keys()):
        continue
    o = sel1[f"{ds}|20"]["TM"]; orig = (o["cfg"], float(o["s"]))
    order = list(dict.fromkeys([orig] + [(c, s) for c in ("AND_L32", "AND_L64") for s in (1.5, 2.5)]))
    v, _, _, c, s = max((N2[(ds, C)][k]["val"], -k[1], -order.index(k), k[0], k[1]) for k in order)
    changed = (c, s) != orig
    out["D2"][f"{ds}|{C}"] = dict(T=C // 2, cfg=c, s=s, val=v, orig_cfg=orig[0], orig_s=orig[1], orig_val=N2[(ds, C)][orig]["val"], changed=changed,
                                  conf_log=f"{S2}/logs/sweep.jsonl" if not changed else f"{S4}/logs/confirmD2.jsonl")
    if changed:
        for sd in range(101, 104):
            J2.append(f"{TR} {ds} {C} {C // 2} {s} {c} {sd} 40 {S4}/logs/confirmD2.jsonl")
json.dump(out, open(f"{S4}/logs/selection4.json", "w"), indent=1)
open(f"{S4}/jobs/confirmD1.txt", "w").write("\n".join(J1) + ("\n" if J1 else ""))
open(f"{S4}/jobs/confirmD2.txt", "w").write("\n".join(J2) + ("\n" if J2 else ""))
for t in out:
    print(t, "cells", len(out[t]), "changed", sum(x["changed"] for x in out[t].values()))
print("confirm jobs D1", len(J1), "D2", len(J2))
