"""Study 5 selection (PROTOCOL.md).
E1: per cell and arm, best seed-1 validation accuracy over the extended family (original runs + s = 1.5 runs);
    ties: smaller s, then the listed cfg order. Changed cells are confirmed with seeds 101-105 (jobs/confirmE1.txt).
E2: the E1 selection of TMR and QM with "_L<L>" removed, seeds 101-103 (jobs/E2.txt).
Only cells whose E1 tuning is complete for that arm are selected. Writes logs/selection5.json."""
import json, collections, re
S1, S2, S5 = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2", "/home/claude/qmtm/study5"
TR = "cd /home/claude/qmtm/study1 && python3 train.py"
FAM = {"TMR": ["AND_L32_fb1", "AND_L64_fb1"], "QM": ["QM90_L32_fb0", "QM95_L32_fb0", "QM95_L64_fb0"],
       "QMR": ["QM90_L32_fb1", "QM95_L32_fb1"]}


def load(path):
    R = collections.defaultdict(dict)
    try:
        for l in open(path):
            d = json.loads(l)
            if d["seed"] == 1 and d["epochs"] == 40:
                R[(d["ds"], d["C"])][(d["cfg"], float(d["s"]))] = d
    except FileNotFoundError:
        pass
    return R


def drop_L(cfg):
    return re.sub(r"_L\d+", "", cfg)


out, JC, JE2 = {"s1": {}, "s2": {}}, [], []
for tag, orig_tune, orig_sel, new_log, conf_log, arms in (
        ("s1", f"{S1}/logs/tune.jsonl", json.load(open(f"{S1}/logs/selection.json")), "tuneE1_s1.jsonl",
         f"{S1}/logs/confirm.jsonl", ("TMR", "QM", "QMR")),
        ("s2", f"{S2}/logs/tune.jsonl", json.load(open(f"{S2}/logs/selection2.json"))["C"], "tuneE1_s2.jsonl",
         f"{S2}/logs/confirmC.jsonl", ("TMR", "QM"))):
    O, N = load(orig_tune), load(f"{S5}/logs/{new_log}")
    for key, o in orig_sel.items():
        ds, C = key.split("|"); C = int(C); T = o["T"]
        for a in arms:
            new = {(c, 1.5) for c in FAM[a]}
            if not new.issubset(N[(ds, C)].keys()):
                continue
            pool = {k: v for k, v in O[(ds, C)].items() if k[0] in FAM[a]}
            pool.update({k: N[(ds, C)][k] for k in new})
            v, _, _, c, s = max((d["val"], -s, -FAM[a].index(c), c, s) for (c, s), d in pool.items())
            changed = not (c == o[a]["cfg"] and s == float(o[a]["s"]))
            out[tag].setdefault(key, {"T": T})[a] = dict(
                cfg=c, s=s, val=v, orig_cfg=o[a]["cfg"], orig_s=float(o[a]["s"]), orig_val=o[a]["val"], changed=changed,
                conf_log=f"{S5}/logs/confirmE1.jsonl" if changed else conf_log)
            if changed:
                JC += [f"{TR} {ds} {C} {T} {s} {c} {sd} 40 {S5}/logs/confirmE1.jsonl" for sd in range(101, 106)]
            if a in ("TMR", "QM"):
                JE2 += [f"{TR} {ds} {C} {T} {s} {drop_L(c)} {sd} 40 {S5}/logs/E2.jsonl" for sd in range(101, 104)]
json.dump(out, open(f"{S5}/logs/selection5.json", "w"), indent=1)
open(f"{S5}/jobs/confirmE1.txt", "w").write("\n".join(JC) + ("\n" if JC else ""))
open(f"{S5}/jobs/E2.txt", "w").write("\n".join(JE2) + ("\n" if JE2 else ""))
for tag in out:
    for a in ("TMR", "QM", "QMR"):
        sel = [x[a] for x in out[tag].values() if a in x]
        if sel:
            print(tag, a, "cells", len(sel), "changed", sum(x["changed"] for x in sel),
                  "s=1.5", sum(x["s"] == 1.5 for x in sel))
print("confirm jobs", len(JC), "E2 jobs", len(JE2))
