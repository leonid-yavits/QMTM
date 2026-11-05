"""Study 2 selection (PROTOCOL.md). Part C: per (dataset, C) and arm family, the (cfg, s) with the highest seed-1 validation
accuracy (ties: smaller s, then first listed cfg); the rule arm is TMR if training density < 3 %, else QM.
Part A: per (dataset, C), the FPTM LF with the highest validation accuracy (ties: smaller LF).
Only cells with complete tuning are selected. Writes logs/selection2.json, jobs/confirmC.txt, jobs/confirmA.txt."""
import json, collections, numpy as np
S1, S2, F = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2", "/home/claude/qmtm/fptm"
FAM = collections.OrderedDict([("TM", ["AND", "AND_L64"]), ("TMR", ["AND_L32_fb1", "AND_L64_fb1"]), ("QM", ["QM90_L32_fb0", "QM95_L32_fb0", "QM95_L64_fb0"])])
out = {"C": {}, "A": {}}
# ---- part C
R = collections.defaultdict(dict)
for l in open(f"{S2}/logs/tune.jsonl"):
    d = json.loads(l); R[(d["ds"], d["C"])][(d["cfg"], d["s"])] = d
jc = []
for (ds, C), got in sorted(R.items()):
    if len(got) < 14:
        continue
    X = np.load(f"/home/claude/data/study2/{ds}.npz")["X_train"]; dens = float(X.mean()); del X
    T = next(iter(got.values()))["T"]; o = dict(T=T, density=dens, rule="TMR" if dens < 0.03 else "QM")
    for arm, cfgs in FAM.items():
        v, _, _, c, s = max((got[(c, s)]["val"], -s, -i, c, s) for i, c in enumerate(cfgs) for (cc, s) in got if cc == c)
        o[arm] = dict(cfg=c, s=s, val=v)
        for sd in range(101, 106):
            jc.append(f"cd {S1} && python3 train.py {ds} {C} {T} {s} {c} {sd} 40 {S2}/logs/confirmC.jsonl")
    out["C"][f"{ds}|{C}"] = o
# ---- part A (FPTM)
Q = collections.defaultdict(dict)
import os
for l in (open(f"{S2}/logs/fptm_tune.jsonl") if os.path.exists(f"{S2}/logs/fptm_tune.jsonl") else []):   # part A is not part of this release
    d = json.loads(l); Q[(d["ds"], d["C"])][d["LF"]] = d
ja = []
for (ds, C), got in sorted(Q.items()):
    if len(got) < 3:
        continue
    v, _, LF = max((got[lf]["val"], -lf, lf) for lf in got)
    out["A"][f"{ds}|{C}"] = dict(LF=LF, val=v, cfg=got[LF]["cfg"], S=int(got[LF]["s"]))
    for sd in range(101, 106):
        ja.append(f"cd {F} && python3 fptm_job.py {ds} {C} 16 {int(got[LF]['s'])} {LF} {sd} 40 {S2}/logs/fptm_confirm.jsonl")
json.dump(out, open(f"{S2}/logs/selection2.json", "w"), indent=1)
open(f"{S2}/jobs/confirmC.txt", "w").write("\n".join(jc) + "\n")
open(f"{S2}/jobs/confirmA.txt", "w").write("\n".join(ja) + "\n")
print(f"C cells {len(out['C'])} jobs {len(jc)}; FPTM cells {len(out['A'])} jobs {len(ja)}")
