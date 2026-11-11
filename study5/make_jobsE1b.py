"""Study 5, E1b job list (PROTOCOL.md): budget sweep, 8 encodings, C = 40, 80, 160, T = C/2.
Per arm (TMR, QM): candidates = the original sweep setting (study 2 part B) plus every family member at s = 1.5.
Seed-1 validation tuning runs for all candidates (the original setting has no seed-1 sweep run). IMDb last.
Writes jobs/tuneE1b.txt."""
import json
S2, S5 = "/home/claude/qmtm/study2", "/home/claude/qmtm/study5"
TR = "cd /home/claude/qmtm/study1 && python3 train.py"
FAM = {"TMR": ["AND_L32_fb1", "AND_L64_fb1"], "QM": ["QM90_L32_fb0", "QM95_L32_fb0", "QM95_L64_fb0"]}
DS = ["adult", "r8_V5000", "r52_V1000", "mnist", "fmnist", "r52_V5000", "r52_V0", "imdb"]
orig = {}
for l in open(f"{S2}/logs/sweep.jsonl"):
    d = json.loads(l)
    for a, f in FAM.items():
        if d["cfg"] in f:
            orig[(d["ds"], d["C"], a)] = (d["cfg"], float(d["s"]))
J = []
for C in (40, 80, 160):
    for ds in DS:
        for a in ("TMR", "QM"):
            cand = [orig[(ds, C, a)]] + [(c, 1.5) for c in FAM[a] if (c, 1.5) != orig[(ds, C, a)]]
            J += [f"{TR} {ds} {C} {C // 2} {s} {c} 1 40 {S5}/logs/tuneE1b.jsonl" for c, s in cand]
open(f"{S5}/jobs/tuneE1b.txt", "w").write("\n".join(J) + "\n")
print(len(J), "jobs")
