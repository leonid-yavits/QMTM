"""Study 5 job list for E1 tuning (PROTOCOL.md). Writes jobs/tuneE1.txt.
Every member of TMR_ext, QM_ext (all 60 cells) and QMR_ext (48 study-1 cells) at s = 1.5, seed 1, 40 epochs.
Jobs are ordered by the measured cost of the vanilla TM at s = 1.5 in study 4 (cheapest dataset first)."""
import json, collections
S1, S2, S4, S5 = (f"/home/claude/qmtm/{d}" for d in ("study1", "study2", "study4", "study5"))
TR = "cd /home/claude/qmtm/study1 && python3 train.py"
FAM = {"TMR": ["AND_L32_fb1", "AND_L64_fb1"], "QM": ["QM90_L32_fb0", "QM95_L32_fb0", "QM95_L64_fb0"],
       "QMR": ["QM90_L32_fb1", "QM95_L32_fb1"]}
cost = collections.defaultdict(float)
for f in ("tuneD1_s1.jsonl", "tuneD1_s2.jsonl"):
    for l in open(f"{S4}/logs/{f}"):
        d = json.loads(l)
        if d["s"] == 1.5:
            cost[d["ds"]] += d["t"]
cells = []
for src, out, arms in ((json.load(open(f"{S1}/logs/selection.json")), "tuneE1_s1.jsonl", ("TMR", "QM", "QMR")),
                       (json.load(open(f"{S2}/logs/selection2.json"))["C"], "tuneE1_s2.jsonl", ("TMR", "QM"))):
    for k, o in src.items():
        ds, C = k.split("|")
        cells.append((cost[ds], ds, int(C), o["T"], out, arms))
J = []
for _, ds, C, T, out, arms in sorted(cells):
    for a in arms:
        for cfg in FAM[a]:
            J.append(f"{TR} {ds} {C} {T} 1.5 {cfg} 1 40 {S5}/logs/{out}")
open(f"{S5}/jobs/tuneE1.txt", "w").write("\n".join(J) + "\n")
print(len(J), "jobs")
