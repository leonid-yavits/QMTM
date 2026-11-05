"""Study 4 job lists (PROTOCOL.md). Writes jobs/tuneD1.txt (study-1 and study-2 C cells) and jobs/tuneD2.txt (sweep)."""
import json, collections
S1, S2, S4 = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2", "/home/claude/qmtm/study4"
TR = "cd /home/claude/qmtm/study1 && python3 train.py"
def grid(path):
    G = collections.defaultdict(set); T = {}
    for l in open(path):
        d = json.loads(l)
        if d["seed"] == 1 and d["epochs"] == 40:
            G[(d["ds"], d["C"])].add(d["s"]); T[(d["ds"], d["C"])] = d["T"]
    return G, T
J1 = []
for src, out in ((f"{S1}/logs/tune.jsonl", "tuneD1_s1.jsonl"), (f"{S2}/logs/tune.jsonl", "tuneD1_s2.jsonl")):
    G, T = grid(src)
    for (ds, C) in sorted(G):
        for s in sorted(G[(ds, C)]):
            J1.append(f"{TR} {ds} {C} {T[(ds, C)]} {s} AND_L32 1 40 {S4}/logs/{out}")
        for cfg in ("AND", "AND_L64", "AND_L32"):
            J1.append(f"{TR} {ds} {C} {T[(ds, C)]} 1.5 {cfg} 1 40 {S4}/logs/{out}")
sel1 = json.load(open(f"{S1}/logs/selection.json"))
J2 = []
for ds in ["adult", "r52_V1000", "r52_V5000", "r8_V5000", "mnist", "r52_V0", "fmnist", "imdb"]:   # cheapest first
    o = sel1[f"{ds}|20"]["TM"]
    for C in (40, 80, 160):
        cand = [(o["cfg"], float(o["s"]))] + [(c, s) for c in ("AND_L32", "AND_L64") for s in (1.5, 2.5)]
        for c, s in dict.fromkeys(cand):
            J2.append(f"{TR} {ds} {C} {C // 2} {s} {c} 1 40 {S4}/logs/tuneD2.jsonl")
open(f"{S4}/jobs/tuneD1.txt", "w").write("\n".join(J1) + "\n")
open(f"{S4}/jobs/tuneD2.txt", "w").write("\n".join(J2) + "\n")
print(len(J1), len(J2))
