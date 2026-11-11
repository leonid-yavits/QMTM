"""Study 5, E1b selection (PROTOCOL.md). Per (encoding, C >= 40, arm): best seed-1 validation accuracy over the
candidates; ties: smaller s, original setting first, then the listed family order. If the selection differs from the
original sweep setting, it is confirmed with seeds 101-103 (jobs/confirmE1b.txt); otherwise the study-2 sweep runs are
reused. Cells with incomplete tuning are skipped. Writes logs/selectionE1b.json and logs/sweepE1b.json (mean test, %)."""
import json, collections
S1, S2, S4, S5 = (f"/home/claude/qmtm/{d}" for d in ("study1", "study2", "study4", "study5"))
TR = "cd /home/claude/qmtm/study1 && python3 train.py"
FAM = {"TMR": ["AND_L32_fb1", "AND_L64_fb1"], "QM": ["QM90_L32_fb0", "QM95_L32_fb0", "QM95_L64_fb0"]}
DS = ["adult", "mnist", "fmnist", "imdb", "r8_V5000", "r52_V1000", "r52_V5000", "r52_V0"]
expect = collections.defaultdict(set)
for l in open(f"{S5}/jobs/tuneE1b.txt"):
    a = l.split("train.py")[1].split(); expect[(a[0], int(a[1]))].add((a[4], float(a[3])))
N = collections.defaultdict(dict)
try:
    for l in open(f"{S5}/logs/tuneE1b.jsonl"):
        d = json.loads(l); N[(d["ds"], d["C"])][(d["cfg"], float(d["s"]))] = d
except FileNotFoundError:
    pass
orig = {}
for l in open(f"{S2}/logs/sweep.jsonl"):
    d = json.loads(l)
    for a, f in FAM.items():
        if d["cfg"] in f:
            orig[(d["ds"], d["C"], a)] = (d["cfg"], float(d["s"]))
A = collections.defaultdict(dict)
for p in (f"{S2}/logs/sweep.jsonl", f"{S5}/logs/confirmE1b.jsonl"):
    try:
        for l in open(p):
            d = json.loads(l); A[(d["ds"], d["C"], d["cfg"], float(d["s"]))][d["seed"]] = d["test"]
    except FileNotFoundError:
        pass
sel, J, sw = {}, [], {}
for ds in DS:
    for C in (40, 80, 160):
        for a in ("TMR", "QM"):
            o = orig[(ds, C, a)]
            cand = [o] + [(c, 1.5) for c in FAM[a] if (c, 1.5) != o]
            if not set(cand).issubset(N[(ds, C)].keys()):
                continue
            rank = lambda k: (N[(ds, C)][k]["val"], -k[1], -cand.index(k))
            c, s = max(cand, key=rank)
            ch = (c, s) != o
            sel[f"{ds}|{C}|{a}"] = dict(cfg=c, s=s, val=N[(ds, C)][(c, s)]["val"], orig_cfg=o[0], orig_s=o[1],
                                        orig_val=N[(ds, C)][o]["val"], changed=ch)
            if ch:
                J += [f"{TR} {ds} {C} {C // 2} {s} {c} {sd} 40 {S5}/logs/confirmE1b.jsonl" for sd in (101, 102, 103)]
            r = A.get((ds, C, c, s), {}); v = [r[x] for x in (101, 102, 103) if x in r]
            sw.setdefault(ds, {}).setdefault(str(C), {})[a] = dict(acc=100 * sum(v) / 3 if len(v) == 3 else None, cfg=c, s=s, changed=ch)
json.dump(sel, open(f"{S5}/logs/selectionE1b.json", "w"), indent=1)
json.dump(sw, open(f"{S5}/logs/sweepE1b.json", "w"), indent=1)
open(f"{S5}/jobs/confirmE1b.txt", "w").write("\n".join(J) + ("\n" if J else ""))
print("selected", len(sel), "changed", sum(x["changed"] for x in sel.values()), "confirm jobs", len(J))
