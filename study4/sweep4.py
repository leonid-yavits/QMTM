"""Study 4, part D2 summary: re-tuned vanilla TM in the budget sweep (8 encodings, C = 5 ... 160).
C <= 20: the extended-TM selection of part D1 (TM_ext), test seeds 101-103.
C >= 40: the part-D2 selection on seed-1 validation, test seeds 101-103 (original sweep runs if unchanged).
Mean test accuracy in %. None if any seed is missing. Writes logs/sweep4.json."""
import json, collections
S1, S2, S4 = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2", "/home/claude/qmtm/study4"
DS = ["adult", "mnist", "fmnist", "imdb", "r8_V5000", "r52_V1000", "r52_V5000", "r52_V0"]
A = collections.defaultdict(dict)
for p in (f"{S1}/logs/confirm.jsonl", f"{S2}/logs/sweep.jsonl", f"{S4}/logs/confirmD1.jsonl", f"{S4}/logs/confirmD2.jsonl"):
    try:
        for l in open(p):
            d = json.loads(l); A[(d["ds"], int(d["C"]), d["cfg"], float(d["s"]))][d["seed"]] = d["test"]
    except FileNotFoundError:
        pass
SEL = json.load(open(f"{S4}/logs/selection4.json"))
out = {}
for ds in DS:
    out[ds] = {}
    for C in (5, 10, 20, 40, 80, 160):
        e = SEL["D1_s1"].get(f"{ds}|{C}") if C <= 20 else SEL["D2"].get(f"{ds}|{C}")
        if e is None:
            out[ds][str(C)] = None; continue
        r = A.get((ds, C, e["cfg"], float(e["s"])), {})
        v = [r[s] for s in (101, 102, 103) if s in r]
        out[ds][str(C)] = dict(acc=100 * sum(v) / 3 if len(v) == 3 else None, cfg=e["cfg"], s=e["s"], changed=e["changed"])
json.dump(out, open(f"{S4}/logs/sweep4.json", "w"), indent=1)
for ds in DS:
    print(ds, " ".join(f"{C}:{'-' if out[ds][C] is None or out[ds][C]['acc'] is None else round(out[ds][C]['acc'], 2)}" for C in out[ds]))
