"""Study 1 selection (PROTOCOL.md). Per (dataset, C, arm): the (cfg, s) with the highest seed-1 validation accuracy
after 40 epochs (ties: smaller s, then first listed cfg). Best QM-family arm and best non-QM arm are chosen on validation
(ties: arm list order). Writes logs/selection.json and jobs/confirm.txt (seeds 101-105) for complete (dataset, C) cells only.
python3 select.py [expected_runs_per_cell_file]"""
import json, collections
ARMS = collections.OrderedDict([
    ("TM", ["AND", "AND_L64"]), ("TMR", ["AND_L32_fb1", "AND_L64_fb1"]),
    ("QM", ["QM90_L32_fb0", "QM95_L32_fb0", "QM95_L64_fb0"]), ("QMR", ["QM90_L32_fb1", "QM95_L32_fb1"]),
    ("QP", ["QP1_L64_fb0", "QP10_L64_fb0"]), ("QPR", ["QP1_L64_fb1", "QP10_L64_fb1"])])
QMF, NONQM = ["QM", "QMR", "QP", "QPR"], ["TM", "TMR"]
# expected tuning runs per (dataset, C): from the job lists (so a partially tuned cell is never selected)
exp = collections.Counter()
for f in ["jobs/tune1.txt", "jobs/tune2.txt", "jobs/tune3.txt"]:
    for l in open(f):
        a = l.split("train.py")[1].split(); exp[(a[0], int(a[1]))] += 1
R = collections.defaultdict(dict)
for l in open("logs/tune.jsonl"):
    d = json.loads(l)
    if d["seed"] == 1 and d["epochs"] == 40:
        R[(d["ds"], d["C"])][(d["cfg"], d["s"])] = d
sel, jobs, incomplete = {}, [], []
for cell in sorted(exp):
    got = R.get(cell, {})
    if len(got) < exp[cell]:
        incomplete.append([*cell, len(got), exp[cell]]); continue
    ds, C = cell; T = next(iter(got.values()))["T"]; out = {}
    for arm, cfgs in ARMS.items():
        cand = [(got[(c, s)]["val"], -s, -i, c, s) for i, c in enumerate(cfgs) for (cc, s) in got if cc == c]
        v, _, _, c, s = max(cand)
        out[arm] = dict(cfg=c, s=s, val=v)
    order = list(ARMS)
    out["bestQM"] = max(QMF, key=lambda a: (out[a]["val"], -order.index(a)))
    out["bestNonQM"] = max(NONQM, key=lambda a: (out[a]["val"], -order.index(a)))
    sel[f"{ds}|{C}"] = dict(T=T, **out)
    for arm in ARMS:
        for sd in range(101, 106):
            jobs.append(f"cd /home/claude/qmtm/study1 && python3 train.py {ds} {C} {T} {out[arm]['s']} {out[arm]['cfg']} {sd} 40 logs/confirm.jsonl")
json.dump(sel, open("logs/selection.json", "w"), indent=1)
open("jobs/confirm.txt", "w").write("\n".join(dict.fromkeys(jobs)) + "\n")
print(f"selected cells {len(sel)}, confirmation jobs {len(dict.fromkeys(jobs))}, incomplete cells {len(incomplete)}")
for x in incomplete:
    print("  incomplete", x)
