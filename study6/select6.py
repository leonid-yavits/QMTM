"""Study 6 E1 selection: per cell, the L_F with the highest seed-1 validation accuracy at the last epoch; ties go to the smaller L_F
(fewer threshold searches). Writes logs/selection6.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
best = {}
for l in open(f"{HERE}/logs/E1_tune.jsonl"):
    d = json.loads(l)
    C = {6: 5, 10: 10}[d["clauses_fptm"]]
    k = f"{d['ds']}|{C}"
    cur = best.get(k)
    if cur is None or d["val"] > cur["val"] or (d["val"] == cur["val"] and d["LF"] < cur["LF"]):
        best[k] = dict(LF=d["LF"], val=d["val"], used_features=d["stats"]["used_features"], nf=d["stats"]["nf"])
assert len(best) == 18, sorted(best)
json.dump(best, open(f"{HERE}/logs/selection6.json", "w"), indent=1)
for k in sorted(best):
    print(k, best[k])
