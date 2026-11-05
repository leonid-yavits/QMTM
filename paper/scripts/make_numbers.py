"""Key study-1 numbers as LaTeX macros (generated, not typed)."""
import json, numpy as np
R = json.load(open("/home/claude/qmtm/study1/logs/stats1.json")); D = R["density"]; C = R["contrasts"]
H = json.load(open("/home/claude/qmtm/hw/qmtm_hw.json"))
m = {}
names = {"TMR-TM": "R", "QM-TM": "Q", "QP-TM": "P", "QMR-TMR": "QR", "QPR-TMR": "PR", "bestQM-bestNonQM": "B"}
for k, n in names.items():
    c = C[k]; m[f"{n}win"] = c["wins"]; m[f"{n}loss"] = c["losses"]
    m[f"{n}rho"] = f"{c['spearman_density'][0]:+.2f}".replace("-", "$-$"); m[f"{n}rhop"] = f"{c['spearman_density'][1]:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
    sp = [r["diff"] for r in c["rows"] if D[r["cell"].split("|")[0]]["density"] < 0.03]; de = [r["diff"] for r in c["rows"] if D[r["cell"].split("|")[0]]["density"] >= 0.03]
    f = lambda v: f"{v:+.2f}".replace("-", "$-$")
    m[f"{n}medsparse"] = f(np.median(sp)); m[f"{n}meddense"] = f(np.median(de))
    m[f"{n}possparse"] = sum(x > 0 for x in sp); m[f"{n}posdense"] = sum(x > 0 for x in de)
    m[f"{n}maxgain"] = f"{max(r['diff'] for r in c['rows']):.2f}"; m[f"{n}mingain"] = f"{min(r['diff'] for r in c['rows']):.2f}".replace("-", "$-$")
m["nsparse"] = len(sp); m["ndense"] = len(de)
hw = [v for ds in H.values() for k, v in ds.items() if isinstance(v, dict)]
for cfg in ["L32", "L64", "QM90_L32", "QM95_L64"]:
    vals = [H[ds][f"{cfg}_s10.0"]["train_overhead_pct"] for ds in H] + [H[ds][f"{cfg}_s2.5"]["train_overhead_pct"] for ds in H]
    m[f"hw{cfg.replace('_','')}min"] = f"{min(vals):.1f}"; m[f"hw{cfg.replace('_','')}max"] = f"{max(vals):.1f}"
area = [H[ds][f"{k}_s10.0"]["area_overhead_pct"] for ds in H for k in ["L32", "L64", "QM90_L32", "QM95_L32", "QM95_L64"]]
m["hwareamax"] = f"{max(area):.3f}"; m["hwareamin"] = f"{min(area):.4f}"
alpha = "abcdefghijklmnopqrstuvwxyz"
def macro(k):
    out = ""
    for ch in k:
        out += ch if ch.isalpha() else ""
    return out
L = []
for k, v in m.items():
    nm = "num" + "".join(ch for ch in k if ch.isalpha()) + "".join(alpha[int(ch)] for ch in k if ch.isdigit())
    L.append(f"\\newcommand{{\\{nm}}}{{{v}}}")
open("/home/claude/qmtm/paper2/tex/numbers.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
