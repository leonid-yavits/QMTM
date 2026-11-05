"""Study 1 statistics (PROTOCOL.md). Unit: seed (101-105), test accuracy. Welch t-test per (dataset, C) cell,
Holm correction within each contrast over all cells. Sparsity: Spearman correlation of the mean gain with density and with
active features per sample (training split), over cells. Writes logs/stats1.json and prints the tables.
python3 stats1.py"""
import json, os, collections, numpy as np
from scipy import stats
ARMS = ["TM", "TMR", "QM", "QMR", "QP", "QPR"]
CON = [("TMR", "TM"), ("QM", "TM"), ("QP", "TM"), ("QMR", "TMR"), ("QPR", "TMR"), ("bestQM", "bestNonQM")]
S = json.load(open("logs/selection.json"))
A = collections.defaultdict(dict)
for l in open("logs/confirm.jsonl"):
    d = json.loads(l)
    A[(d["ds"], d["C"], d["cfg"], d["s"])][d["seed"]] = d["test"]
# density cache (training split)
DC = "logs/density.json"; dens = json.load(open(DC)) if os.path.exists(DC) else {}
for k in S:
    ds = k.split("|")[0]
    if ds not in dens:
        P = {"mnist": "/home/claude/data/proc/mnist.npz", "fmnist": "/home/claude/data/proc/fmnist.npz", "adult": "/home/claude/data/proc/adult.npz",
             "imdb": "/home/claude/data/proc/imdb.npz"}.get(ds, f"/home/claude/data/study1/{ds}.npz")
        X = np.load(P)["X_train"]
        dens[ds] = dict(features=int(X.shape[1]), density=float(X.mean()), active=float(X.sum(1).mean()))
        json.dump(dens, open(DC, "w"), indent=1)


def acc(k, arm):
    ds, C = k.split("|"); a = S[k][S[k][arm]] if arm in ("bestQM", "bestNonQM") else S[k][arm]
    r = A.get((ds, int(C), a["cfg"], float(a["s"])), {})
    return np.array([r[s] for s in range(101, 106) if s in r])


res = {}
for a, b in CON:
    rows = []
    for k in S:
        x, y = acc(k, a), acc(k, b)
        if len(x) < 5 or len(y) < 5:
            continue
        if np.allclose(x, y):
            p = 1.0
        else:
            p = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
        rows.append(dict(cell=k, a=S[k][a] if a == "bestQM" else a, b=S[k][b] if b == "bestNonQM" else b,
                         ma=100 * x.mean(), sa=100 * x.std(ddof=1), mb=100 * y.mean(), sb=100 * y.std(ddof=1), diff=100 * (x.mean() - y.mean()), p=p))
    m = len(rows); order = sorted(range(m), key=lambda i: rows[i]["p"]); run = 0.0
    for r_, i in enumerate(order):                      # Holm step-down adjusted p-values
        run = max(run, min(1.0, (m - r_) * rows[i]["p"])); rows[i]["p_holm"] = run
    ds_ = [dens[r["cell"].split("|")[0]] for r in rows]
    sp_d = stats.spearmanr([d["density"] for d in ds_], [r["diff"] for r in rows]) if m > 2 else None
    sp_a = stats.spearmanr([d["active"] for d in ds_], [r["diff"] for r in rows]) if m > 2 else None
    name = f"{a}-{b}"
    res[name] = dict(rows=rows, n=m, wins=int(sum(bool(r["diff"] > 0 and r["p_holm"] < 0.05) for r in rows)),
                     losses=int(sum(bool(r["diff"] < 0 and r["p_holm"] < 0.05) for r in rows)),
                     spearman_density=None if sp_d is None else [float(sp_d.statistic), float(sp_d.pvalue)],
                     spearman_active=None if sp_a is None else [float(sp_a.statistic), float(sp_a.pvalue)])
    print(f"\n== {name}: {m} cells, significant wins {res[name]['wins']}, losses {res[name]['losses']} (Holm, 0.05)")
    if sp_d is not None:
        print(f"   Spearman(gain, density) rho={sp_d.statistic:+.3f} p={sp_d.pvalue:.3g}; Spearman(gain, active/sample) rho={sp_a.statistic:+.3f} p={sp_a.pvalue:.3g}")
    for r in sorted(rows, key=lambda r: dens[r["cell"].split("|")[0]]["density"]):
        print(f"   {r['cell']:20s} {r['a']:>4s} {r['ma']:6.2f}±{r['sa']:4.2f}  {r['b']:>4s} {r['mb']:6.2f}±{r['sb']:4.2f}  d={r['diff']:+6.2f}  p={r['p']:.3g}  pH={r['p_holm']:.3g}"
              f"  dens={100 * dens[r['cell'].split('|')[0]]['density']:.3f}%")
json.dump(dict(contrasts=res, density=dens), open("logs/stats1.json", "w"), indent=1)
