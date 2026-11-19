"""Fig.: FPTM test accuracy against the number of vote levels, 6 clauses per class, nine datasets.
E2: FPTM trained with L_F in {1, ..., 64} (seeds 101-103).
E1: the selected FPTM models of the C = 5 cells (L_F from select6.py, seeds 101-105) with every clause score clipped at Q
    at inference (Q threshold searches per clause).
Reference: best Study 1 arm with 5 clauses per class (Table "FPTM baseline" of the paper; Study 2 part A summaries in the
supplementary FPTM package, study3/logs/stats3.json).
Shaded: L_F <= 4, where the class sum (3 positive clauses, at most 3 L_F) cannot reach the margin T = 16."""
import json, os, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
OUT = os.environ.get("FIGDIR", os.path.join(HERE, "figures"))
N = {"adult": "UCI Adult", "mnist": "MNIST", "fmnist": "Fashion-MNIST", "imdb": "IMDb", "har": "UCI HAR", "r8_V5000": "R8 ($V$=5k)",
     "r52_V1000": "R52 ($V$=1k)", "news20_V1000": "20NG ($V$=1k)", "sector_V1000": "Sector ($V$=1k)"}
B1 = {"adult|5": 80.60, "mnist|5": 84.19, "fmnist|5": 74.71, "imdb|5": 80.62, "har|5": 90.60, "r8_V5000|5": 88.81,
      "r52_V1000|5": 81.04, "news20_V1000|5": 48.51, "sector_V1000|5": 18.02}
E2, E1 = {}, {}
for l in open(os.path.join(HERE, "logs", "E2_levels.jsonl")):
    d = json.loads(l); E2.setdefault(d["ds"], {}).setdefault(d["LF"], []).append(100 * d["test"])
for l in open(os.path.join(HERE, "logs", "E1_confirm.jsonl")):
    d = json.loads(l)
    if d["clauses_fptm"] != 6: continue
    for Q, a in zip(d["stats"]["q"], d["stats"]["q_test"]):
        E1.setdefault(d["ds"], {}).setdefault(Q, []).append(100 * a)
    E1[d["ds"]]["LF"] = d["LF"]
DS = ["adult", "mnist", "fmnist", "imdb", "har", "r8_V5000", "r52_V1000", "news20_V1000", "sector_V1000"]
X = [1, 2, 4, 8, 16, 32, 64]
C2, C1, CR = "#2a78d6", "#eb6834", "#3d3d3d"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.16, 5.4))
gs = GridSpec(4, 3, height_ratios=[0.12, 1, 1, 1], hspace=0.6, wspace=0.28, figure=fig)
lax = fig.add_subplot(gs[0, :]); lax.axis("off")
for i, ds in enumerate(DS):
    ax = fig.add_subplot(gs[1 + i // 3, i % 3])
    ax.axvspan(0.8, 4.0 * 1.12, color="#eeeeee", lw=0, zorder=0)
    lf = sorted(E2[ds]); m = [st.mean(E2[ds][x]) for x in lf]; s = [st.stdev(E2[ds][x]) for x in lf]
    ax.errorbar(lf, m, yerr=s, color=C2, marker="o", ms=3.2, lw=1.2, capsize=1.5, elinewidth=0.6,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    q = sorted(k for k in E1[ds] if k != "LF"); mq = [st.mean(E1[ds][x]) for x in q]; sq = [st.stdev(E1[ds][x]) for x in q]
    ax.errorbar(q, mq, yerr=sq, color=C1, marker="^", ms=3.2, lw=1.2, capsize=1.5, elinewidth=0.6,
                markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    ax.axhline(B1[f"{ds}|5"], color=CR, lw=0.9, ls=":", zorder=2)
    ax.set_xscale("log", base=2); ax.set_xticks(X); ax.set_xticklabels([str(x) for x in X]); ax.minorticks_off()
    ax.set_xlim(0.8, 80)
    ax.set_title(f"{N[ds]}, selected $L_F$ = {E1[ds]['LF']}", fontsize=7.5, pad=3)
    ax.grid(True, color="#e6e6e6", lw=0.5, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if i % 3 == 0: ax.set_ylabel("Test accuracy (%)")
    if i // 3 == 2: ax.set_xlabel("Vote levels per clause ($L_F$ or $Q$)")
h = [Line2D([], [], color=C2, marker="o", ms=3.5, lw=1.2, markeredgecolor="white", markeredgewidth=0.4, label="FPTM trained with $L_F$ (E2)"),
     Line2D([], [], color=C1, marker="^", ms=3.5, lw=1.2, markeredgecolor="white", markeredgewidth=0.4, label="Selected FPTM, score clipped at $Q$ (E1)"),
     Line2D([], [], color=CR, lw=0.9, ls=":", label="Best Study 1 arm, $C$ = 5"),
     Patch(facecolor="#eeeeee", edgecolor="none", label="E2: class sum below $T$")]
lax.legend(handles=h, loc="center", ncol=4, frameon=False, fontsize=7.2, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)
os.makedirs(OUT, exist_ok=True)
for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    fig.savefig(os.path.join(OUT, f"fig_levels.{ext}"), bbox_inches="tight", metadata=({"CreationDate": None} if ext == "pdf" else {}), **kw)
