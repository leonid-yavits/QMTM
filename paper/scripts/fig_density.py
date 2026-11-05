"""Fig.: study-1 test gain versus input density, per (dataset, C) cell. R1: TMR - TM. QM: best QM-family arm - best non-QM arm.
Filled markers: Holm-significant (0.05, 48 cells within contrast)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
R = json.load(open("/home/claude/qmtm/study1/logs/stats1.json")); D = R["density"]; C = R["contrasts"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(3.5, 2.75))
gs = GridSpec(2, 1, height_ratios=[0.12, 1], hspace=0.05, figure=fig)
lax, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
sty = [("TMR-TM", "R1 (TMR $-$ TM)", "#2a78d6", "o"), ("bestQM-bestNonQM", "QM (best QM arm $-$ best non-QM arm)", "#eb6834", "s")]
h = []
for key, lab, col, mk in sty:
    xs = [100 * D[r["cell"].split("|")[0]]["density"] for r in C[key]["rows"]]; ys = [r["diff"] for r in C[key]["rows"]]
    sig = [r["p_holm"] < 0.05 for r in C[key]["rows"]]
    ax.scatter([x for x, s in zip(xs, sig) if not s], [y for y, s in zip(ys, sig) if not s], s=14, marker=mk, facecolors="white", edgecolors=col, linewidths=0.8, zorder=3)
    h.append(ax.scatter([x for x, s in zip(xs, sig) if s], [y for y, s in zip(ys, sig) if s], s=14, marker=mk, color=col, edgecolors="white", linewidths=0.4, zorder=4, label=lab))
ax.axhline(0, color="#888", lw=0.6, zorder=1); ax.axvline(3, color="#888", lw=0.6, ls="--", zorder=1)
ax.set_xscale("log"); ax.set_xlabel("Training-set input density (%)"); ax.set_ylabel("Test accuracy gain (pp)")
ax.grid(True, which="major", color="#e6e6e6", lw=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
lax.axis("off")
lax.legend(handles=h, loc="center", ncol=1, frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0)
fig.savefig("/home/claude/qmtm/paper2/figures/fig_density.pdf", bbox_inches="tight"); fig.savefig("/home/claude/qmtm/paper2/figures/fig_density.png", dpi=200, bbox_inches="tight")
