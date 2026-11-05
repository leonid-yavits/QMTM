"""Fig.: clause-budget sweep. Mean test accuracy (seeds 101-103) against clauses per class C, per dataset.
TMR, QM and the original TM: C = 5, 10, 20 from the study-1 confirmations, C = 40, 80, 160 from study 2 part B.
Re-tuned TM: study 4 (C <= 20: extended family TM_ext; C >= 40: re-tuned per budget). Missing points are skipped."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
B = json.load(open("/home/claude/qmtm/study2/logs/stats2.json"))["B"]
R = json.load(open("/home/claude/qmtm/study4/logs/sweep4.json"))
for ds in R:
    B[ds]["TMre"] = {C: (v["acc"] if v else None) for C, v in R[ds].items()}
N = json.load(open("/home/claude/qmtm/paper2/data/names.json"))
DS = ["adult", "mnist", "fmnist", "imdb", "r8_V5000", "r52_V1000", "r52_V5000", "r52_V0"]
CS = [5, 10, 20, 40, 80, 160]
STY = [("TM", "TM (original grid)", "#a8a8a8", "o", "--"), ("TMre", "TM (re-tuned)", "#3d3d3d", "D", "-"), ("TMR", "TMR (R1)", "#2a78d6", "s", "-"), ("QM", "QM", "#eb6834", "^", "-")]
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.16, 3.6))
gs = GridSpec(3, 4, height_ratios=[0.12, 1, 1], hspace=0.55, wspace=0.32, figure=fig)
lax = fig.add_subplot(gs[0, :]); lax.axis("off")
for i, ds in enumerate(DS):
    ax = fig.add_subplot(gs[1 + i // 4, i % 4])
    for arm, lab, col, mk, ls in STY:
        pts = [(C, B[ds][arm][str(C)]) for C in CS if B[ds][arm].get(str(C)) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=3.2, lw=1.2, ls=ls,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks(CS); ax.set_xticklabels([str(c) for c in CS])
    ax.minorticks_off()
    ax.set_title(N[ds], fontsize=7.5, pad=3)
    ax.grid(True, color="#e6e6e6", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if i % 4 == 0:
        ax.set_ylabel("Test accuracy (%)")
    if i // 4 == 1:
        ax.set_xlabel("Clauses per class $C$")
h = [Line2D([], [], color=c, marker=m, ms=3.5, lw=1.2, ls=ls, markeredgecolor="white", markeredgewidth=0.4, label=l) for _, l, c, m, ls in STY]
lax.legend(handles=h, loc="center", ncol=4, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.6, borderaxespad=0)
for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    fig.savefig(f"/home/claude/qmtm/paper2/figures/fig_sweep.{ext}", bbox_inches="tight", metadata=({"CreationDate": None} if ext == "pdf" else {}), **kw)
