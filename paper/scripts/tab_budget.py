"""Table: clause budget the vanilla TM needs to reach the rule arm at C = 5 and C = 10 (means of seeds 101-103).
Rule arm per dataset from the study-1 density rule (R1 = TMR below 3 % training density, QM otherwise).
Two vanilla baselines: the original TM (study 2 part B) and the re-tuned TM (study 4, logs/sweep4.json).
Per dataset, a second line (blue) uses the re-tuned rule arm (study 5, E1 selection; see sweep5.py).
Usage: python3 tab_budget.py [OUTDIR]  (writes OUTDIR/tab_budget.tex)."""
import json, sys
from sweep5 import series, CS
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/paper_src"
S = series()
D = json.load(open("/home/claude/qmtm/study1/logs/density.json"))
N = json.load(open("/home/claude/qmtm/paper/data/names.json"))
DS = ["r52_V0", "r8_V5000", "r52_V5000", "r52_V1000", "imdb", "adult", "mnist", "fmnist"]


def need(T, target):
    for C in CS:
        if T[C] is None:
            return "?"
        if T[C] >= target:
            return C
    return None


def cell(n):
    return "?" if n == "?" else (str(n) if n else "$>$160")


def blue(x):
    return "{\\color{blue}" + x + "}" if x else x


L = [r"\begin{tabular}{lrcrrrrrr}", r"\toprule",
     r" & Dens. & Rule & \multicolumn{3}{c}{Rule arm, $C = 5$} & \multicolumn{3}{c}{Rule arm, $C = 10$} \\",
     r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
     r" & & & & \multicolumn{2}{c}{TM $C$ needed} & & \multicolumn{2}{c}{TM $C$ needed} \\",
     r"\cmidrule(lr){5-6}\cmidrule(lr){8-9}",
     r"Dataset & (\%) & arm & Acc. & orig. & re-t. & Acc. & orig. & re-t. \\", r"\midrule"]
ratios = {}
for ds in DS:
    dens = 100 * D[ds]["density"]; arm = "TMR" if dens < 3 else "QM"
    for k, tag in ((arm, "orig"), (arm + "5", "re")):
        cells = []
        for C0 in (5, 10):
            a = S[ds][k][C0]
            if a is None:
                cells += ["--", "--", "--"]; ratios[(ds, tag, C0)] = None; continue
            no, nr = need(S[ds]["TM"], a), need(S[ds]["TMre"], a)
            cells += [f"{a:.2f}", cell(no), cell(nr)]
            ratios[(ds, tag, C0)] = (a, no, nr)
        if tag == "orig":
            L.append(f"{N[ds]} & {dens:.2f} & {'R1' if arm == 'TMR' else 'QM'} & " + " & ".join(cells) + r" \\")
        else:
            L.append(" & ".join([blue(r"\quad re-tuned"), "", ""] + [blue(c) for c in cells]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_budget.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
for k, v in ratios.items():
    if v:
        a, no, nr = v
        print(k, f"{a:.2f}", "orig F", no if no in ("?", None) else no // k[2], "re-t F", nr if nr in ("?", None) else nr // k[2])
