"""Table: clause budget the vanilla TM needs to reach the rule arm at C = 5 and C = 10 (study 2 part B, seeds 101-103 means).
Rule arm per dataset from the study-1 density rule (R1 = TMR below 3 % training density, QM otherwise). Writes tex/tab_budget.tex."""
import json
B = json.load(open("/home/claude/qmtm/study2/logs/stats2.json"))["B"]
D = json.load(open("/home/claude/qmtm/study1/logs/density.json"))
N = json.load(open("/home/claude/qmtm/paper2/data/names.json"))
DS = ["r52_V0", "r8_V5000", "r52_V5000", "r52_V1000", "imdb", "adult", "mnist", "fmnist"]
CS = [5, 10, 20, 40, 80, 160]


def need(ds, target):
    for C in CS:
        v = B[ds]["TM"][str(C)]
        if v is not None and v >= target:
            return C
    return None


L = [r"\begin{tabular}{lrcrrrrr}", r"\toprule",
     r" & Dens. & Rule & \multicolumn{2}{c}{Rule, $C = 5$} & \multicolumn{2}{c}{Rule, $C = 10$} & TM \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Dataset & (\%) & arm & Acc. & TM $C$ & Acc. & TM $C$ & $C{=}160$ \\", r"\midrule"]
ratios = []
for ds in DS:
    dens = 100 * D[ds]["density"]; arm = "TMR" if dens < 3 else "QM"
    cells = []
    for C0 in (5, 10):
        a = B[ds][arm][str(C0)]; n = need(ds, a)
        cells += [f"{a:.2f}", f"{n} ({n // C0}$\\times$)" if n else f"$>$160 ($>$16$\\times$)"]
        ratios.append(n // C0 if n else None)
    L.append(f"{N[ds]} & {dens:.2f} & {'R1' if arm == 'TMR' else 'QM'} & " + " & ".join(cells) + f" & {B[ds]['TM']['160']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("/home/claude/qmtm/paper2/tex/tab_budget.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(ratios)
