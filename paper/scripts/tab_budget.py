"""Table: clause budget the vanilla TM needs to reach the rule arm at C = 5 and C = 10 (means of seeds 101-103).
Rule arm per dataset from the study-1 density rule (R1 = TMR below 3 % training density, QM otherwise).
Two vanilla baselines: the original TM (study 2 part B) and the re-tuned TM (study 4, logs/sweep4.json).
Writes tex/tab_budget.tex and tex/numbers_budget.tex."""
import json
B = json.load(open("/home/claude/qmtm/study2/logs/stats2.json"))["B"]
R = json.load(open("/home/claude/qmtm/study4/logs/sweep4.json"))
D = json.load(open("/home/claude/qmtm/study1/logs/density.json"))
N = json.load(open("/home/claude/qmtm/paper2/data/names.json"))
DS = ["r52_V0", "r8_V5000", "r52_V5000", "r52_V1000", "imdb", "adult", "mnist", "fmnist"]
CS = [5, 10, 20, 40, 80, 160]
TMre = {ds: {C: (R[ds][str(C)]["acc"] if R[ds][str(C)] else None) for C in CS} for ds in DS}
TMor = {ds: {C: B[ds]["TM"][str(C)] for C in CS} for ds in DS}


def need(T, target):
    for C in CS:
        if T[C] is None:
            return "?"
        if T[C] >= target:
            return C
    return None


def cell(n):
    return "?" if n == "?" else (str(n) if n else "$>$160")


L = [r"\begin{tabular}{lrcrrrrrr}", r"\toprule",
     r" & Dens. & Rule & \multicolumn{3}{c}{Rule arm, $C = 5$} & \multicolumn{3}{c}{Rule arm, $C = 10$} \\",
     r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
     r" & & & & \multicolumn{2}{c}{TM $C$ needed} & & \multicolumn{2}{c}{TM $C$ needed} \\",
     r"\cmidrule(lr){5-6}\cmidrule(lr){8-9}",
     r"Dataset & (\%) & arm & Acc. & orig. & re-t. & Acc. & orig. & re-t. \\", r"\midrule"]
M, ratios = {}, {"orig5": [], "re5": [], "orig10": [], "re10": []}
for ds in DS:
    dens = 100 * D[ds]["density"]; arm = "TMR" if dens < 3 else "QM"
    cells = []
    for C0 in (5, 10):
        a = B[ds][arm][str(C0)]; no, nr = need(TMor[ds], a), need(TMre[ds], a)
        cells += [f"{a:.2f}", cell(no), cell(nr)]
        ratios[f"orig{C0}"].append(no if no in ("?", None) else no // C0)
        ratios[f"re{C0}"].append(nr if nr in ("?", None) else nr // C0)
    L.append(f"{N[ds]} & {dens:.2f} & {'R1' if arm == 'TMR' else 'QM'} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open("/home/claude/qmtm/paper2/tex/tab_budget.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(dict(zip(DS, zip(*ratios.values()))))
print("TM re-tuned:", {ds: [None if v is None else round(v, 2) for v in TMre[ds].values()] for ds in DS})
