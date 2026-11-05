"""Table: out-of-sample density rule (study 2 part C). Writes tex/tab_rule.tex and appends rule macros to tex/numbers_rule.tex."""
import json
R = json.load(open("/home/claude/qmtm/study2/logs/stats2.json"))
N = json.load(open("/home/claude/qmtm/paper2/data/names.json"))
rule = {r["cell"]: r for r in R["C:rule-TM"]["rows"]}
oth = {}
for con in ("C:TMR-TM", "C:QM-TM"):
    for r in R[con]["rows"]:
        if r["arm"] != rule[r["cell"]]["arm"]:
            oth[r["cell"]] = r
ARM = {"TMR": "R1", "QM": "QM"}
f = lambda d, p: ((r"\textbf{%+.2f}" if p < 0.05 else "%+.2f") % d).replace("-", "$-$")
L = [r"\begin{tabular}{lrcrrrr}", r"\toprule",
     r"Dataset & Dens. & $C$ & TM & Rule & $\Delta$ rule & $\Delta$ other \\",
     r" & (\%) & & (\%) & (\%) & (pp) & (pp) \\", r"\midrule"]
for k in sorted(rule, key=lambda k: (rule[k]["density"], int(k.split("|")[1]))):
    r, o = rule[k], oth[k]; ds, C = k.split("|")
    L.append(f"{N[ds]} & {100 * r['density']:.2f} & {C} & {r['mb']:.2f} & {ARM[r['arm']]} {r['ma']:.2f} & {f(r['diff'], r['p_holm'])} & {f(o['diff'], o['p_holm'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("/home/claude/qmtm/paper2/tex/tab_rule.tex", "w").write("\n".join(L) + "\n")
c = R["C:rule-TM"]; t = R["C:TMR-TM"]; q = R["C:QM-TM"]
wrong_losses = sum(o["diff"] < 0 and o["p_holm"] < 0.05 for o in oth.values())
wrong_wins = sum(o["diff"] > 0 and o["p_holm"] < 0.05 for o in oth.values())
M = {"ruleN": c["n"], "ruleWin": c["wins"], "ruleLoss": c["losses"], "rulePos": c["positive"], "ruleSignP": f"{c['sign_p']:.3f}",
     "otherWin": wrong_wins, "otherLoss": wrong_losses}
open("/home/claude/qmtm/paper2/tex/numbers_rule.tex", "w").write("".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in M.items()))
print(M)
