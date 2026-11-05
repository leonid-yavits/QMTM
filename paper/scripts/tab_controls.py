"""Table and macros for the baseline controls (study 4, part D1). Reads study4/logs/stats4.json and selection4.json.
Writes tex/tab_controls.tex and tex/numbers_controls.tex. Sparse: training density below 3 %."""
import json, collections, numpy as np
from scipy import stats
O = json.load(open("/home/claude/qmtm/study4/logs/stats4.json"))
SEL = json.load(open("/home/claude/qmtm/study4/logs/selection4.json"))
D = json.load(open("/home/claude/qmtm/study1/logs/density.json"))
dens = lambda r: r["density"] if r.get("density") is not None else D[r["cell"].split("|")[0]]["density"]


def part(rows, lo, hi):
    R = [r for r in rows if lo <= dens(r) < hi]
    return dict(n=len(R), w=sum(r["diff"] > 0 and r["p_holm"] < .05 for r in R), l=sum(r["diff"] < 0 and r["p_holm"] < .05 for r in R),
                pos=sum(r["diff"] > 0 for r in R), med=float(np.median([r["diff"] for r in R])) if R else float("nan"))


def fmt(x):
    return f"{x:+.2f}".replace("-", "$-$")


M = {}
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r" & & \multicolumn{2}{c}{All cells} & \multicolumn{2}{c}{Sparse} & \multicolumn{2}{c}{Dense} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Contrast & Baseline & W/L & Med. & W/L & Med. & W/L & Med. \\", r"\midrule"]
rows_def = [("S1", "TMR", "TMR $-$ TM", "R"), ("S1", "QM", "QM $-$ TM", "Q"), ("S1", "QP", "QP $-$ TM", "P"),
            ("S1", "bestQM", "Best QM $-$ best other", "B"), ("S2C", "rule", "Rule arm $-$ TM", "rule"),
            ("S2C", "TMR", "TMR $-$ TM", "SR"), ("S2C", "QM", "QM $-$ TM", "SQ")]
first2 = True
for st, c, lab, key in rows_def:
    if st == "S2C" and first2:
        L += [r"\midrule", r"\multicolumn{8}{l}{\emph{Study~2, six new datasets, 12 cells}} \\"]; first2 = False
    if st == "S1" and c == "TMR":
        L += [r"\multicolumn{8}{l}{\emph{Study~1, 48 cells}} \\"]
    for base, bl in (("TM", "original"), ("TMext", "extended")):
        rows = O[f"{st}:{c}-{base}"]["rows"]
        a, s, d = part(rows, 0, 1), part(rows, 0, .03), part(rows, .03, 1)
        L.append(f"{lab if base == 'TM' else ''} & {bl} & {a['w']}/{a['l']} & {fmt(a['med'])} & {s['w']}/{s['l']} & {fmt(s['med'])} & {d['w']}/{d['l']} & {fmt(d['med'])} \\\\")
        tag = "orig" if base == "TM" else "ext"
        for nm, v in (("all", a), ("sparse", s), ("dense", d)):
            M[f"ctl{key}{tag}{nm}W"] = v["w"]; M[f"ctl{key}{tag}{nm}L"] = v["l"]; M[f"ctl{key}{tag}{nm}Pos"] = v["pos"]
            M[f"ctl{key}{tag}{nm}N"] = v["n"]; M[f"ctl{key}{tag}{nm}Med"] = f"{v['med']:+.2f}".replace("-", "$-$")
L += [r"\bottomrule", r"\end{tabular}"]
open("/home/claude/qmtm/paper2/tex/tab_controls.tex", "w").write("\n".join(L) + "\n")

# Study 1: the density rule applied to study-1 cells against TM_ext (TMR if sparse, QM if dense)
S1rows = []
for c, lo, hi in (("TMR", 0, .03), ("QM", .03, 1)):
    S1rows += [r for r in O[f"S1:{c}-TMext"]["rows"] if lo <= dens(r) < hi]
r1 = dict(w=sum(r["diff"] > 0 and r["p_holm"] < .05 for r in S1rows), l=sum(r["diff"] < 0 and r["p_holm"] < .05 for r in S1rows),
          pos=sum(r["diff"] > 0 for r in S1rows), n=len(S1rows))
M.update(ctlSoneRuleW=r1["w"], ctlSoneRuleL=r1["l"], ctlSoneRulePos=r1["pos"])
# rule, study 2, sign test vs TM_ext
e = O["S2C:rule-TMext"]
M.update(ctlruleSignP=f"{e['sign_p']:.2f}", ctlrulePos=e["positive"])
# extended-TM selections
for t, nm in (("D1_s1", "One"), ("D1_s2", "Two")):
    v = SEL[t].values()
    M[f"ctlChanged{nm}"] = sum(x["changed"] for x in v); M[f"ctlCells{nm}"] = len(SEL[t])
    M[f"ctlSlow{nm}"] = sum(x["s"] == 1.5 for x in v); M[f"ctlLthree{nm}"] = sum(x["cfg"] == "AND_L32" for x in v)
def nlines(f):
    try:
        return sum(1 for _ in open(f"/home/claude/qmtm/study4/logs/{f}"))
    except FileNotFoundError:
        return 0
M["ctlTuneRuns"] = f"{nlines('tuneD1_s1.jsonl') + nlines('tuneD1_s2.jsonl') + nlines('tuneD2.jsonl'):,}"
M["ctlTestRuns"] = f"{nlines('confirmD1.jsonl') + nlines('confirmD2.jsonl'):,}"
open("/home/claude/qmtm/paper2/tex/numbers_controls.tex", "w").write(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print("\n".join(L)); print({k: v for k, v in M.items() if "all" in k or "ctlS" in k or "ctlC" in k or "rule" in k})
