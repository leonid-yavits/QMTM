"""Study 5 tables and macros (study5/PROTOCOL.md). Reads study5/logs/stats5.json, selection5.json, selectionE1b.json,
sweepE1b.json and study4/logs/stats4.json. Writes, into the paper source directory:
  tab_controls.tex     Table of baseline controls with a third line per contrast, both arms re-tuned (blue).
  tab_noL.tex          Literal-budget ablation (E2).
  numbers_controls5.tex  Macros for the text.
Sparse: training density below 3 %. Holm within each contrast over all cells of the study (as in stats4/stats5)."""
import json, sys, numpy as np
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/paper_src"
Q = "/home/claude/qmtm"
O4 = json.load(open(f"{Q}/study4/logs/stats4.json"))
O5 = json.load(open(f"{Q}/study5/logs/stats5.json"))
SEL5 = json.load(open(f"{Q}/study5/logs/selection5.json"))
D = json.load(open(f"{Q}/study1/logs/density.json"))
SELC = json.load(open(f"{Q}/study2/logs/selection2.json"))["C"]


def dens(r):
    if r.get("density") is not None:
        return r["density"]
    ds = r["cell"].split("|")[0]
    return SELC[r["cell"]]["density"] if r["cell"] in SELC else D[ds]["density"]


def part(rows, lo, hi):
    R = [r for r in rows if lo <= dens(r) < hi]
    return dict(n=len(R), w=sum(r["diff"] > 0 and r["p_holm"] < .05 for r in R),
                l=sum(r["diff"] < 0 and r["p_holm"] < .05 for r in R), pos=sum(r["diff"] > 0 for r in R),
                med=float(np.median([r["diff"] for r in R])) if R else float("nan"))


def fmt(x):
    return "--" if x != x else f"{x:+.2f}".replace("-", "$-$")


def blue(s):
    return "{\\color{blue}" + s + "}" if s else s


M = {}
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r" & & \multicolumn{2}{c}{All cells} & \multicolumn{2}{c}{Sparse} & \multicolumn{2}{c}{Dense} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Contrast & Baseline & W/L & Med. & W/L & Med. & W/L & Med. \\", r"\midrule"]
# (study, study-4 contrast, study-5 contrast or None, label, macro key)
rows_def = [("S1", "TMR", "E1:S1:TMRext-TMext", "TMR $-$ TM", "R"), ("S1", "QM", "E1:S1:QMext-TMext", "QM $-$ TM", "Q"),
            ("S1", "QP", None, "QP $-$ TM", "P"), ("S1", "bestQM", "E1:S1:bestQM-bestNonQM", "Best QM $-$ best other", "B"),
            ("S2C", "rule", "E1:S2C:rule-TMext", "Rule arm $-$ TM", "rule"), ("S2C", "TMR", "E1:S2C:TMRext-TMext", "TMR $-$ TM", "SR"),
            ("S2C", "QM", "E1:S2C:QMext-TMext", "QM $-$ TM", "SQ")]
first2 = True
for st, c, c5, lab, key in rows_def:
    if st == "S2C" and first2:
        L += [r"\midrule", r"\multicolumn{8}{l}{\emph{Study~2, six new datasets, 12 cells}} \\"]; first2 = False
    if st == "S1" and c == "TMR":
        L += [r"\multicolumn{8}{l}{\emph{Study~1, 48 cells}} \\"]
    lines = [("TM", "original", O4[f"{st}:{c}-TM"]["rows"], "orig", False),
             ("TMext", "extended", O4[f"{st}:{c}-TMext"]["rows"], "ext", False)]
    if c5:
        lines.append(("both", "both re-tuned", O5[c5]["rows"], "both", True))
    for base, bl, rows, tag, new in lines:
        a, s, d = part(rows, 0, 1), part(rows, 0, .03), part(rows, .03, 1)
        cells = [lab if base == "TM" else "", bl, f"{a['w']}/{a['l']}", fmt(a["med"]), f"{s['w']}/{s['l']}", fmt(s["med"]),
                 f"{d['w']}/{d['l']}", fmt(d["med"])]
        if new:
            cells = [blue(x) for x in cells]
        L.append(" & ".join(cells) + r" \\")
        if new:
            for nm, v in (("all", a), ("sparse", s), ("dense", d)):
                M[f"ctl{key}{tag}{nm}W"] = v["w"]; M[f"ctl{key}{tag}{nm}L"] = v["l"]; M[f"ctl{key}{tag}{nm}Pos"] = v["pos"]
                M[f"ctl{key}{tag}{nm}N"] = v["n"]; M[f"ctl{key}{tag}{nm}Med"] = fmt(v["med"])
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUTDIR}/tab_controls.tex", "w").write("\n".join(L) + "\n")

# Study 1: density rule against TM_ext, both re-tuned (TMR_ext if sparse, QM_ext if dense; Holm from those contrasts)
S1rows = []
for c, lo, hi in (("E1:S1:TMRext-TMext", 0, .03), ("E1:S1:QMext-TMext", .03, 1)):
    S1rows += [r for r in O5[c]["rows"] if lo <= dens(r) < hi]
M.update(ctlSoneRuleBothW=sum(r["diff"] > 0 and r["p_holm"] < .05 for r in S1rows),
         ctlSoneRuleBothL=sum(r["diff"] < 0 and r["p_holm"] < .05 for r in S1rows),
         ctlSoneRuleBothPos=sum(r["diff"] > 0 for r in S1rows), ctlSoneRuleBothN=len(S1rows))
e = O5["E1:S2C:rule-TMext"]
M.update(ctlruleBothSignP=f"{e['sign_p']:.2f}" if e["sign_p"] is not None else "--", ctlruleBothPos=e["positive"],
         ctlruleBothN=e["n"])
# effect of s = 1.5 on the modified arm itself, and QMR_ext against TMR_ext
for c5, key in (("E1:S1:TMRext-TMR", "RselfS"), ("E1:S1:QMext-QM", "QselfS"), ("E1:S1:QMRext-TMRext", "QRR")):
    r = O5[c5]
    M.update({f"ctl{key}W": r["wins"], f"ctl{key}L": r["losses"], f"ctl{key}Pos": r["positive"], f"ctl{key}N": r["n"],
              f"ctl{key}Med": fmt(r["median_diff"] if r["median_diff"] is not None else float("nan"))})
# selection counts
for tag, nm in (("s1", "One"), ("s2", "Two")):
    for a, an in (("TMR", "R"), ("QM", "Q"), ("QMR", "QR")):
        sel = [x[a] for x in SEL5[tag].values() if a in x]
        if not sel:
            continue
        M[f"selFive{an}{nm}N"] = len(sel); M[f"selFive{an}{nm}Changed"] = sum(x["changed"] for x in sel)
        M[f"selFive{an}{nm}Slow"] = sum(x["s"] == 1.5 for x in sel)

# E2: literal-budget ablation
T = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r" & & \multicolumn{4}{c}{No budget $-$ budget} & \multicolumn{2}{c}{Literals per clause} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
     r"Study & Arm & Cells & W/L & Med. & Within 1~pp & Budget & No budget \\", r"\midrule"]
for S, sn in (("S1", "1"), ("S2C", "2")):
    for arm, an in (("TMR", "R"), ("QM", "Q")):
        r = O5[f"E2:{S}:{arm}noL-{arm}L"]; r2 = O5[f"E2:{S}:{arm}noL-TMext"]
        iL = [x["inc_L"] for x in r["rows"] if x.get("inc_L") is not None]
        iN = [x["inc_noL"] for x in r["rows"] if x.get("inc_noL") is not None]
        mL = float(np.median(iL)) if iL else float("nan"); mN = float(np.median(iN)) if iN else float("nan")
        med = r["median_diff"] if r["median_diff"] is not None else float("nan")
        T.append(f"{sn} & {'TMR' if arm == 'TMR' else 'QM'} & {r['n']} & {r['wins']}/{r['losses']} & {fmt(med)} & "
                 f"{r['within1']} & {mL:.1f} & {mN:.1f} \\\\")
        ok = r["n"] > 0 and r["losses"] == 0 and med > -1 and r["within1"] >= 0.75 * r["n"]
        k = f"noL{an}{'One' if S == 'S1' else 'Two'}"
        M.update({f"{k}N": r["n"], f"{k}W": r["wins"], f"{k}L": r["losses"], f"{k}Med": fmt(med),
                  f"{k}Within": r["within1"], f"{k}Pos": r["positive"], f"{k}IncL": f"{mL:.1f}", f"{k}IncN": f"{mN:.1f}",
                  f"{k}Works": "yes" if ok else "no", f"{k}TmW": r2["wins"], f"{k}TmL": r2["losses"],
                  f"{k}TmMed": fmt(r2["median_diff"] if r2["median_diff"] is not None else float("nan"))})
        sp = [x for x in r["rows"] if dens(x) < .03]; dn = [x for x in r["rows"] if dens(x) >= .03]
        for nm, R in (("Sparse", sp), ("Dense", dn)):
            M[f"{k}{nm}Med"] = fmt(float(np.median([x["diff"] for x in R])) if R else float("nan"))
            M[f"{k}{nm}L"] = sum(x["diff"] < 0 and x["p_holm"] < .05 for x in R); M[f"{k}{nm}N"] = len(R)
            M[f"{k}{nm}Within"] = sum(abs(x["diff"]) <= 1 for x in R)
T += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUTDIR}/tab_noL.tex", "w").write("\n".join(T) + "\n")


def nlines(p):
    try:
        return sum(1 for _ in open(p))
    except FileNotFoundError:
        return 0


S5 = f"{Q}/study5/logs"
M["fiveTuneRuns"] = f"{nlines(f'{S5}/tuneE1_s1.jsonl') + nlines(f'{S5}/tuneE1_s2.jsonl') + nlines(f'{S5}/tuneE1b.jsonl'):,}"
M["fiveTestRuns"] = f"{nlines(f'{S5}/confirmE1.jsonl') + nlines(f'{S5}/E2.jsonl') + nlines(f'{S5}/confirmE1b.jsonl'):,}"
try:
    SB = json.load(open(f"{S5}/selectionE1b.json"))
    M["sweepFiveN"] = len(SB); M["sweepFiveChanged"] = sum(x["changed"] for x in SB.values())
    M["sweepFiveSlow"] = sum(x["s"] == 1.5 for x in SB.values())
except FileNotFoundError:
    pass
open(f"{OUTDIR}/numbers_controls5.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print("\n".join(L)); print("\n".join(T))
for k, v in M.items():
    print(k, v)
