"""Generate LaTeX tables from the result logs (no hand transcription)."""
import json
D = json.load(open("/home/claude/qmtm/paper2/data/datasets.json"))
R = json.load(open("/home/claude/qmtm/study1/logs/stats1.json"))["contrasts"]
NAME = {"adult": "UCI Adult", "mnist": "MNIST", "fmnist": "Fashion-MNIST", "imdb": "IMDb", "har": "UCI HAR", "phishing": "UCI Phishing",
        "r8_V5000": "R8 ($V$=5k)", "r52_V1000": "R52 ($V$=1k)", "r52_V5000": "R52 ($V$=5k)", "r52_V0": "R52 (full)",
        "dorothea_V5000": "Dorothea ($V$=5k)", "dorothea_V0": "Dorothea (full)", "sector_V1000": "Sector ($V$=1k)", "sector_V5000": "Sector ($V$=5k)",
        "sector_V20000": "Sector ($V$=20k)", "news20_V1000": "20NG ($V$=1k)", "news20_V5000": "20NG ($V$=5k)", "news20_V20000": "20NG ($V$=20k)",
        "rcv1_V5000": "RCV1 ($V$=5k)", "ledgar_V5000": "LEDGAR ($V$=5k)", "agnews": "AG News", "trec": "TREC-6", "emotion": "Emotion",
        "usps": "USPS", "letter": "Letter", "satimage": "Satimage"}
json.dump(NAME, open("/home/claude/qmtm/paper2/data/names.json", "w"), indent=1)
S1 = [k for k in D if k not in ("agnews", "trec", "emotion", "usps", "letter", "satimage")]
order = sorted(S1, key=lambda k: D[k]["density"])
L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Dataset & $K$ & Train/val/test & Bits $n$ & Density (\\%) & Active & $C$ \\\\", "\\midrule"]
for k in order:
    d = D[k]; cs = "5/10/20" if k in ("adult", "mnist", "fmnist", "imdb", "r8_V5000", "r52_V1000", "r52_V5000", "r52_V0") else "5/10"
    L.append(f"{NAME[k]} & {d['K']} & {d['train']:,}/{d['val']:,}/{d['test']:,} & {d['features']:,} & {100*d['density']:.2f} & {d['active']:.0f} & {cs} \\\\")
L.append("\\midrule")
for k in sorted(["agnews", "trec", "emotion", "usps", "letter", "satimage"], key=lambda k: D[k]["density"]):
    d = D[k]; L.append(f"{NAME[k]}$^\\dagger$ & {d['K']} & {d['train']:,}/{d['val']:,}/{d['test']:,} & {d['features']:,} & {100*d['density']:.2f} & {d['active']:.0f} & 5/10 \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("/home/claude/qmtm/paper2/tex/tab_datasets.tex", "w").write("\n".join(L) + "\n")
# study-1 per-cell table
g = lambda con, cell: next(r for r in R[con]["rows"] if r["cell"] == cell)
def f(r, which):
    v = r["ma"] if which == "a" else r["mb"]; return f"{v:.2f}"
def d(r):
    s = f"{r['diff']:+.2f}".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if r["p_holm"] < 0.05 else s
L = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
     "Dataset & $C$ & TM & TMR & QM & Best QM arm & $\\Delta_{\\mathrm{R1}}$ & $\\Delta_{\\mathrm{QM}}$ & $\\Delta_{\\mathrm{best}}$ \\\\", "\\midrule"]
cells = sorted({r["cell"] for r in R["TMR-TM"]["rows"]}, key=lambda c: (D[c.split("|")[0]]["density"], c.split("|")[0], int(c.split("|")[1])))
prev = None
for c in cells:
    ds, C = c.split("|"); a = g("TMR-TM", c); b = g("QM-TM", c); e = g("bestQM-bestNonQM", c)
    L.append(f"{NAME[ds] if ds != prev else ''} & {C} & {f(a,'b')} & {f(a,'a')} & {f(b,'a')} & {e['ma']:.2f} ({e['a']}) & {d(a)} & {d(b)} & {d(e)} \\\\")
    prev = ds
L += ["\\bottomrule", "\\end{tabular}"]
open("/home/claude/qmtm/paper2/tex/tab_study1.tex", "w").write("\n".join(L) + "\n")
print("ok", len(cells))
