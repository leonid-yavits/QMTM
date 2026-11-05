"""Study 4 statistics (PROTOCOL.md). Test accuracy, seeds 101-105. Welch per cell, Holm within each contrast.
D1, study 1: TMR, QM, QP against TM_ext; best QM-family arm against the best of {TM_ext, TMR}.
D1, study 2 part C: rule arm, TMR and QM against TM_ext; sign test over cells as in study 2.
Each contrast is reported next to the original one (TM instead of TM_ext) over the same cells.
Cells whose TM_ext confirmation is incomplete are skipped and counted. Writes logs/stats4.json.
D2 (sweep with re-tuned TM) is summarized separately in sweep4.py."""
import json, collections, numpy as np
from scipy import stats
S1, S2, S4 = "/home/claude/qmtm/study1", "/home/claude/qmtm/study2", "/home/claude/qmtm/study4"


def load(*paths):
    A = collections.defaultdict(dict)
    for p in paths:
        try:
            for l in open(p):
                d = json.loads(l); A[(d["ds"], int(d["C"]), d["cfg"], float(d["s"]))][d["seed"]] = d["test"]
        except FileNotFoundError:
            pass
    return A


def vec(A, ds, C, cfg, s):
    r = A.get((ds, int(C), cfg, float(s)), {})
    return np.array([r[sd] for sd in range(101, 106) if sd in r])


def welch(x, y):
    return 1.0 if np.allclose(x, y) else float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def holm(rows):
    m = len(rows); run = 0.0
    for r_, i in enumerate(sorted(range(m), key=lambda i: rows[i]["p"])):
        run = max(run, min(1.0, (m - r_) * rows[i]["p"])); rows[i]["p_holm"] = run


def summarize(rows):
    holm(rows)
    npos = sum(r["diff"] > 0 for r in rows); nneg = sum(r["diff"] < 0 for r in rows)
    return dict(n=len(rows), wins=int(sum(r["diff"] > 0 and r["p_holm"] < 0.05 for r in rows)),
                losses=int(sum(r["diff"] < 0 and r["p_holm"] < 0.05 for r in rows)), positive=int(npos), negative=int(nneg),
                mean_diff=float(np.mean([r["diff"] for r in rows])) if rows else None,
                sign_p=float(stats.binomtest(npos, npos + nneg, 0.5, alternative="greater").pvalue) if npos + nneg else None, rows=rows)


SEL4 = json.load(open(f"{S4}/logs/selection4.json"))
A1 = load(f"{S1}/logs/confirm.jsonl", f"{S4}/logs/confirmD1.jsonl")
AC = load(f"{S2}/logs/confirmC.jsonl", f"{S4}/logs/confirmD1.jsonl")
OUT = {}


def contrast(tag, sel, A, arm_of, cells_note):
    """arm_of(k, o) -> (label, cfg, s) or None. Builds the contrast against TM_ext and against the original TM."""
    rows_ext, rows_orig, skipped = [], [], 0
    for k, o in sel.items():
        ds, C = k.split("|"); e = SEL4[tag].get(k)
        if e is None:
            skipped += 1; continue
        a = arm_of(k, o)
        x = vec(A, ds, C, a[1], a[2]); te = vec(A, ds, C, e["cfg"], e["s"]); to = vec(A, ds, C, o["TM"]["cfg"], o["TM"]["s"])
        if len(te) < 5:
            skipped += 1; continue
        b = a[3] if len(a) > 3 else None                     # optional comparator for the best-vs-best contrast
        for rows, y in ((rows_ext, te if b is None else b(k, o, te, ext=True)), (rows_orig, to if b is None else b(k, o, to, ext=False))):
            rows.append(dict(cell=k, arm=a[0], ma=100 * x.mean(), sa=100 * x.std(ddof=1), mb=100 * y.mean(), sb=100 * y.std(ddof=1),
                             diff=100 * (x.mean() - y.mean()), p=welch(x, y), density=o.get("density")))
    return rows_ext, rows_orig, skipped


# Study 1 contrasts against TM_ext
S = json.load(open(f"{S1}/logs/selection.json"))


def best_nonqm(k, o, tm_vec, ext):
    """Best of {TM_ext or TM, TMR} chosen by seed-1 validation, as in study 1."""
    ds, C = k.split("|"); e = SEL4["D1_s1"][k]
    tm_val = e["val"] if ext else o["TM"]["val"]
    if o["TMR"]["val"] > tm_val:
        return vec(A1, ds, C, o["TMR"]["cfg"], o["TMR"]["s"])
    return tm_vec


for name, arm_of in (
        ("TMR", lambda k, o: ("TMR", o["TMR"]["cfg"], o["TMR"]["s"])),
        ("QM", lambda k, o: ("QM", o["QM"]["cfg"], o["QM"]["s"])),
        ("QP", lambda k, o: ("QP", o["QP"]["cfg"], o["QP"]["s"])),
        ("bestQM", lambda k, o: (o["bestQM"], o[o["bestQM"]]["cfg"], o[o["bestQM"]]["s"], best_nonqm))):
    ext, orig, sk = contrast("D1_s1", S, A1, arm_of, "study 1")
    OUT[f"S1:{name}-TMext"] = summarize(ext); OUT[f"S1:{name}-TM"] = summarize(orig); OUT[f"S1:{name}-TMext"]["skipped"] = sk

# Study 2 part C contrasts against TM_ext
SC = json.load(open(f"{S2}/logs/selection2.json"))["C"]
for name, arm_of in (
        ("rule", lambda k, o: (o["rule"], o[o["rule"]]["cfg"], o[o["rule"]]["s"])),
        ("TMR", lambda k, o: ("TMR", o["TMR"]["cfg"], o["TMR"]["s"])),
        ("QM", lambda k, o: ("QM", o["QM"]["cfg"], o["QM"]["s"]))):
    ext, orig, sk = contrast("D1_s2", SC, AC, arm_of, "study 2 C")
    OUT[f"S2C:{name}-TMext"] = summarize(ext); OUT[f"S2C:{name}-TM"] = summarize(orig); OUT[f"S2C:{name}-TMext"]["skipped"] = sk

json.dump(OUT, open(f"{S4}/logs/stats4.json", "w"), indent=1)
for key in OUT:
    if key.endswith("TMext"):
        e, o = OUT[key], OUT[key.replace("TMext", "TM")]
        f = lambda r: f"n={r['n']:2d} wins {r['wins']:2d} losses {r['losses']:2d} pos {r['positive']:2d}/{r['n']:2d} mean {str(r['mean_diff'] if r['mean_diff'] is None else round(r['mean_diff'], 2)):>6} sign p={r['sign_p'] if r['sign_p'] is None else round(r['sign_p'], 4)}"
        print(f"{key:18s} vs TM_ext: {f(e)}  (skipped {e['skipped']})")
        print(f"{'':18s} vs TM    : {f(o)}")
