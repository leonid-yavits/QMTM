"""Sweep data with TMR and QM re-tuned (study 5). Mean test accuracy (%) over seeds 101-103.
C <= 20: the E1 selection (study5/logs/selection5.json, s1), test runs from its confirmation log.
C >= 40: the E1b selection (study5/logs/sweepE1b.json).
Missing cells (IMDb, incomplete runs) are None. The original arms come from study 2 part B (stats2.json)."""
import json, collections
Q = "/home/claude/qmtm"
B = json.load(open(f"{Q}/study2/logs/stats2.json"))["B"]
R4 = json.load(open(f"{Q}/study4/logs/sweep4.json"))
SEL = json.load(open(f"{Q}/study5/logs/selection5.json"))["s1"]
try:
    SW = json.load(open(f"{Q}/study5/logs/sweepE1b.json"))
except FileNotFoundError:
    SW = {}
DS = ["adult", "mnist", "fmnist", "imdb", "r8_V5000", "r52_V1000", "r52_V5000", "r52_V0"]
CS = [5, 10, 20, 40, 80, 160]
_T = {}


def _tests(log):
    if log not in _T:
        A = collections.defaultdict(dict)
        for l in open(log):
            d = json.loads(l)
            if d["epochs"] == 40:
                A[(d["ds"], d["C"], d["cfg"], float(d["s"]))][d["seed"]] = d["test"]
        _T[log] = A
    return _T[log]


def mean3(log, ds, C, cfg, s):
    r = _tests(log).get((ds, C, cfg, float(s)), {})
    v = [r[x] for x in (101, 102, 103) if x in r]
    return 100 * sum(v) / 3 if len(v) == 3 else None


def retuned(ds, arm, C):
    """(acc, cfg, s, changed) of the re-tuned arm, or None."""
    if C <= 20:
        x = SEL.get(f"{ds}|{C}", {}).get(arm)
        if not x:
            return None
        return mean3(x["conf_log"], ds, C, x["cfg"], x["s"]), x["cfg"], x["s"], x["changed"]
    x = SW.get(ds, {}).get(str(C), {}).get(arm)
    if not x:
        return None
    return x["acc"], x["cfg"], x["s"], x["changed"]


def series():
    """S[ds][key][C] -> acc or None; keys TM, TMre, TMR, QM (original) and TMR5, QM5 (re-tuned)."""
    S = {}
    for ds in DS:
        S[ds] = {"TM": {C: B[ds]["TM"].get(str(C)) for C in CS},
                 "TMre": {C: (R4[ds][str(C)]["acc"] if R4[ds].get(str(C)) else None) for C in CS},
                 "TMR": {C: B[ds]["TMR"].get(str(C)) for C in CS}, "QM": {C: B[ds]["QM"].get(str(C)) for C in CS}}
        for a in ("TMR", "QM"):
            S[ds][a + "5"] = {C: (lambda r: r[0] if r else None)(retuned(ds, a, C)) for C in CS}
    return S


if __name__ == "__main__":
    S = series()
    for ds in DS:
        for k in S[ds]:
            print(f"{ds:10s} {k:5s}", " ".join("  --  " if v is None else f"{v:6.2f}" for v in S[ds][k].values()))
    # consistency: original arms at C <= 20 recomputed from the study-1 confirmation log
    for ds in DS:
        for a in ("TMR", "QM"):
            for C in (5, 10, 20):
                x = SEL.get(f"{ds}|{C}", {}).get(a)
                if x:
                    m = mean3(f"{Q}/study1/logs/confirm.jsonl", ds, C, x["orig_cfg"], x["orig_s"])
                    if m is None or abs(m - B[ds][a][str(C)]) > 1e-6:
                        print("MISMATCH", ds, a, C, m, B[ds][a][str(C)])
