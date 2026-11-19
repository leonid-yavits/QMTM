"""Study 6 job lists (PROTOCOL.md). Run from the repository root.
  python3 study6/make_jobs6.py tune      -> jobs/E1_tune.txt, jobs/E2_levels.txt, jobs/E4_binary.txt
  python3 study6/make_jobs6.py confirm   -> jobs/E1_confirm.txt, jobs/E3_clauses.txt (needs logs/selection6.json from select6.py)
Paths: repository QMTM (default /home/claude/qmtm). Each line is one shell command; run with run6.sh."""
import sys, os, json, math, statistics as st
R = os.environ.get("QMTM", "/home/claude/qmtm"); S6 = f"{R}/study6"
HERE = os.path.dirname(os.path.abspath(__file__))
DS9 = ["adult", "mnist", "fmnist", "har", "imdb", "r8_V5000", "r52_V1000", "news20_V1000", "sector_V1000"]
DS6 = ["adult", "r8_V5000", "r52_V1000", "mnist", "fmnist", "imdb"]          # budget-sweep encodings of Section 6.5
CL = {5: 6, 10: 10}                                                           # FPTM clauses per class for C = 5, 10 (as in Study 2 part A)
T_FP, EP = 16, 40
FP = f"python3 {S6}/fptm6_job.py"


def w(name, lines):
    os.makedirs(f"{HERE}/jobs", exist_ok=True)
    open(f"{HERE}/jobs/{name}", "w").write("\n".join(lines) + "\n"); print(name, len(lines))


def tune():
    # E1 tuning: L_F in {4, 16, 64}, seed 1, validation
    w("E1_tune.txt", [f"{FP} {ds} {CL[C]} {T_FP} {lf} 1 {EP} {S6}/logs/E1_tune.jsonl" for ds in DS9 for C in (5, 10) for lf in (4, 16, 64)])
    # E2: vote levels, L_F in {1, 2, 4, 8, 16, 32, 64}, 6 clauses per class, seeds 101-103, test
    w("E2_levels.txt", [f"{FP} {ds} 6 {T_FP} {lf} {sd} {EP} {S6}/logs/E2_levels.jsonl"
                        for ds in DS9 for lf in (1, 2, 4, 8, 16, 32, 64) for sd in (101, 102, 103)])
    # E4: binary arms at 320 and 640 clauses per class, configuration selected at C = 160 in the existing sweep logs
    logs = [f"{HERE}/../study2/logs/sweep.jsonl", f"{HERE}/../study4/logs/confirmD2.jsonl", f"{HERE}/../study5/logs/confirmE1b.jsonl"]
    acc = {}
    for f in logs:
        for l in open(f):
            d = json.loads(l)
            if d["C"] == 160 and d["ds"] in ("mnist", "fmnist", "imdb") and d["seed"] >= 101:
                acc.setdefault((d["ds"], d["cfg"], d["s"]), []).append(d["test"])
    J = []
    for ds in ("mnist", "fmnist", "imdb"):
        cand = {k: st.mean(v) for k, v in acc.items() if k[0] == ds}
        best = max(cand, key=cand.get)                                              # most accurate binary configuration at C = 160
        van = max((k for k in cand if k[1].startswith("AND") and "_fb1" not in k[1]), key=cand.get)   # most accurate vanilla TM
        for k in dict.fromkeys([best, van]):
            print("E4", ds, k, round(100 * cand[k], 2))
            for C in (320, 640):
                for sd in (101, 102, 103):
                    J.append(f"cd {R}/study1 && python3 train.py {ds} {C} {C // 2} {k[2]} {k[1]} {sd} {EP} {S6}/logs/E4_binary.jsonl")
    w("E4_binary.txt", J)


def confirm():
    sel = json.load(open(f"{HERE}/logs/selection6.json"))
    w("E1_confirm.txt", [f"{FP} {ds} {CL[C]} {T_FP} {sel[f'{ds}|{C}']['LF']} {sd} {EP} {S6}/logs/E1_confirm.jsonl"
                         for ds in DS9 for C in (5, 10) for sd in (101, 102, 103, 104, 105)])
    # E3: clause sweep at the L_F selected for C = 5
    w("E3_clauses.txt", [f"{FP} {ds} {cl} {T_FP} {sel[f'{ds}|5']['LF']} {sd} {EP} {S6}/logs/E3_clauses.jsonl"
                         for ds in DS6 for cl in (2, 4, 6, 10, 20, 40) for sd in (101, 102, 103)])


{"tune": tune, "confirm": confirm}[sys.argv[1]]()
