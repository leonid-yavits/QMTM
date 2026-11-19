"""One study-6 FPTM run (reference FPTM, graded vote) with model logging. Skips keys already in the log.
python3 fptm6_job.py <ds> <clauses_fptm> <T> <LF> <seed> <epochs> <out.jsonl>
clauses_fptm is FPTM's clauses_num (per class, split equally by polarity). L = 2*LF. S = 1000 if n >= 1000 else n.
Environment: QMTM_DATA (exported datasets, default /home/claude/data/fptm), JULIA (default julia)."""
import sys, json, os, subprocess
ds, CL, T, LF, sd, EP, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
DATA = os.environ.get("QMTM_DATA", "/home/claude/data/fptm"); JL = os.environ.get("JULIA", "julia")
nf = int(open(f"{DATA}/{ds}/meta.txt").read().split()[0]); S = 1000 if nf >= 1000 else nf
L = 2 * LF
cfg = f"FP_LF{LF}_S{S}"
key = (ds, CL, T, cfg, sd, EP)
if os.path.exists(out):
    for l in open(out):
        d = json.loads(l)
        if (d["ds"], d["clauses_fptm"], d["T"], d["cfg"], d["seed"], d["epochs"]) == key:
            sys.exit(0)
here = os.path.dirname(os.path.abspath(__file__))
r = subprocess.run([JL, "-t", "1", f"{here}/fptm/run_fptm6.jl", DATA, ds, str(CL), str(T), str(S), str(L), str(LF), str(sd), str(EP), "5"],
                   capture_output=True, text=True)
res = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
st = [l for l in r.stdout.splitlines() if l.startswith("STATS ")]
if r.returncode != 0 or not res or not st:
    sys.stderr.write(f"FAILED {key}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}\n"); sys.exit(1)
_, curve, te, tt, s_int = res[0].split()
curve = [float(x) for x in curve.split(",")]
stats = json.loads(st[0][6:])
# the quantized vote with Q = LF is the full graded vote: it must reproduce the test accuracy
assert abs(stats["q_test"][-1] - float(te)) < 1e-9, (stats["q_test"][-1], te)
with open(out, "a") as f:
    f.write(json.dumps(dict(ds=ds, C=CL, T=T, s=float(S), cfg=cfg, arm="FPTM", seed=sd, epochs=EP, clauses_fptm=CL, L=L, LF=LF, s_int=int(s_int),
                            val=curve[-1], val_curve=curve, test=float(te), t=float(tt), stats=stats)) + "\n")
