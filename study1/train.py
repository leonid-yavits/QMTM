"""Study 1: one TM training run on a Booleanized dataset (npz with X_train/y_train[/X_val/y_val]/X_test/y_test).
If the npz has no validation split, a stratified 10 % of the training split (seed 0) is held out.
python3 train.py <ds> <C> <T> <s> <cfg> <seed> <epochs> <out.jsonl>
cfg: AND | AND_L<L> | AND_L<L>_fb<fb> | QM<theta%>_L<L>_fb<fb> (relative QM over all included literals)
     | QP<theta%>_L<L>_fb<fb> (polarity-aware QM over included positive literals; negated literals strict)
fb bit 1 = R1 (targeted Type Ib). Logs the validation curve, final validation and test accuracy, and clause statistics."""
import sys, json, os, re, time

DS, C, T, s, cfg, sd, EP, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
key = (DS, C, T, s, cfg, sd, EP)
if os.path.exists(out):                      # skip completed keys before the heavy imports
    for l in open(out):
        d = json.loads(l)
        if (d["ds"], d["C"], d["T"], d["s"], d["cfg"], d["seed"], d["epochs"]) == key:
            sys.exit(0)
import numpy as np
sys.path.insert(0, "/home/claude/qmtm")
import qmtm
from sklearn.model_selection import train_test_split
m = re.fullmatch(r"AND(?:_L(\d+))?(?:_fb(\d+))?|(?:QM|QP)(\d+)_L(\d+)_fb(\d+)", cfg)
if cfg.startswith("AND"):
    mode, th, L, fb = 0, 1000, (int(m.group(1)) if m.group(1) else None), (int(m.group(2)) if m.group(2) else 0)
else:
    mode, th, L, fb = (3 if cfg.startswith("QP") else 1), 10 * int(m.group(3)), int(m.group(4)), int(m.group(5))
P = {"mnist": "/home/claude/data/proc/mnist.npz", "fmnist": "/home/claude/data/proc/fmnist.npz", "adult": "/home/claude/data/proc/adult.npz",
     "imdb": "/home/claude/data/proc/imdb.npz"}.get(DS, f"/home/claude/data/study1/{DS}.npz")
if not os.path.exists(P):                    # study-2 datasets (PROTOCOL.md of study 2, part C)
    P = f"/home/claude/data/study2/{DS}.npz"
z = np.load(P)
X, Y, Xt, Yt = z["X_train"], z["y_train"].astype(np.uint32), z["X_test"], z["y_test"].astype(np.uint32)
if "X_val" in z.files:
    Xv, Yv = z["X_val"], z["y_val"].astype(np.uint32)
else:
    a, b = train_test_split(np.arange(len(Y)), test_size=0.1, random_state=0, stratify=Y)
    X, Xv, Y, Yv = X[a], X[b], Y[a], Y[b]
qmtm.seed(sd)
EVERY = int(os.environ.get("MT_EVAL_EVERY", "5"))
tm = qmtm.QMTM(C, T, s, mode=mode, theta_pm=th, max_included_literals=L, fb=fb, fast=int(os.environ.get("QMTM_FAST", "1")))
i0 = int(np.argmax(Y))                       # init_machine needs one row carrying the largest label; inputs stay uint8 (packed by encode)
tm.init_machine(np.ascontiguousarray(X[i0:i0 + 1], dtype=np.uint32), Y[i0:i0 + 1])
E, Ev, Et = tm.encode(X), tm.encode(Xv), tm.encode(Xt)
del X, Xv, Xt
t0 = time.time(); curve = []
for e in range(1, EP + 1):
    tm.train_epoch_encoded(E, Y)
    if e % EVERY == 0 or e == EP:
        curve.append(round(float(np.mean(tm.predict_encoded(Ev) == Yv)), 4))
pt = tm.predict_encoded(Et)
cls_, cl_, lit_ = tm.include_sets(); Mh = tm.number_of_features // 2; K = tm.number_of_classes
comp = dict(inc_per_clause=float(len(lit_) / (K * C)), pos_per_clause=float((lit_ < Mh).sum() / (K * C)),
            empty_clauses=float(1 - len(set(zip(cls_.tolist(), cl_.tolist()))) / (K * C)))
with open(out, "a") as f:
    f.write(json.dumps(dict(ds=DS, C=C, T=T, s=s, cfg=cfg, arm={0: "AND", 1: "QM", 3: "QP"}[mode], seed=sd, epochs=EP, val=curve[-1], val_curve=curve,
                            test=float(np.mean(pt == Yt)), pred=pt.astype(int).tolist() if os.environ.get("MT_PRED") else None,
                            comp=comp, t=round(time.time() - t0, 1))) + "\n")
