"""Re-run train.py's procedure for the six QM-winning cells (selected QM-family config) and record the largest clause
(included literals) every 5 epochs. Seeds given on the command line (1 = tuning seed). Validation accuracy at epoch 40 is
printed so that the run can be matched to tune.jsonl / confirm.jsonl."""
import sys, json, numpy as np
sys.path.insert(0, "/home/claude/qmtm")
import qmtm
from sklearn.model_selection import train_test_split
CELLS = [("mnist", 10, 5, 10.0, 1, 950, 64, 0), ("fmnist", 5, 3, 10.0, 1, 900, 32, 0), ("fmnist", 10, 5, 10.0, 1, 900, 32, 0),
         ("fmnist", 20, 10, 10.0, 1, 900, 32, 0), ("har", 5, 3, 2.5, 1, 900, 32, 1), ("har", 10, 5, 2.5, 1, 900, 32, 1)]
seeds = [int(x) for x in sys.argv[1:]] or [1]
P = {"mnist": "/home/claude/data/proc/mnist.npz", "fmnist": "/home/claude/data/proc/fmnist.npz", "har": "/home/claude/data/study1/har.npz"}
for ds, C, T, s, mode, th, L, fb in CELLS:
    z = np.load(P[ds]); X, Y = z["X_train"], z["y_train"].astype(np.uint32)
    if "X_val" in z.files: Xv, Yv = z["X_val"], z["y_val"].astype(np.uint32)
    else:
        a, b = train_test_split(np.arange(len(Y)), test_size=0.1, random_state=0, stratify=Y); X, Xv, Y, Yv = X[a], X[b], Y[a], Y[b]
    for sd in seeds:
        qmtm.seed(sd)
        tm = qmtm.QMTM(C, T, s, mode=mode, theta_pm=th, max_included_literals=L, fb=fb, fast=1)
        i0 = int(np.argmax(Y)); tm.init_machine(np.ascontiguousarray(X[i0:i0 + 1], dtype=np.uint32), Y[i0:i0 + 1])
        E, Ev = tm.encode(X), tm.encode(Xv); mx = []
        for e in range(1, 41):
            tm.train_epoch_encoded(E, Y)
            if e % 5 == 0:
                cls, cl, lit = tm.include_sets()
                cnt = np.bincount(cls * C + cl, minlength=tm.number_of_classes * C)
                mx.append(int(cnt.max()))
                if e in (5, 10, 15, 20, 25, 30, 35, 40) and e % 5 == 0:
                    v = round(float(np.mean(tm.predict_encoded(Ev) == Yv)), 4)
        print(json.dumps(dict(ds=ds, C=C, theta=th / 1000, seed=sd, max_per_5ep=mx, overall=max(mx), limit=59 if th == 900 else 119, val40=v)), flush=True)
