"""Run a short seeded MNIST training and dump predictions + TA states.
Usage: QMTM_LIB=<lib> python3 validate.py <mode> <theta_pm> <abs_tol> <out.npz>"""
import sys, time, numpy as np
import qmtm

mode, th, tol, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
z = np.load("/home/claude/data/proc/mnist.npz")
X, Y, Xt, Yt = z["X_train"][:10000], z["y_train"][:10000], z["X_test"][:2000], z["y_test"][:2000]
qmtm.seed(7)
tm = qmtm.QMTM(200, 10, 7.5, mode=mode, theta_pm=th, abs_tol=tol)
tm.init_machine(X, Y)
t = time.time()
for e in range(3):
    tm.train_epoch(X, Y)
p = tm.predict(Xt)
st = np.concatenate([np.concatenate(s) for s in tm.get_state()])
print(f"mode={mode} th={th} tol={tol} acc={np.mean(p == Yt):.4f} time={time.time()-t:.1f}s")
np.savez(out, p=p, st=st)
