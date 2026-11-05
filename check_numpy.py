"""Independent NumPy reference of the QM clause rule vs. the C kernel's predict."""
import numpy as np, qmtm

z = np.load("/home/claude/data/proc/mnist.npz")
X, Y, Xt = z["X_train"][:5000], z["y_train"][:5000], z["X_test"][:1000]
qmtm.seed(3)
tm = qmtm.QMTM(100, 10, 7.5, mode=1, theta_pm=900)
tm.init_machine(X, Y)
for _ in range(3):
    tm.train_epoch(X, Y)

lits = np.hstack([Xt, 1 - Xt]).astype(bool)                # (n, 1568), x then ~x
nb, nf = tm.number_of_state_bits, tm.number_of_features
def includes(c):
    _, ta = tm.get_state()[c]
    msb = ta.reshape(tm.number_of_clauses, tm.number_of_ta_chunks, nb)[:, :, nb - 1]
    bits = np.unpackbits(msb.astype("<u4").view(np.uint8), axis=1, bitorder="little")
    return bits[:, :nf].astype(bool)                         # (clauses, 1568)

def predict_ref(mode, th, tol):
    sums = np.zeros((len(Xt), tm.number_of_classes), int)
    for c in range(tm.number_of_classes):
        inc = includes(c)
        ninc = inc.sum(1)
        viol = (~lits).astype(np.int32) @ inc.T.astype(np.int32)      # (n, clauses)
        t = ((1000 - th) * ninc) // 1000 if mode == 1 else np.full_like(ninc, tol)
        out = (viol <= t[None, :]) & (ninc[None, :] > 0)
        pol = np.where(np.arange(tm.number_of_clauses) % 2 == 0, 1, -1)
        sums[:, c] = np.clip(out.astype(int) @ pol, -tm.T, tm.T)
    return sums.argmax(1)

for cfg in [(1, 900, 0), (1, 800, 0), (2, 1000, 3), (0, 1000, 0)]:
    tm.set_qm(*cfg)
    pc = tm.predict(Xt)
    pr = predict_ref(cfg[0] if cfg[0] else 1, cfg[1], cfg[2])
    print(cfg, "C vs NumPy predictions identical:", np.array_equal(pc, pr), " agree %d/%d" % ((pc == pr).sum(), len(pc)))
