"""Study 6 copy of the study-3 export. Export a study-1 dataset to bit-packed files for the FPTM Julia driver, with the SAME train/val/test split as study1/train.py.
Writes /home/claude/data/fptm/<ds>/{Xtr,Xva,Xte}.bits (np.packbits, bitorder little, rows padded to whole bytes), {ytr,yva,yte}.i16, meta.txt.
python3 export.py <ds>"""
import sys, os, numpy as np
from sklearn.model_selection import train_test_split
DS = sys.argv[1]
P = {"mnist": "/home/claude/data/proc/mnist.npz", "fmnist": "/home/claude/data/proc/fmnist.npz", "adult": "/home/claude/data/proc/adult.npz",
     "imdb": "/home/claude/data/proc/imdb.npz"}.get(DS, f"/home/claude/data/study1/{DS}.npz")
z = np.load(P)
X, Y, Xt, Yt = z["X_train"], z["y_train"].astype(np.uint32), z["X_test"], z["y_test"].astype(np.uint32)
if "X_val" in z.files:
    Xv, Yv = z["X_val"], z["y_val"].astype(np.uint32)
else:
    a, b = train_test_split(np.arange(len(Y)), test_size=0.1, random_state=0, stratify=Y)
    X, Xv, Y, Yv = X[a], X[b], Y[a], Y[b]
o = os.path.join(os.environ.get("QMTM_DATA", "/home/claude/data/fptm"), DS) + "/"; os.makedirs(o, exist_ok=True)
for nm, A, y in [("tr", X, Y), ("va", Xv, Yv), ("te", Xt, Yt)]:
    np.packbits(np.ascontiguousarray(A, dtype=np.uint8) > 0, axis=1, bitorder="little").tofile(o + f"X{nm}.bits")
    y.astype(np.int16).tofile(o + f"y{nm}.i16")
open(o + "meta.txt", "w").write(f"{X.shape[1]} {len(Y)} {len(Yv)} {len(Yt)} {int(max(Y.max(), Yv.max(), Yt.max())) + 1}\n")
print(DS, X.shape[1], len(Y), len(Yv), len(Yt))
