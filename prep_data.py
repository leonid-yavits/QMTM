"""Binarize MNIST, Fashion-MNIST and UCI Adult following the AP(TM) paper's
stated encodings (Section 6.1). Output: /home/claude/data/proc/<name>.npz
with uint8 X_train, y_train, X_test, y_test."""
import gzip, numpy as np, pandas as pd, cv2, os
from sklearn.model_selection import train_test_split

RAW = "/home/claude/data/raw"
OUT = "/home/claude/data/proc"
os.makedirs(OUT, exist_ok=True)

def idx(path):
    with gzip.open(path, "rb") as f:
        b = f.read()
    magic = int.from_bytes(b[0:4], "big")
    nd = magic & 0xFF
    dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(dims)

def load_mnist_like(d):
    return (idx(f"{d}/train-images-idx3-ubyte.gz"), idx(f"{d}/train-labels-idx1-ubyte.gz"),
            idx(f"{d}/t10k-images-idx3-ubyte.gz"), idx(f"{d}/t10k-labels-idx1-ubyte.gz"))

# MNIST: global threshold 75
xtr, ytr, xte, yte = load_mnist_like(f"{RAW}/mnist_a")
np.savez_compressed(f"{OUT}/mnist.npz",
                    X_train=(xtr.reshape(-1, 784) > 75).astype(np.uint8), y_train=ytr,
                    X_test=(xte.reshape(-1, 784) > 75).astype(np.uint8), y_test=yte)

# Fashion-MNIST: adaptive Gaussian thresholding, window 11, offset 2
xtr, ytr, xte, yte = load_mnist_like(f"{RAW}/fmnist")
def agt(X):
    return np.stack([cv2.adaptiveThreshold(im, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2) for im in X]).reshape(-1, 784)
np.savez_compressed(f"{OUT}/fmnist.npz", X_train=agt(xtr).astype(np.uint8), y_train=ytr,
                    X_test=agt(xte).astype(np.uint8), y_test=yte)

# UCI Adult: one-hot categorical (incl. '?'), 10-bit thermometer numerical
# (uniform thresholds between min and max), stratified 80/20 split.
df = pd.read_csv(f"{RAW}/adult-all.csv", header=None)
df = df.apply(lambda c: c.str.strip() if c.dtype == object else c)
y = (df[14] == ">50K").astype(np.uint8).values
cat = [1, 3, 5, 6, 7, 8, 9, 13]
num = [0, 2, 4, 10, 11, 12]
parts = [pd.get_dummies(df[c].astype(str), prefix=str(c)).values.astype(np.uint8) for c in cat]
for c in num:
    v = df[c].values.astype(float)
    th = np.linspace(v.min(), v.max(), 12)[1:-1]          # 10 interior thresholds
    parts.append((v[:, None] > th[None, :]).astype(np.uint8))
X = np.hstack(parts)
assert X.shape[1] == 162, X.shape
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
np.savez_compressed(f"{OUT}/adult.npz", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)

for n in ["mnist", "fmnist", "adult"]:
    z = np.load(f"{OUT}/{n}.npz")
    print(n, z["X_train"].shape, z["X_test"].shape, "ones-frac %.3f" % z["X_train"].mean(),
          "classes", np.bincount(z["y_train"]))
