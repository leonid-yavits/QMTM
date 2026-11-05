"""Booleanize the downloaded datasets for study 1. Output: /home/claude/data/study1/<name>.npz (uint8 X, uint16 y).
- news20, sector (sector.scale files), rcv1 (LIBSVM multiclass): binary presence of each feature; validation = stratified 10 % of the training
  file (seed 0); chi-square selection to V features on the reduced training part only. rcv1: official 518,571-document
  test file subsampled to 50,000 documents (seed 0), because the dense TM input of the full file does not fit in memory.
- dorothea (UCI / NIPS 2003): train (800) -> stratified 85/15 train/val (seed 0); the public validation set (350, labelled) is the test set.
  Binary features as given (100,000).
- har (UCI HAR): 561 features, thermometer code with 10 quantile bins from the training split; official train/test split.
- phishing (UCI): 30 categorical features one-hot encoded; stratified 80/20 train/test split (seed 0).
python3 prep_more.py <name> [V ...]"""
import bz2, collections, sys, numpy as np, scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
D = "/home/claude/data/study1/"; DL = D + "dl/"
name, Vs = sys.argv[1], [int(v) for v in sys.argv[2:]] or [0]


def split_val(X, y, frac, seed=0):
    cnt = collections.Counter(y); idx = np.arange(len(y)); rare = np.array([cnt[v] < 2 for v in y])
    a, b = train_test_split(idx[~rare], test_size=frac, random_state=seed, stratify=y[~rare])
    return np.concatenate([a, idx[rare]]), b


def save(nm, Xa, ya, Xb, yb, Xt, yt, V):
    if V and V < Xa.shape[1]:
        sc, _ = chi2(Xa, ya); sel = np.sort(np.argsort(-np.nan_to_num(sc), kind="stable")[:V])
    else:
        sel = np.arange(Xa.shape[1])
    f = lambda X: (np.asarray(X[:, sel].todense()) if sp.issparse(X) else X[:, sel]).astype(np.uint8)
    o = dict(X_train=f(Xa), y_train=ya.astype(np.uint16), X_val=f(Xb), y_val=yb.astype(np.uint16), X_test=f(Xt), y_test=yt.astype(np.uint16))
    tag = f"{nm}_V{V}" if nm in ("news20", "sector", "rcv1", "dorothea") else nm
    np.savez_compressed(D + tag + ".npz", **o)
    a = o["X_train"]
    print(f"{tag}: K={len(np.unique(ya))} train {len(ya)} val {len(yb)} test {len(yt)} features {a.shape[1]} density {100 * a.mean():.3f} % active/sample {a.sum(1).mean():.1f}", flush=True)


if name in ("news20", "sector", "rcv1"):
    fn = {"news20": ("news20.bz2", "news20.t.bz2"), "sector": ("sector.scale.bz2", "sector.t.scale.bz2"), "rcv1": ("rcv1_train.multiclass.bz2", "rcv1_test.multiclass.bz2")}[name]
    Xtr, ytr = load_svmlight_file(bz2.open(DL + fn[0]))
    Xte, yte = load_svmlight_file(bz2.open(DL + fn[1]), n_features=Xtr.shape[1])
    if Xte.shape[1] > Xtr.shape[1]:
        Xte = Xte[:, :Xtr.shape[1]]
    lab = {v: i for i, v in enumerate(np.unique(ytr))}
    ytr = np.array([lab[v] for v in ytr]); keep = np.array([v in lab for v in yte]); Xte, yte = Xte[keep], np.array([lab[v] for v in yte[keep]])
    if name == "rcv1":
        s = np.sort(np.random.default_rng(0).choice(len(yte), 50000, replace=False)); Xte, yte = Xte[s], yte[s]
    Xtr.data[:] = 1; Xte.data[:] = 1; Xtr, Xte = Xtr.tocsc().astype(np.uint8), Xte.tocsc().astype(np.uint8)
    a, b = split_val(Xtr, ytr, 0.1)
    for V in Vs:
        save(name, Xtr[a], ytr[a], Xtr[b], ytr[b], Xte, yte, V)
elif name == "dorothea":
    def rd(p, n=100000):
        rows = [list(map(int, l.split())) for l in open(p)]
        X = np.zeros((len(rows), n), np.uint8)
        for i, r in enumerate(rows):
            X[i, np.array(r) - 1] = 1
        return X
    X = rd(DL + "dorothea/DOROTHEA/dorothea_train.data"); y = (np.loadtxt(DL + "dorothea/DOROTHEA/dorothea_train.labels") > 0).astype(int)
    Xt = rd(DL + "dorothea/DOROTHEA/dorothea_valid.data"); yt = (np.loadtxt(DL + "dorothea/dorothea_valid.labels") > 0).astype(int)
    a, b = split_val(X, y, 0.15)
    for V in Vs:
        save(name, sp.csc_matrix(X[a]), y[a], sp.csc_matrix(X[b]), y[b], sp.csc_matrix(Xt), yt, V)
elif name == "har":
    H = DL + "har/UCI HAR Dataset/"
    X, y = np.loadtxt(H + "train/X_train.txt"), np.loadtxt(H + "train/y_train.txt").astype(int) - 1
    Xt, yt = np.loadtxt(H + "test/X_test.txt"), np.loadtxt(H + "test/y_test.txt").astype(int) - 1
    q = np.quantile(X, np.linspace(0.05, 0.95, 10), axis=0)         # 10 thresholds at training quantiles 5, 15, ..., 95 %: 10 bits per feature (561 x 10 = 5,610)
    th = lambda Z: (Z[:, :, None] > q.T[None]).astype(np.uint8).reshape(len(Z), -1)
    B, Bt = th(X), th(Xt); a, b = split_val(B, y, 0.1)
    save(name, B[a], y[a], B[b], y[b], Bt, yt, 0)
elif name == "phishing":
    rows = [l.strip().split(",") for l in open(DL + "phishing/Training Dataset.arff") if l.strip() and not l.startswith("@")]
    A = np.array(rows, dtype=int); X, y = A[:, :-1], (A[:, -1] > 0).astype(int)
    cols = [np.unique(X[:, j]) for j in range(X.shape[1])]
    B = np.concatenate([(X[:, j:j + 1] == c[None]).astype(np.uint8) for j, c in enumerate(cols)], axis=1)
    i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)
    a, b = split_val(B[i_tr], y[i_tr], 0.1)
    save(name, B[i_tr][a], y[i_tr][a], B[i_tr][b], y[i_tr][b], B[i_te], y[i_te], 0)
