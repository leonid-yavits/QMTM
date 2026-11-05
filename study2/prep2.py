"""Study 2C datasets -> /home/claude/data/study2/<name>.npz (uint8 X, uint16 y), splits as in PROTOCOL.md (C).
Text (agnews, trec, emotion): lowercase alphabetic tokens of length >= 2, NLTK English stopwords removed (not for TREC, amendment A1),
binary bag of words, document frequency >= 2 on the training part, chi-square selection to 5,000 features on the training part.
Dense (usps, letter, satimage; LIBSVM): 10 thresholds at training quantiles 5, 15, ..., 95 % per feature (as HAR in study 1).
Validation: official split if present (emotion; letter/satimage .val with .tr as training), else stratified 10 % of train (seed 0).
python3 prep2.py <name>"""
import bz2, collections, re, sys, numpy as np, pyarrow.parquet as pq
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
DL, OUT = "/home/claude/data/study2/dl/", "/home/claude/data/study2/"
name = sys.argv[1]


def split_val(n, y, frac=0.1):
    cnt = collections.Counter(y); idx = np.arange(n); rare = np.array([cnt[v] < 2 for v in y])
    a, b = train_test_split(idx[~rare], test_size=frac, random_state=0, stratify=y[~rare])
    return np.concatenate([a, idx[rare]]), b


def save(Xa, ya, Xb, yb, Xt, yt):
    np.savez_compressed(OUT + name + ".npz", X_train=Xa, y_train=ya.astype(np.uint16), X_val=Xb, y_val=yb.astype(np.uint16), X_test=Xt, y_test=yt.astype(np.uint16))
    print(f"{name}: K={len(np.unique(ya))} train {len(ya)} val {len(yb)} test {len(yt)} features {Xa.shape[1]} "
          f"density {100 * Xa.mean():.3f} % active/sample {Xa.sum(1).mean():.1f} empty test rows {(Xt.sum(1) == 0).sum()}", flush=True)


if name in ("agnews", "trec", "emotion"):
    stop = set(open("/home/claude/data/study1/reuters/stopwords").read().split()) if name != "trec" else set()
    tok = lambda s: [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in stop]
    col = "label_coarse" if name == "trec" else "label"
    rd = lambda s: pq.read_table(DL + f"{name}/{s}.parquet").to_pydict()
    tr, te = rd("train"), rd("test")
    T, y = tr["text"], np.array(tr[col]); Tt, yt = te["text"], np.array(te[col])
    if name == "emotion":
        va = rd("validation"); Ta, ya, Tb, yb = T, y, va["text"], np.array(va[col])
    else:
        a, b = split_val(len(y), y); Ta, ya, Tb, yb = [T[i] for i in a], y[a], [T[i] for i in b], y[b]
    cv = CountVectorizer(analyzer=tok, binary=True, min_df=2)
    Xa = cv.fit_transform(Ta); Xb = cv.transform(Tb); Xt = cv.transform(Tt)
    V = 5000
    if Xa.shape[1] > V:
        sc, _ = chi2(Xa, ya); sel = np.sort(np.argsort(-np.nan_to_num(sc), kind="stable")[:V])
    else:
        sel = np.arange(Xa.shape[1])
    f = lambda X: np.asarray(X[:, sel].todense(), dtype=np.uint8)
    save(f(Xa), ya, f(Xb), yb, f(Xt), yt)
else:
    L = DL + "libsvm/"
    if name == "usps":
        Xr, y = load_svmlight_file(bz2.open(L + "usps.bz2")); Xt, yt = load_svmlight_file(bz2.open(L + "usps.t.bz2"), n_features=Xr.shape[1])
        Xr, Xt = Xr.toarray(), Xt.toarray(); a, b = split_val(len(y), y); Xa, ya, Xb, yb = Xr[a], y[a], Xr[b], y[b]
    else:
        Xa, ya = load_svmlight_file(L + f"{name}.scale.tr"); n = Xa.shape[1]
        Xb, yb = load_svmlight_file(L + f"{name}.scale.val", n_features=n); Xt, yt = load_svmlight_file(L + f"{name}.scale.t", n_features=n)
        Xa, Xb, Xt = Xa.toarray(), Xb.toarray(), Xt.toarray()
    lab = {v: i for i, v in enumerate(np.unique(ya))}
    ya, yb, yt = (np.array([lab[v] for v in z]) for z in (ya, yb, yt))
    q = np.quantile(Xa, np.linspace(0.05, 0.95, 10), axis=0)
    th = lambda Z: (Z[:, :, None] > q.T[None]).astype(np.uint8).reshape(len(Z), -1)
    save(th(Xa), ya, th(Xb), yb, th(Xt), yt)
