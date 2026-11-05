"""Reuters-21578 ApteMod (NLTK copy) -> R8 and R52 single-label sets (Cardoso-Cachopo's definition: documents with
exactly one topic; R52 = topics with >= 1 such training and >= 1 such test document; R8 = the 8 most frequent).
Binary bag of words: lowercase alphabetic tokens of length >= 2, NLTK English stopwords removed, document frequency
>= 2 in the training split. Validation = stratified 15 % of the training split (seed 0). Chi-square feature selection
on the reduced training split only (V features, V in argv). Writes /home/claude/data/study1/<name>_V<V>.npz.
python3 prep_reuters.py <V> [<V> ...]   (V = 0 keeps the whole vocabulary)"""
import collections, re, sys, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
R = "/home/claude/data/study1/reuters/"
stop = set(open(R + "stopwords").read().split())
cats = {}
for l in open(R + "cats.txt"):
    f = l.split(); cats[f[0]] = f[1:]
single = {d: c[0] for d, c in cats.items() if len(c) == 1}
tr = collections.Counter(c for d, c in single.items() if d.startswith("training/")); te = collections.Counter(c for d, c in single.items() if d.startswith("test/"))
r52 = sorted(c for c in tr if te[c] > 0); r8 = sorted(sorted(r52, key=lambda c: -(tr[c] + te[c]))[:8])
text = lambda d: open(R + d, encoding="latin-1").read()
tok = lambda s: [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in stop]
for name, keep in [("r8", r8), ("r52", r52)]:
    D = sorted(d for d, c in single.items() if c in keep); lab = {c: i for i, c in enumerate(keep)}
    trd = [d for d in D if d.startswith("training/")]; ted = [d for d in D if d.startswith("test/")]
    ytr_all = np.array([lab[single[d]] for d in trd]); yte = np.array([lab[single[d]] for d in ted])
    print(name, len(keep), "classes", len(trd), "train", len(ted), "test")
    # rare classes (one training doc) cannot be stratified: put them in the training part
    cnt = collections.Counter(ytr_all); idx = np.arange(len(trd)); rare = np.array([cnt[y] < 2 for y in ytr_all])
    a, b = train_test_split(idx[~rare], test_size=0.15, random_state=0, stratify=ytr_all[~rare]); a = np.concatenate([a, idx[rare]])
    cv = CountVectorizer(analyzer=tok, binary=True, min_df=2)
    Xa = cv.fit_transform([text(trd[i]) for i in a]); Xb = cv.transform([text(trd[i]) for i in b]); Xt = cv.transform([text(d) for d in ted])
    ya, yb = ytr_all[a], ytr_all[b]
    for V in map(int, sys.argv[1:]):
        if V and V < Xa.shape[1]:
            sc, _ = chi2(Xa, ya); sc = np.nan_to_num(sc); sel = np.sort(np.argsort(-sc, kind="stable")[:V])
        else:
            sel = np.arange(Xa.shape[1])
        f = lambda X: np.asarray(X[:, sel].todense(), dtype=np.uint8)
        out = dict(X_train=f(Xa), y_train=ya.astype(np.uint16), X_val=f(Xb), y_val=yb.astype(np.uint16), X_test=f(Xt), y_test=yte.astype(np.uint16))
        np.savez_compressed(f"/home/claude/data/study1/{name}_V{V}.npz", **out)
        dens = out["X_train"].mean()
        print(f"  V={V or Xa.shape[1]} features, density {100 * dens:.3f} %, active/sample {out['X_train'].sum(1).mean():.1f}, empty test docs {(out['X_test'].sum(1) == 0).sum()}")
