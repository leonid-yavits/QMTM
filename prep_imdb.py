"""IMDb encoding per AP(TM) Section 6.1: keras imdb (num_words=5000, index_from=2),
binary unigrams + bigrams, 20,000 most frequent train n-grams as candidates,
5,000 features selected by chi2 on train. Two stages:
  stage 'dump' (run with the venv that has keras): writes flat int arrays
  stage 'encode' (system python): builds the binary feature matrices"""
import sys, numpy as np

RAW = "/home/claude/data/raw/imdb"
TMP = "/home/claude/data/proc/imdb_seq.npz"

if sys.argv[1] == "dump":
    import os, shutil
    os.environ.setdefault("KERAS_BACKEND", "numpy")
    cache = os.path.expanduser("~/.keras/datasets"); os.makedirs(cache, exist_ok=True)
    for f in ["imdb.npz", "imdb_word_index.json"]:
        shutil.copy(f"{RAW}/{f}", f"{cache}/{f}")
    import keras
    (xtr, ytr), (xte, yte) = keras.datasets.imdb.load_data(num_words=5000, index_from=2)
    wi = keras.datasets.imdb.get_word_index()
    def flat(x):
        return np.concatenate([np.asarray(s, np.int32) for s in x]), np.cumsum([0] + [len(s) for s in x])
    a, ao = flat(xtr); b, bo = flat(xte)
    words = np.array(list(wi.keys())); ids = np.array(list(wi.values()), np.int32)
    np.savez(TMP, xtr=a, xtr_off=ao, ytr=np.asarray(ytr, np.uint8), xte=b, xte_off=bo,
             yte=np.asarray(yte, np.uint8), words=words, ids=ids)
    print("dumped", len(ao) - 1, len(bo) - 1)
    sys.exit()

from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.feature_selection import SelectKBest, chi2

z = np.load(TMP)
id2w = {int(i) + 2: str(w) for w, i in zip(z["words"], z["ids"])}
id2w.update({0: "<PAD>", 1: "<START>", 2: "<UNK>"})
def docs(x, off):
    return [[id2w[int(t)] for t in x[off[i]:off[i + 1]]] for i in range(len(off) - 1)]
def grams(terms):
    return set(terms) | {terms[j] + " " + terms[j + 1] for j in range(len(terms) - 1)}
dtr = docs(z["xtr"], z["xtr_off"])
tr = [grams(d) for d in dtr]
te = [grams(d) for d in docs(z["xte"], z["xte_off"])]
cnt = Counter()
for t in dtr:                                  # corpus frequency, as in the pyTM demo
    cnt.update(t); cnt.update(t[j] + " " + t[j + 1] for j in range(len(t) - 1))
cand = [p for p, _ in cnt.most_common(20000)]
col = {p: k for k, p in enumerate(cand)}
def mat(G):
    rows, cols = [], []
    for i, g in enumerate(G):
        for p in g:
            k = col.get(p)
            if k is not None:
                rows.append(i); cols.append(k)
    return csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(len(G), len(cand)))
Xtr, Xte = mat(tr), mat(te)
skb = SelectKBest(chi2, k=5000).fit(Xtr, z["ytr"])
Xtr, Xte = skb.transform(Xtr).toarray().astype(np.uint8), skb.transform(Xte).toarray().astype(np.uint8)
np.savez_compressed("/home/claude/data/proc/imdb.npz", X_train=Xtr, y_train=z["ytr"], X_test=Xte, y_test=z["yte"])
print("imdb", Xtr.shape, Xte.shape, "candidates", len(cand), "ones-frac %.4f" % Xtr.mean(),
      "classes", np.bincount(z["ytr"]), np.bincount(z["yte"]))
