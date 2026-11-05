"""LEDGAR (LexGLUE, coastalcph/lex_glue on Hugging Face): 100 provision classes, official train/validation/test splits
(60,000 / 10,000 / 10,000). Binary bag of words with the tokenizer of prep_reuters.py (lowercase alphabetic tokens of
length >= 2, NLTK English stopwords removed), document frequency >= 2 in the training split, chi-square selection to V
features on the training split only. python3 prep_ledgar.py <V> [<V> ...]"""
import re, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
D = "/home/claude/data/study1/"; stop = set(open(D + "reuters/stopwords").read().split())
tok = lambda s: [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in stop]
S = {s: pd.read_parquet(D + f"dl/ledgar/{s}.parquet") for s in ["train", "validation", "test"]}
cv = CountVectorizer(analyzer=tok, binary=True, min_df=2)
Xa = cv.fit_transform(S["train"]["text"]); Xb = cv.transform(S["validation"]["text"]); Xt = cv.transform(S["test"]["text"])
ya, yb, yt = (S[s]["label"].to_numpy() for s in ["train", "validation", "test"])
sc, _ = chi2(Xa, ya); sc = np.nan_to_num(sc)
for V in map(int, sys.argv[1:]):
    sel = np.sort(np.argsort(-sc, kind="stable")[:V]) if V else np.arange(Xa.shape[1])
    f = lambda X: np.asarray(X[:, sel].todense(), dtype=np.uint8)
    o = dict(X_train=f(Xa), y_train=ya.astype(np.uint16), X_val=f(Xb), y_val=yb.astype(np.uint16), X_test=f(Xt), y_test=yt.astype(np.uint16))
    np.savez_compressed(D + f"ledgar_V{V}.npz", **o)
    print(f"ledgar_V{V}: K={len(np.unique(ya))} train {len(ya)} val {len(yb)} test {len(yt)} features {o['X_train'].shape[1]} (vocabulary {Xa.shape[1]}) density {100 * o['X_train'].mean():.3f} % active/sample {o['X_train'].sum(1).mean():.1f}")
