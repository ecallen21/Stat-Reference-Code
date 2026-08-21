"""TF-IDF and BM25 (Reference §25.2).

Given a document corpus D of N documents over a vocabulary V:

  TF-IDF:
      tf(t, d)          = raw count of term t in d
      idf(t)             = log( (N + 1) / (df(t) + 1) ) + 1   (sklearn smoothing)
      tf-idf(t, d)       = tf(t, d) * idf(t)                   (typically L2-normalised)

  Okapi BM25 (Robertson-Sparck Jones et al. 1994):
      BM25(t, d) = idf_bm25(t) * tf(t, d) * (k1 + 1)
                    / ( tf(t, d) + k1 * (1 - b + b * |d| / avg|d|) )
      idf_bm25(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )      (with clip at 0)
  k1 in [1.2, 2.0]; b ~ 0.75 typical.

BM25 saturates tf (no reward past ~k1 repetitions) and length-normalises,
which is why it beats plain TF-IDF for retrieval.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def build_vocab(docs) -> dict:
    vocab = {}
    for d in docs:
        for w in d:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def tf_matrix(docs, vocab):
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            X[i, vocab[w]] = c
    return X


def tfidf(docs, l2_normalise: bool = True) -> dict:
    vocab = build_vocab(docs)
    X = tf_matrix(docs, vocab)
    N, V = X.shape
    df = (X > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1                # sklearn convention
    W = X * idf[None, :]
    if l2_normalise:
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = np.where(norms > 0, W / norms, 0.0)
    return {"vocab": vocab, "tfidf": W, "idf": idf}


def bm25_score(query, docs, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    N = len(docs)
    doc_lens = np.array([len(d) for d in docs])
    avgdl = doc_lens.mean() if N else 0.0
    # doc-term counts
    df_map = Counter()
    tf_maps = [Counter(d) for d in docs]
    for tm in tf_maps:
        for t in tm:
            df_map[t] += 1
    scores = np.zeros(N)
    for t in query:
        if t not in df_map:
            continue
        df = df_map[t]
        idf = max(math.log((N - df + 0.5) / (df + 0.5) + 1.0), 0.0)
        for i, tm in enumerate(tf_maps):
            tf = tm.get(t, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * doc_lens[i] / avgdl)
            scores[i] += idf * tf * (k1 + 1) / denom
    return scores


if __name__ == "__main__":
    docs = [
        "the quick brown fox jumps over the lazy dog".split(),
        "never trust a computer that you cannot throw out a window".split(),
        "a stitch in time saves nine".split(),
        "the early bird catches the worm early in the morning".split(),
        "the quick brown fox is quick and the brown fox is brown".split(),
    ]

    res = tfidf(docs)
    print(f"=== TF-IDF on 5 documents, |V| = {len(res['vocab'])} ===")
    print(f"  IDF of 'the'          = {res['idf'][res['vocab']['the']]:.3f}   "
          f"(low — very common)")
    print(f"  IDF of 'stitch'       = {res['idf'][res['vocab']['stitch']]:.3f}   (high)")
    print(f"  L2-norm of doc 0      = "
          f"{float(np.linalg.norm(res['tfidf'][0])):.3f}   (should be 1)")

    # query with BM25
    query = "quick brown fox".split()
    scores = bm25_score(query, docs, k1=1.5, b=0.75)
    print(f"\n=== BM25 scores for query {query!r} ===")
    for i, s in enumerate(scores):
        print(f"  doc {i} (|d|={len(docs[i])}): {s:.3f}   \"{' '.join(docs[i])[:60]}...\"")
    rank = np.argsort(-scores)
    print(f"\n  ranking: {rank.tolist()}   (doc 4 has repeated query terms → high)")

    print("\n--- library cross-check (sklearn TfidfVectorizer / rank_bm25) ---")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(tokenizer=lambda x: x.split(), token_pattern=None,
                            lowercase=False, norm="l2")
        X = v.fit_transform([" ".join(d) for d in docs])
        print(f"  sklearn tfidf shape = {X.shape}   L2-norm doc0 = "
              f"{float(np.linalg.norm(X.toarray()[0])):.3f}")
    except ImportError:
        print("  (sklearn not installed)")
    try:
        from rank_bm25 import BM25Okapi
        bm = BM25Okapi(docs)
        print(f"  rank_bm25 scores for query = {[round(x, 3) for x in bm.get_scores(query).tolist()]}")
    except ImportError:
        print("  (rank_bm25 not installed — install with 'pip install rank-bm25' for cross-check)")
