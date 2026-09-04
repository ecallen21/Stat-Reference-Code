"""Document embedding + similarity (Reference Sec 42.9).

Represent a document as a dense vector, then compare via cosine
similarity.  Options:

  * TF-IDF (sparse but classical baseline).
  * Doc2Vec / SBERT (dense neural embeddings; not shown -- external
    heavy deps).
  * MEAN of word-vector components as a compact demo.

Retrieval scoring:
  * COSINE similarity of embeddings.
  * BM25 (Robertson-Sparck-Jones) -- classical IR ranking function.
"""
from __future__ import annotations    # stdlib

import re
from math import log

import numpy as np    # numerical arrays


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def tfidf_matrix(docs):
    tok = [tokenize(d) for d in docs]
    vocab = sorted({w for d in tok for w in d})
    idx = {w: i for i, w in enumerate(vocab)}
    N = len(docs); V = len(vocab)
    tf = np.zeros((N, V))
    for i, toks in enumerate(tok):
        for w in toks:
            tf[i, idx[w]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1
    tfidf = tf / (tf.sum(axis=1, keepdims=True) + 1e-12) * idf
    return tfidf, vocab


def cosine_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return Xn @ Xn.T


def bm25_score(query, docs, k1=1.5, b=0.75):
    tokd = [tokenize(d) for d in docs]
    N = len(docs)
    avgdl = np.mean([len(t) for t in tokd])
    df = {}
    for t in tokd:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    qtok = tokenize(query)
    scores = []
    for i, t in enumerate(tokd):
        s = 0.0
        for w in qtok:
            if w not in df: continue
            idf = log((N - df[w] + 0.5) / (df[w] + 0.5) + 1)
            tf = t.count(w)
            s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(float(s))
    return scores


if __name__ == "__main__":
    print("=== Document embedding + similarity: TF-IDF + cosine + BM25 ===\n")
    docs = [
        "Patient started on aspirin for chest pain.",
        "Started aspirin for prophylaxis after chest pain episode.",
        "The soccer team won the game last night.",
        "Insulin dose was increased for diabetes management.",
    ]
    X, vocab = tfidf_matrix(docs)
    print(f"  vocabulary size = {len(vocab)}")
    C = cosine_matrix(X)
    print(f"  cosine similarity matrix (rows/cols in doc order):\n{np.round(C, 3)}\n")

    query = "aspirin chest pain"
    print(f"  BM25 scores for query {query!r}:")
    for i, s in enumerate(bm25_score(query, docs)):
        print(f"    doc {i}: {s:.3f}   -- {docs[i]!r}")

    print("\n--- library cross-check (R text2vec, quanteda; Python sentence-transformers, sklearn) ---")
