"""Latent semantic analysis / indexing (Reference Sec 42.13).

Deerwester et al. 1990.  Truncated SVD of a term-document matrix M:

    M ~ U_k * Sigma_k * V_k^T

  * ROW  vectors of U_k Sigma_k -> latent word vectors.
  * COL  vectors of V_k Sigma_k -> latent document vectors.

Documents that share NO literal terms can still have high cosine
similarity in the latent space (synonymy capture).

Query -> latent space:  q_hat = q^T * U_k * Sigma_k^{-1}
"""
from __future__ import annotations    # stdlib

import re

import numpy as np    # numerical arrays


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def _tdm(docs):
    tok = [tokenize(d) for d in docs]
    vocab = sorted({w for d in tok for w in d})
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(vocab), len(docs)))
    for j, t in enumerate(tok):
        for w in t:
            M[idx[w], j] += 1
    return M, vocab, idx


def lsa(docs, k=2):
    M, vocab, idx = _tdm(docs)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    U_k = U[:, :k]; S_k = S[:k]; V_k = Vt[:k, :].T
    doc_vecs = V_k * S_k
    return {"U": U_k, "S": S_k, "V": V_k, "doc_vecs": doc_vecs,
            "vocab": vocab, "idx": idx}


def query_lsa(model, q_text):
    q = np.zeros(len(model["vocab"]))
    for w in tokenize(q_text):
        if w in model["idx"]:
            q[model["idx"][w]] += 1
    q_hat = q @ model["U"] / (model["S"] + 1e-12)
    return q_hat


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


if __name__ == "__main__":
    print("=== LSA / LSI: truncated SVD of term-document matrix ===\n")
    docs = [
        "human interface computer",
        "user survey computer response",
        "system engineering testing",
        "eps user interface system",
        "trees graph minor",
        "graph minor survey",
    ]
    m = lsa(docs, k=2)
    print(f"  vocabulary size = {len(m['vocab'])}")
    print(f"  singular values = {np.round(m['S'], 3)}\n")
    print(f"  Doc-doc similarity in latent space (2D):")
    D = m["doc_vecs"]
    for i in range(len(docs)):
        row = "  ".join(f"{cos(D[i], D[j]):+.2f}" for j in range(len(docs)))
        print(f"    d{i}:  {row}")

    print("\n  Query 'human computer interaction' -> nearest documents:")
    q = query_lsa(m, "human computer interaction")
    sims = [cos(q, D[j]) for j in range(len(docs))]
    order = np.argsort(-np.array(sims))
    for j in order:
        print(f"    d{j}: cos = {sims[j]:+.3f}   -- {docs[j]!r}")

    print("\n--- library cross-check (R lsa/text2vec/irlba; Python sklearn.TruncatedSVD, gensim.LsiModel) ---")
