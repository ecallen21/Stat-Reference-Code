"""Product Quantization for ANN (Reference Sec 47.84).

Jegou, Douze & Schmid 2011 'Product quantization for nearest
neighbor search', IEEE TPAMI 33(1). Split each d-dim vector into
M sub-vectors of dimension d/M; k-means each sub-space
independently to 2^b codewords. A vector is stored as M codebook
indices -- M * b bits, typically 16 * 8 = 128 bits per vector.

Asymmetric-distance approximation:  query is not compressed; each
subvector distance to the 2^b codewords is precomputed once, then
scoring a database vector = M table lookups + sum.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.cluster import KMeans    # per-subspace clustering


def pq_train(X, M, K):
    n, d = X.shape
    assert d % M == 0
    sub = d // M
    codebooks = []
    codes = np.zeros((n, M), dtype=np.int32)
    for m in range(M):
        km = KMeans(n_clusters=K, n_init=10, random_state=m).fit(X[:, m * sub:(m + 1) * sub])
        codebooks.append(km.cluster_centers_)
        codes[:, m] = km.labels_
    return {"codebooks": codebooks, "codes": codes, "sub": sub, "M": M, "K": K}


def pq_dist(query, pq):
    """Return array of approximate distances from query to all coded vectors."""
    n = len(pq["codes"])
    M = pq["M"]; sub = pq["sub"]
    d_tables = np.zeros((M, pq["K"]))
    for m in range(M):
        q_sub = query[m * sub:(m + 1) * sub]
        d_tables[m] = ((pq["codebooks"][m] - q_sub) ** 2).sum(axis=1)
    d_approx = np.zeros(n)
    for m in range(M):
        d_approx += d_tables[m, pq["codes"][:, m]]
    return d_approx


if __name__ == "__main__":
    print("=== Product Quantization (Jegou-Douze-Schmid 2011) ===\n")
    rng = np.random.default_rng(0)
    n = 10_000; d = 64
    X = rng.normal(size=(n, d))
    Q = rng.normal(size=(50, d))

    for M in [4, 8, 16]:
        K = 256
        pq = pq_train(X, M=M, K=K)
        # bits per vector = M * log2(K)
        bits = M * int(np.log2(K))
        # Compare top-10 recall: PQ NN vs exact NN
        recall = 0.0
        for q in Q:
            d_true = ((X - q) ** 2).sum(-1)
            true_top = set(np.argsort(d_true)[:10].tolist())
            d_pq = pq_dist(q, pq)
            pq_top = set(np.argsort(d_pq)[:10].tolist())
            recall += len(true_top & pq_top) / 10
        recall /= len(Q)
        print(f"  M={M:2d}, K={K}  ->  {bits:3d} bits/vec ({bits/8:.0f} B)   "
              f"Recall@10 = {recall:.3f}")

    print("\n  Full-precision baseline: d=64 float32 = 256 bytes/vec.")
    print("  PQ with M=16 compresses ~16x to 16 bytes/vec; recall grows in M.")
    print("\n--- library cross-check (RcppFaiss / rnnR R; faiss / scann Python) ---")
