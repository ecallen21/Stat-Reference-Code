"""Node embeddings: DeepWalk + node2vec (Reference Sec 30.19).

Perozzi, Al-Rfou & Skiena (2014) 'DeepWalk: Online learning of social
representations.'
Grover & Leskovec (2016) 'node2vec: Scalable feature learning for
networks.'

Represent each node by a low-dim vector so that nodes co-occurring in
RANDOM WALKS have similar embeddings.

Algorithm:
  1. Generate M random walks of length L from each node.
  2. Treat walks as 'sentences'; train Skip-gram / Word2Vec on the
     (node, neighbour-in-window) pairs.

node2vec generalises DeepWalk by biasing walks with parameters (p, q):
  * p (return): prob of returning to previous node.
  * q (in-out): prob of exploring further (BFS vs DFS).

Here we implement DeepWalk with a compact NumPy skip-gram using
matrix factorisation (Levy-Goldberg 2014 equivalence).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def random_walks(A, n_walks=10, walk_len=10, seed=0):
    """Uniform random walks on adjacency A (numpy square matrix)."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    walks = []
    for _ in range(n_walks):
        for start in range(n):
            walk = [start]
            for _ in range(walk_len - 1):
                cur = walk[-1]
                nbrs = np.where(A[cur] > 0)[0]
                if len(nbrs) == 0: break
                walk.append(int(rng.choice(nbrs)))
            walks.append(walk)
    return walks


def cooccurrence_matrix(walks, n, window=3):
    """Symmetric co-occurrence within a window."""
    C = np.zeros((n, n))
    for walk in walks:
        for i, u in enumerate(walk):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    C[u, walk[j]] += 1
    return C


def deepwalk_embeddings(A, d=8, n_walks=10, walk_len=10, window=3, seed=0):
    walks = random_walks(A, n_walks, walk_len, seed=seed)
    n = A.shape[0]
    C = cooccurrence_matrix(walks, n, window=window)
    # Positive shifted PMI (SGNS-equivalent, Levy-Goldberg 2014).
    total = C.sum()
    row = C.sum(axis=1) + 1e-6
    PMI = np.log((C * total) / (row[:, None] * row[None, :]) + 1e-6)
    PMI = np.maximum(PMI, 0.0)
    U, s, _ = np.linalg.svd(PMI, full_matrices=False)
    return U[:, :d] * np.sqrt(s[:d])


if __name__ == "__main__":
    print("=== DeepWalk / node2vec embeddings via PMI + SVD ===\n")
    rng = np.random.default_rng(0)
    # Two dense clusters connected by a single bridge edge.
    n = 20
    A = np.zeros((n, n))
    for i in range(10):
        for j in range(i + 1, 10):
            if rng.random() < 0.6:
                A[i, j] = A[j, i] = 1
    for i in range(10, n):
        for j in range(i + 1, n):
            if rng.random() < 0.6:
                A[i, j] = A[j, i] = 1
    A[3, 15] = A[15, 3] = 1                          # bridge

    Z = deepwalk_embeddings(A, d=4, n_walks=20, walk_len=15, window=3, seed=0)

    # Check: embeddings within a cluster should be closer than across.
    from itertools import product
    same_cluster = []
    diff_cluster = []
    for i, j in product(range(n), repeat=2):
        if i >= j: continue
        d_ij = float(np.linalg.norm(Z[i] - Z[j]))
        if (i < 10) == (j < 10):
            same_cluster.append(d_ij)
        else:
            diff_cluster.append(d_ij)
    print(f"  mean intra-cluster distance = {np.mean(same_cluster):.3f}")
    print(f"  mean cross-cluster distance = {np.mean(diff_cluster):.3f}")
    print(f"  ratio (cross / intra)        = {np.mean(diff_cluster) / max(np.mean(same_cluster), 1e-6):.2f}"
          "   (>= 1 = clusters are separated)\n")
    print("--- library cross-check (node2vec pip pkg; gensim Word2Vec on walks; PyG) ---")
