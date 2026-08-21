"""Community detection: modularity, greedy agglomerative, spectral (Reference §24.3).

Modularity (Newman-Girvan) of a partition {c(i)}:

    Q = (1 / 2m) * sum_ij [A_ij - k_i k_j / (2m)] * 1{c(i) == c(j)}

Greedy modularity (Clauset-Newman-Moore): start with each node in its own
community; repeatedly merge the pair whose merge maximises delta_Q; stop when
no merge helps.

Spectral community detection: sign of the leading eigenvector of the
modularity matrix B_ij = A_ij - k_i k_j / (2m) gives a 2-way split;
recurse on each side for more communities.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def modularity(A, labels) -> float:
    A = np.asarray(A, dtype=float); labels = np.asarray(labels)
    m = A.sum() / 2.0
    if m == 0:
        return 0.0
    k = A.sum(axis=1)
    B = A - np.outer(k, k) / (2 * m)
    same = (labels[:, None] == labels[None, :]).astype(float)
    return float((B * same).sum() / (2 * m))


def greedy_modularity(A, max_iter: int = 500) -> dict:
    A = np.asarray(A, dtype=float); n = A.shape[0]
    labels = np.arange(n)
    best_Q = modularity(A, labels)
    for _ in range(max_iter):
        unique = np.unique(labels); improved = False
        best_pair = None; best_delta = 0.0
        for i, ci in enumerate(unique):
            for cj in unique[i + 1:]:
                trial = np.where(labels == cj, ci, labels)
                Q = modularity(A, trial)
                if Q - best_Q > best_delta + 1e-12:
                    best_delta = Q - best_Q; best_pair = (ci, cj)
                    improved = True
        if not improved:
            break
        ci, cj = best_pair
        labels = np.where(labels == cj, ci, labels)
        best_Q += best_delta
    # relabel 0..k-1
    _, labels = np.unique(labels, return_inverse=True)
    return {"labels": labels, "modularity": float(best_Q),
            "n_communities": int(labels.max() + 1),
            "method": "greedy agglomerative modularity"}


def spectral_split(A) -> np.ndarray:
    """Newman's leading-eigenvector modularity split (2-way)."""
    A = np.asarray(A, dtype=float); n = A.shape[0]
    m = A.sum() / 2.0; k = A.sum(axis=1)
    B = A - np.outer(k, k) / (2 * m)
    w, V = np.linalg.eigh(B)
    v = V[:, -1]
    return (v > 0).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3-block SBM: within-p 0.8, between-p 0.05
    sizes = [10, 10, 10]; n = sum(sizes); truth = np.repeat(range(len(sizes)), sizes)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.8 if truth[i] == truth[j] else 0.05
            A[i, j] = A[j, i] = int(rng.uniform() < p)

    print("=== 3-block SBM (10 nodes each; within 0.80, between 0.05) ===")
    res = greedy_modularity(A)
    print(f"  greedy modularity Q = {res['modularity']:.4f}")
    print(f"  # detected communities = {res['n_communities']}")

    # confusion vs truth
    from collections import Counter
    def _accuracy(labels, truth):
        # best label alignment: majority vote inside each detected community
        acc = 0
        for c in np.unique(labels):
            mask = labels == c
            top = Counter(truth[mask]).most_common(1)[0][0]
            acc += int((truth[mask] == top).sum())
        return acc / len(truth)
    print(f"  clustering accuracy vs truth = {_accuracy(res['labels'], truth):.3f}")

    sp = spectral_split(A)
    print(f"\n  spectral 2-way split modularity = {modularity(A, sp):.4f}")

    print("\n--- library cross-check (networkx greedy_modularity_communities) ---")
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities, modularity as _nxmod
        G = nx.from_numpy_array(A)
        comms = list(greedy_modularity_communities(G))
        Q_nx = _nxmod(G, comms)
        print(f"  nx greedy Q = {Q_nx:.4f}   nx # comms = {len(comms)}")
    except ImportError:
        print("  (networkx not installed)")
