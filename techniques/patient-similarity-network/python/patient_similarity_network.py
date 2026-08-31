"""Patient similarity network (Reference Sec 30.25).

Build a network of patients where edges reflect FEATURE SIMILARITY;
run community detection to discover SUBTYPES.

Pipeline:
  1. Standardise the patient x feature matrix.
  2. Compute pairwise similarity (Gaussian / cosine / mixed).
  3. K-nearest-neighbour graph (sparsify).
  4. Community detection (Louvain-style modularity here).
  5. Report subtypes + within-subtype feature profiles.

Widely used in precision medicine (Li 2015, Ching 2018) to stratify
heterogeneous cohorts.

Here we implement Gaussian similarity + KNN + a simple label-
propagation community detection on synthetic mixed-diagnosis cohort.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_similarity_network(X, k=5, sigma=None):
    """Gaussian similarity + k-NN sparsification (symmetric)."""
    n = X.shape[0]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))
    if sigma is None:
        sigma = float(np.median(D[D > 0]))
    S = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(S, 0)
    W = np.zeros_like(S)
    for i in range(n):
        nn = np.argsort(-S[i])[:k]
        W[i, nn] = S[i, nn]
    W = np.maximum(W, W.T)     # symmetric
    return W


def label_propagation(W, max_iter=50, seed=0):
    """Raghavan-Albert-Kumara label propagation for community detection."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    labels = np.arange(n)
    for _ in range(max_iter):
        order = rng.permutation(n)
        changed = False
        for i in order:
            nbrs = np.where(W[i] > 0)[0]
            if len(nbrs) == 0: continue
            votes = np.zeros(n)
            for j in nbrs:
                votes[labels[j]] += W[i, j]
            best = int(votes.argmax())
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed: break
    # Relabel to consecutive ids
    uniq, inv = np.unique(labels, return_inverse=True)
    return inv


def cluster_purity(pred, truth):
    K = pred.max() + 1
    hits = 0
    for k in range(K):
        m = pred == k
        if not m.any(): continue
        modal = np.bincount(truth[m]).argmax()
        hits += (truth[m] == modal).sum()
    return hits / len(pred)


if __name__ == "__main__":
    print("=== Patient similarity network + label propagation ===\n")
    rng = np.random.default_rng(0)
    n_per = 20
    d = 6
    # Three subtypes with distinct feature centroids.
    centres = np.array([[3, 0, -1, 1, 0, 0],
                          [-2, 2, 1, -1, 0, 0],
                          [0, -3, 2, 0, 1, -1]])
    truth = np.repeat(np.arange(3), n_per)
    X = np.vstack([centres[k] + rng.normal(0, 1.0, (n_per, d)) for k in range(3)])
    # Standardise
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    W = build_similarity_network(X, k=5)
    labels = label_propagation(W, max_iter=100, seed=0)
    n_communities = len(np.unique(labels))
    purity = cluster_purity(labels, truth)
    print(f"  n patients = {len(X)}   features = {d}   k-NN = 5")
    print(f"  communities discovered = {n_communities}")
    print(f"  cluster purity vs true subtype = {purity:.3f}")
    print("\n  Patient similarity + community detection stratifies the cohort into subtypes.\n")
    print("--- library cross-check (SNFtool R; PSN Python; scikit-network) ---")
