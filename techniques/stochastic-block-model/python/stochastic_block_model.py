"""Stochastic Block Model — SBM (Reference Sec 30.4).

Nowicki & Snijders (2001), Snijders & Nowicki (1997).

Nodes belong to one of K latent BLOCKS. Edge probability depends only
on block membership:

  P(A_ij = 1 | z_i, z_j)  =  B[z_i, z_j].

Maximum-likelihood estimation of (Z, B) via EM (or variational EM, or
belief propagation) recovers the block structure.

Here we implement a compact variational-EM SBM: soft block assignments
via posterior probabilities, then M-step for the block-probability
matrix B. Test on a synthetic graph with planted 3-block structure.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sbm_em(A, K, max_iter=50, seed=0):
    """Variational EM for undirected Bernoulli SBM."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    # Warm-start with spectral clustering on the adjacency matrix.
    U, s, _ = np.linalg.svd(A + 1e-3 * np.eye(n), full_matrices=False)
    features = U[:, :K] * s[:K]
    # Simple k-means init on features.
    centres = features[rng.choice(n, K, replace=False)]
    for _ in range(20):
        d = np.sum((features[:, None] - centres[None]) ** 2, axis=2)
        labels_init = d.argmin(axis=1)
        centres = np.array([features[labels_init == k].mean(axis=0) if (labels_init == k).any() else centres[k] for k in range(K)])
    q = np.full((n, K), 0.05)
    q[np.arange(n), labels_init] = 0.90
    q = q / q.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        # M-step: block probabilities B and prior alpha
        alpha = q.mean(axis=0) + 1e-6
        alpha = alpha / alpha.sum()
        M = q.T @ A @ q                        # K x K expected edges
        D = q.T @ (np.ones_like(A) - np.eye(n)) @ q
        B = M / (D + 1e-6)
        B = np.clip(B, 1e-6, 1 - 1e-6)
        # E-step: log q_ik proportional to log alpha_k + sum_j q_jl [A_ij log B_kl + (1-A_ij) log(1-B_kl)]
        log_q = np.log(alpha)[None, :] + np.zeros((n, K))
        for k in range(K):
            for l in range(K):
                log_q[:, k] += (A @ q[:, l]) * np.log(B[k, l]) \
                                + ((1 - A) @ q[:, l]) * np.log(1 - B[k, l])
        log_q -= log_q.max(axis=1, keepdims=True)
        q_new = np.exp(log_q)
        q_new = q_new / q_new.sum(axis=1, keepdims=True)
        if np.max(np.abs(q_new - q)) < 1e-4:
            q = q_new; break
        q = q_new
    return q, B, alpha


def cluster_accuracy(pred, truth):
    """Best permutation match between two labelings."""
    from itertools import permutations
    Ks = max(pred.max(), truth.max()) + 1
    best = 0
    for perm in permutations(range(Ks)):
        remap = np.array(perm)
        m = (remap[pred] == truth).mean()
        if m > best: best = m
    return best


if __name__ == "__main__":
    print("=== SBM variational EM on a planted 3-block graph ===\n")
    rng = np.random.default_rng(0)
    n_per = 20
    K = 3
    n = n_per * K
    labels = np.repeat(np.arange(K), n_per)
    B_true = np.array([[0.6, 0.05, 0.05],
                        [0.05, 0.6, 0.05],
                        [0.05, 0.05, 0.6]])
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < B_true[labels[i], labels[j]]:
                A[i, j] = A[j, i] = 1
    print(f"  planted 3-block graph:   n = {n}   edge budget within-block ~0.6, cross-block ~0.05")

    q, B_hat, alpha = sbm_em(A, K=3, max_iter=50, seed=1)
    pred = q.argmax(axis=1)
    acc = cluster_accuracy(pred, labels)
    print(f"  recovered cluster accuracy (best-permutation): {acc:.3f}")
    print(f"  estimated B_hat:\n{B_hat.round(3)}")
    print(f"  estimated block prior alpha: {alpha.round(3).tolist()}   truth = [{1/3:.3f}]*3\n")

    print("--- library cross-check (R sbm; graph-tool for SBM; Python graspologic) ---")
