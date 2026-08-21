"""Stochastic Block Model — SBM (Reference §24.6).

Generative model for graphs with community structure:

    z_i ~ Categorical(pi)              (block label of node i)
    A_ij | z_i, z_j ~ Bernoulli(B[z_i, z_j])

We fit by:

  * SIMULATION: draw an SBM with known blocks (planted-partition).
  * ESTIMATION: hard EM (label switching) — given labels, MLE for B is
    the empirical within/between edge density; given B, reassign each
    node to the block maximising its complete-data likelihood contribution.
    Iterate.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def simulate_sbm(sizes, B, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = sum(sizes); z = np.repeat(range(len(sizes)), sizes)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < B[z[i], z[j]]:
                A[i, j] = A[j, i] = 1
    return A, z


def _mle_B(A, z, K):
    """MLE of block probabilities given labels."""
    B = np.zeros((K, K)); counts = np.zeros((K, K))
    for r in range(K):
        for s in range(K):
            idx_r = np.where(z == r)[0]; idx_s = np.where(z == s)[0]
            if r == s:
                if len(idx_r) < 2:
                    B[r, s] = 0.0; continue
                sub = A[np.ix_(idx_r, idx_r)]
                total_pairs = len(idx_r) * (len(idx_r) - 1) / 2
                B[r, s] = float(sub.sum()) / (2 * total_pairs) if total_pairs > 0 else 0.0
            else:
                sub = A[np.ix_(idx_r, idx_s)]
                total_pairs = len(idx_r) * len(idx_s)
                B[r, s] = float(sub.sum()) / total_pairs if total_pairs > 0 else 0.0
    return B


def _reassign(A, z, K, B):
    """Reassign each node greedily to the block maximising its log-lik contribution."""
    n = A.shape[0]; z_new = z.copy()
    Bc = np.clip(B, 1e-6, 1 - 1e-6)
    logB = np.log(Bc); log1B = np.log(1 - Bc)
    for i in range(n):
        best = -np.inf; best_k = z[i]
        for k in range(K):
            ll = 0.0
            for j in range(n):
                if i == j:
                    continue
                ll += A[i, j] * logB[k, z_new[j]] + (1 - A[i, j]) * log1B[k, z_new[j]]
            if ll > best:
                best = ll; best_k = k
        z_new[i] = best_k
    return z_new


def fit_sbm(A, K: int, n_iter: int = 20, seed: int = 0) -> dict:
    """Hard EM starting from spectral (leading Laplacian eigenvectors) init."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    # spectral warm start: k-means on top-K eigenvectors of A
    w, V = np.linalg.eigh(A.astype(float))
    U = V[:, -K:]
    # k-means (simple)
    idx = rng.choice(n, K, replace=False); centres = U[idx]
    z = np.zeros(n, dtype=int)
    for _ in range(30):
        d2 = ((U[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        z_new = d2.argmin(axis=1)
        if (z_new == z).all():
            break
        z = z_new
        for k in range(K):
            if (z == k).any():
                centres[k] = U[z == k].mean(axis=0)
    # EM
    for it in range(n_iter):
        B = _mle_B(A, z, K)
        z_new = _reassign(A, z, K, B)
        if (z_new == z).all():
            break
        z = z_new
    B = _mle_B(A, z, K)
    return {"labels": z, "B": B, "K": K, "n_iter": it + 1,
            "method": "SBM hard EM (spectral warm start)"}


def match_labels(z_true, z_hat):
    """Best permutation of z_hat to match z_true (via majority vote per detected block)."""
    K = int(max(z_true.max(), z_hat.max())) + 1
    from collections import Counter
    remap = {}
    for k in range(K):
        mask = z_hat == k
        if mask.any():
            remap[k] = Counter(z_true[mask]).most_common(1)[0][0]
    return np.array([remap.get(zi, zi) for zi in z_hat])


if __name__ == "__main__":
    sizes = [15, 15, 15]; K = len(sizes)
    B_true = np.array([[0.8, 0.05, 0.05],
                       [0.05, 0.7, 0.05],
                       [0.05, 0.05, 0.9]])
    A, z_true = simulate_sbm(sizes, B_true, seed=0)

    print(f"=== SBM fit (n=45, K=3, planted within-p 0.7-0.9, between-p 0.05) ===")
    fit = fit_sbm(A, K, n_iter=20, seed=1)
    print(f"  iterations = {fit['n_iter']}")
    z_matched = match_labels(z_true, fit["labels"])
    acc = float((z_matched == z_true).mean())
    print(f"  block-recovery accuracy = {acc:.3f}")
    print(f"  estimated B (rows/cols in original label order):")
    for r in range(K):
        print("   " + "  ".join(f"{fit['B'][r, s]:6.3f}" for s in range(K)))
    print(f"  true B:")
    for r in range(K):
        print("   " + "  ".join(f"{B_true[r, s]:6.3f}" for s in range(K)))

    print("\n--- library cross-check (R blockmodels; Python graph-tool) ---")
