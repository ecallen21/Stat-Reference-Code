"""Independent Component Analysis (ICA) — Reference Sec 25.1.

Hyvarinen & Oja (2000) 'Independent Component Analysis: Algorithms and
Applications' (FastICA).

Given mixed observations X = A * S where A is an unknown mixing matrix
and S are STATISTICALLY INDEPENDENT sources, ICA recovers S up to
permutation + scale by maximising the NON-GAUSSIANITY of the sources
(central-limit theorem: sums of independents are more Gaussian).

FastICA per-component:

  1. Whiten X:  Z = V * (X - mean).
  2. Initialise w; iterate  w <- E[Z g(w'Z)] - E[g'(w'Z)] w  (g = tanh).
  3. Deflate to force orthogonality of the recovered components.

Here we mix two INDEPENDENT non-Gaussian sources (a sine wave + a
sawtooth), verify that FastICA recovers them up to sign / permutation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _whiten(X):
    """Center + PCA-whiten so cov(Z) = I."""
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt / s[:, None] * np.sqrt(len(X))
    Z = Xc @ W.T
    return Z, W


def _g(u): return np.tanh(u)
def _dg(u): return 1.0 - np.tanh(u) ** 2


def fast_ica(X, n_components=None, max_iter=200, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    Z, W_white = _whiten(X)
    n, d = Z.shape
    if n_components is None:
        n_components = d
    W = rng.normal(0, 1, (n_components, d))
    for i in range(n_components):
        w = W[i] / np.linalg.norm(W[i])
        for _ in range(max_iter):
            wZ = Z @ w
            w_new = (Z * _g(wZ)[:, None]).mean(axis=0) - _dg(wZ).mean() * w
            # Deflation: orthogonalise against previous
            for j in range(i):
                w_new -= (w_new @ W[j]) * W[j]
            w_new /= np.linalg.norm(w_new)
            if abs(abs(w_new @ w) - 1) < tol:
                w = w_new; break
            w = w_new
        W[i] = w
    S_hat = Z @ W.T
    A_hat = np.linalg.pinv(W @ W_white)
    return S_hat, W, A_hat


if __name__ == "__main__":
    print("=== FastICA on a two-source mixture (Hyvarinen 2000) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    t = np.linspace(0, 8, n)
    s1 = np.sin(2 * t)                              # sinusoid
    s2 = np.sign(np.sin(3 * t + 0.5))               # square wave
    S = np.stack([s1, s2], axis=1)
    A = np.array([[1.0, 0.5], [0.3, 1.2]])          # mixing matrix
    X = S @ A.T + 0.02 * rng.normal(0, 1, (n, 2))

    S_hat, W_ica, A_hat = fast_ica(X, n_components=2)
    # Match components by max abs correlation
    corr = np.abs(np.corrcoef(S.T, S_hat.T)[:2, 2:])
    # Greedy match
    order = corr.argmax(axis=1)
    print(f"  correlations |corr(S, S_hat)|:\n{np.round(corr, 3)}")
    print(f"  best match sources -> recovered indices: {order.tolist()}")
    for src in range(2):
        print(f"    source {src+1}  <->  recovered {order[src]}   |corr|={corr[src, order[src]]:.3f}")

    print("\n  Recovered sources should have |corr| ~ 1 with true sources (up to sign).\n")
    print("--- library cross-check (sklearn.decomposition.FastICA; R fastICA) ---")
