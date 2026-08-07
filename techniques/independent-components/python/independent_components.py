"""Independent Component Analysis via FastICA (Reference §9.9).

ICA separates a multivariate signal x = A s into STATISTICALLY INDEPENDENT
non-Gaussian source components s.  Cocktail-party problem: recover
individual speakers from microphone mixtures.

    x = A s      (observed mixtures; A is the unknown mixing matrix)
    s: independent, at most one Gaussian component.

Model assumptions
    - Independence (not just decorrelation).
    - Non-Gaussianity: Gaussian mixtures are indeterminate (any rotation
      of independent Gaussian sources is still Gaussian and independent).

FastICA (Hyvarinen 1999)
    1. Center and WHITEN x -> z (E[zz^T] = I).
    2. Find w such that y = w^T z has MAXIMAL non-Gaussianity, measured
       by NEGENTROPY  J(y) ~ (E[G(y)] - E[G(nu)])^2, nu ~ N(0, 1).
    3. Fixed-point update:
        w <- E[z g(w^T z)] - E[g'(w^T z)] w
        with G(u) = log cosh(u), g(u) = tanh(u).
    4. Symmetric decorrelation across components (orthogonalize the W matrix).

Common contrast functions
    G(u) = log cosh(u)       (default; robust)
    G(u) = -exp(-u^2/2)      (kurtosis-based)

The demo below recovers 3 mixed sources; matches sklearn.decomposition.FastICA.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def whiten(X):
    """Center X and project onto principal components with unit variance."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / len(Xc)
    d, E = np.linalg.eigh(C)
    d = np.maximum(d, 1e-12)
    D_inv_sqrt = np.diag(1 / np.sqrt(d))
    W_white = D_inv_sqrt @ E.T
    Z = Xc @ W_white.T
    return Z, W_white, mu


def _g_tanh(u): return np.tanh(u)
def _gp_tanh(u): return 1 - np.tanh(u) ** 2


def fast_ica(X, n_components: int = None, max_iter: int = 200, tol: float = 1e-6,
             seed: int = 0) -> dict:
    """FastICA with logcosh nonlinearity and symmetric decorrelation."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); n, p = X.shape
    k = n_components or p
    Z, W_white, mu = whiten(X)
    W = rng.normal(size=(k, k))
    for _ in range(max_iter):
        Wu = W @ Z.T           # k x n
        g = _g_tanh(Wu)
        gp = _gp_tanh(Wu)
        W_new = (g @ Z) / n - (gp.mean(1)[:, None] * W)
        # Symmetric decorrelation: W <- (W W^T)^(-1/2) W
        U, S, Vt = np.linalg.svd(W_new)
        W_new = U @ np.diag(1 / S) @ U.T @ W_new
        if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < tol: break
        W = W_new
    S_est = (W @ Z.T).T
    A_est = np.linalg.pinv(W @ W_white)  # observation-space mixing matrix
    return {"sources": S_est,
            "unmixing_matrix": W,
            "mixing_matrix_estimate": A_est,
            "whitening_matrix": W_white,
            "mean": mu,
            "n_components": int(k),
            "method": "FastICA (logcosh, symmetric decorrelation)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000; t = np.linspace(0, 8, n)
    # Three non-Gaussian sources
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))            # square wave
    s3 = rng.laplace(size=n)                # heavy-tailed noise
    S_true = np.column_stack([s1, s2, s3])
    A = np.array([[1.0, 0.5, 0.3],
                  [0.4, 1.0, 0.5],
                  [0.2, 0.3, 1.0]])
    X = S_true @ A.T

    print("=== FastICA on 3 mixed sources ===")
    r = fast_ica(X, n_components=3, seed=1)
    S_hat = r["sources"]
    # Match each estimated component to its best-correlated true source (permutation + sign)
    scores = np.abs(np.corrcoef(S_hat.T, S_true.T)[:3, 3:])
    perm = np.argmax(scores, axis=1)
    for k in range(3):
        rho = np.corrcoef(S_hat[:, k], S_true[:, perm[k]])[0, 1]
        print(f"  estimated component {k} <-> true source {perm[k]}: |corr| = {abs(rho):.3f}")

    print("\n--- library cross-check (sklearn FastICA) ---")
    try:
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=3, random_state=1)
        S_sk = ica.fit_transform(X)
        scores = np.abs(np.corrcoef(S_sk.T, S_true.T)[:3, 3:])
        perm = np.argmax(scores, axis=1)
        for k in range(3):
            rho = np.corrcoef(S_sk[:, k], S_true[:, perm[k]])[0, 1]
            print(f"  sklearn component {k} <-> true source {perm[k]}: |corr| = {abs(rho):.3f}")
    except Exception as ex:
        print(f"  (sklearn FastICA unavailable: {ex})")
