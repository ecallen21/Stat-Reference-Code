"""Robust PCA (Reference Sec 25.13).

Candes, Li, Ma & Wright (2011) 'Robust principal component analysis?'

Decompose an observed matrix M into a LOW-RANK part L and a SPARSE
part S:

  min_{L, S}  ||L||_*  +  lambda * ||S||_1     s.t.  M = L + S.

Solved via ADMM / IALM (Inexact Augmented Lagrange Multiplier).
Recovers the low-rank component even under GROSS but SPARSE
corruption (e.g. video background modelling, sensor spikes,
face-illumination artefacts).

Here we implement a compact ADMM for RPCA, apply to a corrupted rank-2
matrix and verify recovery.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _soft(x, tau): return np.sign(x) * np.maximum(0, np.abs(x) - tau)


def _svt(X, tau):
    """Singular value thresholding: soft-threshold singular values."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_th = np.maximum(0, s - tau)
    return U @ np.diag(s_th) @ Vt, int((s_th > 0).sum())


def robust_pca(M, lam=None, max_iter=200, tol=1e-5, mu=None):
    """Principal Component Pursuit (Candes 2011) via ADMM."""
    n, d = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))
    if mu is None:
        mu = 0.25 * n * d / max(1e-9, float(np.sum(np.abs(M))))
    L = np.zeros_like(M); S = np.zeros_like(M); Y = np.zeros_like(M)
    for it in range(max_iter):
        L, rank_L = _svt(M - S + Y / mu, 1.0 / mu)
        S = _soft(M - L + Y / mu, lam / mu)
        Y = Y + mu * (M - L - S)
        err = float(np.linalg.norm(M - L - S, ord="fro") / max(1e-9, np.linalg.norm(M, ord="fro")))
        if err < tol:
            break
    return L, S, {"rank_L": rank_L, "iterations": it + 1, "rel_error": err}


if __name__ == "__main__":
    print("=== Robust PCA (Candes-Li-Ma-Wright 2011) ===\n")
    rng = np.random.default_rng(0)
    n, d, r = 40, 30, 2
    # Low-rank truth
    U = rng.normal(0, 1, (n, r))
    V = rng.normal(0, 1, (d, r))
    L_true = U @ V.T
    # Sparse corruption
    S_true = np.zeros((n, d))
    corrupt_mask = rng.random((n, d)) < 0.10
    S_true[corrupt_mask] = rng.choice([-3, 3], size=int(corrupt_mask.sum()))
    M = L_true + S_true

    L_hat, S_hat, info = robust_pca(M)
    rank_hat = info["rank_L"]

    err_L = float(np.linalg.norm(L_hat - L_true, ord="fro") / np.linalg.norm(L_true, ord="fro"))
    tp = int(((np.abs(S_hat) > 0.5) & corrupt_mask).sum())
    fp = int(((np.abs(S_hat) > 0.5) & ~corrupt_mask).sum())
    fn = int(((np.abs(S_hat) <= 0.5) & corrupt_mask).sum())

    print(f"  true rank(L) = {r}   estimated rank(L) = {rank_hat}")
    print(f"  Frobenius relative error of L: {err_L:.4f}")
    print(f"  sparse corruption recovery:    TP={tp}   FP={fp}   FN={fn}"
          f"   (n_corrupt = {int(corrupt_mask.sum())})")
    print(f"  iterations = {info['iterations']}   final |M - L - S|/|M| = {info['rel_error']:.2e}\n")

    print("--- library cross-check (R rpca; Python fbpca + custom; splitpca) ---")
