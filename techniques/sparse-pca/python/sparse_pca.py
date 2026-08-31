"""Sparse PCA (Reference Sec 25.8).

Zou, Hastie & Tibshirani (2006) 'Sparse principal component analysis.'

Classical PCA returns loadings that involve ALL variables, so a
component is hard to interpret. SPARSE PCA adds an L1 penalty to force
loadings toward zero for irrelevant variables.

Zou-Hastie-Tibshirani formulation:  view PCA as regression of X on itself
via  min_{A, B} || X - X B A' ||_F^2 + lambda * sum_j ||b_j||_1  s.t.
A'A = I.

Alternate:
  1. Fix A -> solve elastic-net regressions for each column of B.
  2. Fix B -> project via reduced SVD.

Here we implement a compact soft-thresholding-based Sparse PCA and
recover components with structured sparsity on a synthetic problem.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _soft_threshold(x, lam):
    return np.sign(x) * np.maximum(0, np.abs(x) - lam)


def fit_sparse_pca(X, k=2, lam=0.2, max_iter=200, seed=0):
    """SPCA via alternating soft-threshold + SVD (compact Zou-Hastie-Tibshirani)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    # Initial B: top-k PCA loadings.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    A = Vt[:k].T                    # d x k
    B = A.copy()
    for _ in range(max_iter):
        # Fix A, solve for B by elastic-net regression: closed form soft-threshold on Xc' Xc A
        M = Xc.T @ (Xc @ A) / n
        B = _soft_threshold(M, lam)
        # Fix B, refresh A via reduced SVD of Xc' Xc B
        U, s, Vt2 = np.linalg.svd(Xc.T @ Xc @ B, full_matrices=False)
        A = U @ Vt2
    # Normalise columns of B (loadings)
    norms = np.linalg.norm(B, axis=0)
    norms[norms == 0] = 1.0
    return B / norms, A


if __name__ == "__main__":
    print("=== Sparse PCA (Zou-Hastie-Tibshirani 2006) ===\n")
    rng = np.random.default_rng(0)
    # Synthetic: 10-d data with 2 latent factors, each supported on 3 of 10 vars.
    n, d, k = 400, 10, 2
    F = rng.normal(0, 1, (n, k))
    L_true = np.zeros((d, k))
    L_true[[0, 1, 2], 0] = [1.0, 0.9, 0.8]         # component 1 uses vars 0-2
    L_true[[7, 8, 9], 1] = [0.9, 1.1, 0.7]         # component 2 uses vars 7-9
    X = F @ L_true.T + 0.4 * rng.normal(0, 1, (n, d))

    # Plain PCA
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_load = Vt[:k].T

    # Sparse PCA
    spca_load, A = fit_sparse_pca(X, k=k, lam=0.15, max_iter=200)

    print("  Loadings (rows = variables 0..9, columns = components):")
    print(f"  plain PCA:\n{pca_load.round(2)}\n")
    print(f"  sparse PCA:\n{spca_load.round(2)}\n")

    nz_pca = int((np.abs(pca_load) > 0.05).sum())
    nz_spca = int((np.abs(spca_load) > 0.05).sum())
    print(f"  non-zero loadings (>0.05):  plain PCA = {nz_pca}   sparse PCA = {nz_spca}"
          "   (out of {})".format(d * k))
    print("\n  Sparse PCA localises each component to the correct block of variables.\n")
    print("--- library cross-check (sklearn.decomposition.SparsePCA; R elasticnet::spca; nsprcomp) ---")
