"""Gaussian Graphical Model / Graphical LASSO (Reference Sec 30.8).

Friedman, Hastie & Tibshirani (2008) 'Sparse inverse covariance
estimation with the graphical LASSO.'

For X ~ N(0, Sigma), Omega = Sigma^{-1} has an intuitive graph
interpretation:

  Omega_{ij} = 0  iff  X_i is CONDITIONALLY INDEPENDENT of X_j
              given all other variables.

So learning a sparse Omega recovers a network of conditional
dependencies among features.

Graphical LASSO objective:

  min_Omega  -log det(Omega) + tr(S Omega) + rho * sum_{i != j} |Omega_ij|.

Here we implement a coordinate-descent variant (the block-coordinate
graphical LASSO) using sklearn's fast implementation and demonstrate
recovery of a planted precision-matrix structure.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def graphical_lasso_fit(X, rho=0.1):
    from sklearn.covariance import GraphicalLasso as GL
    gl = GL(alpha=rho, max_iter=200)
    gl.fit(X)
    return gl.precision_


if __name__ == "__main__":
    print("=== Gaussian Graphical Model / Graphical LASSO ===\n")
    rng = np.random.default_rng(0)
    d = 6
    # Truth: precision matrix with a sparse pattern.
    Omega_true = np.array([
        [ 2.0, -0.5,  0.0,  0.0,  0.0,  0.0],
        [-0.5,  2.0, -0.5,  0.0,  0.0,  0.0],
        [ 0.0, -0.5,  2.0, -0.5,  0.0,  0.0],
        [ 0.0,  0.0, -0.5,  2.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  2.0, -0.5],
        [ 0.0,  0.0,  0.0,  0.0, -0.5,  2.0],
    ])
    Sigma_true = np.linalg.inv(Omega_true)
    n = 400
    X = rng.multivariate_normal(np.zeros(d), Sigma_true, n)

    Omega_hat = graphical_lasso_fit(X, rho=0.05)
    print("  true precision (edges where |Omega| > 0):")
    print((np.abs(Omega_true) > 1e-6).astype(int))
    print("\n  estimated precision (|Omega| > 0.02):")
    print((np.abs(Omega_hat) > 0.02).astype(int))
    # Off-diagonal support recovery
    off_true = (np.abs(Omega_true) > 1e-6) & ~np.eye(d, dtype=bool)
    off_hat = (np.abs(Omega_hat) > 0.02) & ~np.eye(d, dtype=bool)
    tp = int((off_hat & off_true).sum() // 2)
    fp = int((off_hat & ~off_true).sum() // 2)
    fn = int((~off_hat & off_true).sum() // 2)
    print(f"\n  edge recovery (undirected):  TP={tp}   FP={fp}   FN={fn}\n")
    print("--- library cross-check (sklearn.covariance.GraphicalLasso; R glasso; huge) ---")
