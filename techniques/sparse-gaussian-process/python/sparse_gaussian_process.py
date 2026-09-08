"""Sparse Gaussian Process via inducing points (Reference Sec 47.89).

Snelson & Ghahramani 2005 'Sparse Gaussian processes using pseudo-
inputs' (FITC); Titsias 2009 'Variational learning of inducing
variables in sparse Gaussian processes' (VFE/SVGP). Full GP scales
O(n^3); replace it with M << n inducing pseudo-inputs Z:

    q(f) ~ GP with covariance   Q_nn = K_nm K_mm^{-1} K_mn.

FITC posterior:
    mu = K_nm (K_mm + K_mn Lam^{-1} K_nm)^{-1} K_mn Lam^{-1} y
where Lam = diag(K_nn - Q_nn) + sigma^2 I.

Reduces cost to O(n M^2 + M^3); with M = 50-500 handles n = 10^5+.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rbf(X, Y, ell, sf2):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return sf2 * np.exp(-0.5 * sq / (ell ** 2))


def full_gp(X, y, Xs, ell, sf2, sn2):
    K = rbf(X, X, ell, sf2) + sn2 * np.eye(len(X))
    Ks = rbf(X, Xs, ell, sf2)
    Kss = rbf(Xs, Xs, ell, sf2)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = Kss - v.T @ v
    return mu, np.diag(var) + sn2


def fitc(X, y, Xs, Z, ell, sf2, sn2):
    Kmm = rbf(Z, Z, ell, sf2) + 1e-8 * np.eye(len(Z))
    Kmn = rbf(Z, X, ell, sf2)
    Knm = Kmn.T
    Kmm_inv_Kmn = np.linalg.solve(Kmm, Kmn)
    Q_nn = (Knm @ Kmm_inv_Kmn)     # cheap: only diagonal needed
    diag_Knn = np.full(len(X), sf2)     # RBF diagonal
    diag_Qnn = np.einsum("ij,ji->i", Knm, Kmm_inv_Kmn)
    Lam = diag_Knn - diag_Qnn + sn2
    # Posterior mean / variance at Xs
    Ksm = rbf(Xs, Z, ell, sf2)
    Kss_diag = np.full(len(Xs), sf2)
    Sigma_inv = Kmm + Kmn @ (Kmn.T / Lam[:, None])
    L_S = np.linalg.cholesky(Sigma_inv)
    alpha = np.linalg.solve(L_S.T, np.linalg.solve(L_S, Kmn @ (y / Lam)))
    mu = Ksm @ alpha
    v = np.linalg.solve(L_S, Ksm.T)
    var = Kss_diag - np.einsum("ij,ji->i", Ksm, np.linalg.solve(Kmm, Ksm.T)) \
          + np.einsum("ij,ji->i", v.T, v) + sn2
    return mu, var


if __name__ == "__main__":
    print("=== Sparse GP inducing points (Snelson-Ghahramani; Titsias) ===\n")
    rng = np.random.default_rng(0)
    n = 500
    X = np.sort(rng.uniform(-5, 5, size=(n, 1)), axis=0)
    y = (np.sin(X[:, 0]) + 0.1 * rng.normal(size=n))
    Xs = np.linspace(-5, 5, 200)[:, None]

    ell, sf2, sn2 = 1.0, 1.0, 0.01
    mu_full, var_full = full_gp(X, y, Xs, ell, sf2, sn2)

    for M in [10, 30, 100]:
        Z = np.linspace(-5, 5, M)[:, None]
        mu_s, var_s = fitc(X, y, Xs, Z, ell, sf2, sn2)
        rel = float(np.linalg.norm(mu_s - mu_full) / np.linalg.norm(mu_full))
        print(f"  M = {M:3d} inducing pts:  ||mu_FITC - mu_full|| / ||mu_full|| = {rel:.4f}")

    print("\n  Reduces O(n^3) to O(n M^2); M ~ 30-100 usually suffices.")
    print("\n--- library cross-check (mlegp / kernlab R; GPflow / gpytorch Python) ---")
