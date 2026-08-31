"""Latent-space network model (Reference Sec 30.5).

Hoff, Raftery & Handcock (2002) 'Latent space approaches to social
network analysis.'

Each node has a LATENT POSITION z_i in R^d; edge probability shrinks
with distance:

  logit P(A_ij = 1)  =  alpha  -  ||z_i - z_j||.

Fit via MLE (or MCMC in the Bayesian version). Positions capture
transitivity, homophily, and clustering without an explicit block
structure.

Here we implement a light gradient-descent MLE for the distance model on
a synthetic graph with a hidden 2-D latent geometry (two clusters),
then verify recovered positions align with the truth.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_latent_space(A, d=2, lr=0.05, epochs=800, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    Z = rng.normal(0, 0.3, (n, d))
    alpha = 0.0
    for _ in range(epochs):
        # Distance matrix
        D = np.sqrt(np.maximum(0, ((Z[:, None] - Z[None]) ** 2).sum(axis=2)))
        logit = alpha - D
        P = _sigmoid(logit)
        # Gradient of log-likelihood wrt Z (undirected, no self-loops)
        mask = 1 - np.eye(n)
        E = (A - P) * mask                              # (n, n)
        d_alpha = float(E.sum() / 2)
        # d logit / dz_i = -(z_i - z_j) / D_ij; summed
        d_Z = np.zeros_like(Z)
        with np.errstate(invalid="ignore", divide="ignore"):
            for i in range(n):
                diff = Z[i] - Z                          # (n, d)
                dist = D[i] + 1e-6
                d_Z[i] = -np.sum(E[i][:, None] * diff / dist[:, None], axis=0)
        alpha += lr * d_alpha / (n * (n - 1) / 2)
        Z += lr * d_Z / n
    return Z, alpha


def procrustes_align(A, B):
    """Rigid alignment of A onto B (rotation + reflection allowed)."""
    A0 = A - A.mean(axis=0); B0 = B - B.mean(axis=0)
    U, s, Vt = np.linalg.svd(A0.T @ B0)
    R = U @ Vt
    return A0 @ R


if __name__ == "__main__":
    print("=== Hoff-Raftery-Handcock latent-space network (2002) ===\n")
    rng = np.random.default_rng(0)
    n = 40
    # Hidden latent positions: two clusters in the plane.
    z_true = np.vstack([rng.normal([-2, 0], 0.5, (n // 2, 2)),
                          rng.normal([+2, 0], 0.5, (n - n // 2, 2))])
    D_true = np.sqrt(((z_true[:, None] - z_true[None]) ** 2).sum(axis=2))
    alpha_true = 3.0
    P_true = _sigmoid(alpha_true - D_true)
    A = (rng.random(P_true.shape) < P_true).astype(float)
    A = np.triu(A, 1); A = A + A.T
    np.fill_diagonal(A, 0)

    Z_hat, alpha_hat = fit_latent_space(A, d=2, lr=0.05, epochs=600)
    Z_aligned = procrustes_align(Z_hat, z_true)
    residual = float(np.mean(np.sqrt(((Z_aligned - (z_true - z_true.mean(axis=0))) ** 2).sum(axis=1))))
    print(f"  alpha true = {alpha_true:.2f}   alpha_hat = {alpha_hat:.2f}")
    print(f"  Procrustes-aligned mean position error: {residual:.3f}")
    print(f"  (should be small vs cluster separation ~ 4)\n")
    print("--- library cross-check (R latentnet; Python graspologic latent-position) ---")
