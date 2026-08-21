"""Conditional Autoregressive (CAR) / Intrinsic CAR (ICAR) model (Reference §23.10).

Alternative to SAR for spatial random-effects modelling.  Specifies full
CONDITIONAL distributions:

    u_i | u_{-i} ~ N( sum_{j ~ i} b_ij u_j / m_i,   tau^2 / m_i )

where m_i is the number of neighbours of i.  Common choice b_ij = 1
(so conditional mean is the AVERAGE of neighbours' u).

Joint distribution:  u ~ N(0, (tau^2) (D - alpha W)^-1) where D is
diagonal number-of-neighbours and alpha in (0, 1) controls autocorrelation
(alpha = 1 gives the "improper" ICAR, standard for hierarchical Bayesian
disease-mapping / Besag-York-Mollie BYM).

Rate estimation (BYM style)
    y_i ~ Poisson(E_i * exp(alpha + u_i + v_i))
        u_i : spatially-structured CAR component
        v_i : unstructured Normal random effect
    Requires MCMC or INLA in practice.

The demo below just constructs the CAR precision matrix and generates a
draw from a CAR field for illustration.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def car_precision(W_binary, alpha: float = 0.95, tau2: float = 1.0) -> np.ndarray:
    """CAR precision matrix from a BINARY adjacency (rook / queen)."""
    W = np.asarray(W_binary, dtype=float)
    D = np.diag(W.sum(1))
    Q = (D - alpha * W) / tau2
    return Q


def simulate_car(W_binary, alpha: float = 0.95, tau2: float = 1.0, seed: int = 0) -> np.ndarray:
    """Draw one field from N(0, Q^{-1}) with Q the CAR precision."""
    Q = car_precision(W_binary, alpha, tau2)
    # Add small ridge for numerical stability
    Q = Q + 1e-4 * np.eye(len(Q))
    L = np.linalg.cholesky(np.linalg.inv(Q))
    rng = np.random.default_rng(seed)
    return L @ rng.normal(size=len(Q))


def icar_penalty(u, W_binary) -> float:
    """Compute the ICAR quadratic penalty sum_{i~j} (u_i - u_j)^2."""
    u = np.asarray(u)
    W = np.asarray(W_binary)
    return float(0.5 * np.sum(W * (u[:, None] - u[None, :]) ** 2))


if __name__ == "__main__":
    # 8x8 grid with rook adjacency
    m = 8; n = m * m
    coords = np.array([(i, j) for i in range(m) for j in range(m)])
    D_ij = np.abs(coords[:, None] - coords[None, :]).sum(-1)
    W = (D_ij == 1).astype(float)

    Q = car_precision(W, alpha=0.95, tau2=1.0)
    print(f"=== CAR precision matrix (m x m = 64) ===")
    print(f"  shape: {Q.shape}, min eig = {np.linalg.eigvalsh(Q).min():.4f}")
    print(f"  diag mean = {np.diag(Q).mean():.3f}")

    u = simulate_car(W, alpha=0.95, tau2=1.0, seed=0)
    print(f"\n  CAR field draw: sd = {u.std():.3f}   spatial autocorr (Moran-like corr with mean neighbour):")
    neigh_mean = (W @ u) / W.sum(1)
    print(f"    cor(u_i, avg neighbour u) = {np.corrcoef(u, neigh_mean)[0, 1]:.3f}")

    print(f"\n=== ICAR penalty on the drawn field: {icar_penalty(u, W):.3f}")

    print("\n--- library cross-check (R CARBayes / spdep) ---")
    print("  R: CARBayes::S.CARleroux()  -- CAR / Leroux prior with MCMC")
    print("     INLA::inla(...) with graph =  -- BYM / CAR in one line")
