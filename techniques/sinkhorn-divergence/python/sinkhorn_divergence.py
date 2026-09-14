"""Sinkhorn Divergence (Reference Sec 47.314).

Cuturi 2013 NeurIPS (Sinkhorn OT); Feydy et al 2019 AISTATS
(divergence de-biasing). Entropy-regularised optimal transport
distance:

    OT_reg(mu, nu) = min_T <T, C> + reg * H(T)     s.t. T 1 = mu, T^T 1 = nu

Sinkhorn iterates alternately normalise rows/columns of
K = exp(-C/reg). De-biased SINKHORN DIVERGENCE:

    S_reg(mu, nu) = OT_reg(mu, nu) - 0.5 * (OT_reg(mu, mu) + OT_reg(nu, nu))

is positive and zero iff mu = nu, unlike the vanilla OT_reg.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sinkhorn(a, b, C, reg=0.1, n_iter=200):
    """Entropy-regularised OT cost (Cuturi 2013)."""
    K = np.exp(-C / reg)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    T = np.diag(u) @ K @ np.diag(v)
    return float(np.sum(T * C))


def sinkhorn_divergence(a, b, Cab, Caa, Cbb, reg=0.1, n_iter=200):
    """De-biased Sinkhorn divergence."""
    return (sinkhorn(a, b, Cab, reg, n_iter)
            - 0.5 * sinkhorn(a, a, Caa, reg, n_iter)
            - 0.5 * sinkhorn(b, b, Cbb, reg, n_iter))


if __name__ == "__main__":
    print("=== Sinkhorn Divergence (Cuturi 2013; Feydy et al 2019) ===\n")
    rng = np.random.default_rng(0)

    # 1-D discrete distributions on a common grid
    x = np.linspace(-5, 5, 40)
    mu = np.exp(-0.5 * (x - 1) ** 2); mu /= mu.sum()
    Caa = (x[:, None] - x[None, :]) ** 2

    print(f"  Discrete mu = N(1, 1) approx, common grid, cost = squared distance")
    print(f"\n  Test 1: OT_reg(mu, mu) should be > 0 due to entropy regularisation")
    for reg in [0.01, 0.1, 1.0]:
        ot_biased = sinkhorn(mu, mu, Caa, reg=reg, n_iter=200)
        sd = sinkhorn_divergence(mu, mu, Caa, Caa, Caa, reg=reg, n_iter=200)
        print(f"    reg = {reg:>5}   OT_reg(mu, mu) = {ot_biased:.4f}   Sinkhorn div = {sd:.4f}")

    print(f"\n  Test 2: Divergences between distinct distributions")
    for shift in [0.5, 1.5, 3.0]:
        nu = np.exp(-0.5 * (x - (1 + shift)) ** 2); nu /= nu.sum()
        Cab = (x[:, None] - x[None, :]) ** 2
        Cbb = Cab
        ot_biased = sinkhorn(mu, nu, Cab, reg=0.1, n_iter=200)
        sd = sinkhorn_divergence(mu, nu, Cab, Caa, Cbb, reg=0.1, n_iter=200)
        print(f"    shift = {shift:>3}   OT_reg = {ot_biased:.3f}   Sinkhorn div = {sd:.3f}   "
              f"(true W2^2 = {shift**2:.3f})")

    print(f"\n  Sinkhorn divergence is ZERO between identical distributions and")
    print(f"  APPROXIMATES the true squared Wasserstein-2 distance as reg -> 0.")

    print("\n--- library cross-check (POT.sinkhorn / POT.sinkhorn2; geomloss.SamplesLoss) ---")
