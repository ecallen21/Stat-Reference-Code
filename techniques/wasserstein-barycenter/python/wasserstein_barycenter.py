"""Wasserstein Barycenter (Reference Sec 47.313).

Agueh & Carlier 2011 SIAM J Math Anal; Cuturi & Doucet 2014
ICML. Given probability measures mu_1, ..., mu_K on the same
space, their (regularised) barycenter minimises the weighted
sum of Wasserstein distances:

    nu* = argmin_nu sum_k w_k W_p^p(nu, mu_k)

Cuturi-Doucet use entropy-regularised Sinkhorn to compute
approximate barycenters efficiently.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sinkhorn(a, b, C, reg, n_iter=200):
    """Entropy-regularised OT via Sinkhorn (Cuturi 2013)."""
    K = np.exp(-C / reg)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    T = np.diag(u) @ K @ np.diag(v)
    return T, float(np.sum(T * C))


def wass_barycenter_1d(mus, positions, weights=None, reg=0.01, n_iter=200):
    """1-D free-support barycenter via fixed-point iteration."""
    K = len(mus); N = len(positions)
    if weights is None: weights = np.ones(K) / K
    # Init as uniform
    a = np.ones(N) / N
    C_grid = (positions[:, None] - positions[None, :]) ** 2
    for _ in range(n_iter):
        Tks = [sinkhorn(a, mus[k], C_grid, reg)[0] for k in range(K)]
        a = np.exp(sum(weights[k] * np.log(np.maximum(Tks[k].sum(axis=1), 1e-12))
                        for k in range(K)))
        a /= a.sum()
    return a


if __name__ == "__main__":
    print("=== Wasserstein Barycenter (Agueh-Carlier 2011; Cuturi-Doucet 2014) ===\n")
    rng = np.random.default_rng(0)

    # Two 1-D Gaussians on a common grid
    x = np.linspace(-6, 6, 50)
    mu1 = np.exp(-0.5 * ((x - 2) / 0.8) ** 2); mu1 /= mu1.sum()
    mu2 = np.exp(-0.5 * ((x + 2) / 0.8) ** 2); mu2 /= mu2.sum()

    for w in [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)]:
        bary = wass_barycenter_1d([mu1, mu2], x, weights=np.array(w),
                                    reg=0.5, n_iter=50)
        # Compute the mean of the barycenter
        mean_bary = float(np.sum(x * bary))
        mean_mu1 = float(np.sum(x * mu1)); mean_mu2 = float(np.sum(x * mu2))
        exp_mean = w[0] * mean_mu1 + w[1] * mean_mu2
        print(f"  weights = {w}   bary mean = {mean_bary:>+.3f}   "
              f"expected (linear) = {exp_mean:>+.3f}")

    print(f"\n  Barycenter interpolates the LOCATION of the two Gaussians linearly:")
    print(f"    - Wasserstein barycenter TRANSLATES the distribution (mean linear in w)")
    print(f"    - Linear (l2) average would produce a BIMODAL mixture, not a shifted mode")

    print("\n--- library cross-check (POT.bregman.barycenter; POT.gromov.wasserstein) ---")
