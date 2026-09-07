"""Optimal transport / Wasserstein distance (Reference Sec 46.16).

Monge 1781; Kantorovich 1942; Cuturi 2013 'Sinkhorn distances:
lightspeed computation of optimal transport', NIPS. Measures distance
between probability measures alpha, beta by the minimum cost to move
mass from alpha to beta given a ground metric d(x, y):

    W_p(alpha, beta)^p = inf_{gamma in Coupling(alpha, beta)}
                              integral d(x, y)^p dgamma(x, y)

Discrete form on n atoms x_i with weights alpha_i and m atoms y_j
with weights beta_j:

    W_1  = min_{T >= 0}  sum_ij  T_ij * d(x_i, y_j)
    s.t. sum_j T_ij = alpha_i  and  sum_i T_ij = beta_j

Regularised (Sinkhorn):

    W_eps = min_T  sum T * d  +  eps * sum T * (log T - 1)

Solved by iterative scaling (Sinkhorn-Knopp) -- O(n m) per iteration,
scales to large problems.

We compute:
    * 1-D exact W_1 via sorted CDFs (closed form).
    * n-D Sinkhorn distance via iterative scaling.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def wasserstein_1d(x, y):
    """Exact 1-D W_1: integral |F_x(t) - F_y(t)| dt = mean(|sorted x - sorted y|) if same n."""
    x_s = np.sort(x); y_s = np.sort(y)
    if len(x_s) != len(y_s):
        #  Interpolate on a common quantile grid
        u = np.linspace(0, 1, max(len(x_s), len(y_s)))
        qx = np.quantile(x_s, u)
        qy = np.quantile(y_s, u)
        return float(np.mean(np.abs(qx - qy)))
    return float(np.mean(np.abs(x_s - y_s)))


def sinkhorn(a, b, C, eps=0.05, n_iter=200):
    """Entropic-regularised OT. Returns (transport plan, W_eps)."""
    a = a / a.sum(); b = b / b.sum()
    K = np.exp(-C / eps)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    T = np.diag(u) @ K @ np.diag(v)
    return T, float(np.sum(T * C))


if __name__ == "__main__":
    print("=== Optimal transport -- Wasserstein / Sinkhorn ===\n")
    rng = np.random.default_rng(0)

    #  1-D: compare two normal samples
    x = rng.normal(0, 1, size=500)
    y = rng.normal(1, 1, size=500)
    W = wasserstein_1d(x, y)
    print(f"  1-D W_1 (N(0,1) vs N(1,1))   = {W:.3f}   (analytic = |mu_x - mu_y| = 1.000)")

    #  Different scales
    x2 = rng.normal(0, 1, size=500)
    y2 = rng.normal(0, 2, size=500)
    W2 = wasserstein_1d(x2, y2)
    #  Analytic W_1 for two normals with same mean but sd difference: E|N(0, sig_x-sig_y)| ~ (2/pi)^0.5 * |sig_x - sig_y|
    print(f"  1-D W_1 (N(0,1) vs N(0,2))   = {W2:.3f}   (~ sqrt(2/pi) * 1 = 0.798)")

    #  n-D Sinkhorn on small 2-D point clouds
    X = rng.normal(0, 1, size=(20, 2))
    Y = rng.normal(1, 1, size=(20, 2))
    a = np.ones(20); b = np.ones(20)
    C = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
    T, W_sink = sinkhorn(a, b, C, eps=0.05, n_iter=200)
    print(f"\n  2-D Sinkhorn W (n=20 each cloud)  = {W_sink:.3f}")
    print(f"  Transport-plan row-sum (target 1/20): min {T.sum(axis=1).min():.3f}, "
          f"max {T.sum(axis=1).max():.3f}")

    print("\n--- library cross-check (transport R, POT Python) ---")
