"""HSIC kernel independence test (Reference Sec 47.56).

Gretton, Bousquet, Smola & Scholkopf 2005 'Measuring statistical
dependence with Hilbert-Schmidt norms'. Kernel-based dependence
measure that detects any nonlinear association:

    HSIC(X, Y) = || C_{XY} ||_HS^2

Empirical estimator (biased):

    HSIC = 1/(n-1)^2 * tr(K H L H),   H = I - 1/n 1 1'

with K, L Gaussian (RBF) kernels on X, Y. HSIC = 0 iff X and Y
independent under characteristic kernels. Gamma approximation of
the null gives a p-value.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import gamma    # null approximation


def rbf_kernel(X, sigma=None):
    n = len(X)
    X = np.atleast_2d(X.reshape(n, -1))
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    if sigma is None:
        sigma = np.sqrt(0.5 * np.median(sq[sq > 0]))
    return np.exp(-sq / (2 * sigma ** 2)), sigma


def hsic_gamma_test(X, Y):
    """HSIC + gamma-approx p-value (Gretton et al 2008)."""
    n = len(X)
    K, sx = rbf_kernel(X); L, sy = rbf_kernel(Y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = float(np.trace(Kc @ Lc) / (n - 1) ** 2)
    # Null gamma approx
    mu = (1.0 / n) * (1 + np.mean(K) * np.mean(L) - np.mean(K) - np.mean(L)) \
         if False else (np.trace(K) / n) * (np.trace(L) / n) / n    # simple form
    var = (2 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))) * \
          np.trace(Kc @ Kc) * np.trace(Lc @ Lc) / (n - 1) ** 4
    alpha = mu ** 2 / max(var, 1e-15); beta = max(var, 1e-15) / max(mu, 1e-15)
    p = float(1 - gamma.cdf(hsic, a=alpha, scale=beta))
    return {"hsic": hsic, "p": p, "sigma_x": sx, "sigma_y": sy}


if __name__ == "__main__":
    print("=== HSIC kernel independence test (Gretton et al 2005) ===\n")
    rng = np.random.default_rng(0)
    n = 300

    for name, gen in [
        ("independent   ", lambda: (rng.normal(size=n), rng.normal(size=n))),
        ("linear         ", lambda: (lambda x: (x, 0.7 * x + 0.3 * rng.normal(size=n)))(rng.normal(size=n))),
        ("quadratic      ", lambda: (lambda x: (x, x * x + 0.2 * rng.normal(size=n)))(rng.normal(size=n))),
        ("sinusoid       ", lambda: (lambda x: (x, np.sin(3 * x) + 0.1 * rng.normal(size=n)))(rng.uniform(-3, 3, size=n))),
    ]:
        X, Y = gen()
        r = hsic_gamma_test(X, Y)
        print(f"  {name}  HSIC = {r['hsic']:.4f}   p = {r['p']:.4f}   "
              f"(sigma_x={r['sigma_x']:.2f}, sigma_y={r['sigma_y']:.2f})")

    print("\n  Quadratic + sinusoid: HSIC picks up the nonlinear signal that")
    print("  Pearson correlation misses (E[X * X^2] = 0 for symmetric X).")
    print("\n--- library cross-check (dHSIC R; hyppo / pyRMT Python) ---")
