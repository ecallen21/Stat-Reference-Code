"""Subgradient Method (Reference Sec 47.338).

Shor 1985; Boyd, Xiao & Mutapcic 2004. Minimise a convex
NON-SMOOTH f by iterating in ANY subgradient direction:

    g_k in ∂f(x_k)
    x_{k+1} = x_k - alpha_k * g_k

Diminishing step size (alpha_k = c / sqrt(k)) guarantees
convergence of the RUNNING BEST value; fixed step guarantees
a neighbourhood.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def subgradient_L1_regression(X, y, lam=0.1, n_iter=1000):
    """Fit LAD (L1) regression via subgradient method:
    min sum |y_i - X_i^T beta| + lam * ||beta||_1."""
    n, d = X.shape
    beta = np.zeros(d)
    best_obj = np.inf; best_beta = beta.copy()
    hist = []
    for k in range(1, n_iter + 1):
        alpha = 0.1 / np.sqrt(k)
        resid = y - X @ beta
        # Subgradient of sum |r_i|: -X^T sign(r)
        g_data = -X.T @ np.sign(resid)
        # Subgradient of L1 penalty: lam * sign(beta), any value in [-1,1] at 0
        g_pen = lam * np.sign(beta)
        beta = beta - alpha * (g_data + g_pen)
        obj = float(np.abs(y - X @ beta).sum() + lam * np.abs(beta).sum())
        if obj < best_obj: best_obj = obj; best_beta = beta.copy()
        hist.append(best_obj)
    return best_beta, hist


if __name__ == "__main__":
    print("=== Subgradient Method (Shor 1985; Boyd-Xiao-Mutapcic 2004) ===\n")
    rng = np.random.default_rng(0)

    n, d = 200, 10
    X = rng.standard_normal((n, d))
    beta_true = np.array([2, -1, 0, 0, 3, 0, 0, -0.5, 0, 1])
    y = X @ beta_true + rng.standard_normal(n) * 0.3
    # Inject outliers
    y[10] += 20; y[50] -= 25

    beta_hat, hist = subgradient_L1_regression(X, y, lam=0.1, n_iter=3000)

    print(f"  L1 (LAD) regression with L1 penalty via subgradient")
    print(f"  n = {n}, d = {d}, 2 outliers added at rows 10, 50\n")
    print(f"  {'j':>3}  {'true':>7}  {'est':>7}")
    for j in range(d):
        print(f"  {j:>3}  {beta_true[j]:>+7.2f}  {beta_hat[j]:>+7.2f}")

    print(f"\n  Objective trajectory (running best):")
    for k in [10, 100, 500, 1000, 2000, 2999]:
        print(f"    iter {k:>4}   best obj = {hist[k]:.4f}")

    print(f"\n  Subgradient is SLOW (O(1/sqrt(k))) but works for ANY convex non-smooth f.")
    print(f"  For proximable non-smooth pieces, use PROXIMAL GRADIENT (much faster).")

    print("\n--- library cross-check (cvxpy for exact LAD; quantreg::rq R for L1 regression) ---")
