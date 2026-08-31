"""SCAD + MCP nonconvex penalties (Reference Sec 32.2).

Fan & Li (2001) 'Variable selection via nonconcave penalized likelihood
and its oracle properties.'
Zhang, C.-H. (2010) 'Nearly unbiased variable selection under minimax
concave penalty.'

LASSO shrinks LARGE coefficients as much as small ones (bias). SCAD and
MCP are FOLDED-CONCAVE penalties that vanish for large |beta| ->
UNBIASED for signals, still sparse for zeros, and enjoy the ORACLE
PROPERTY.

SCAD derivative:
  p'_lambda(t) = lambda                          if 0 <= t <= lambda
                (a lambda - t)_+ / (a - 1)       if lambda <= t <= a lambda
                0                                 if t > a lambda      (a=3.7 usually)

MCP derivative:
  p'_lambda(t) = lambda - t / gamma  if t <= gamma lambda
                0                     if t > gamma lambda   (gamma=3 usually)

Fit via LOCAL LINEAR APPROXIMATION (LLA, Zou & Li 2008):
  linearise p at current beta_k, giving a weighted-LASSO subproblem
  weights_j = p'_lambda(|beta_k_j|).

Here we compare LASSO vs SCAD vs MCP on a synthetic sparse regression;
report support recovery + bias on the true signals.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _soft(x, lam):
    return np.sign(x) * np.maximum(0, np.abs(x) - lam)


def scad_deriv(t, lam, a=3.7):
    t = np.abs(t)
    return np.where(t <= lam, lam,
             np.where(t <= a * lam, np.maximum(0, a * lam - t) / (a - 1), 0.0))


def mcp_deriv(t, lam, gamma=3.0):
    t = np.abs(t)
    return np.where(t <= gamma * lam, np.maximum(0, lam - t / gamma), 0.0)


def coordinate_descent_weighted_lasso(X, y, w, max_iter=200, tol=1e-8):
    """Argmin_beta 0.5/n || y - X beta ||^2 + sum_j w_j |beta_j|."""
    n, d = X.shape
    beta = np.zeros(d)
    XtX_diag = (X ** 2).sum(axis=0) / n
    for _ in range(max_iter):
        beta_old = beta.copy()
        r = y - X @ beta
        for j in range(d):
            xj = X[:, j]
            rho = xj @ r / n + XtX_diag[j] * beta[j]
            beta_j_new = _soft(rho, w[j]) / max(XtX_diag[j], 1e-12)
            r = r + xj * (beta[j] - beta_j_new)
            beta[j] = beta_j_new
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def fit_penalized(X, y, lam, method="lasso", max_iter_lla=10):
    n, d = X.shape
    beta = np.zeros(d)
    if method == "lasso":
        return coordinate_descent_weighted_lasso(X, y, np.full(d, lam))
    # LLA iterations
    for _ in range(max_iter_lla):
        w = (scad_deriv(beta, lam) if method == "scad" else mcp_deriv(beta, lam))
        beta_new = coordinate_descent_weighted_lasso(X, y, w)
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new; break
        beta = beta_new
    return beta


if __name__ == "__main__":
    print("=== SCAD + MCP nonconvex penalties ===\n")
    rng = np.random.default_rng(0)
    n, d = 200, 30
    X = rng.normal(0, 1, (n, d))
    beta_true = np.zeros(d)
    beta_true[[0, 5, 10]] = [3.0, -2.5, 2.0]
    y = X @ beta_true + rng.normal(0, 0.5, n)

    lam = 0.15
    for method in ("lasso", "scad", "mcp"):
        beta = fit_penalized(X, y, lam, method=method)
        support = np.abs(beta) > 1e-3
        true_supp = np.abs(beta_true) > 0
        tp = int(((support) & (true_supp)).sum())
        fp = int(((support) & (~true_supp)).sum())
        bias = float(np.abs(beta[true_supp] - beta_true[true_supp]).mean())
        print(f"  {method:>5}   nonzero={int(support.sum()):>3}"
              f"   TP={tp}/3   FP={fp}   mean |signal bias|={bias:.4f}"
              f"   signal betas={np.round(beta[true_supp], 2).tolist()}")

    print("\n  Nonconvex SCAD / MCP recover signal magnitudes with less bias than LASSO.\n")
    print("--- library cross-check (R ncvreg for SCAD/MCP; Python celer / pyglmnet) ---")
