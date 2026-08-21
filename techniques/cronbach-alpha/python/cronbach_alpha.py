"""Cronbach's alpha + McDonald's omega (Reference §22.3).

Internal-consistency reliability of a K-item scale (Cronbach 1951):

    alpha = (K / (K - 1)) * (1 - sum_j var(x_j) / var(sum_j x_j))

Interpretation
    alpha in [0, 1] (can be negative if items are anti-correlated).
    0.7 acceptable for research, 0.8+ good for individual decisions.

Alpha assumes ESSENTIAL TAU-EQUIVALENCE (all items measure the same trait
with equal loadings).  McDonald's OMEGA relaxes this using a factor model:

    omega = (sum lambda_j)^2 / ((sum lambda_j)^2 + sum theta_jj)

with loadings lambda_j and residual variances theta_jj from a
1-factor CFA on the item scores.

Alpha with item deleted: recompute after dropping each item -> reveals
items that hurt reliability.

Standardized alpha: use item correlations rather than covariances (item
variances forced equal).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def cronbach_alpha(X) -> dict:
    """Cronbach's alpha + alpha-with-item-deleted diagnostics."""
    X = np.asarray(X, dtype=float)
    n, K = X.shape
    total_var = float(np.var(X.sum(axis=1), ddof=1))
    var_items = X.var(axis=0, ddof=1)
    alpha = (K / (K - 1)) * (1 - var_items.sum() / total_var)
    # Alpha with each item deleted
    alpha_del = []
    for j in range(K):
        Xd = np.delete(X, j, axis=1)
        v_items = Xd.var(axis=0, ddof=1)
        v_total = np.var(Xd.sum(axis=1), ddof=1)
        Kd = K - 1
        alpha_del.append(float((Kd / (Kd - 1)) * (1 - v_items.sum() / v_total)) if Kd > 1 else float("nan"))
    # Standardized alpha (mean inter-item correlation)
    C = np.corrcoef(X, rowvar=False)
    r_bar = float((C.sum() - K) / (K * (K - 1)))
    std_alpha = K * r_bar / (1 + (K - 1) * r_bar)
    return {"alpha": float(alpha), "standardized_alpha": std_alpha,
            "mean_inter_item_r": r_bar,
            "alpha_if_deleted": alpha_del,
            "n_items": int(K), "n": int(n),
            "method": "Cronbach alpha + standardized alpha + item-deletion"}


def mcdonald_omega(X) -> dict:
    """Omega via 1-factor CFA loadings + residual variances."""
    X = np.asarray(X, dtype=float); n, K = X.shape
    S = np.cov(X, rowvar=False, ddof=1)
    # Simple eigendecomposition-based single-factor estimate (rough PCA-style)
    d, V = np.linalg.eigh(S)
    lam = V[:, -1] * math.sqrt(max(d[-1], 0))    # first-PC loadings (sign arbitrary)
    if lam.mean() < 0: lam = -lam                # flip for interpretability
    theta = np.diag(S) - lam ** 2
    omega = (lam.sum() ** 2) / (lam.sum() ** 2 + np.maximum(theta, 0).sum())
    return {"omega": float(omega), "loadings": lam,
            "residual_var": np.maximum(theta, 0),
            "method": "McDonald omega via 1-factor eigendecomposition"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, K = 300, 6
    theta = rng.normal(0, 1, n)
    lam = np.array([0.7, 0.7, 0.7, 0.8, 0.6, 0.65])
    X = theta[:, None] * lam[None, :] + rng.normal(0, 0.5, (n, K))

    r = cronbach_alpha(X)
    print(f"=== Cronbach alpha, K = {K}, n = {n} ===")
    print(f"  alpha              = {r['alpha']:.4f}")
    print(f"  standardized alpha = {r['standardized_alpha']:.4f}")
    print(f"  mean r_ij          = {r['mean_inter_item_r']:.4f}")
    print(f"  alpha if item deleted: {[round(a, 3) for a in r['alpha_if_deleted']]}")

    o = mcdonald_omega(X)
    print(f"\n=== McDonald omega ===")
    print(f"  omega = {o['omega']:.4f}")
    print(f"  loadings: {o['loadings'].round(3)}")

    print("\n--- library cross-check (R psych::alpha / omega) ---")
    print("  psych::alpha(X); psych::omega(X)")
