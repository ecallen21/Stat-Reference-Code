"""Tukey biweight (bisquare) M-estimator (Beaton-Tukey 1974).

Loss (redescending):
    rho(u) = (c^2 / 6) * (1 - (1 - (u/c)^2)^3)    if |u| <= c
    rho(u) = c^2 / 6                              otherwise

Influence psi(u) = u * (1 - (u/c)^2)^2 REDESCENDS to zero for
|u| > c — extreme outliers get zero influence. Standard tuning
c = 4.685 for 95% Gaussian efficiency.

Requires good initialisation (LTS / MCD start) because rho is
non-convex.
"""

import numpy as np    # arrays + linalg


def biweight_w(u, c):
    """Biweight weight function u |-> psi(u)/u."""
    mask = np.abs(u) < c
    w = np.zeros_like(u)
    w[mask] = (1 - (u[mask] / c) ** 2) ** 2
    return w


def s_scale_initial(X, y, beta):
    """Simple MAD-based initial scale."""
    r = y - X @ beta
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def tukey_regression(X, y, beta0, c=4.685, max_iter=100, tol=1e-8):
    beta = beta0.copy()
    for _ in range(max_iter):
        r = y - X @ beta
        sigma = s_scale_initial(X, y, beta) + 1e-10
        u = r / sigma
        w = biweight_w(u, c)
        W = np.sqrt(w)[:, None]
        beta_new = np.linalg.lstsq(W * X, np.sqrt(w) * y, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def demo():
    print("=== Tukey biweight M-estimator (Beaton-Tukey 1974) ===")
    rng = np.random.default_rng(2026)
    n, p = 200, 4
    beta_true = np.array([2.0, -1.5, 0.8, 3.0])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    outlier_idx = rng.choice(n, size=40, replace=False)
    y[outlier_idx] = y[outlier_idx] + rng.uniform(-40, 40, size=40)    # 20% wild outliers

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    # Use Tukey with a Huber-like small c first to get a decent start,
    # then run redescending Tukey with c = 4.685.
    beta_huber_like = tukey_regression(X, y, beta0=beta_ols, c=1.345)
    beta_tukey = tukey_regression(X, y, beta0=beta_huber_like, c=4.685)

    err_ols = np.linalg.norm(beta_ols - beta_true)
    err_warm = np.linalg.norm(beta_huber_like - beta_true)
    err_tukey = np.linalg.norm(beta_tukey - beta_true)
    print(f"  True beta       : {beta_true}")
    print(f"  OLS             : {np.round(beta_ols, 3)}   err = {err_ols:.4f}")
    print(f"  Warm (c=1.345)  : {np.round(beta_huber_like, 3)}   err = {err_warm:.4f}")
    print(f"  Tukey (c=4.685) : {np.round(beta_tukey, 3)}   err = {err_tukey:.4f}")


if __name__ == "__main__":
    demo()
