"""Huber M-estimator (Huber 1964).

Loss:
    rho(u) = 0.5 u^2                     if |u| <= k
    rho(u) = k * |u| - 0.5 k^2           otherwise

Influence function psi(u) = min(k, max(-k, u)) is BOUNDED,
so a single outlier cannot dominate the estimate. Combines OLS
efficiency at Gaussian (95% at k = 1.345 sigma) with robustness
to moderate outliers.
"""

import numpy as np    # arrays + linalg


def huber_psi(u, k):
    return np.clip(u, -k, k)


def huber_regression(X, y, k=1.345, max_iter=50, tol=1e-8):
    """Iteratively-reweighted LS for the Huber M-estimator."""
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        r = y - X @ beta
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-10
        u = r / sigma
        # weights w_i = psi(u) / u  (with w = 1 for small |u|)
        w = np.where(np.abs(u) < k, 1.0, k / (np.abs(u) + 1e-10))
        W = np.sqrt(w)[:, None]
        beta_new = np.linalg.lstsq(W * X, (W.flatten()) * y, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def demo():
    print("=== Huber M-estimator (Huber 1964) ===")
    rng = np.random.default_rng(2026)
    n, p = 200, 4
    beta_true = np.array([2.0, -1.5, 0.8, 3.0])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    outlier_idx = rng.choice(n, size=30, replace=False)
    y[outlier_idx] = y[outlier_idx] + rng.uniform(-15, 15, size=30)    # 15% outliers

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_huber = huber_regression(X, y, k=1.345)

    err_ols = np.linalg.norm(beta_ols - beta_true)
    err_huber = np.linalg.norm(beta_huber - beta_true)
    print(f"  True beta   : {beta_true}")
    print(f"  OLS  beta   : {np.round(beta_ols, 3)}, err = {err_ols:.4f}")
    print(f"  Huber beta  : {np.round(beta_huber, 3)}, err = {err_huber:.4f}")

    try:
        from sklearn.linear_model import HuberRegressor
        reg = HuberRegressor(epsilon=1.345, fit_intercept=False, max_iter=500)
        reg.fit(X, y)
        err_sk = np.linalg.norm(reg.coef_ - beta_true)
        print(f"  sklearn HR  : {np.round(reg.coef_, 3)}, err = {err_sk:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    demo()
