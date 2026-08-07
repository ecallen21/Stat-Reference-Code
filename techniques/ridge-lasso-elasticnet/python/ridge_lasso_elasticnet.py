"""Ridge, LASSO, and Elastic Net regression (Reference §5.9, §5.10).

Penalized regression that trades a bit of bias for a lot of variance
reduction.  All three shrink the OLS coefficients toward zero:

    Ridge   (Hoerl-Kennard 1970): minimize ||y - X beta||^2 + lambda ||beta||_2^2
        Closed form: beta = (X'X + lambda I)^-1 X' y
        Shrinks all coefficients smoothly; never sets to exactly zero.

    LASSO   (Tibshirani 1996): minimize ||y - X beta||^2 + lambda ||beta||_1
        Solved by coordinate descent.  Sparsity: some coefs become
        exactly zero, giving VARIABLE SELECTION.

    Elastic Net (Zou-Hastie 2005): minimize (1/2) ||y - X beta||^2 +
                                   lambda (alpha ||beta||_1 + (1 - alpha)/2 ||beta||_2^2)
        alpha in [0, 1] interpolates: alpha=0 -> ridge, alpha=1 -> LASSO.
        Handles correlated predictors better than pure LASSO.

Regularization path
    Fit at a grid of lambda values (typically 100 log-spaced from lambda_max
    down to a fraction of lambda_max); pick lambda by cross-validation
    (see cross-validation Batch 9).

    lambda_max = max_j |X_j' y| / (n * alpha)   for standardized X.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _standardize(X):
    mu = X.mean(0); sd = X.std(0, ddof=1); sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


def ridge_regression(X, y, lam: float) -> dict:
    """Ridge (L2) regression, closed form.  Intercept centered separately."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    ybar = y.mean()
    Xs, mu, sd = _standardize(X)
    p = X.shape[1]
    beta_s = np.linalg.solve(Xs.T @ Xs + lam * np.eye(p), Xs.T @ (y - ybar))
    beta_raw = beta_s / sd
    intercept = ybar - mu @ beta_raw
    return {"intercept": float(intercept), "beta": beta_raw,
            "lambda": float(lam),
            "method": "Ridge regression (closed form)"}


def _soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)


def elastic_net(X, y, lam: float, alpha: float = 1.0,
                max_iter: int = 500, tol: float = 1e-6) -> dict:
    """Elastic Net via coordinate descent.  alpha = 1 -> LASSO; alpha = 0 -> ridge."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    ybar = y.mean(); yc = y - ybar
    Xs, mu, sd = _standardize(X)
    n, p = Xs.shape
    beta = np.zeros(p)
    col_norm_sq = np.sum(Xs ** 2, axis=0)  # each equals (n - 1) after standardization
    r = yc.copy()
    for it in range(max_iter):
        beta_prev = beta.copy()
        for j in range(p):
            r_j = r + Xs[:, j] * beta[j]
            rho = Xs[:, j] @ r_j / n
            denom = col_norm_sq[j] / n + lam * (1 - alpha)
            beta[j] = _soft_threshold(rho, lam * alpha) / denom
            r = r_j - Xs[:, j] * beta[j]
        if np.max(np.abs(beta - beta_prev)) < tol: break
    beta_raw = beta / sd
    intercept = ybar - mu @ beta_raw
    return {"intercept": float(intercept), "beta": beta_raw,
            "n_nonzero": int(np.sum(np.abs(beta) > 1e-8)),
            "lambda": float(lam), "alpha": float(alpha),
            "iterations": int(it + 1),
            "method": f"Elastic Net (alpha={alpha}) via coordinate descent"}


def lasso(X, y, lam: float, **kw): return elastic_net(X, y, lam=lam, alpha=1.0, **kw)


def regularization_path(X, y, alpha: float = 1.0, n_lambda: int = 100) -> dict:
    """Fit at a log-spaced grid of lambda from lambda_max down to lambda_max/1000."""
    Xs, _, _ = _standardize(X); yc = y - y.mean()
    lam_max = float(np.max(np.abs(Xs.T @ yc)) / (len(y) * max(alpha, 1e-6)))
    lam_grid = np.exp(np.linspace(math.log(lam_max), math.log(lam_max / 1e3), n_lambda))
    betas = []
    for lam in lam_grid:
        r = elastic_net(X, y, lam=lam, alpha=alpha, max_iter=100)
        betas.append(r["beta"])
    return {"lambdas": lam_grid,
            "betas": np.array(betas), "alpha": float(alpha),
            "method": f"Regularization path (alpha={alpha})"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X = rng.normal(size=(n, p))
    # Sparse true beta: only 5 non-zero coefs
    beta_true = np.zeros(p); beta_true[:5] = [3, -2, 1.5, -1, 0.5]
    y = X @ beta_true + rng.normal(0, 1.0, n)

    print("=== Ridge (lam = 10) ===")
    r = ridge_regression(X, y, lam=10.0)
    print(f"  first 6 coefs: {r['beta'][:6].round(3)}")

    print("\n=== LASSO (lam = 0.05) ===")
    r = lasso(X, y, lam=0.05)
    print(f"  first 6 coefs: {r['beta'][:6].round(3)}")
    print(f"  n_nonzero = {r['n_nonzero']}")

    print("\n=== Elastic Net (alpha = 0.5, lam = 0.05) ===")
    r = elastic_net(X, y, lam=0.05, alpha=0.5)
    print(f"  first 6 coefs: {r['beta'][:6].round(3)}")

    print("\n=== LASSO path (first / mid / last lambda) ===")
    path = regularization_path(X, y, alpha=1.0, n_lambda=30)
    for j in (0, 15, 29):
        nz = int(np.sum(np.abs(path["betas"][j]) > 1e-8))
        print(f"  lambda = {path['lambdas'][j]:.4f}: n_nonzero = {nz}")

    print("\n--- library cross-check (sklearn Lasso) ---")
    try:
        from sklearn.linear_model import Lasso
        m = Lasso(alpha=0.05, fit_intercept=True).fit(X, y)
        print(f"  sklearn Lasso first 6 coefs: {m.coef_[:6].round(3)}, n_nonzero = {int((m.coef_ != 0).sum())}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
