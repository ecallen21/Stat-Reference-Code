"""Regularized linear regression: ridge, lasso, elastic net (Reference §5.9, §5.17).

All three minimize  RSS(beta) + lambda * Penalty(beta) :

  Ridge (L2)  : Penalty = sum_j beta_j^2                  -- has a closed form
  Lasso (L1)  : Penalty = sum_j |beta_j|                  -- coordinate descent
  Elastic net : Penalty = alpha * |beta|_1 + (1 - alpha) * |beta|_2^2 / 2
                                                          -- mixes ridge and lasso

Ridge shrinks coefficients toward 0 but never *to* 0. Lasso shrinks AND
performs variable selection (many coefficients are exactly zero at the
optimum). Elastic net handles "correlated groups" of predictors more
gracefully than lasso (which arbitrarily picks one of a correlated group).

Standardize the predictors first
--------------------------------
Penalties act on the *coefficients*; if x1 is in meters and x2 in kilometers,
they get penalized differently. Standard practice: center and scale the X
columns, fit, then back-transform if needed. We do that here.

What this file implements from scratch
--------------------------------------
- ``ridge`` via the closed form  beta = (X' X + lambda I)^{-1} X' y
- ``lasso`` via cyclic coordinate descent with soft-thresholding
- ``elastic_net`` via the same coordinate descent with the EN soft-threshold
- ``cv_ridge`` / ``cv_lasso`` : k-fold CV to pick lambda
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _standardize(X):
    mu = X.mean(axis=0)
    s = X.std(axis=0, ddof=0)
    s = np.where(s == 0, 1, s)
    return (X - mu) / s, mu, s


def ridge(X, y, lam: float, include_intercept: bool = True) -> dict:
    """Ridge regression. lam = lambda; predictors are standardized internally."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    Xs, mu_X, s_X = _standardize(X)
    y_mean = y.mean() if include_intercept else 0.0
    yc = y - y_mean
    p = Xs.shape[1]
    beta_std = np.linalg.solve(Xs.T @ Xs + lam * np.eye(p), Xs.T @ yc)
    beta = beta_std / s_X
    intercept = y_mean - mu_X @ beta if include_intercept else 0.0
    return {"intercept": float(intercept), "beta": beta.tolist(),
            "beta_standardized": beta_std.tolist(), "lambda": lam}


def _soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso(X, y, lam: float, max_iter: int = 1000, tol: float = 1e-7) -> dict:
    """Lasso via cyclic coordinate descent with soft-thresholding.

    Predictors are standardized internally (so each x_j has SS_j = n).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    Xs, mu_X, s_X = _standardize(X)
    y_mean = y.mean(); yc = y - y_mean
    n, p = Xs.shape
    beta = np.zeros(p)
    # Precompute SS_j = sum x_ij^2 = n for standardized X
    for _ in range(max_iter):
        beta_old = beta.copy()
        r = yc - Xs @ beta
        for j in range(p):
            # add back the j-th contribution: r += Xs[:, j] * beta_j
            r += Xs[:, j] * beta[j]
            z_j = Xs[:, j] @ r / n
            beta[j] = _soft_threshold(z_j, lam / n)
            r -= Xs[:, j] * beta[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    coef = beta / s_X
    intercept = y_mean - mu_X @ coef
    return {"intercept": float(intercept), "beta": coef.tolist(),
            "beta_standardized": beta.tolist(), "lambda": lam,
            "nonzero": int((np.abs(beta) > 1e-10).sum())}


def elastic_net(X, y, lam: float, alpha: float = 0.5,
                max_iter: int = 1000, tol: float = 1e-7) -> dict:
    """Elastic net: penalty = lambda * (alpha * |beta|_1 + (1 - alpha)/2 * |beta|_2^2).

    alpha=1 -> lasso, alpha=0 -> ridge (up to scaling), alpha in (0, 1) -> mix.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    Xs, mu_X, s_X = _standardize(X)
    y_mean = y.mean(); yc = y - y_mean
    n, p = Xs.shape
    beta = np.zeros(p)
    l1 = lam * alpha; l2 = lam * (1 - alpha)
    for _ in range(max_iter):
        beta_old = beta.copy()
        r = yc - Xs @ beta
        for j in range(p):
            r += Xs[:, j] * beta[j]
            z_j = Xs[:, j] @ r / n
            beta[j] = _soft_threshold(z_j, l1 / n) / (1 + l2 / n)
            r -= Xs[:, j] * beta[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    coef = beta / s_X
    intercept = y_mean - mu_X @ coef
    return {"intercept": float(intercept), "beta": coef.tolist(),
            "lambda": lam, "alpha": alpha,
            "nonzero": int((np.abs(beta) > 1e-10).sum())}


def cv_regularization(X, y, fit_fn, lambdas, n_folds: int = 5, seed: int = 0) -> dict:
    """Generic k-fold CV: returns mean MSE per lambda and the best lambda."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    mse_per_lam = []
    for lam in lambdas:
        fold_mse = []
        for k in range(n_folds):
            test = folds[k]; train = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            res = fit_fn(X[train], y[train], lam)
            yhat = res["intercept"] + X[test] @ np.array(res["beta"])
            fold_mse.append(float(((y[test] - yhat) ** 2).mean()))
        mse_per_lam.append(float(np.mean(fold_mse)))
    j = int(np.argmin(mse_per_lam))
    return {"lambdas": list(lambdas), "mse": mse_per_lam,
            "best_lambda": lambdas[j], "best_mse": mse_per_lam[j]}


def library_versions(X, y, lam=1.0):
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    r = Ridge(alpha=lam).fit(X, y)
    l = Lasso(alpha=lam / X.shape[0]).fit(X, y)
    e = ElasticNet(alpha=lam / X.shape[0], l1_ratio=0.5).fit(X, y)
    return {"sklearn Ridge coefs": r.coef_.tolist(),
            "sklearn Lasso coefs": l.coef_.tolist(),
            "sklearn ElasticNet coefs": e.coef_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    n, p = 100, 10
    X = rng.normal(0, 1, (n, p))
    true_beta = np.array([2.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.3])
    y = X @ true_beta + rng.normal(0, 1, n)

    print("=== Ridge (lambda = 10) ===")
    r = ridge(X, y, lam=10.0)
    print(f"  intercept = {r['intercept']:+.4f}")
    print(f"  beta      = {[round(b, 3) for b in r['beta']]}")

    print("\n=== Lasso (lambda = 15) -- enough to zero out small effects ===")
    l = lasso(X, y, lam=15.0)
    print(f"  intercept = {l['intercept']:+.4f}")
    print(f"  beta      = {[round(b, 3) for b in l['beta']]}")
    print(f"  non-zero  = {l['nonzero']}")

    print("\n=== Elastic net (lambda = 15, alpha = 0.5) ===")
    e = elastic_net(X, y, lam=15.0, alpha=0.5)
    print(f"  beta      = {[round(b, 3) for b in e['beta']]}")
    print(f"  non-zero  = {e['nonzero']}")

    print("\n=== 5-fold CV for lasso lambda ===")
    cv = cv_regularization(X, y, lasso, lambdas=[1.0, 3.0, 10.0, 30.0, 100.0])
    print(f"  best lambda = {cv['best_lambda']}   best MSE = {cv['best_mse']:.4f}")

    print(f"\n  true beta = {true_beta.tolist()}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, y, lam=10.0).items():
        print(f"  {k}: {[round(c, 3) for c in v]}")
