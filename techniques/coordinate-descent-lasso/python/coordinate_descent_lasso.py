"""Coordinate descent for lasso (Reference Sec 47.67).

Friedman, Hastie & Tibshirani 2010 'Regularization paths for
generalized linear models via coordinate descent', JSS 33(1). For
standardized X, cycle j = 1, ..., p updating

    beta_j <- S(  X_j' r_partial / n , lambda ) / (X_j' X_j / n)

where r_partial = y - X beta + X_j beta_j and S is the soft-
thresholding operator  S(z, gamma) = sign(z) max(|z| - gamma, 0).

With warm starts, strong-rule screening and covariance updates,
this powers `glmnet` -- solves the full lambda path in seconds.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)


def lasso_coord_descent(X, y, lam, max_iter=1000, tol=1e-6):
    """Coordinate descent for lasso. Assumes X is column-standardised."""
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()
    for it in range(max_iter):
        max_dbeta = 0.0
        for j in range(p):
            r_partial = r + X[:, j] * beta[j]
            z = X[:, j] @ r_partial / n
            beta_new = soft_threshold(z, lam) / (X[:, j] @ X[:, j] / n)
            db = beta_new - beta[j]
            if abs(db) > max_dbeta:
                max_dbeta = abs(db)
            r -= X[:, j] * db
            beta[j] = beta_new
        if max_dbeta < tol:
            break
    return {"beta": beta, "iters": it + 1}


def lasso_path(X, y, lams):
    """Warm-start path over a decreasing lambda grid."""
    n, p = X.shape
    Xs = (X - X.mean(0)) / X.std(0)
    yc = y - y.mean()
    betas = np.zeros((len(lams), p))
    b = np.zeros(p)
    for k, lam in enumerate(lams):
        r = yc - Xs @ b
        for _ in range(200):
            max_d = 0.0
            for j in range(p):
                rp = r + Xs[:, j] * b[j]
                z = Xs[:, j] @ rp / n
                bn = soft_threshold(z, lam) / (Xs[:, j] @ Xs[:, j] / n)
                d = bn - b[j]
                if abs(d) > max_d:
                    max_d = abs(d)
                r -= Xs[:, j] * d
                b[j] = bn
            if max_d < 1e-6:
                break
        betas[k] = b
    return betas


if __name__ == "__main__":
    print("=== Coordinate descent lasso (Friedman-Hastie-Tibshirani 2010) ===\n")
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p); beta_true[[0, 3, 7, 15]] = [1.5, -1.0, 0.8, -0.6]
    y = X @ beta_true + 0.4 * rng.normal(size=n)

    Xs = (X - X.mean(0)) / X.std(0)
    yc = y - y.mean()
    for lam in [0.01, 0.05, 0.15, 0.3]:
        r = lasso_coord_descent(Xs, yc, lam)
        n_nz = int((np.abs(r["beta"]) > 1e-4).sum())
        top4 = np.argsort(np.abs(r["beta"]))[::-1][:4]
        print(f"  lam = {lam:.2f}   iters = {r['iters']:3d}   n_nonzero = {n_nz}   "
              f"top4 = {[(int(i), round(r['beta'][i], 3)) for i in top4]}")

    print("\n  True support = {0, 3, 7, 15} with coefs (1.5, -1.0, 0.8, -0.6).")
    print("\n--- library cross-check (glmnet R; sklearn.linear_model.Lasso Python) ---")
