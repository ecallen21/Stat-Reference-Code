"""Partial Least Squares regression via NIPALS (Reference §5.31; Wold 1966).

Regression method that constructs LATENT COMPONENTS which maximize the
COVARIANCE between X and Y (contrast with PCA: maximizes X variance only,
without regard to Y).  Especially useful when:
    - p >> n (many correlated predictors)
    - X columns are highly collinear
    - Multi-output regression (Y is multivariate)

NIPALS algorithm (Wold 1966) for PLS1 (single-output y):
    For each component h = 1, ..., H:
        1. w = X^T y / ||X^T y||           weight vector (X-loadings for this direction)
        2. t = X w                          latent score for observations
        3. p = X^T t / (t^T t)              X loadings
        4. b = t^T y / (t^T t)              regression coefficient
        5. Deflate X <- X - t p^T
              y <- y - b t

Prediction: beta = W (P^T W)^-1 [b_1, ..., b_H]

Number of components H chosen by cross-validation.

Compare with Principal Component Regression (PCR): PCR uses X-only PCs;
PLS uses Y-supervised components, usually needing fewer for same accuracy.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def pls1_nipals(X, y, n_components: int) -> dict:
    """PLS1 for a single response via NIPALS."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    x_mean = X.mean(0); y_mean = y.mean()
    Xc = X - x_mean; yc = y - y_mean
    W = np.zeros((p, n_components))
    P = np.zeros((p, n_components))
    T = np.zeros((n, n_components))
    b = np.zeros(n_components)
    for h in range(n_components):
        w = Xc.T @ yc; w /= np.linalg.norm(w)
        t = Xc @ w
        p_h = Xc.T @ t / (t @ t)
        b_h = t @ yc / (t @ t)
        W[:, h] = w; P[:, h] = p_h; T[:, h] = t; b[h] = b_h
        Xc = Xc - np.outer(t, p_h)
        yc = yc - b_h * t
    beta = W @ np.linalg.solve(P.T @ W, b)
    intercept = float(y_mean - x_mean @ beta)
    return {"beta": beta, "intercept": intercept,
            "W": W, "P": P, "T": T,
            "n_components": int(n_components),
            "method": "PLS1 via NIPALS"}


def predict(pls, X_new):
    return pls["intercept"] + np.asarray(X_new, dtype=float) @ pls["beta"]


def pls_cv(X, y, max_components: int = 10, n_folds: int = 5, seed: int = 0) -> dict:
    """CV-based selection of number of components."""
    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.permutation(n); folds = np.array_split(idx, n_folds)
    rmses = []
    for h in range(1, max_components + 1):
        preds = np.zeros(n)
        for k in range(n_folds):
            te = folds[k]; tr = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            fit = pls1_nipals(X[tr], y[tr], h)
            preds[te] = predict(fit, X[te])
        rmses.append(float(np.sqrt(np.mean((y - preds) ** 2))))
    best = int(np.argmin(rmses)) + 1
    return {"rmse_per_h": rmses, "best_H": best}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 100, 20
    # p >> effective dimension: latent-factor DGP
    F = rng.normal(size=(n, 3))
    beta_lat = np.array([2.0, -1.0, 0.5])
    y = F @ beta_lat + rng.normal(0, 0.5, n)
    load = rng.normal(size=(3, p))
    X = F @ load + rng.normal(0, 0.5, (n, p))

    cv = pls_cv(X, y, max_components=8)
    print("=== 5-fold CV RMSE by number of components ===")
    for h, r in enumerate(cv["rmse_per_h"], 1):
        print(f"  H = {h}: RMSE = {r:.4f}")
    print(f"  best H = {cv['best_H']}")

    fit = pls1_nipals(X, y, n_components=cv["best_H"])
    print(f"\n  in-sample RMSE  = {np.sqrt(np.mean((y - predict(fit, X)) ** 2)):.4f}")

    print("\n--- library cross-check (sklearn PLSRegression) ---")
    try:
        from sklearn.cross_decomposition import PLSRegression
        m = PLSRegression(n_components=cv["best_H"]).fit(X, y)
        print(f"  sklearn PLSRegression in-sample RMSE = {np.sqrt(np.mean((y - m.predict(X).ravel()) ** 2)):.4f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
