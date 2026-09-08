"""Spatial Error Model (SEM) (Reference Sec 47.48).

Anselin 1988 'Spatial Econometrics'. Contrasts with the Spatial
Autoregressive (SAR) LAG model. SEM allows spatial dependence in the
DISTURBANCES rather than the outcome:

    y = X beta + u
    u = lambda * W u + eps,     eps ~ N(0, sigma^2 I)

so   y = X beta + (I - lambda W)^{-1} eps.

Coefficients beta remain interpretable as usual OLS effects; the
lambda term corrects for spatial correlation of unobservables --
ignoring it inflates SE and biases inference (not point estimates).

Fitted here by profile-likelihood over lambda.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar    # profile lambda


def sem_profile(lam, X, y, W):
    n = len(y)
    A = np.eye(n) - lam * W
    Xs = A @ X
    ys = A @ y
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    resid = ys - Xs @ beta
    sig2 = float((resid ** 2).sum() / n)
    sign, logdet = np.linalg.slogdet(A)
    return -(-(n / 2) * np.log(2 * np.pi * sig2) - (resid ** 2).sum() / (2 * sig2) + logdet)


def fit_sem(X, y, W):
    res = minimize_scalar(sem_profile, args=(X, y, W), bounds=(-0.99, 0.99), method="bounded")
    lam = float(res.x)
    A = np.eye(len(y)) - lam * W
    beta, *_ = np.linalg.lstsq(A @ X, A @ y, rcond=None)
    resid = A @ y - (A @ X) @ beta
    sig2 = float((resid ** 2).sum() / len(y))
    return {"lambda": lam, "beta": beta, "sigma2": sig2, "loglik": -float(res.fun)}


if __name__ == "__main__":
    print("=== Spatial Error Model (Anselin 1988) ===\n")
    rng = np.random.default_rng(0)
    n_side = 12; n = n_side * n_side
    # Row-normalised queen-contiguity W on a grid
    coord = np.array([(i, j) for i in range(n_side) for j in range(n_side)])
    dist = np.linalg.norm(coord[:, None] - coord[None, :], axis=-1)
    W = ((dist > 0) & (dist <= np.sqrt(2))).astype(float)
    W = W / W.sum(axis=1, keepdims=True)

    true_beta = np.array([1.0, 2.0, -0.5])
    true_lam = 0.6
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    eps = rng.normal(size=n) * 0.5
    u = np.linalg.solve(np.eye(n) - true_lam * W, eps)
    y = X @ true_beta + u

    # Naive OLS
    ols_beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  Naive OLS beta = {np.round(ols_beta, 3)}  (truth {true_beta})")

    fit = fit_sem(X, y, W)
    print(f"  SEM MLE lambda = {fit['lambda']:.3f}  (truth {true_lam})")
    print(f"  SEM beta       = {np.round(fit['beta'], 3)}  (truth {true_beta})")
    print(f"  sigma^2        = {fit['sigma2']:.3f}  (truth 0.25)")
    print(f"  log-lik        = {fit['loglik']:.2f}")

    print("\n--- library cross-check (spatialreg / spdep R; spreg / pysal Python) ---")
