"""Spatial GLM: Poisson regression with CAR / ICAR spatial random effects (Reference §23.x extra).

Model (Poisson-lognormal with spatial random effects):

    y_i ~ Poisson( E_i * exp( x_i^T beta + u_i ) )
    u   ~ N(0, sigma^2 * (D - alpha W)^{-1})     (CAR)

We fit by MAP / PIRLS: iterate
  (1) Newton step for beta given u,
  (2) block-Newton step for u given beta with the CAR prior penalty,
  (3) update sigma^2 by generalised-cross-validation-ish rule
      sigma^2_hat = u^T (D - alpha W) u / (n - trace(H))    (rough).

Full Bayesian fits (CARBayes, INLA) sample the joint posterior — see the R stub.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def _build_grid_W(n_side: int):
    n = n_side ** 2
    W = np.zeros((n, n), dtype=int)
    for i in range(n_side):
        for j in range(n_side):
            k = i * n_side + j
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n_side and 0 <= jj < n_side:
                    W[k, ii * n_side + jj] = 1
    return W


def fit_poisson_car(y, X, W, E=None, alpha: float = 0.95,
                    n_iter: int = 40, tol: float = 1e-6) -> dict:
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    n, p = X.shape
    if E is None:
        E = np.ones(n)
    D = np.diag(W.sum(axis=1))
    Q = D - alpha * W                                    # CAR precision (unscaled by sigma^2)
    # start
    beta = np.zeros(p); u = np.zeros(n); sig2 = 1.0
    log_lik_prev = -np.inf
    for it in range(n_iter):
        # --- beta update: Newton on Poisson log-lik with u fixed ---
        eta = X @ beta + u
        mu = E * np.exp(np.clip(eta, -30, 30))
        g = X.T @ (y - mu)
        H = (X.T * mu) @ X
        step = np.linalg.solve(H + 1e-6 * np.eye(p), g)
        beta += np.clip(step, -1.0, 1.0)
        # --- u update: penalised IRLS with CAR prior u^T Q u / sig2 ---
        eta = X @ beta + u
        mu = E * np.exp(np.clip(eta, -30, 30))
        g_u = (y - mu) - Q @ u / sig2
        H_u = np.diag(mu) + Q / sig2
        step_u = np.linalg.solve(H_u + 1e-6 * np.eye(n), g_u)
        u += np.clip(step_u, -1.0, 1.0)
        # --- sigma^2 update: closed-form conditional posterior mean
        # under inverse-Gamma(0.001, 0.001) prior => sigma^2 = u^T Q u / n
        sig2 = float(max(u @ Q @ u / n, 1e-4))
        # convergence
        eta = X @ beta + u
        mu = E * np.exp(np.clip(eta, -30, 30))
        log_lik = float((y * np.log(mu + 1e-12) - mu).sum())
        if abs(log_lik - log_lik_prev) < tol:
            break
        log_lik_prev = log_lik
    return {"beta": beta, "u": u, "sigma2": sig2,
            "log_lik": log_lik, "n_iter": it + 1,
            "alpha": alpha,
            "method": "Poisson-CAR spatial GLM (PIRLS + CAR penalty)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_side = 10; n = n_side ** 2
    W = _build_grid_W(n_side)
    # true spatial field u (CAR draw)
    D = np.diag(W.sum(axis=1))
    alpha_true = 0.95
    Q = D - alpha_true * W
    Sigma_u = 1.5 * np.linalg.inv(Q + 1e-4 * np.eye(n))
    Lu = np.linalg.cholesky(Sigma_u)
    u_true = Lu @ rng.normal(size=n)

    # covariate
    x = rng.normal(size=n)
    beta_true = np.array([-0.5, 0.7])                   # intercept + x
    X = np.column_stack([np.ones(n), x])
    E = np.full(n, 20.0)                                 # expected counts
    log_mu = X @ beta_true + u_true
    y = rng.poisson(E * np.exp(log_mu))

    fit = fit_poisson_car(y, X, W, E=E, alpha=0.95, n_iter=30)
    print(f"=== Poisson-CAR spatial GLM (10x10 grid, alpha={fit['alpha']}) ===")
    print(f"  iterations = {fit['n_iter']}   log-lik = {fit['log_lik']:.1f}")
    print(f"  beta_hat = {np.round(fit['beta'], 3).tolist()}   "
          f"true beta = {beta_true.tolist()}")
    print(f"  sigma^2_hat = {fit['sigma2']:.3f}   true = 1.5")
    print(f"  cor(u_hat, u_true) = "
          f"{float(np.corrcoef(fit['u'], u_true)[0, 1]):+.3f}")

    # contrast: OLS ignoring spatial dependence
    from numpy.linalg import lstsq
    b_ols, *_ = lstsq(X, np.log((y + 0.5) / E), rcond=None)
    print(f"\n  OLS-ignoring-space beta = {np.round(b_ols, 3).tolist()}")

    print("\n--- library cross-check (R CARBayes::S.CARleroux / INLA::inla family='poisson') ---")
