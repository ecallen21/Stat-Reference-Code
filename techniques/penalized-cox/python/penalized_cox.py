"""Penalized Cox regression (L1 / L2 / elastic-net) (Reference §11.21).

Objective: maximize the partial log-likelihood MINUS a penalty:

    -logPL(beta)  +  lambda * ( (1 - alpha) * ||beta||_2^2 / 2  +  alpha * ||beta||_1 )

    alpha = 0    -> ridge (L2)
    alpha = 1    -> lasso (L1)      -> automatic variable selection
    0 < alpha < 1 -> elastic net    -> selection + grouping of correlated predictors

Solved via coordinate descent (Simon-Friedman-Hastie-Tibshirani 2011): for each
coordinate j, do a quadratic approximation to the partial log-likelihood at the
current beta and solve the resulting L1-penalized univariate LS via soft-
thresholding. Iterate until convergence.

This module implements a pedagogical coordinate-descent solver for demonstration.
For production high-dimensional Cox use R's `glmnet` or Python's `sksurv.linear_model.CoxnetSurvivalAnalysis`.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cox-ph", "python"))
from cox_ph import fit_cox    # techniques/cox-ph/python/cox_ph.py::fit_cox


def _soft_threshold(z, gamma):
    """Soft-thresholding operator used in coordinate-descent L1 problems."""
    if z > gamma:  return z - gamma
    if z < -gamma: return z + gamma
    return 0.0


def _cox_working_response(times, events, X, beta):
    """One IRLS-style working response (z, w) at current beta for coordinate descent.

    Uses the Newton step decomposition of the partial log-likelihood: score U(beta)
    and Hessian H(beta), then treats z = X beta + H^-1 U as the pseudo-response
    with weights = diag(H). Approximation, but standard for coordinate descent.
    """
    from cox_ph import _partial_lik_and_grad
    nll, ng, nH = _partial_lik_and_grad(beta, np.zeros_like(times), times, events, X, ties="breslow")
    # Newton step is - H^-1 g (where H = negative Hessian returned)
    # Work with per-observation "weights": use diag(X'HX)/(X'X) approximation on beta scale
    return -ng, nH               # score and negative-Hessian


def penalized_cox(times, events, X, lambda_: float, alpha: float = 1.0,
                   max_iter: int = 200, tol: float = 1e-6) -> dict:
    """Coordinate-descent penalized Cox with elastic-net penalty.

    Parameters
    ----------
    lambda_ : overall regularization strength (>= 0).
    alpha   : L1 fraction (0 = ridge, 1 = lasso, 0.5 = elastic net).
    """
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float); events = np.asarray(events, dtype=int)
    n, p = X.shape
    # Standardize columns for the penalty to be scale-consistent
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    beta_std = np.zeros(p)

    for it in range(max_iter):
        # Quadratic approximation at current beta (Newton step on unpenalized loss)
        from cox_ph import _partial_lik_and_grad
        nll, ng, nH = _partial_lik_and_grad(beta_std, np.zeros_like(times), times, events, Xs, ties="breslow")
        H_diag = np.diag(nH)
        # Working response z_j = beta_j + score_j / hessian_jj
        # (component-wise Newton step)
        z = beta_std + np.where(H_diag > 0, -ng / H_diag, 0.0)

        beta_new = np.zeros_like(beta_std)
        for j in range(p):
            # Coordinate update for elastic-net penalized quadratic
            partial = H_diag[j] * z[j]
            denom = H_diag[j] + n * lambda_ * (1 - alpha)
            if denom <= 0:
                beta_new[j] = 0.0
                continue
            beta_new[j] = _soft_threshold(partial / n, lambda_ * alpha) * n / denom
        if np.max(np.abs(beta_new - beta_std)) < tol:
            beta_std = beta_new; break
        beta_std = beta_new

    # Undo standardization: beta_original_scale = beta_std / sd
    beta = beta_std / sd
    return {"beta": beta.tolist(),
            "beta_standardized": beta_std.tolist(),
            "n_nonzero": int(np.sum(np.abs(beta) > 1e-8)),
            "lambda": lambda_, "alpha": alpha,
            "n_iter": it + 1,
            "method": f"penalized Cox (alpha={alpha}) via coordinate descent"}


def path_over_lambda(times, events, X, lambdas, alpha: float = 1.0) -> dict:
    """Solve penalized Cox at a sequence of lambdas (regularization path)."""
    return {"lambdas": list(lambdas),
             "coef_paths": [penalized_cox(times, events, X, lam, alpha)["beta"] for lam in lambdas],
             "alpha": alpha}


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    n = 200
    p_dim = 10
    X = rng.normal(0, 1, size=(n, p_dim))
    beta_true = np.zeros(p_dim)
    beta_true[:3] = [0.7, -0.5, 0.4]                # only first 3 predictors matter
    T_event = -np.log(rng.uniform(0, 1, n)) / (0.1 * np.exp(X @ beta_true))
    C = rng.uniform(0, 15, n)
    times = np.minimum(T_event, C); events = (T_event <= C).astype(int)

    print(f"=== Lasso Cox (alpha=1, lambda=0.05); true nonzeros = [0,1,2] ===")
    fit_lasso = penalized_cox(times, events, X, lambda_=0.05, alpha=1.0)
    for j, b in enumerate(fit_lasso["beta"]):
        marker = " *" if abs(b) > 1e-8 else ""
        print(f"  x{j}: beta = {b:+.4f}{marker}")
    print(f"  {fit_lasso['n_nonzero']} nonzero coefficients")

    print(f"\n=== Ridge Cox (alpha=0, lambda=0.05) ===")
    fit_ridge = penalized_cox(times, events, X, lambda_=0.05, alpha=0.0)
    for j, b in enumerate(fit_ridge["beta"]):
        print(f"  x{j}: beta = {b:+.4f}")

    print(f"\n=== Compare: ordinary Cox (unpenalized) ===")
    fit_ord = fit_cox(times, events, X)
    for j, b in enumerate(fit_ord["beta"]):
        print(f"  x{j}: beta = {b:+.4f}")

    print(f"\n=== Lasso path over lambda ∈ {{0.01, 0.02, 0.05, 0.1, 0.2}} ===")
    p = path_over_lambda(times, events, X, [0.01, 0.02, 0.05, 0.1, 0.2], alpha=1.0)
    for lam, coefs in zip(p["lambdas"], p["coef_paths"]):
        nz = sum(abs(c) > 1e-8 for c in coefs)
        print(f"  lambda={lam:5.3f}: n_nonzero = {nz}")
