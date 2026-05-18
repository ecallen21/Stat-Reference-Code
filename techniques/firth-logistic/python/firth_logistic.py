"""Firth's penalized likelihood logistic regression (Reference §7.7, §7.51).

When a predictor (or combination) PERFECTLY SEPARATES the outcome, the
maximum-likelihood logistic estimate diverges (beta -> +/- infinity, SEs blow
up). Symptoms: huge coefficients, huge SEs, "did not converge" warnings.

Firth (1993) proposes maximizing a PENALIZED log-likelihood:

    L*(beta) = L(beta) + 0.5 * log |I(beta)|

where I(beta) is the Fisher information. The Jeffreys-prior-style penalty
shrinks beta away from the boundary by an O(1/n) correction, giving FINITE
MLE-like estimates even with separation.

Algorithm: modified score equations. For logistic with weight w_i = pi_i(1 - pi_i)
and hat values h_i = diag(X (X'WX)^{-1} X'W), the Firth correction adds
(h_i / 2) * (1 - 2 * pi_i) to the working response of standard IRLS.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _logistic(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))


def fit(X, y, max_iter: int = 100, tol: float = 1e-7) -> dict:
    """Firth's penalized-likelihood logistic regression."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p); converged = False; it = 0
    for it in range(1, max_iter + 1):
        eta = X @ beta; pi = _logistic(eta)
        w = pi * (1 - pi); w = np.clip(w, 1e-12, None)
        sw = np.sqrt(w); Xw = X * sw[:, None]
        # Hat matrix diagonal h = diag(X (X'WX)^-1 X' W)
        XtWX_inv = np.linalg.pinv(X.T @ (X * w[:, None]))
        h = np.einsum("ij,jk,ik->i", X, XtWX_inv, X) * w
        # Firth working response: y_adj = y + h*(0.5 - pi); standard IRLS on it
        y_adj = y + h * (0.5 - pi)
        z = eta + (y_adj - pi) / w
        beta_new, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; converged = True; break
        beta = beta_new
    eta = X @ beta; pi = _logistic(eta)
    w = np.clip(pi * (1 - pi), 1e-12, None)
    var_beta = np.linalg.pinv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(var_beta))
    z_stats = beta / se; p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    return {"beta": beta.tolist(), "se": se.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "odds_ratio": np.exp(beta).tolist(),
            "iterations": it, "converged": converged}


def standard_logistic(X, y, max_iter=50):
    """Plain ML logistic for comparison (no Firth correction)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    p = X.shape[1]; beta = np.zeros(p); diverged = False
    for it in range(max_iter):
        eta = X @ beta; pi = _logistic(eta)
        w = pi * (1 - pi); w = np.clip(w, 1e-12, None)
        z = eta + (y - pi) / w
        sw = np.sqrt(w); Xw = X * sw[:, None]; zw = z * sw
        try:
            beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        except Exception:
            diverged = True; break
        if np.any(np.abs(beta_new) > 50):
            diverged = True; beta = beta_new; break
        if np.max(np.abs(beta_new - beta)) < 1e-7:
            beta = beta_new; break
        beta = beta_new
    return {"beta": beta.tolist(), "diverged": diverged, "iter": it + 1}


def library_versions(X, y):
    out = {}
    try:
        from firthlogist import FirthLogisticRegression
        flr = FirthLogisticRegression().fit(X[:, 1:], y)  # drop intercept col
        out["firthlogist coefs"] = ([float(flr.intercept_)] +
                                     [float(c) for c in flr.coef_])
    except Exception as exc:
        out["firthlogist"] = f"not used ({exc}); try 'pip install firthlogist'"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 50
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    # CONSTRUCT separation: every obs with x1 > 1.5 has y = 1, all others y = 0 noise
    y = (x1 > 0.5).astype(int)
    X = np.column_stack([np.ones(n), x1, x2])

    print("=== Standard ML logistic (should diverge or give huge betas) ===")
    std = standard_logistic(X, y)
    print(f"  beta = {[round(b, 4) for b in std['beta']]}")
    print(f"  diverged = {std['diverged']}  iter = {std['iter']}")

    print("\n=== Firth penalized logistic (should converge with reasonable betas) ===")
    res = fit(X, y)
    for j, nm in enumerate(["intercept", "x1", "x2"]):
        print(f"  {nm:10s} beta = {res['beta'][j]:+.4f}  SE = {res['se'][j]:.4f}  "
              f"OR = {res['odds_ratio'][j]:.3f}  p = {res['p_values'][j]:.4g}")
    print(f"  converged = {res['converged']}  iter = {res['iterations']}")

    print("\n--- library (firthlogist) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
