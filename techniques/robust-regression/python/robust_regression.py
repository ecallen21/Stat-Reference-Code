"""Robust regression: Huber M-estimator, MM-estimator, LTS (Reference §5.11).

OLS minimizes sum(e_i^2). A single bad point can dominate the fit because
squared loss grows without bound. Robust methods replace the squared loss with
something that grows linearly (or stops growing) in the tail:

  Huber rho(r) = { r^2 / 2          if |r| <= k
                 { k|r| - k^2 / 2   if |r| > k

The M-estimator minimizes sum(rho(e_i / sigma_hat)). Implemented via IRLS:
  weights w_i = rho'(r_i) / r_i  (Huber: 1 for |r| <= k, k / |r| outside)
  beta <- (X' W X)^{-1} X' W y     (WLS step)
  repeat until convergence.

This is essentially the same loop as ``techniques/robust-location-scale``, but
in the multivariate-regression setting.

Higher-breakdown options (mentioned in the README):
  - MM-estimator   (Yohai 1987): starts from a high-breakdown S-estimator,
                                 then takes M-steps. 50% breakdown + high
                                 efficiency.
  - LTS (Least Trimmed Squares): minimize sum of the h smallest squared
                                 residuals, h = floor((n + p + 1) / 2).
                                 Exact algorithm is NP-hard; use FAST-LTS heuristic.

Both are available in R's ``robustbase`` (``lmrob``, ``ltsReg``) and Python's
``statsmodels.robust.robust_linear_model``. We implement Huber from scratch and
mention the others.
"""
from __future__ import annotations

import math

import numpy as np


def _mad_scale(x):
    """1.4826 * median(|x - median(x)|)  --  consistent estimate of sigma at the normal."""
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def huber_regression(X, y, k: float = 1.345, max_iter: int = 100, tol: float = 1e-7) -> dict:
    """Huber M-estimator of beta via IRLS.

    Parameters
    ----------
    X : design matrix (with intercept column).
    y : response.
    k : Huber tuning constant (1.345 -> ~95% efficiency at the normal).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    # OLS start
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for it in range(max_iter):
        r = y - X @ beta
        sigma = _mad_scale(r) or 1.0
        u = r / sigma
        w = np.where(np.abs(u) <= k, 1.0, k / np.maximum(np.abs(u), 1e-15))
        # WLS step
        sw = np.sqrt(w)
        Xw = X * sw[:, None]; yw = y * sw
        beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol * np.maximum(1.0, np.max(np.abs(beta))):
            beta = beta_new
            break
        beta = beta_new
    # Final weights / residuals / SE (M-estimator SE: sigma * sqrt(diag((X'X)^-1)) corrected
    # by an asymptotic factor; we use the simple SE approximation here)
    r = y - X @ beta
    sigma = _mad_scale(r) or 1.0
    var_beta = sigma ** 2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    return {"beta": beta.tolist(), "se": se.tolist(),
            "sigma_hat": float(sigma), "iterations": it + 1,
            "residuals": r.tolist(), "weights": w.tolist()}


def library_versions(X, y):
    out = {}
    try:
        import statsmodels.api as sm
        rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        out["statsmodels RLM (Huber) params"] = rlm.params.tolist()
        out["statsmodels RLM (Huber) bse"] = rlm.bse.tolist()
    except Exception as exc:
        out["statsmodels"] = f"error: {exc}"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 1 + 2 * x + rng.normal(0, 1.0, n)
    # Add a few bad y values
    y[0] = 50; y[5] = -40
    X = np.column_stack([np.ones(n), x])

    print("=== OLS (contaminated) ===")
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  beta = {beta_ols.tolist()}")

    print("\n=== Huber M-regression (k = 1.345) ===")
    res = huber_regression(X, y)
    print(f"  beta = {res['beta']}   SE = {res['se']}")
    print(f"  iterations = {res['iterations']}   sigma_hat = {res['sigma_hat']:.4f}")
    print(f"  smallest weights: {sorted(res['weights'])[:5]}")

    print(f"\n  true intercept = 1.0, true slope = 2.0")
    print("  Huber recovers ~ the true coefficients; OLS is pulled by the outliers.")

    print("\n--- library (statsmodels RLM) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
