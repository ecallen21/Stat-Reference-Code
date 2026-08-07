"""Tobit regression for censored continuous outcomes (Reference §5.19).

Continuous outcomes with a pile-up at a known limit (bottom-coded at 0 for
spending, top-coded at 100 for percentage scores).  Latent-variable formulation:

    y_i^*     = X_i beta + eps_i,   eps_i ~ N(0, sigma^2)
    y_i       = max(L, min(y_i^*, U))     (observed censored value)

Standard Tobit (Tobin 1958): left-censoring at 0.  Type-II ("heckit") allows
separate selection and outcome equations.

Log-likelihood
    Contributions from three regimes:
        y_i = L        : Pr(y_i^* <= L)                            = Phi((L - X_i beta) / sigma)
        y_i in (L, U)  : (1 / sigma) phi((y_i - X_i beta) / sigma)
        y_i = U        : Pr(y_i^* >= U) = 1 - Phi((U - X_i beta) / sigma)

MLE by BFGS on (beta, log sigma).

Contrast with OLS-on-uncensored: naive OLS on the observed y is biased
downward (for right-censoring) because the tail is compressed.  Two-part
models (see zero-inflated-regression) offer an alternative if the pile-up
is really a distinct "no participation" process.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def tobit_fit(X, y, lower: float = None, upper: float = None) -> dict:
    """Tobit MLE with optional left (lower) and/or right (upper) censoring."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    left_mask = None if lower is None else (y <= lower + 1e-12)
    right_mask = None if upper is None else (y >= upper - 1e-12)
    if left_mask is None and right_mask is None:
        raise ValueError("provide at least one of lower/upper")
    def neg_ll(theta):
        beta = theta[:p]; log_sigma = theta[p]
        sigma = math.exp(log_sigma)
        mu = X @ beta
        ll = 0.0
        # Uncensored obs
        mid = np.ones(n, dtype=bool)
        if left_mask is not None: mid &= ~left_mask
        if right_mask is not None: mid &= ~right_mask
        if mid.any():
            ll += np.sum(stats.norm.logpdf(y[mid], loc=mu[mid], scale=sigma))
        # Left-censored
        if left_mask is not None and left_mask.any():
            ll += np.sum(stats.norm.logcdf(lower, loc=mu[left_mask], scale=sigma))
        # Right-censored
        if right_mask is not None and right_mask.any():
            ll += np.sum(stats.norm.logsf(upper, loc=mu[right_mask], scale=sigma))
        return -ll
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta0 = np.concatenate([beta0, [math.log(y.std())]])
    res = minimize(neg_ll, theta0, method="BFGS")
    beta_hat = res.x[:p]; sigma_hat = math.exp(res.x[p])
    se = np.sqrt(np.diag(res.hess_inv))
    return {"beta": beta_hat, "sigma": sigma_hat,
            "se_beta": se[:p], "se_log_sigma": float(se[p]),
            "log_lik": float(-res.fun),
            "n": int(n), "n_censored_left": int(left_mask.sum() if left_mask is not None else 0),
            "n_censored_right": int(right_mask.sum() if right_mask is not None else 0),
            "method": "Tobit MLE (BFGS)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n); X = np.column_stack([np.ones(n), x])
    beta_true = np.array([0.5, 1.0]); sigma_true = 1.0
    y_star = X @ beta_true + rng.normal(0, sigma_true, n)
    y = np.maximum(y_star, 0)  # left-censored at 0

    print("=== Tobit vs OLS on left-censored data ===")
    print(f"  n_censored (y = 0): {int((y == 0).sum())} of {n}")
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  OLS beta         = {beta_ols.round(3)}  (biased toward 0)")
    r = tobit_fit(X, y, lower=0.0)
    print(f"  Tobit beta       = {r['beta'].round(3)}  (true {beta_true})")
    print(f"  Tobit sigma      = {r['sigma']:.3f}  (true {sigma_true})")
    print(f"  Tobit log-lik    = {r['log_lik']:.2f}")

    print("\n=== Tobit with two-sided censoring [0, 5] ===")
    y2 = np.clip(y_star, 0, 5)
    print(f"  n left-censored = {int((y2 == 0).sum())}, n right-censored = {int((y2 == 5).sum())}")
    r = tobit_fit(X, y2, lower=0.0, upper=5.0)
    print(f"  Tobit beta = {r['beta'].round(3)}  (true {beta_true})")

    print("\n--- library cross-check (statsmodels or lifelines) ---")
    try:
        import statsmodels.api as sm
        # statsmodels does not have a first-class Tobit; use OLS on truncated + report
        print(f"  statsmodels OLS truncated:  {sm.OLS(y[y > 0], X[y > 0]).fit().params.round(3)}")
    except Exception as ex:
        print(f"  (statsmodels error: {ex})")
