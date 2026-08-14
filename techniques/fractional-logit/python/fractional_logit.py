"""Fractional response / fractional-logit regression (Reference §5.26).

Outcome in [0, 1] including the boundary values 0 and 1 (unlike Beta
regression which is strictly interior).  Examples: fraction of budget
spent, percentage of votes won, participation rate.

Papke-Wooldridge (1996) proposed a QUASI-LIKELIHOOD approach:
    E[y | x] = G(x' beta)                 G = logistic (or probit)
    Bernoulli-like log-likelihood evaluated on the fractional outcome:
        l(beta) = sum_i (y_i log G(x_i' beta) + (1 - y_i) log(1 - G(x_i' beta)))

Even though y is not Bernoulli, this QUASI-MLE is consistent for the
conditional mean function.  Standard errors need a SANDWICH correction
(HC1 or HC3).

Contrast with beta-regression: beta-reg requires y in (0, 1) and models
the full distribution; fractional logit only models the conditional mean
but is boundary-friendly.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def fractional_logit(X, y) -> dict:
    """Papke-Wooldridge fractional logit quasi-MLE with HC0 sandwich SEs."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    def neg_ql(beta):
        z = X @ beta
        return -np.sum(y * z - np.logaddexp(0, z))
    res = minimize(neg_ql, np.zeros(p), method="BFGS")
    beta = res.x
    # HC0 sandwich SE
    z = X @ beta; mu = 1 / (1 + np.exp(-z))
    A = X.T @ (X * (mu * (1 - mu))[:, None])
    resid = y - mu
    B = X.T @ (X * (resid ** 2)[:, None])
    cov = np.linalg.pinv(A) @ B @ np.linalg.pinv(A)
    se = np.sqrt(np.diag(cov))
    return {"beta": beta, "se_HC0": se,
            "z": beta / se, "p_value": 2 * stats.norm.sf(np.abs(beta / se)),
            "log_quasi_lik": float(-res.fun),
            "method": "Fractional logit (Papke-Wooldridge quasi-MLE) with HC0 SEs"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta_true = np.array([0.0, 0.8, -0.4])
    mu = 1 / (1 + np.exp(-(X @ beta_true)))
    # Fractional outcome: beta-distributed around mu with concentration
    y = rng.beta(mu * 20, (1 - mu) * 20)
    # Mix in boundary zeros/ones (~10%)
    idx0 = rng.choice(n, 30, replace=False); y[idx0] = 0
    idx1 = rng.choice(n, 20, replace=False); y[idx1] = 1

    print(f"=== Fractional logit (n = {n}, {int((y == 0).sum())} zeros, {int((y == 1).sum())} ones) ===")
    r = fractional_logit(X, y)
    for i, name in enumerate(("intercept", "x1", "x2")):
        print(f"  {name}: beta = {r['beta'][i]:6.3f}   HC0 SE = {r['se_HC0'][i]:.3f}   "
              f"z = {r['z'][i]:.3f}   p = {r['p_value'][i]:.4f}   true = {beta_true[i]}")

    print("\n--- library cross-check (statsmodels GLM binomial with fractional y) ---")
    try:
        import statsmodels.api as sm
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC0")
        print(f"  statsmodels beta: {m.params.round(3)}")
        print(f"  statsmodels HC0 SE: {m.bse.round(3)}")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
