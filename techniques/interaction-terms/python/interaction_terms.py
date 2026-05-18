"""Interaction terms in regression (Reference §5.16, §5.24, §5.25, §5.37).

An interaction means the effect of one predictor depends on the level of
another. Add an x1 * x2 product column to the design matrix:

  y = b0 + b1 x1 + b2 x2 + b3 (x1 * x2) + eps

The effect of x1 on y is then (b1 + b3 x2) -- it MOVES with x2.

What this file shows
--------------------
1. Continuous-by-continuous interaction (x1 * x2).
   - Centering x1 and x2 before forming the product hugely reduces collinearity
     between the main effects and the product (§5.25). The fitted values and R^2
     are unchanged; the COEFFICIENTS and their SEs are far more interpretable.

2. Categorical-by-continuous interaction (group * x).
   - The 'simple slope' for each group: beta_x + beta_{group_k * x}.
   - Used to ask "does the slope differ across groups?"

3. Marginal-effects helper: given a fitted vector of betas and the design row's
   structure, return the derivative dE[y]/dx_j as a function of the other
   predictors -- the practical "what does the coefficient mean" check.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fit_ols_with_se(X, y):
    """Helper: fit y = X beta; return betas with SEs."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    rss = float(e @ e); sigma2 = rss / (n - p)
    var_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    t = beta / se; p_vals = 2 * stats.t.sf(np.abs(t), n - p)
    return {"beta": beta, "se": se, "t": t, "p": p_vals,
            "rss": rss, "sigma_hat": float(math.sqrt(sigma2)),
            "df_residual": n - p, "vcov": var_beta}


def continuous_by_continuous(x1, x2, y, center: bool = False) -> dict:
    """Fit y ~ x1 + x2 + x1:x2. Optionally center x1, x2 first."""
    x1 = np.asarray(x1, dtype=float); x2 = np.asarray(x2, dtype=float)
    if center:
        x1 = x1 - x1.mean(); x2 = x2 - x2.mean()
    X = np.column_stack([np.ones_like(x1), x1, x2, x1 * x2])
    out = fit_ols_with_se(X, y)
    out["names"] = ["intercept", "x1", "x2", "x1:x2"]
    out["centered"] = center
    return out


def categorical_by_continuous(x, group, y) -> dict:
    """Fit y ~ x + group + x:group via dummy coding.

    ``group`` is a 1-D label vector. We dummy-code with the first level as the
    reference, then form interaction columns for the other levels.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    levels = sorted(set(group))
    ref = levels[0]; dummies = []
    for lev in levels[1:]:
        dummies.append(np.array([1.0 if g == lev else 0.0 for g in group]))
    D = np.column_stack(dummies) if dummies else np.zeros((len(x), 0))
    inter = np.column_stack([d * x for d in dummies]) if dummies else np.zeros((len(x), 0))
    X = np.column_stack([np.ones_like(x), x, D, inter])
    out = fit_ols_with_se(X, y)
    names = ["intercept", "x"]
    names += [f"group[{lev}]" for lev in levels[1:]]
    names += [f"x:group[{lev}]" for lev in levels[1:]]
    out["names"] = names
    # Simple-slope summary per level
    slopes = {ref: out["beta"][1]}
    for k, lev in enumerate(levels[1:]):
        slopes[lev] = out["beta"][1] + out["beta"][2 + len(dummies) + k]
    out["simple_slopes"] = slopes
    return out


def marginal_effect_x1(beta, x2_values):
    """For y = b0 + b1 x1 + b2 x2 + b3 x1 x2, dE[y]/dx1 = b1 + b3 x2."""
    return (beta[1] + beta[3] * np.asarray(x2_values)).tolist()


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 200

    print("=== Continuous-by-continuous interaction ===")
    x1 = rng.normal(10, 3, n); x2 = rng.normal(5, 2, n)
    y = 1 + 0.5 * x1 - 0.3 * x2 + 0.1 * x1 * x2 + rng.normal(0, 1, n)
    print("\n-- raw (uncentered) -- look at the SEs on the main effects:")
    raw = continuous_by_continuous(x1, x2, y, center=False)
    for nm, b, se, p in zip(raw["names"], raw["beta"], raw["se"], raw["p"]):
        print(f"  {nm:10s}: beta = {b:+.4f}  SE = {se:.4f}  p = {p:.4g}")
    print("\n-- centered (subtract means before the product) --")
    cen = continuous_by_continuous(x1, x2, y, center=True)
    for nm, b, se, p in zip(cen["names"], cen["beta"], cen["se"], cen["p"]):
        print(f"  {nm:10s}: beta = {b:+.4f}  SE = {se:.4f}  p = {p:.4g}")
    print(f"  (R^2 should be identical: raw {1 - raw['rss']/((y - y.mean())**2).sum():.4f},"
          f" centered {1 - cen['rss']/((y - y.mean())**2).sum():.4f})")

    print("\n=== Marginal effect of x1 at x2 = -1, 0, +1 (centered fit) ===")
    print(marginal_effect_x1(cen["beta"], [-1, 0, 1]))

    print("\n=== Categorical (3 levels) by continuous ===")
    group = rng.choice(["A", "B", "C"], n)
    slope_by_group = {"A": 0.2, "B": 0.6, "C": 1.0}
    y2 = 1.0 + np.array([slope_by_group[g] for g in group]) * x1 + rng.normal(0, 1, n)
    res = categorical_by_continuous(x1, group, y2)
    for nm, b, se, p in zip(res["names"], res["beta"], res["se"], res["p"]):
        print(f"  {nm:18s}: beta = {b:+.4f}  SE = {se:.4f}  p = {p:.4g}")
    print(f"  simple slopes by group: {res['simple_slopes']}")
    print(f"  (true slopes were {slope_by_group})")
