"""Standardized regression coefficients and dominance analysis (Reference §5.38).

Two related ways to answer "which predictor matters most?"

STANDARDIZED COEFFICIENTS ("beta weights")
------------------------------------------
Standardize y and each x_j (subtract mean, divide by SD) and refit OLS.
The resulting coefficients are unit-free and comparable in magnitude:

    beta_j^* = beta_j * SD(x_j) / SD(y)

A beta weight of 0.4 means "a 1-SD increase in x_j shifts y by 0.4 SD."
Useful when predictors are on wildly different scales (income in $ vs.
age in years vs. binary indicators), but PROBLEMATIC when predictors are
correlated -- the standardized coefficients reflect the unique
contribution given the rest of the model, not the variable's importance
in isolation.

DOMINANCE ANALYSIS (Budescu 1993)
---------------------------------
Decomposes the model's R^2 into per-predictor "general dominance" contributions
that sum to total R^2. For each subset S not containing j, compute R^2(S union j)
- R^2(S); average over all S of a given size; then average over all sizes.

  general_dominance_j = mean over subset sizes k of [
      mean over subsets S of size k not containing j of
          R^2(S union j) - R^2(S)
  ]

This is the "Shapley value" of x_j in the cooperative game where the payoff is
R^2(S). It is COMPUTATIONALLY EXPENSIVE (2^p model fits); feasible for p <= ~15.
For larger p, sample subsets randomly ("permutation importance" with R^2 as
the metric).

The TWO methods can disagree: a predictor with high beta-weight might have low
dominance if much of its apparent effect is shared with correlated predictors.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def _ols_r2(X, y):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if X.shape[1] == 0:
        return 0.0
    Xc = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    e = y - Xc @ beta
    rss = float(e @ e); tss = float(((y - y.mean()) ** 2).sum())
    return 1 - rss / tss if tss > 0 else float("nan")


def standardized_coefficients(X, y) -> dict:
    """Refit OLS on z-scored predictors and response; return beta weights."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    mu_x = X.mean(axis=0); sd_x = X.std(axis=0, ddof=1)
    mu_y = y.mean(); sd_y = y.std(ddof=1)
    sd_x = np.where(sd_x == 0, 1, sd_x); sd_y = sd_y if sd_y else 1
    Xz = (X - mu_x) / sd_x
    yz = (y - mu_y) / sd_y
    beta_std, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), Xz]), yz, rcond=None)
    # Also report the raw OLS coefficients and the conversion factor
    beta_raw, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)
    return {"intercept_standardized": float(beta_std[0]),
            "beta_standardized": beta_std[1:].tolist(),
            "beta_raw": beta_raw[1:].tolist(),
            "sd_x": sd_x.tolist(), "sd_y": float(sd_y)}


def dominance_analysis(X, y, names=None) -> dict:
    """General-dominance Shapley decomposition of R^2 across predictors.

    Each predictor's contribution is its average gain in R^2 from being added
    to a subset of the other predictors, averaged uniformly over subset sizes.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if names is None:
        names = [f"x{j+1}" for j in range(p)]
    # Total R^2 of the full model
    total_r2 = _ols_r2(X, y)
    # By-size averages of "R^2 gain when adding predictor j to a size-k subset"
    contributions = {j: [] for j in range(p)}
    for k in range(p):
        for subset in combinations(range(p), k):
            r2_S = _ols_r2(X[:, list(subset)], y)
            for j in range(p):
                if j in subset: continue
                r2_Sj = _ols_r2(X[:, list(subset) + [j]], y)
                contributions[j].append((k, r2_Sj - r2_S))
    # General dominance: mean over subset SIZES of mean-within-size gain
    gd = {}
    for j in range(p):
        by_size = {}
        for k, gain in contributions[j]:
            by_size.setdefault(k, []).append(gain)
        gd[j] = float(np.mean([np.mean(v) for v in by_size.values()]))
    return {"names": names,
            "general_dominance": {names[j]: gd[j] for j in range(p)},
            "total_r_squared": total_r2,
            "sum_of_dominance": float(sum(gd.values()))}


def library_versions(X, y):
    out = {}
    try:
        import statsmodels.api as sm
        Xz = (X - X.mean(0)) / X.std(0, ddof=1)
        yz = (y - y.mean()) / y.std(ddof=1)
        res = sm.OLS(yz, sm.add_constant(Xz)).fit()
        out["statsmodels OLS on z-scored data"] = res.params.tolist()
    except Exception as exc:
        out["statsmodels"] = f"error: {exc}"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    n, p = 200, 4
    # Correlated predictors so beta weights and dominance can disagree
    A = np.array([[1.0, 0.7, 0.0, 0.0],
                  [0.7, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.3],
                  [0.0, 0.0, 0.3, 1.0]])
    L = np.linalg.cholesky(A)
    X = rng.normal(0, 1, (n, p)) @ L.T
    true_beta = np.array([1.0, 0.5, 0.8, -0.4])
    y = X @ true_beta + rng.normal(0, 1, n)
    names = ["x1", "x2", "x3", "x4"]

    print("=== Standardized coefficients (z-score then OLS) ===")
    sc = standardized_coefficients(X, y)
    for nm, br, bs in zip(names, sc["beta_raw"], sc["beta_standardized"]):
        print(f"  {nm}: raw beta = {br:+.4f}   standardized = {bs:+.4f}")

    print("\n=== Dominance analysis (Shapley decomposition of R^2) ===")
    da = dominance_analysis(X, y, names)
    for nm, contrib in da["general_dominance"].items():
        print(f"  {nm}: general dominance = {contrib:+.4f}")
    print(f"  sum of contributions = {da['sum_of_dominance']:.4f}")
    print(f"  total R^2            = {da['total_r_squared']:.4f}")
    print("  (sum should equal total R^2, by construction)")

    print("\n--- library (statsmodels on z-scored data) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
