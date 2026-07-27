"""Permutation / randomization tests (Reference §10.7, §10.16).

Under H0 (no effect), the group labels are EXCHANGEABLE with the outcomes.
So if H0 is true, permuting labels should produce a test statistic distributed
like the observed one.

Two-sample mean difference:
    Observed T_obs = mean(y | group=A) - mean(y | group=B)
    Under H0: randomly shuffle the group labels; recompute T on the shuffle.
    Repeat B times to build the null distribution.
    p-value = fraction of |T_perm| >= |T_obs|      (two-sided)

Extends to any statistic where the null-hypothesis symmetry is clear:
    - two-sample mean / median / t-stat
    - correlation between two variables (permute one)
    - regression coefficient (permute residuals -- see wild-bootstrap or the
      "residual randomization" approach below)

Exact vs. approximate:
    Exact  : enumerate all n!/(n1! n2!) label assignments -- feasible only when
             n is small (< ~15).
    Monte-Carlo (default) : sample B random permutations -- p-value is unbiased
             for any B (small B just widens its CI).

Add-1 rule (Phipson & Smyth 2010) for a valid p-value:
    p_hat = (1 + #{|T_perm| >= |T_obs|}) / (1 + B)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def permutation_two_sample(x1, x2, statistic=None, n_perm: int = 5000,
                            alternative: str = "two-sided", seed: int = 0) -> dict:
    """Permutation test for two independent samples.

    ``statistic(a, b)`` should return a scalar. Default: mean(a) - mean(b).
    """
    x1 = np.asarray(x1, dtype=float); x2 = np.asarray(x2, dtype=float)
    n1, n2 = x1.size, x2.size; n = n1 + n2
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x1, x2])
    t_obs = float(statistic(x1, x2))
    t_perm = np.empty(n_perm)
    for b in range(n_perm):
        idx = rng.permutation(n)
        a = combined[idx[:n1]]; c = combined[idx[n1:]]
        t_perm[b] = float(statistic(a, c))
    if alternative == "two-sided":
        extreme = np.abs(t_perm) >= abs(t_obs)
    elif alternative == "greater":
        extreme = t_perm >= t_obs
    elif alternative == "less":
        extreme = t_perm <= t_obs
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    p = (1 + int(extreme.sum())) / (1 + n_perm)         # Phipson-Smyth
    return {"T_obs": t_obs,
            "p_value": float(p),
            "alternative": alternative,
            "n_perm": n_perm, "n1": n1, "n2": n2,
            "method": "two-sample permutation test (Phipson-Smyth add-1)"}


def permutation_correlation(x, y, n_perm: int = 5000, seed: int = 0) -> dict:
    """Permutation test for Pearson correlation (permute y)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = x.size
    r_obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm)
    for b in range(n_perm):
        r_perm[b] = float(np.corrcoef(x, rng.permutation(y))[0, 1])
    extreme = np.abs(r_perm) >= abs(r_obs)
    p = (1 + int(extreme.sum())) / (1 + n_perm)
    return {"r_obs": r_obs, "p_value": float(p),
            "n_perm": n_perm, "n": n,
            "method": "permutation test for correlation (two-sided)"}


def permutation_regression_coef(X, y, coef_index: int, n_perm: int = 5000, seed: int = 0) -> dict:
    """Permutation test for a regression coefficient via response permutation.

    NOTE: full-response permutation tests the JOINT null of ALL coefficients = 0
    (not just the one you named), unless you use residual permutation (Freedman-
    Lane). This function does the simple full-response version; use with care.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n = X.shape[0]
    beta_obs, *_ = np.linalg.lstsq(X, y, rcond=None)
    obs = float(beta_obs[coef_index])
    rng = np.random.default_rng(seed)
    coef_perm = np.empty(n_perm)
    for b in range(n_perm):
        y_perm = rng.permutation(y)
        beta_p, *_ = np.linalg.lstsq(X, y_perm, rcond=None)
        coef_perm[b] = beta_p[coef_index]
    extreme = np.abs(coef_perm) >= abs(obs)
    p = (1 + int(extreme.sum())) / (1 + n_perm)
    return {"coef_obs": obs, "p_value": float(p),
            "coef_index": coef_index,
            "n_perm": n_perm,
            "method": "permutation test for regression coef (response permutation)"}


def library_versions(x1, x2):
    from scipy.stats import permutation_test
    res = permutation_test((x1, x2), lambda a, b: np.mean(a) - np.mean(b),
                            n_resamples=5000, permutation_type="independent",
                            alternative="two-sided", random_state=0)
    return {"scipy.stats.permutation_test":
            {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}}


if __name__ == "__main__":
    rng = np.random.default_rng(37)
    x1 = rng.normal(0, 1, 40); x2 = rng.normal(0.5, 1, 45)         # real shift
    print("=== Two-sample permutation test (mean diff) ===")
    out = permutation_two_sample(x1, x2)
    for k, v in out.items(): print(f"  {k:15s}: {v}")

    x = rng.normal(0, 1, 100); y = 0.4 * x + rng.normal(0, 1, 100)
    print("\n=== Permutation correlation (true r > 0) ===")
    out = permutation_correlation(x, y)
    for k, v in out.items(): print(f"  {k:15s}: {v}")

    n = 200
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    y = 1.0 + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)
    print("\n=== Permutation test for regression coef of x1 (true = 0.5) ===")
    out = permutation_regression_coef(X, y, coef_index=1)
    for k, v in out.items(): print(f"  {k:15s}: {v}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(x1, x2).items():
        print(f"  {k}: {v}")
