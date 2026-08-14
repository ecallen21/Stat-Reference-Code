"""Analysis of Covariance (Reference §6.16).

Combines ANOVA (categorical predictor) with linear regression on a
continuous COVARIATE (e.g. baseline value).  Adjusts group means for the
covariate to reduce within-group variance and increase power.

Model:
    y_ij = mu + alpha_i + beta * (x_ij - x_bar) + eps_ij
        alpha_i : group effect (i-th treatment)
        beta    : common slope on covariate x

Assumption of PARALLEL SLOPES (no group x covariate interaction):
    Test by fitting the interaction model  y ~ group + x + group:x  and
    checking group:x term.  If not parallel, ANCOVA is invalid;
    report Johnson-Neyman regions of significance.

Adjusted means (least-squares means):
    mu_i_hat + beta_hat * (x_bar_overall - x_bar_i)
    Report these instead of raw group means.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def ancova(y, group, x) -> dict:
    """One-covariate ANCOVA + parallel-slopes test + adjusted means."""
    y = np.asarray(y, dtype=float); group = np.asarray(group); x = np.asarray(x, dtype=float)
    groups = np.unique(group); k = len(groups); n = len(y)
    x_bar = float(x.mean())

    # Full ANCOVA design: dummy-coded groups + covariate (parallel slopes)
    D = np.column_stack([np.ones(n)] +
                         [(group == g).astype(float) for g in groups[1:]] +
                         [x - x_bar])
    beta_full, resid_full = _ols(D, y)
    RSS_full = float(resid_full @ resid_full)
    df_full = n - D.shape[1]

    # Restricted (no group effect): intercept + covariate
    D_res = np.column_stack([np.ones(n), x - x_bar])
    _, resid_res = _ols(D_res, y)
    RSS_res = float(resid_res @ resid_res)
    df_res = n - 2

    # F test for group effect
    F_group = ((RSS_res - RSS_full) / (k - 1)) / (RSS_full / df_full)
    p_group = float(stats.f.sf(F_group, k - 1, df_full))

    # Parallel-slopes test: full-full + group:x interactions
    D_int = np.column_stack([D] + [(group == g).astype(float) * (x - x_bar) for g in groups[1:]])
    _, resid_int = _ols(D_int, y)
    RSS_int = float(resid_int @ resid_int)
    df_int = n - D_int.shape[1]
    F_slope = ((RSS_full - RSS_int) / (k - 1)) / (RSS_int / df_int)
    p_slope = float(stats.f.sf(F_slope, k - 1, df_int))

    # Adjusted means
    beta_cov = beta_full[-1]
    means_adj = {}
    for i, g in enumerate(groups):
        mask = group == g
        y_bar_i = y[mask].mean(); x_bar_i = x[mask].mean()
        means_adj[str(g)] = float(y_bar_i + beta_cov * (x_bar - x_bar_i))
    raw_means = {str(g): float(y[group == g].mean()) for g in groups}

    return {"F_group": float(F_group), "df_group": (k - 1, int(df_full)),
            "p_group": p_group,
            "F_parallel_slopes": float(F_slope),
            "p_parallel_slopes": p_slope,
            "beta_covariate": float(beta_cov),
            "adjusted_means": means_adj, "raw_means": raw_means,
            "n": int(n), "k_groups": int(k),
            "method": "ANCOVA (one covariate, parallel slopes)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_per = 25; groups = np.repeat(["A", "B", "C"], n_per)
    x = rng.normal(size=3 * n_per)
    # Group effect + covariate effect + noise
    y = np.concatenate([1 + 0.5 * x[:n_per] + rng.normal(0, 1, n_per),
                        2 + 0.5 * x[n_per:2*n_per] + rng.normal(0, 1, n_per),
                        3 + 0.5 * x[2*n_per:] + rng.normal(0, 1, n_per)])

    print("=== ANCOVA (3 groups, common slope 0.5, group effects 1/2/3) ===")
    r = ancova(y, groups, x)
    print(f"  F group   = {r['F_group']:.3f}, df = {r['df_group']}, p = {r['p_group']:.4f}")
    print(f"  beta cov  = {r['beta_covariate']:.3f} (true 0.5)")
    print(f"  parallel-slopes test: F = {r['F_parallel_slopes']:.3f}, p = {r['p_parallel_slopes']:.4f}")
    print(f"  raw means:      {r['raw_means']}")
    print(f"  adjusted means: {r['adjusted_means']}")
    print("\n--- library cross-check (statsmodels.formula.api ols) ---")
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
        df = pd.DataFrame({"y": y, "group": groups, "x": x})
        fit = smf.ols("y ~ group + x", data=df).fit()
        print(f"  statsmodels beta_x = {fit.params['x']:.3f}")
        print(f"  statsmodels F test for group:\n{fit.f_test('group[T.B] = group[T.C] = 0')}")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
