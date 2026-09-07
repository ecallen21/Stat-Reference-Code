"""Fisher's method for combining p-values (Reference Sec 22.16).

Fisher 1932 'Statistical Methods for Research Workers'. Combines
k INDEPENDENT p-values into one overall p-value by summing log-p's:

    X_F = -2 * sum_i log(p_i)  ~  Chi^2_{2k}  under H0_global.

Related combiners:
    * STOUFFER (weighted-Z):    Z = sum w_i * Phi^{-1}(1 - p_i) / sqrt(sum w_i^2)
    * BROWN:                     Fisher corrected for correlated p's (uses cov)
    * CAUCHY (CCT):              robust to arbitrary dependence

Applications:
    * Meta-analysis of studies with only p-values reported.
    * Combining independent gene-set / SNP tests.
    * Multiple-endpoint clinical trials (rare).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def fisher_combine(p_values):
    p = np.asarray(p_values)
    X = -2 * np.sum(np.log(p))
    k = len(p)
    return {"X": float(X), "df": 2 * k,
            "p_combined": float(1 - stats.chi2.cdf(X, 2 * k))}


def stouffer_combine(p_values, weights=None):
    p = np.asarray(p_values)
    if weights is None:
        weights = np.ones(len(p))
    z = stats.norm.ppf(1 - p)
    Z = (weights * z).sum() / np.sqrt((weights ** 2).sum())
    return {"Z": float(Z), "p_combined": float(1 - stats.norm.cdf(Z))}


def cauchy_combine(p_values, weights=None):
    """Cauchy combination test (Liu & Xie 2020) -- robust to arbitrary dependence."""
    p = np.asarray(p_values)
    if weights is None:
        weights = np.ones(len(p)) / len(p)
    T = np.sum(weights * np.tan((0.5 - p) * np.pi))
    p_c = float(0.5 - np.arctan(T) / np.pi)
    return {"T_cauchy": float(T), "p_combined": p_c}


if __name__ == "__main__":
    print("=== Combining p-values: Fisher / Stouffer / Cauchy ===\n")
    rng = np.random.default_rng(0)

    scenarios = {
        "null (5 uniform p)":      rng.uniform(size=5),
        "signal (5 small p)":       [0.001, 0.02, 0.05, 0.10, 0.15],
        "one strong + 4 null":      [1e-4, 0.5, 0.7, 0.3, 0.4],
        "10 mildly-significant":    rng.uniform(0, 0.1, size=10).tolist(),
    }
    for label, p in scenarios.items():
        f = fisher_combine(p)
        s = stouffer_combine(p)
        c = cauchy_combine(p)
        print(f"  {label}")
        print(f"    p vector    = {np.round(p, 4)}")
        print(f"    Fisher p    = {f['p_combined']:.4g}   (X = {f['X']:.2f}, df = {f['df']})")
        print(f"    Stouffer p  = {s['p_combined']:.4g}")
        print(f"    Cauchy p    = {c['p_combined']:.4g}")
        print()

    print("--- library cross-check (metap R, poolr R; scipy.stats.combine_pvalues Python) ---")
