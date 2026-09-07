"""Woolf test of homogeneity of odds ratios (Reference Sec 4.17).

Woolf 1955 'On estimating the relationship between blood group and
disease', Ann Hum Genet. Tests whether K stratum-specific odds
ratios are HOMOGENEOUS -- a prerequisite for reporting a single
pooled OR (Mantel-Haenszel / Cochran-Mantel-Haenszel).

Statistic:
    X_W = sum_k w_k * (log OR_k - log OR_pool)^2   ~   Chi^2_{K-1}
    w_k = 1 / Var(log OR_k) = (1/a_k + 1/b_k + 1/c_k + 1/d_k)^{-1}

Reject H0 (homogeneity) if X_W > chi^2 critical.

Related: Breslow-Day (implemented separately) is a better small-sample
alternative; Woolf is asymptotic but simple.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def woolf_test(tables):
    """tables : list of (a, b, c, d) 2x2 tuples per stratum."""
    logOR = []; w = []
    for a, b, c, d in tables:
        #  Haldane correction if any zero
        if 0 in (a, b, c, d):
            a += 0.5; b += 0.5; c += 0.5; d += 0.5
        lor = np.log((a * d) / (b * c))
        var = 1 / a + 1 / b + 1 / c + 1 / d
        logOR.append(lor); w.append(1 / var)
    logOR = np.array(logOR); w = np.array(w)
    lor_pool = float(np.sum(w * logOR) / np.sum(w))
    X = float(np.sum(w * (logOR - lor_pool) ** 2))
    df = len(tables) - 1
    p = 1 - stats.chi2.cdf(X, df)
    return {"X_W": X, "df": df, "p": float(p),
            "logOR_stratum": logOR.tolist(),
            "OR_stratum": np.exp(logOR).tolist(),
            "logOR_pooled": lor_pool,
            "OR_pooled": float(np.exp(lor_pool))}


if __name__ == "__main__":
    print("=== Woolf test of homogeneity of odds ratios ===\n")
    #  Case 1: 3 strata with the SAME true OR
    homog = [(30, 70, 20, 80), (40, 60, 25, 75), (50, 50, 35, 65)]
    r1 = woolf_test(homog)
    print(f"  Homogeneous OR case:")
    print(f"    Stratum ORs = {[round(v, 2) for v in r1['OR_stratum']]}")
    print(f"    Pooled OR   = {r1['OR_pooled']:.3f}")
    print(f"    X_W = {r1['X_W']:.3f}  df = {r1['df']}  p = {r1['p']:.3f}   "
          f"(should NOT reject homogeneity)")

    #  Case 2: 3 strata with very different ORs (interaction!)
    hetero = [(40, 20, 20, 40), (30, 30, 30, 30), (20, 40, 40, 20)]
    r2 = woolf_test(hetero)
    print(f"\n  Heterogeneous OR case (interaction):")
    print(f"    Stratum ORs = {[round(v, 2) for v in r2['OR_stratum']]}")
    print(f"    Pooled OR   = {r2['OR_pooled']:.3f}")
    print(f"    X_W = {r2['X_W']:.3f}  df = {r2['df']}  p = {r2['p']:.3g}   "
          f"(REJECT homogeneity -> don't report pooled OR)")

    print("\n--- library cross-check (metafor / DescTools::WoolfTest R;\n"
          "                          scipy from-scratch Python) ---")
