"""Levene / Brown-Forsythe variance-homogeneity tests (Sec 47.61).

Levene 1960 'Robust tests for equality of variances' (in Olkin et
al., eds., 'Contributions to probability and statistics'); Brown &
Forsythe 1974 'Robust tests for the equality of variances', JASA.

Both apply a one-way ANOVA F-test to the absolute deviations
Z_{ij} = |Y_{ij} - center_j| where center_j is:

    * mean of group j          -> Levene
    * median of group j        -> Brown-Forsythe (more robust; recommended)
    * trimmed mean of group j  -> Brown-Forsythe variant

More robust to non-normality than the classical F- or Bartlett tests.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # F-distribution CDF


def levene_bf(*groups, center="median"):
    """Levene / Brown-Forsythe test. center in {mean, median, trimmed}."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    if center == "mean":
        c = [np.mean(g) for g in groups]
    elif center == "median":
        c = [np.median(g) for g in groups]
    elif center == "trimmed":
        c = [stats.trim_mean(g, 0.1) for g in groups]
    else:
        raise ValueError("center must be mean/median/trimmed")

    Z = [np.abs(g - ci) for g, ci in zip(groups, c)]
    Zbar = [np.mean(z) for z in Z]
    Zbar_all = np.concatenate(Z).mean()
    num = sum(len(z) * (zb - Zbar_all) ** 2 for z, zb in zip(Z, Zbar)) / (k - 1)
    den = sum(((z - zb) ** 2).sum() for z, zb in zip(Z, Zbar)) / (N - k)
    W = num / den
    p = float(1 - stats.f.cdf(W, k - 1, N - k))
    return {"W": float(W), "p": p, "df": (k - 1, N - k), "center": center}


if __name__ == "__main__":
    print("=== Levene / Brown-Forsythe variance-homogeneity (1960, 1974) ===\n")
    rng = np.random.default_rng(0)

    # Case 1: equal variances (should NOT reject)
    g1 = rng.normal(0, 1.0, 50)
    g2 = rng.normal(0, 1.0, 50)
    g3 = rng.normal(0, 1.0, 50)
    for center in ["mean", "median", "trimmed"]:
        r = levene_bf(g1, g2, g3, center=center)
        print(f"  equal var    ({center:8s}): W = {r['W']:.3f}   p = {r['p']:.4f}")

    print()
    # Case 2: unequal variances (should reject)
    g1 = rng.normal(0, 1.0, 50)
    g2 = rng.normal(0, 2.0, 50)
    g3 = rng.normal(0, 3.5, 50)
    for center in ["mean", "median", "trimmed"]:
        r = levene_bf(g1, g2, g3, center=center)
        print(f"  unequal var  ({center:8s}): W = {r['W']:.3f}   p = {r['p']:.6f}")

    print()
    # Case 3: heavy-tailed t3 (Levene inflated, Brown-Forsythe robust)
    g1 = rng.standard_t(3, 50) * 1.0
    g2 = rng.standard_t(3, 50) * 1.0
    g3 = rng.standard_t(3, 50) * 1.0
    for center in ["mean", "median"]:
        r = levene_bf(g1, g2, g3, center=center)
        print(f"  t3 same var  ({center:8s}): W = {r['W']:.3f}   p = {r['p']:.4f}")

    print("\n--- library cross-check (car::leveneTest R; scipy.stats.levene Python) ---")
