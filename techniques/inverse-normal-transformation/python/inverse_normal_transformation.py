"""Rank-based inverse normal transformation (Reference Sec 41.3).

Rank observations then apply the inverse normal CDF -- forces EXACT
normality by construction.  Widely used in GWAS phenotype
preprocessing and biomarker analysis.

Blom (1958)          : Phi^-1((r - 3/8) / (n + 1/4))
Tukey (1962)         : Phi^-1((r - 1/3) / (n + 1/3))
Van der Waerden      : Phi^-1(r / (n + 1))
Rankit (Bliss)       : Phi^-1((r - 1/2) / n)

r = midrank of observation (ties broken by average).  Preserves the
ranking; discards the metric information beyond the ordering.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def int_transform(y, method="blom"):
    r = stats.rankdata(y, method="average")
    n = len(y)
    if method == "blom":
        q = (r - 3 / 8) / (n + 1 / 4)
    elif method == "tukey":
        q = (r - 1 / 3) / (n + 1 / 3)
    elif method == "waerden":
        q = r / (n + 1)
    elif method == "rankit":
        q = (r - 0.5) / n
    else:
        raise ValueError(method)
    return stats.norm.ppf(q)


if __name__ == "__main__":
    print("=== Rank-based inverse normal transformation (INT) ===\n")
    rng = np.random.default_rng(0)
    n = 500
    y = rng.exponential(scale=2.0, size=n)     # heavy right skew

    print(f"  Raw          skew = {stats.skew(y):+.3f}   kurt = {stats.kurtosis(y):+.3f}")
    for method in ("blom", "tukey", "waerden", "rankit"):
        yt = int_transform(y, method=method)
        _, p = stats.shapiro(yt)
        print(f"  INT ({method:>8s})  skew = {stats.skew(yt):+.3f}   kurt = {stats.kurtosis(yt):+.3f}"
              f"   Shapiro p = {p:.3f}")

    print("\n  INT forces exact normality by construction; suitable for downstream methods")
    print("  that ASSUME normality (linear regression residuals, GWAS quantitative traits).\n")
    print("--- library cross-check (R RNOmni/bestNormalize; Python custom + scipy) ---")
