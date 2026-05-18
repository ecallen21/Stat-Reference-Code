"""Tests for homogeneity of variance across k groups (Reference §3.20, §3.55).

These check the ANOVA / Student's-t assumption that all groups have equal
variances. Three classical tests, in decreasing order of sensitivity to
non-normality:

  - Levene's test      : one-way ANOVA on |x_ij - mean(x_i)|   (center = mean)
  - Brown-Forsythe     : one-way ANOVA on |x_ij - median(x_i)| (center = median)
                         More robust to non-normality than classic Levene.
  - Bartlett's test    : likelihood-ratio against equal variances. Most powerful
                         when groups ARE normal; very sensitive to non-normality
                         otherwise -- don't use unless you've checked normality.
  - F-test of variances: for the two-sample case (s1^2 / s2^2 ~ F(n1-1, n2-1)).
                         Same non-normality caveat as Bartlett.

If any test is significant -> use Welch's ANOVA / Welch's t / Games-Howell
post-hoc instead of the equal-variance versions.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(g): return sum(g) / len(g)


def _median(g):
    s = sorted(g); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _levene_brown_forsythe(groups, center="median"):
    """Levene's test with configurable center.

    center = "mean"   -> Levene's original test
    center = "median" -> Brown-Forsythe (more robust)
    """
    k = len(groups)
    ns = [len(g) for g in groups]
    centers = [(_mean(g) if center == "mean" else _median(g)) for g in groups]
    # within-group absolute deviations
    z = [[abs(v - c) for v in g] for g, c in zip(groups, centers)]
    N = sum(ns)
    grand_z = sum(sum(zi) for zi in z) / N
    z_means = [sum(zi) / n for zi, n in zip(z, ns)]
    ss_b = sum(n * (zm - grand_z) ** 2 for n, zm in zip(ns, z_means))
    ss_w = sum((v - zm) ** 2 for zi, zm in zip(z, z_means) for v in zi)
    df1, df2 = k - 1, N - k
    if ss_w == 0:
        return {"statistic": float("inf"), "df1": df1, "df2": df2, "p_value": 0.0,
                "center": center}
    F = (ss_b / df1) / (ss_w / df2)
    p = float(stats.f.sf(F, df1, df2))
    return {"statistic": F, "df1": df1, "df2": df2, "p_value": p, "center": center}


def levene(groups: Sequence[Sequence[float]]) -> dict:
    """Levene's test (center = mean)."""
    return _levene_brown_forsythe(groups, center="mean")


def brown_forsythe(groups: Sequence[Sequence[float]]) -> dict:
    """Brown-Forsythe variance test (Levene with center = median). Robust default."""
    return _levene_brown_forsythe(groups, center="median")


def bartlett(groups: Sequence[Sequence[float]]) -> dict:
    """Bartlett's test (likelihood-ratio). High power IF groups are normal."""
    k = len(groups)
    ns = [len(g) for g in groups]
    vars_ = [sum((v - _mean(g)) ** 2 for v in g) / (n - 1) for g, n in zip(groups, ns)]
    N = sum(ns)
    sp2 = sum((n - 1) * v for n, v in zip(ns, vars_)) / (N - k)
    num = (N - k) * math.log(sp2) - sum((n - 1) * math.log(v) for n, v in zip(ns, vars_))
    denom = 1 + (1 / (3 * (k - 1))) * (
        sum(1 / (n - 1) for n in ns) - 1 / (N - k)
    )
    chi2 = num / denom
    p = float(stats.chi2.sf(chi2, k - 1))
    return {"statistic": chi2, "df": k - 1, "p_value": p}


def f_test_two_variances(x1: Sequence[float], x2: Sequence[float],
                         alternative: str = "two-sided") -> dict:
    """F-test of H0: var(x1) == var(x2). Tiny power loss vs. Bartlett for k=2."""
    n1, n2 = len(x1), len(x2)
    v1 = sum((v - _mean(x1)) ** 2 for v in x1) / (n1 - 1)
    v2 = sum((v - _mean(x2)) ** 2 for v in x2) / (n2 - 1)
    F = v1 / v2
    df1, df2 = n1 - 1, n2 - 1
    if alternative == "two-sided":
        p = 2 * min(stats.f.cdf(F, df1, df2), stats.f.sf(F, df1, df2))
    elif alternative == "greater":
        p = float(stats.f.sf(F, df1, df2))
    elif alternative == "less":
        p = float(stats.f.cdf(F, df1, df2))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {"F": F, "df1": df1, "df2": df2, "p_value": min(1.0, p), "var1": v1, "var2": v2}


def library_versions(groups):
    return {
        "scipy.stats.levene (median)": stats.levene(*groups, center="median"),
        "scipy.stats.levene (mean)":   stats.levene(*groups, center="mean"),
        "scipy.stats.bartlett":         stats.bartlett(*groups),
    }


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(4)
    # group C has noticeably larger variance
    a = rng.normal(50, 10, 30).tolist()
    b = rng.normal(50, 11, 28).tolist()
    c = rng.normal(50, 22, 32).tolist()
    groups = [a, b, c]

    print("=== Levene's test (center=mean) ===")
    print(levene(groups))
    print("\n=== Brown-Forsythe (center=median) ===")
    print(brown_forsythe(groups))
    print("\n=== Bartlett's test ===")
    print(bartlett(groups))
    print("\n=== F-test (two-sample): a vs c ===")
    print(f_test_two_variances(a, c))

    print("\n--- library (scipy) ---")
    for k, v in library_versions(groups).items():
        print(f"  {k}: {v}")
