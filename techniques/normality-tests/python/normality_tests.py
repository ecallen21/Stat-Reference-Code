"""Formal tests for normality (Reference §3.19, §3.40).

Tests included
--------------
- Shapiro-Wilk           : ratio of two estimators of variance based on order
                           statistics. Most powerful general-purpose test; default.
- Jarque-Bera            : (n/6) * (g1^2 + g2^2 / 4); from skewness g1 and excess
                           kurtosis g2. Asymptotic chi^2_2. Bad for small n.
- D'Agostino-Pearson K^2 : transforms g1 and g2 to standard normals (omnibus on
                           skewness AND kurtosis); good for n in [20, 1000+].
- Lilliefors             : Kolmogorov-Smirnov against N(mean(x), var(x)) with
                           estimated parameters -- uses Lilliefors' critical values
                           (the plain K-S is too liberal when you estimate
                           parameters from the data).
- Anderson-Darling       : weighted CDF distance, with extra weight in the tails;
                           especially sensitive to deviations there.

Caveats
-------
With very large n, every test rejects -- *any* real-world data deviates from
"exactly normal." Always pair with a Q-Q plot and a skewness/kurtosis check.
With very small n (< ~20), all of them have low power -- failure to reject is
not evidence of normality. The reference doc's §1.29 covers Q-Q interpretation.

This file's from-scratch versions cover Jarque-Bera and a simple Lilliefors via
the K-S statistic with Monte-Carlo p-values; Shapiro-Wilk, D'Agostino-Pearson,
and Anderson-Darling are deferred to scipy because they need precomputed
constants / coefficient tables.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(x): return sum(x) / len(x)


def _central_moment(x, k):
    m = _mean(x); return sum((v - m) ** k for v in x) / len(x)


def jarque_bera_scratch(x: Sequence[float]) -> dict:
    """JB = (n/6)*(g1^2 + g2^2/4), chi^2_2 under H0. Poor for n < ~100."""
    n = len(x)
    m2 = _central_moment(x, 2)
    g1 = _central_moment(x, 3) / m2 ** 1.5             # skewness
    g2 = _central_moment(x, 4) / m2 ** 2 - 3.0         # excess kurtosis
    jb = (n / 6.0) * (g1 ** 2 + g2 ** 2 / 4.0)
    p = float(stats.chi2.sf(jb, df=2))
    return {"statistic": jb, "df": 2, "p_value": p,
            "skewness": g1, "excess_kurtosis": g2}


def lilliefors_scratch(x: Sequence[float], n_mc: int = 5000, rng=None) -> dict:
    """Lilliefors test: K-S against fitted normal with parameter Monte-Carlo p-value.

    We use a Monte-Carlo p-value (resample many samples of size n from N(0,1),
    standardize, and compute the K-S statistic against the fitted normal) so we
    don't need Lilliefors' tabulated critical values.
    """
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(x, dtype=float)
    n = arr.size
    mu = arr.mean(); sigma = arr.std(ddof=1)
    if sigma == 0:
        return {"statistic": float("nan"), "p_value": float("nan")}
    ks_obs = stats.kstest((arr - mu) / sigma, "norm").statistic

    sims = rng.standard_normal((n_mc, n))
    sims_z = (sims - sims.mean(axis=1, keepdims=True)) / sims.std(axis=1, ddof=1, keepdims=True)
    sim_stats = np.array([stats.kstest(row, "norm").statistic for row in sims_z])
    p = float((sim_stats >= ks_obs).mean())
    return {"statistic": float(ks_obs), "p_value": p,
            "note": f"Monte-Carlo p ({n_mc} replications)"}


def library_versions(x):
    arr = np.asarray(x, dtype=float)
    out = {}
    try:
        sw = stats.shapiro(arr)
        out["Shapiro-Wilk (scipy.stats.shapiro)"] = (float(sw.statistic), float(sw.pvalue))
    except Exception as exc:
        out["Shapiro-Wilk"] = f"error: {exc}"
    jb = stats.jarque_bera(arr)
    out["Jarque-Bera (scipy.stats.jarque_bera)"] = (float(jb.statistic), float(jb.pvalue))
    dp = stats.normaltest(arr)
    out["D'Agostino-Pearson K^2 (scipy.stats.normaltest)"] = (float(dp.statistic), float(dp.pvalue))
    ad = stats.anderson(arr, dist="norm")
    out["Anderson-Darling A^2 (scipy.stats.anderson)"] = (
        float(ad.statistic),
        f"critical values at 15/10/5/2.5/1% = {list(ad.critical_values)}")
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(11)

    print("=== Sample 1: N(0, 1), n = 80 (should NOT be rejected) ===")
    a = rng.normal(0, 1, 80).tolist()
    print("Jarque-Bera (scratch):", jarque_bera_scratch(a))
    print("Lilliefors (scratch) :", lilliefors_scratch(a, n_mc=2000, rng=rng))
    print("\n--- library ---")
    for k, v in library_versions(a).items():
        print(f"  {k}: {v}")

    print("\n=== Sample 2: log-normal(0, 0.7), n = 80 (should be rejected) ===")
    b = rng.lognormal(0, 0.7, 80).tolist()
    print("Jarque-Bera (scratch):", jarque_bera_scratch(b))
    print("Lilliefors (scratch) :", lilliefors_scratch(b, n_mc=2000, rng=rng))
    print("\n--- library ---")
    for k, v in library_versions(b).items():
        print(f"  {k}: {v}")
