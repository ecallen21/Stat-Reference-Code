"""Monte Carlo simulation for power and CI coverage (Reference §10.9).

Two canonical uses of Monte Carlo in applied statistics:

1. POWER ANALYSIS
   Under a specified ALTERNATIVE hypothesis (a chosen effect size, sample
   size, and analysis method), simulate many datasets, run the test on each,
   and compute the fraction of times you correctly reject H0.
   power_hat = mean(p_value < alpha)   with SE = sqrt(p (1 - p) / n_sim).

2. CI COVERAGE
   Under a specified data-generating process (with known true parameter),
   simulate many datasets, compute the chosen CI on each, and count how often
   the true parameter falls inside. Should equal 1 - alpha for a well-calibrated
   procedure. Coverage below nominal is a warning sign.

Both are just "simulate N datasets, run the analysis, aggregate the answer."
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint for functions (like Callable[[int], float])

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def power_simulation(sample_fn: Callable, test_fn: Callable, n_sim: int = 2000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """Simulate ``n_sim`` datasets under an ALTERNATIVE, run ``test_fn``, and
    report empirical power.

    sample_fn(rng) -> a dataset object.
    test_fn(dataset) -> p-value in [0, 1].
    """
    rng = np.random.default_rng(seed)
    reject = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        ds = sample_fn(rng)
        p = float(test_fn(ds))
        reject[i] = p < alpha
    power_hat = float(reject.mean())
    se = math.sqrt(power_hat * (1 - power_hat) / n_sim)
    return {"power_hat": power_hat,
            "MC_SE": se,
            "CI95_wilson": _wilson_ci(reject.sum(), n_sim, 0.95),
            "n_sim": n_sim, "alpha": alpha,
            "method": "Monte Carlo power simulation"}


def coverage_simulation(sample_fn: Callable, ci_fn: Callable, true_param: float,
                         n_sim: int = 2000, seed: int = 0) -> dict:
    """Simulate ``n_sim`` datasets from a known-parameter DGP, compute CIs, and
    report empirical coverage.

    sample_fn(rng) -> dataset.
    ci_fn(dataset) -> (lower, upper) tuple.
    """
    rng = np.random.default_rng(seed)
    covers = np.zeros(n_sim, dtype=bool)
    widths = np.empty(n_sim)
    for i in range(n_sim):
        ds = sample_fn(rng)
        lo, hi = ci_fn(ds)
        covers[i] = (lo <= true_param <= hi)
        widths[i] = hi - lo
    cov_hat = float(covers.mean())
    return {"coverage_hat": cov_hat,
            "MC_SE": math.sqrt(cov_hat * (1 - cov_hat) / n_sim),
            "CI95_wilson": _wilson_ci(covers.sum(), n_sim, 0.95),
            "mean_CI_width": float(widths.mean()),
            "n_sim": n_sim, "true_param": true_param,
            "method": "Monte Carlo CI coverage simulation"}


def _wilson_ci(x, n, conf):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = x / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return {"lower": max(0.0, center - half), "upper": min(1.0, center + half)}


if __name__ == "__main__":
    # --- Power: two-sample t-test, n=30/group, true delta = 0.5, sigma = 1 ---
    def sample_two_sample(rng):
        x1 = rng.normal(0, 1, 30)
        x2 = rng.normal(0.5, 1, 30)
        return (x1, x2)
    def test_two_sample(ds):
        x1, x2 = ds
        return float(stats.ttest_ind(x1, x2, equal_var=False).pvalue)

    print("=== Power of two-sample t-test at delta=0.5, n=30/group (alpha=0.05) ===")
    pw = power_simulation(sample_two_sample, test_two_sample, n_sim=2000)
    print(f"  empirical power = {pw['power_hat']:.4f}  (MC SE = {pw['MC_SE']:.4f})")
    print(f"  Wilson 95% CI: {pw['CI95_wilson']}")
    # analytical for reference
    from scipy.stats import nct
    from math import sqrt
    nc = 0.5 / (1 * sqrt(2 / 30))
    tcrit = stats.t.ppf(0.975, 58)
    theo = float(1 - nct.cdf(tcrit, 58, nc) + nct.cdf(-tcrit, 58, nc))
    print(f"  theoretical power (Welch approx): {theo:.4f}")

    # --- Coverage: normal-based 95% CI for the mean, small n = 10 ---
    def sample_small_normal(rng):
        return rng.normal(5.0, 2.0, size=10)
    def ci_mean(ds):
        m = ds.mean(); s = ds.std(ddof=1); n = ds.size
        h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
        return (m - h, m + h)

    print("\n=== Coverage of Student's t 95% CI for the mean (n=10) ===")
    cv = coverage_simulation(sample_small_normal, ci_mean, true_param=5.0, n_sim=2000)
    print(f"  empirical coverage = {cv['coverage_hat']:.4f}  (target = 0.95)")
    print(f"  mean CI width = {cv['mean_CI_width']:.4f}")
    print(f"  Wilson 95% CI on coverage: {cv['CI95_wilson']}")
