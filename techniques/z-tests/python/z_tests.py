"""z-tests for means and proportions (Reference §3.7, §3.22).

The z-test is the large-sample / known-variance cousin of the t-test:

  z = (estimate - null) / SE_under_null

with z compared to the standard normal. Variants below:

  - one_sample_mean_z   : known sigma; small-n possible if sigma really is known
  - two_sample_mean_z   : independent samples, both sigmas known
  - one_proportion_z    : H0: p == p0; SE uses p0 (Wald uses p_hat -- noted)
  - two_proportion_z    : pooled-proportion SE under H0: p1 == p2
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _mean(x):
    return sum(x) / len(x)


def _z_pvalue(z: float, alternative: str) -> float:
    if alternative == "two-sided":
        return 2 * stats.norm.sf(abs(z))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    if alternative == "less":
        return float(stats.norm.cdf(z))
    raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")


def _z_ci(estimate: float, se: float, conf: float):
    z = stats.norm.ppf(0.5 + conf / 2)
    return estimate - z * se, estimate + z * se


def one_sample_mean_z(x: Sequence[float], mu0: float, sigma: float,
                      alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """One-sample z-test for the mean, sigma known.

    ``sigma`` is the *population* SD (assumed known, hence z and not t).
    """
    n = len(x)
    m = _mean(x)
    se = sigma / math.sqrt(n)
    z = (m - mu0) / se
    lo, hi = _z_ci(m, se, conf)
    return {"mean": m, "se": se, "z": z, "p_value": _z_pvalue(z, alternative),
            "ci_lower": lo, "ci_upper": hi}


def two_sample_mean_z(x1: Sequence[float], x2: Sequence[float],
                      sigma1: float, sigma2: float,
                      alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Two-sample z-test for the difference in means, both sigmas known."""
    n1, n2 = len(x1), len(x2)
    diff = _mean(x1) - _mean(x2)
    se = math.sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)
    z = diff / se
    lo, hi = _z_ci(diff, se, conf)
    return {"mean_diff": diff, "se": se, "z": z,
            "p_value": _z_pvalue(z, alternative), "ci_lower": lo, "ci_upper": hi}


def one_proportion_z(x: int, n: int, p0: float, alternative: str = "two-sided",
                     conf: float = 0.95, continuity: bool = False) -> dict:
    """One-proportion z-test: H0: p == p0.

    SE under H0 uses ``p0``: ``sqrt(p0(1-p0)/n)``. (Wald uses ``p_hat`` in the SE,
    which differs from the score test below -- this implementation uses the score
    form, which is what ``prop.test`` / ``scipy.stats.binomtest`` align with.)

    Parameters
    ----------
    x : count of successes.
    n : total trials.
    p0 : null-hypothesis proportion.
    continuity : if True, apply Yates' continuity correction (subtract 1/(2n) from
        |p_hat - p0|, floored at zero) -- recommended for small n.
    """
    p_hat = x / n
    se0 = math.sqrt(p0 * (1 - p0) / n)
    diff = abs(p_hat - p0)
    if continuity:
        diff = max(0.0, diff - 1.0 / (2 * n))
    z = math.copysign(diff, p_hat - p0) / se0 if se0 > 0 else 0.0
    lo, hi = _z_ci(p_hat, math.sqrt(p_hat * (1 - p_hat) / n), conf)  # Wald CI
    return {"p_hat": p_hat, "se_null": se0, "z": z,
            "p_value": _z_pvalue(z, alternative),
            "ci_lower_wald": max(0.0, lo), "ci_upper_wald": min(1.0, hi)}


def two_proportion_z(x1: int, n1: int, x2: int, n2: int,
                     alternative: str = "two-sided", conf: float = 0.95,
                     continuity: bool = False) -> dict:
    """Two-proportion z-test: H0: p1 == p2, pooled-proportion SE.

    Pooled-p form (what scipy's ``proportions_ztest`` and R's ``prop.test`` use under
    the null): ``p_pool = (x1 + x2) / (n1 + n2)``;
    ``SE = sqrt(p_pool (1 - p_pool) (1/n1 + 1/n2))``.
    """
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    diff = p1 - p2
    if continuity:
        cc = 0.5 * (1 / n1 + 1 / n2)
        adj_diff = math.copysign(max(0.0, abs(diff) - cc), diff)
    else:
        adj_diff = diff
    z = adj_diff / se if se > 0 else 0.0
    # CI on the difference uses the unpooled (Wald) SE -- standard practice
    se_unpool = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = _z_ci(diff, se_unpool, conf)
    return {"p1": p1, "p2": p2, "diff": diff, "p_pool": p_pool, "se_pooled": se,
            "z": z, "p_value": _z_pvalue(z, alternative),
            "ci_lower": lo, "ci_upper": hi}


def library_versions(x1, n1, x2, n2):
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint

    z_stat, p_val = proportions_ztest([x1, x2], [n1, n2])
    return {
        "two-prop z (statsmodels)": (float(z_stat), float(p_val)),
        "p1 95% Wilson CI": proportion_confint(x1, n1, method="wilson"),
        "p2 95% Wilson CI": proportion_confint(x2, n2, method="wilson"),
    }


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(1)
    x = rng.normal(101, 15, 50).tolist()

    print("=== one-sample mean z  (mu0 = 100, sigma = 15) ===")
    for k, v in one_sample_mean_z(x, mu0=100, sigma=15).items():
        print(f"  {k:14s}: {v}")

    a = rng.normal(105, 12, 60).tolist()
    b = rng.normal(100, 18, 55).tolist()
    print("\n=== two-sample mean z  (sigma1=12, sigma2=18) ===")
    for k, v in two_sample_mean_z(a, b, sigma1=12, sigma2=18).items():
        print(f"  {k:14s}: {v}")

    print("\n=== one-proportion z  (x=42, n=100, p0=0.5) ===")
    for k, v in one_proportion_z(42, 100, p0=0.5).items():
        print(f"  {k:14s}: {v}")

    print("\n=== two-proportion z  (42/100 vs 30/100) ===")
    for k, v in two_proportion_z(42, 100, 30, 100).items():
        print(f"  {k:14s}: {v}")

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(42, 100, 30, 100).items():
        print(f"  {k:24s}: {v}")
