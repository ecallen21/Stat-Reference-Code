"""Double (iterated) bootstrap for CI calibration (Reference §10.11).

The plain (single) bootstrap gives a CI whose coverage may differ from the
nominal level in finite samples. The DOUBLE bootstrap uses a second-level
bootstrap to CALIBRATE the level.

Algorithm (calibrated percentile CI):
    1. Sample B1 bootstrap replicates x*_1..x*_{B1} from x.
    2. For each x*_b:
        a. Compute theta*_b.
        b. Draw B2 SECOND-level bootstrap resamples x**_1..x**_{B2} from x*_b.
        c. Compute the fraction of theta**_c that fall inside the plain
           percentile CI [Q_{alpha/2}(theta**), Q_{1-alpha/2}(theta**)] built
           from x*_b.
    3. Adjust the nominal level so the average of those inner-coverage
       estimates equals 1 - alpha.

We implement a simpler "calibrate the percentile CI level" version:
    - Build the plain-percentile CI at nominal level alpha_0.
    - For each first-level b, ask: does the "true" statistic (proxied by the
      full-sample theta_hat) fall inside the CI built from x*_b at level alpha?
      That coverage should be 1 - alpha; if not, adjust alpha.

Full second-level bootstrap is O(B1 * B2) so use small B1, B2 for demonstration.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint for functions

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def double_bootstrap_calibrated_percentile(
    x, statistic: Callable, n_boot_outer: int = 200, n_boot_inner: int = 200,
    conf: float = 0.95, seed: int = 0) -> dict:
    """Double bootstrap CI with a coverage-calibration adjustment.

    Uses each outer replicate as a proxy for the "true" statistic to estimate
    what nominal level actually delivers ``conf`` coverage.
    """
    x = np.asarray(x, dtype=float); n = x.size
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    # First-level bootstrap
    outer_theta = np.empty(n_boot_outer)
    covered = np.zeros(n_boot_outer)
    inner_los = np.empty(n_boot_outer)
    inner_his = np.empty(n_boot_outer)
    for b in range(n_boot_outer):
        x_star = x[rng.integers(0, n, size=n)]
        outer_theta[b] = float(statistic(x_star))
        # inner bootstrap FROM x_star to build a nominal-level CI on theta_star
        inner_theta = np.empty(n_boot_inner)
        for c in range(n_boot_inner):
            xs = x_star[rng.integers(0, n, size=n)]
            inner_theta[c] = float(statistic(xs))
        alpha0 = 1 - conf
        lo, hi = np.quantile(inner_theta, [alpha0 / 2, 1 - alpha0 / 2])
        inner_los[b] = lo; inner_his[b] = hi
        # does the plain CI capture the ORIGINAL sample's estimate?
        covered[b] = int(lo <= theta_hat <= hi)
    empirical_coverage = float(covered.mean())
    # One-step Beran (1987) calibration: if the plain CI at nominal alpha_0
    # actually achieves alpha_emp = 1 - empirical_coverage, use
    #     alpha_calibrated = alpha_0^2 / alpha_emp
    # which is < alpha_0 (wider CI) when alpha_emp > alpha_0 (undercoverage).
    alpha0 = 1 - conf
    alpha_emp = max(1e-6, 1 - empirical_coverage)
    alpha_calibrated = max(1e-6, min(0.5, alpha0 * alpha0 / alpha_emp))

    # Plain outer-level percentile CI
    plain_lo, plain_hi = np.quantile(outer_theta, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    # Calibrated outer-level percentile CI
    cal_lo, cal_hi = np.quantile(outer_theta, [alpha_calibrated / 2, 1 - alpha_calibrated / 2])
    return {"theta_hat": theta_hat,
            "empirical_inner_coverage": empirical_coverage,
            "nominal_level": conf,
            "CI_plain_percentile": {"lower": float(plain_lo), "upper": float(plain_hi)},
            "CI_calibrated":        {"lower": float(cal_lo),   "upper": float(cal_hi)},
            "alpha_calibrated": float(alpha_calibrated),
            "n_boot_outer": n_boot_outer, "n_boot_inner": n_boot_inner,
            "method": "double bootstrap with coverage-calibration adjustment"}


if __name__ == "__main__":
    rng = np.random.default_rng(53)
    x = rng.exponential(2.0, size=50)
    print("=== Double bootstrap: calibrated percentile CI for the MEDIAN ===")
    out = double_bootstrap_calibrated_percentile(x, np.median,
                                                    n_boot_outer=200, n_boot_inner=200)
    for k, v in out.items():
        print(f"  {k:26s}: {v}")
