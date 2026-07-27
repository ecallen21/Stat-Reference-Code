"""BCa bootstrap CI + comparison of bootstrap CI methods (Reference §10.3, §10.14).

BCa = Bias-Corrected and accelerated. Two adjustments to the percentile CI:
    z0 (bias correction): the fraction of bootstrap replicates below theta_hat,
        mapped to a z-value.  z0 = Phi^{-1}(p),  p = mean(theta* < theta_hat)
    a  (acceleration)   : measures skewness of the sampling distribution;
        estimated from JACKKNIFE deviations of theta_hat.
        a = sum (mean(J) - J_i)^3  /  (6 * (sum (mean(J) - J_i)^2)^(3/2))
        where J_i = theta_hat on the sample with obs i removed.

Adjusted percentiles:
    alpha1 = Phi( z0 + (z0 + z_{alpha/2})   / (1 - a (z0 + z_{alpha/2})) )
    alpha2 = Phi( z0 + (z0 + z_{1-alpha/2}) / (1 - a (z0 + z_{1-alpha/2})) )

Then CI = [ Q_{alpha1}(theta*), Q_{alpha2}(theta*) ].

BCa is second-order accurate (better small-sample coverage than percentile)
and transformation-respecting -- usually the recommended default.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _bootstrap_replicates_1d(x, statistic, n_boot, rng):
    n = x.size
    return np.array([statistic(x[rng.integers(0, n, size=n)]) for _ in range(n_boot)])


def _jackknife_replicates_1d(x, statistic):
    n = x.size
    idx = np.arange(n)
    return np.array([statistic(x[idx != i]) for i in range(n)])


def bca_ci_1d(x, statistic, n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> dict:
    """BCa confidence interval for a scalar statistic on a 1-D sample."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_star = _bootstrap_replicates_1d(x, statistic, n_boot, rng)
    # z0: bias correction
    p = float(np.mean(theta_star < theta_hat))
    p = min(max(p, 1e-12), 1 - 1e-12)         # avoid infinity in Phi^-1
    z0 = stats.norm.ppf(p)
    # a: acceleration from jackknife
    J = _jackknife_replicates_1d(x, statistic)
    Jbar = J.mean()
    num = ((Jbar - J) ** 3).sum()
    den = 6.0 * ((Jbar - J) ** 2).sum() ** 1.5
    a_hat = float(num / den) if den > 0 else 0.0
    # Adjusted percentiles
    alpha = 1 - conf
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    def adjusted(z):
        val = z0 + (z0 + z) / (1 - a_hat * (z0 + z))
        return float(stats.norm.cdf(val))
    a1 = adjusted(z_lo); a2 = adjusted(z_hi)
    lo, hi = np.quantile(theta_star, [a1, a2])
    return {"theta_hat": theta_hat,
            "z0_bias_correction": float(z0),
            "a_acceleration": a_hat,
            "adjusted_percentiles": {"lower": a1, "upper": a2},
            "CI_BCa": {"lower": float(lo), "upper": float(hi)},
            "n_boot": n_boot, "conf": conf,
            "method": "BCa bootstrap CI (Efron 1987)"}


def compare_ci_methods(x, statistic, n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> dict:
    """Compare percentile / basic / normal / BCa CIs on the SAME bootstrap replicates
    (uses the same random seed, so replicates are shared for a fair comparison)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_star = _bootstrap_replicates_1d(x, statistic, n_boot, rng)
    alpha = 1 - conf
    q_lo, q_hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    z = stats.norm.ppf(1 - alpha / 2)
    se = float(theta_star.std(ddof=1))

    # BCa
    p = float(np.mean(theta_star < theta_hat))
    p = min(max(p, 1e-12), 1 - 1e-12)
    z0 = stats.norm.ppf(p)
    J = _jackknife_replicates_1d(x, statistic)
    Jbar = J.mean()
    num = ((Jbar - J) ** 3).sum()
    den = 6.0 * ((Jbar - J) ** 2).sum() ** 1.5
    a_hat = float(num / den) if den > 0 else 0.0
    def adjusted(zq):
        val = z0 + (z0 + zq) / (1 - a_hat * (z0 + zq))
        return float(stats.norm.cdf(val))
    a1 = adjusted(stats.norm.ppf(alpha / 2))
    a2 = adjusted(stats.norm.ppf(1 - alpha / 2))
    bca_lo, bca_hi = np.quantile(theta_star, [a1, a2])

    def width(pair): return pair["upper"] - pair["lower"]
    percentile = {"lower": float(q_lo), "upper": float(q_hi)}
    basic      = {"lower": float(2 * theta_hat - q_hi), "upper": float(2 * theta_hat - q_lo)}
    normal     = {"lower": float(theta_hat - z * se),   "upper": float(theta_hat + z * se)}
    bca        = {"lower": float(bca_lo), "upper": float(bca_hi)}

    return {"theta_hat": theta_hat, "SE_bootstrap": se,
            "z0_bias_correction": float(z0), "a_acceleration": a_hat,
            "CI_percentile": percentile, "width_percentile": width(percentile),
            "CI_basic":      basic,      "width_basic":      width(basic),
            "CI_normal":     normal,     "width_normal":     width(normal),
            "CI_BCa":        bca,        "width_BCa":        width(bca),
            "n_boot": n_boot, "conf": conf,
            "method": "comparison of bootstrap CI methods (shared replicates)"}


def library_versions(x):
    from scipy.stats import bootstrap as scipy_boot
    out = {}
    for method in ("percentile", "basic", "BCa"):
        res = scipy_boot((x,), np.median, n_resamples=2000,
                         confidence_level=0.95, method=method, random_state=0)
        out[f"scipy {method}"] = {"lower": float(res.confidence_interval.low),
                                   "upper": float(res.confidence_interval.high)}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    x = rng.exponential(2.0, size=80)         # skewed sample

    print("=== BCa CI for the median ===")
    out = bca_ci_1d(x, np.median, n_boot=2000)
    for k, v in out.items():
        print(f"  {k:22s}: {v}")

    print("\n=== Compare all four CI methods (median, same B=2000 replicates) ===")
    cmp = compare_ci_methods(x, np.median, n_boot=2000)
    for method in ("percentile", "basic", "normal", "BCa"):
        pair = cmp[f"CI_{method}"]; w = cmp[f"width_{method}"]
        print(f"  {method:10s}: [{pair['lower']:.4f}, {pair['upper']:.4f}]  width={w:.4f}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(x).items():
        print(f"  {k}: {v}")
