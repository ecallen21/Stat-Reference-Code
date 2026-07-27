"""Nonparametric bootstrap: case resampling (Reference §10.1).

Basic idea:
    Given a sample x_1..x_n and a statistic theta_hat = T(x_1..x_n), estimate
    the sampling distribution of theta_hat by resampling the DATA with
    replacement B times, computing T on each resample, and using the
    empirical distribution of the resulting theta*_1..theta*_B.

Confidence intervals covered here:
    - percentile         : take the alpha/2 and 1 - alpha/2 quantiles of theta*
    - basic (pivotal)    : 2 theta_hat - Q_{1-alpha/2}, 2 theta_hat - Q_{alpha/2}
    - normal             : theta_hat +/- z * SE_bootstrap
For BCa (bias- and skewness-adjusted) see techniques/bca-bootstrap.

Works with any statistic that takes a 1-D sample (mean, median, correlation,
regression coefficient computed on a resampled data matrix, ...).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def bootstrap_1d(x: Sequence[float], statistic: Callable,
                 n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap a scalar statistic on a 1-D sample.

    Parameters
    ----------
    x : the sample.
    statistic : callable that takes a 1-D array and returns a scalar (e.g. ``np.median``).
    n_boot : number of bootstrap replicates.
    conf : confidence level for the returned CIs.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        theta_star[b] = float(statistic(x[rng.integers(0, n, size=n)]))
    alpha = 1 - conf
    lo_q, hi_q = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    z = stats.norm.ppf(1 - alpha / 2)
    se = float(theta_star.std(ddof=1))
    return {"theta_hat": theta_hat,
            "bootstrap_SE": se,
            "CI_percentile": {"lower": float(lo_q), "upper": float(hi_q)},
            "CI_basic":      {"lower": float(2 * theta_hat - hi_q),
                               "upper": float(2 * theta_hat - lo_q)},
            "CI_normal":     {"lower": float(theta_hat - z * se),
                               "upper": float(theta_hat + z * se)},
            "n_boot": n_boot, "n": n, "conf": conf,
            "method": "nonparametric bootstrap (case resampling)"}


def bootstrap_2d(data, statistic: Callable, n_boot: int = 2000,
                  conf: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap over ROWS of a 2-D matrix. ``statistic(data)`` -> scalar.

    Common uses: correlation between two columns, regression coefficient on
    a design matrix, any function of a whole n x p data matrix.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(data))
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        theta_star[b] = float(statistic(data[idx]))
    alpha = 1 - conf
    lo_q, hi_q = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    z = stats.norm.ppf(1 - alpha / 2)
    se = float(theta_star.std(ddof=1))
    return {"theta_hat": theta_hat, "bootstrap_SE": se,
            "CI_percentile": {"lower": float(lo_q), "upper": float(hi_q)},
            "CI_basic":      {"lower": float(2 * theta_hat - hi_q),
                               "upper": float(2 * theta_hat - lo_q)},
            "CI_normal":     {"lower": float(theta_hat - z * se),
                               "upper": float(theta_hat + z * se)},
            "n_boot": n_boot, "n": n, "conf": conf,
            "method": "nonparametric bootstrap over rows"}


def library_versions(x):
    from scipy.stats import bootstrap as scipy_boot
    res = scipy_boot((x,), np.median, n_resamples=2000,
                     confidence_level=0.95, method="percentile", random_state=0)
    return {"scipy.stats.bootstrap median percentile CI":
            {"lower": float(res.confidence_interval.low),
             "upper": float(res.confidence_interval.high),
             "SE": float(res.standard_error)}}


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    # Skewed sample: bootstrap the MEDIAN
    x = rng.exponential(2.0, size=100)
    print("=== Bootstrap median (n=100, B=2000) ===")
    out = bootstrap_1d(x, np.median, n_boot=2000)
    print(f"  theta_hat = {out['theta_hat']:.4f}")
    print(f"  SE_boot   = {out['bootstrap_SE']:.4f}")
    print(f"  CI (percentile): [{out['CI_percentile']['lower']:.4f}, {out['CI_percentile']['upper']:.4f}]")
    print(f"  CI (basic):      [{out['CI_basic']['lower']:.4f}, {out['CI_basic']['upper']:.4f}]")
    print(f"  CI (normal):     [{out['CI_normal']['lower']:.4f}, {out['CI_normal']['upper']:.4f}]")

    # Bootstrap a regression slope
    n = 200
    X = rng.normal(0, 1, size=(n, 2))
    y = 1.5 + 0.8 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(0, 0.5, n)
    data = np.column_stack([X, y])
    def slope_x1(data):
        Xd = np.column_stack([np.ones(data.shape[0]), data[:, 0], data[:, 1]])
        yd = data[:, 2]
        beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        return beta[1]                                # slope on x1
    print("\n=== Bootstrap regression slope on x1 (true = 0.80) ===")
    out2 = bootstrap_2d(data, slope_x1, n_boot=2000)
    print(f"  theta_hat = {out2['theta_hat']:.4f}")
    print(f"  SE_boot   = {out2['bootstrap_SE']:.4f}")
    print(f"  CI (percentile): [{out2['CI_percentile']['lower']:.4f}, {out2['CI_percentile']['upper']:.4f}]")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(x).items():
        print(f"  {k}: {v}")
