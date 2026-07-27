"""Subsampling and m-out-of-n bootstrap (Reference §10.10, §10.15).

Subsampling (Politis-Romano-Wolf 1999)
--------------------------------------
Draw sub-samples of size m << n WITHOUT REPLACEMENT. Under mild conditions,
subsampling produces asymptotically valid CIs even when the bootstrap
distribution is INCONSISTENT (e.g. extreme-value statistics like max, or
statistics at boundaries of the parameter space where the bootstrap breaks
down).

m-out-of-n bootstrap
--------------------
Same idea but WITH replacement -- draws of size m from the original n. Choosing
m << n restores consistency in some pathological cases (e.g. extreme-value
statistics, unit-root time series). Bickel-Goetze-van Zwet (1997) showed
m-out-of-n bootstrap can consistently estimate the sampling distribution when
the ordinary n-out-of-n bootstrap fails.

Both use the same driver: computed statistic distribution on many size-m
subsamples/resamples, scaled appropriately if needed for the studentized
version.

For subsampling with rate scaling factor:
    Under the null hypothesis, T_n = a_n (theta_hat_n - theta_0) ->_d L
    Then subsample statistics T_{n,m}^{(b)} = a_m (theta_hat_m^{(b)} - theta_hat_n)
    approximate L asymptotically; use their quantiles / SE for inference.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint for functions

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def subsampling_1d(x, statistic: Callable, m: int, n_sub: int = 2000,
                    conf: float = 0.95, rate_pow: float = 0.5, seed: int = 0) -> dict:
    """Subsampling CI (Politis-Romano-Wolf) for a scalar statistic.

    Parameters
    ----------
    m : subsample size (must be < n).
    rate_pow : convergence rate exponent (default sqrt-n rate, rate_pow=0.5).
    """
    x = np.asarray(x, dtype=float); n = x.size
    if m >= n:
        raise ValueError("m must be strictly less than n")
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_sub = np.empty(n_sub)
    for b in range(n_sub):
        idx = rng.choice(n, size=m, replace=False)
        theta_sub[b] = float(statistic(x[idx]))
    # centered & rate-scaled subsample statistics
    a_m = m ** rate_pow; a_n = n ** rate_pow
    T_sub = a_m * (theta_sub - theta_hat)              # ~ null-limit distribution
    alpha = 1 - conf
    lo_T, hi_T = np.quantile(T_sub, [alpha / 2, 1 - alpha / 2])
    # Invert to a CI on theta:  theta_hat +/- T / a_n
    return {"theta_hat": theta_hat,
            "subsample_SE_at_m": float(theta_sub.std(ddof=1)),
            "CI_subsampling": {"lower": float(theta_hat - hi_T / a_n),
                                "upper": float(theta_hat - lo_T / a_n)},
            "m": m, "n": n, "n_sub": n_sub, "conf": conf, "rate_pow": rate_pow,
            "method": "Politis-Romano-Wolf subsampling CI (without replacement)"}


def m_out_of_n_bootstrap(x, statistic: Callable, m: int, n_boot: int = 2000,
                          conf: float = 0.95, seed: int = 0) -> dict:
    """m-out-of-n bootstrap (WITH replacement) for a scalar statistic."""
    x = np.asarray(x, dtype=float); n = x.size
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=m)
        theta_star[b] = float(statistic(x[idx]))
    alpha = 1 - conf
    lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    return {"theta_hat": theta_hat,
            "bootstrap_SE_at_m": float(theta_star.std(ddof=1)),
            "CI_percentile": {"lower": float(lo), "upper": float(hi)},
            "m": m, "n": n, "n_boot": n_boot, "conf": conf,
            "method": "m-out-of-n bootstrap (with replacement)"}


if __name__ == "__main__":
    rng = np.random.default_rng(43)
    # Bootstrap of the MAX is inconsistent for the classic n-out-of-n bootstrap
    # (it converges to a mixture instead of the true extreme-value law).
    # Subsampling gives a consistent CI.
    x = rng.exponential(1.0, size=200)
    print("=== Subsampling CI for the sample MAX (a case where ordinary bootstrap fails) ===")
    out = subsampling_1d(x, np.max, m=50, n_sub=2000)
    print(f"  theta_hat = {out['theta_hat']:.4f}")
    print(f"  CI (subsampling): [{out['CI_subsampling']['lower']:.4f}, {out['CI_subsampling']['upper']:.4f}]")

    print("\n=== m-out-of-n bootstrap of the MAX ===")
    out2 = m_out_of_n_bootstrap(x, np.max, m=50, n_boot=2000)
    print(f"  CI: [{out2['CI_percentile']['lower']:.4f}, {out2['CI_percentile']['upper']:.4f}]")

    print("\n=== Naive n-out-of-n bootstrap of MAX (would be inconsistent) ===")
    B = 2000
    theta_nn = np.array([float(np.max(x[rng.integers(0, x.size, size=x.size)])) for _ in range(B)])
    lo, hi = np.quantile(theta_nn, [0.025, 0.975])
    print(f"  CI: [{lo:.4f}, {hi:.4f}]")
    print(f"  (Note the CI touches the observed max; that's the inconsistency signature.)")
