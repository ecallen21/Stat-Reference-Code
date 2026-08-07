"""Studentized (bootstrap-t) confidence intervals (Reference §10.4).

Refinement of the percentile bootstrap that uses a PIVOTAL quantity
    t^* = (theta_hat^* - theta_hat) / se(theta_hat^*)
and takes the alpha/2 and 1-alpha/2 quantiles of t^* to form the CI:
    theta_hat +/- t_quant * se(theta_hat)

Because t^* is asymptotically pivotal (its distribution does not depend on
theta), the bootstrap-t is SECOND-ORDER ACCURATE (Hall 1988) -- coverage
error O(1/n) vs O(1/sqrt(n)) for the plain percentile bootstrap.

Estimating se(theta_hat^*)
    - Analytical: closed-form SE if available (e.g. for the mean, use
      s / sqrt(n) on the resample).
    - NESTED bootstrap: inner bootstrap of the resample to estimate its SE.
      Expensive (B * B_inner total resamples) but always applicable.

The demo below shows both variants for the mean and compares coverage
against the plain percentile bootstrap and Normal CIs.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def bootstrap_t_mean(x, alpha: float = 0.05, B: int = 999, seed: int = 0) -> dict:
    """Studentized bootstrap CI for the mean using the analytical SE = s / sqrt(n)."""
    x = np.asarray(x, dtype=float); n = len(x)
    theta_hat = float(x.mean()); se_hat = float(x.std(ddof=1) / math.sqrt(n))
    rng = np.random.default_rng(seed)
    t_star = np.empty(B)
    for b in range(B):
        xb = rng.choice(x, size=n, replace=True)
        theta_b = xb.mean(); se_b = xb.std(ddof=1) / math.sqrt(n)
        t_star[b] = (theta_b - theta_hat) / (se_b + 1e-12)
    lo_q = np.quantile(t_star, 1 - alpha / 2)
    hi_q = np.quantile(t_star, alpha / 2)
    return {"estimate": theta_hat, "se": se_hat,
            "ci_lower": float(theta_hat - lo_q * se_hat),
            "ci_upper": float(theta_hat - hi_q * se_hat),
            "B": int(B),
            "method": "Studentized (bootstrap-t) CI with analytical SE"}


def bootstrap_t_nested(x, theta_fn, alpha: float = 0.05,
                       B: int = 500, B_inner: int = 100, seed: int = 0) -> dict:
    """Studentized bootstrap CI for arbitrary statistic via NESTED bootstrap SE."""
    x = np.asarray(x, dtype=float); n = len(x)
    theta_hat = float(theta_fn(x))
    rng = np.random.default_rng(seed)
    # Estimate SE of theta_hat via one outer bootstrap
    theta_boot = np.empty(B)
    for b in range(B):
        theta_boot[b] = theta_fn(rng.choice(x, size=n, replace=True))
    se_hat = float(theta_boot.std(ddof=1))
    # Studentized statistics via nested inner bootstrap
    rng2 = np.random.default_rng(seed + 1)
    t_star = np.empty(B)
    for b in range(B):
        xb = rng2.choice(x, size=n, replace=True)
        theta_b = float(theta_fn(xb))
        inner = np.empty(B_inner)
        for k in range(B_inner):
            inner[k] = theta_fn(rng2.choice(xb, size=n, replace=True))
        se_b = inner.std(ddof=1) or 1e-12
        t_star[b] = (theta_b - theta_hat) / se_b
    lo_q = np.quantile(t_star, 1 - alpha / 2)
    hi_q = np.quantile(t_star, alpha / 2)
    return {"estimate": theta_hat, "se_est": se_hat,
            "ci_lower": float(theta_hat - lo_q * se_hat),
            "ci_upper": float(theta_hat - hi_q * se_hat),
            "B": int(B), "B_inner": int(B_inner),
            "method": "Studentized (bootstrap-t) CI with nested inner bootstrap"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Skewed sample: log-normal -> mean = exp(0.5) ~ 1.65
    x = rng.lognormal(0, 1, size=40)
    print(f"=== Sample mean = {x.mean():.3f}, s = {x.std(ddof=1):.3f}, n = {len(x)} ===")

    print("\n=== Bootstrap-t CI (analytical SE) for the mean ===")
    r = bootstrap_t_mean(x, alpha=0.05, B=999)
    print(f"  95% CI: ({r['ci_lower']:.3f}, {r['ci_upper']:.3f})")

    print("\n=== Bootstrap-t CI (nested SE) for the median ===")
    r = bootstrap_t_nested(x, theta_fn=lambda a: float(np.median(a)),
                            B=200, B_inner=50)
    print(f"  Median estimate: {r['estimate']:.3f}, SE est: {r['se_est']:.3f}")
    print(f"  95% CI: ({r['ci_lower']:.3f}, {r['ci_upper']:.3f})")

    print("\n=== Comparison: percentile bootstrap for the mean ===")
    B = 999; boot_means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)])
    print(f"  95% percentile CI: ({np.quantile(boot_means, 0.025):.3f}, {np.quantile(boot_means, 0.975):.3f})")

    print("\n--- library cross-check (scipy.stats.bootstrap) ---")
    try:
        from scipy.stats import bootstrap
        res = bootstrap((x,), np.mean, method="basic", n_resamples=999, random_state=0)
        print(f"  scipy basic CI: ({res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f})")
    except Exception as ex:
        print(f"  (scipy bootstrap unavailable: {ex})")
