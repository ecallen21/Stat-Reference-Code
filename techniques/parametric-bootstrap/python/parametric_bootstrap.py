"""Parametric bootstrap (Reference §10.2).

Difference from nonparametric bootstrap:
    - Nonparametric: resample the OBSERVED data with replacement.
    - Parametric   : fit a parametric model to the data, then simulate new
                     samples from the FITTED distribution.

Trade-off:
    - If your parametric model is (nearly) correct, parametric bootstrap is
      more efficient (smaller SE for the same B).
    - If the model is wrong, parametric bootstrap gives misleadingly tight CIs.

Common use cases:
    - CI for a parameter of a specific distribution (e.g. gamma shape).
    - Small-sample inference where the sampling distribution has known form.
    - Diagnostic simulations under a fitted GLM (residual patterns, PPCs).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def parametric_bootstrap(x, fit_fn: Callable, sample_fn: Callable,
                          statistic: Callable, n_boot: int = 2000,
                          conf: float = 0.95, seed: int = 0) -> dict:
    """Generic parametric-bootstrap driver.

    Parameters
    ----------
    x         : observed sample.
    fit_fn    : callable(x) -> parameter tuple/dict describing the fitted model.
    sample_fn : callable(params, n, rng) -> new sample of size n from that model.
    statistic : callable(sample) -> scalar (same statistic on original and each resample).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    params = fit_fn(x)
    theta_hat = float(statistic(x))
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        xb = sample_fn(params, n, rng)
        theta_star[b] = float(statistic(xb))
    alpha = 1 - conf
    lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    z = stats.norm.ppf(1 - alpha / 2)
    se = float(theta_star.std(ddof=1))
    return {"theta_hat": theta_hat, "fitted_params": params,
            "bootstrap_SE": se,
            "CI_percentile": {"lower": float(lo), "upper": float(hi)},
            "CI_basic":      {"lower": float(2 * theta_hat - hi),
                               "upper": float(2 * theta_hat - lo)},
            "CI_normal":     {"lower": float(theta_hat - z * se),
                               "upper": float(theta_hat + z * se)},
            "n_boot": n_boot, "n": n, "conf": conf,
            "method": "parametric bootstrap"}


# --- Common parametric families ------------------------------------------

def normal_fit(x):
    return {"mu": float(np.mean(x)), "sigma": float(np.std(x, ddof=1))}

def normal_sample(params, n, rng):
    return rng.normal(params["mu"], params["sigma"], size=n)


def gamma_fit(x):
    # MLE via scipy
    shape, loc, scale = stats.gamma.fit(x, floc=0)
    return {"shape": float(shape), "scale": float(scale)}

def gamma_sample(params, n, rng):
    return rng.gamma(params["shape"], params["scale"], size=n)


def exponential_fit(x):
    # MLE:  rate = 1 / mean
    return {"rate": 1.0 / float(np.mean(x))}

def exponential_sample(params, n, rng):
    return rng.exponential(1.0 / params["rate"], size=n)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # Simulate gamma; estimate the shape parameter with parametric bootstrap
    true_shape, true_scale = 2.5, 1.3
    x = rng.gamma(true_shape, true_scale, size=80)
    print("=== Parametric bootstrap of gamma shape (true = 2.5) ===")
    out = parametric_bootstrap(
        x,
        fit_fn=gamma_fit,
        sample_fn=gamma_sample,
        statistic=lambda z: stats.gamma.fit(z, floc=0)[0],   # MLE shape
        n_boot=1000,
    )
    print(f"  theta_hat (MLE shape) = {out['theta_hat']:.4f}")
    print(f"  fitted params: {out['fitted_params']}")
    print(f"  SE_boot = {out['bootstrap_SE']:.4f}")
    print(f"  CI (percentile): [{out['CI_percentile']['lower']:.4f}, {out['CI_percentile']['upper']:.4f}]")

    # Normal median CI (parametric)
    print("\n=== Parametric bootstrap of normal median (true mu = 5.0) ===")
    x2 = rng.normal(5.0, 2.0, size=100)
    out2 = parametric_bootstrap(x2, normal_fit, normal_sample, np.median, n_boot=2000)
    print(f"  theta_hat (median) = {out2['theta_hat']:.4f}")
    print(f"  fitted (mu, sigma) = {out2['fitted_params']}")
    print(f"  CI (percentile): [{out2['CI_percentile']['lower']:.4f}, {out2['CI_percentile']['upper']:.4f}]")
