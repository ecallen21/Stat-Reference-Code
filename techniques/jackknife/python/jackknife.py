"""Jackknife SE, bias correction, and jackknife-after-bootstrap (Reference §10.6, §10.17).

The jackknife is the pre-bootstrap resampling procedure: compute the statistic
on each leave-one-out (LOO) subsample and use the spread of those replicates.

Leave-one-out estimates:
    J_i  =  theta_hat computed on x with obs i removed        i = 1..n
    Jbar =  mean of J_i

Jackknife SE:
    SE_jack  =  sqrt( (n - 1) / n  *  sum_i (J_i - Jbar)^2 )

Bias estimate & bias-corrected estimator:
    bias_hat = (n - 1) * (Jbar - theta_hat)
    theta_bc = theta_hat - bias_hat
             = n * theta_hat - (n - 1) * Jbar

Compared to bootstrap:
    - Jackknife is deterministic (no random seed).
    - Only n replicates (vs B for bootstrap). Cheaper for small n.
    - Works poorly for non-smooth statistics (e.g. median, quantiles) -- the
      LOO sensitivities are too discrete. Bootstrap handles those better.
    - Jackknife SE is a first-order approximation; bootstrap can capture
      higher-order features (skewness, etc.).

Jackknife-after-bootstrap (Efron 1992) is a diagnostic: measures the
INFLUENCE of each observation on the bootstrap distribution -- useful for
finding leverage / influential points.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def jackknife_1d(x, statistic: Callable) -> dict:
    """Jackknife SE + bias + bias-corrected estimator for a 1-D sample."""
    x = np.asarray(x, dtype=float); n = x.size
    theta_hat = float(statistic(x))
    idx = np.arange(n)
    J = np.array([float(statistic(x[idx != i])) for i in range(n)])
    Jbar = float(J.mean())
    SE_jack = float(math.sqrt((n - 1) / n * ((J - Jbar) ** 2).sum()))
    bias_hat = float((n - 1) * (Jbar - theta_hat))
    theta_bc = float(theta_hat - bias_hat)
    return {"theta_hat": theta_hat,
            "jackknife_mean": Jbar,
            "SE_jackknife": SE_jack,
            "bias_estimate": bias_hat,
            "theta_bias_corrected": theta_bc,
            "loo_replicates_head": J[:10].tolist(),
            "n": n, "method": "jackknife (Quenouille-Tukey)"}


def jackknife_2d(data, statistic: Callable) -> dict:
    """Jackknife over ROWS of a 2-D data matrix."""
    data = np.asarray(data, dtype=float); n = data.shape[0]
    theta_hat = float(statistic(data))
    idx = np.arange(n)
    J = np.array([float(statistic(data[idx != i])) for i in range(n)])
    Jbar = float(J.mean())
    SE_jack = float(math.sqrt((n - 1) / n * ((J - Jbar) ** 2).sum()))
    bias_hat = float((n - 1) * (Jbar - theta_hat))
    return {"theta_hat": theta_hat,
            "SE_jackknife": SE_jack,
            "bias_estimate": bias_hat,
            "theta_bias_corrected": float(theta_hat - bias_hat),
            "n": n, "method": "jackknife over rows"}


def jackknife_after_bootstrap(x, statistic: Callable, n_boot: int = 2000, seed: int = 0) -> dict:
    """Efron's jackknife-after-bootstrap: influence of each obs on the bootstrap SE.

    For each i, compute the bootstrap SE of theta* restricted to bootstrap
    samples that DO NOT contain obs i. The bigger the change from the full
    bootstrap SE, the more influential obs i.
    """
    x = np.asarray(x, dtype=float); n = x.size
    rng = np.random.default_rng(seed)
    # Store the boot indices to check membership per obs later
    all_samples = np.empty((n_boot, n), dtype=int)
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        all_samples[b] = rng.integers(0, n, size=n)
        theta_star[b] = float(statistic(x[all_samples[b]]))
    se_full = float(theta_star.std(ddof=1))
    # For each obs, filter to bootstrap replicates NOT containing i
    influence = np.empty(n)
    n_kept = np.empty(n, dtype=int)
    for i in range(n):
        mask = ~np.any(all_samples == i, axis=1)
        n_kept[i] = int(mask.sum())
        if n_kept[i] > 1:
            influence[i] = float(theta_star[mask].std(ddof=1)) - se_full
        else:
            influence[i] = float("nan")
    # rank observations by absolute influence
    order = np.argsort(-np.abs(np.nan_to_num(influence)))
    return {"SE_full_bootstrap": se_full,
            "per_obs_influence_on_SE": influence.tolist(),
            "n_replicates_excluding_obs": n_kept.tolist(),
            "top_5_influential_obs": order[:5].tolist(),
            "n_boot": n_boot, "n": n,
            "method": "jackknife-after-bootstrap (Efron 1992)"}


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    x = rng.gamma(2.0, 1.0, size=50)          # skewed positive sample
    print("=== Jackknife SE + bias of the MEAN (should have bias ~ 0) ===")
    out = jackknife_1d(x, np.mean)
    for k, v in out.items():
        print(f"  {k:26s}: {v if not isinstance(v, list) else v}")

    print("\n=== Jackknife SE + bias of the VARIANCE (biased for finite n) ===")
    # note: np.var default ddof=0 is biased; use ddof=1 for unbiased sample variance
    out = jackknife_1d(x, lambda z: np.var(z, ddof=0))    # deliberately biased
    print(f"  theta_hat (var, ddof=0): {out['theta_hat']:.4f}")
    print(f"  bias_estimate:           {out['bias_estimate']:.4f}")
    print(f"  bias_corrected:          {out['theta_bias_corrected']:.4f}")
    print(f"  compare unbiased var:    {np.var(x, ddof=1):.4f}")

    print("\n=== Jackknife-after-bootstrap for mean (identifies influential obs) ===")
    jab = jackknife_after_bootstrap(x, np.mean, n_boot=2000)
    print(f"  full bootstrap SE = {jab['SE_full_bootstrap']:.4f}")
    print(f"  top-5 influential obs (indices): {jab['top_5_influential_obs']}")
    print(f"  their x values: {[float(x[i]) for i in jab['top_5_influential_obs']]}")
