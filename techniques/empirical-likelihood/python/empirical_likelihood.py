"""Empirical likelihood (Reference Sec 33.5).

Owen (1988, 1990) 'Empirical likelihood ratio confidence regions.'

Nonparametric method that mimics parametric likelihood-ratio inference
WITHOUT specifying a distribution. For a target theta = E[Y]:

  Maximise    prod_i p_i
  s.t.        p_i >= 0,  sum_i p_i = 1,  sum_i p_i * (Y_i - theta) = 0.

Owen's theorem: -2 log(EL(theta)) converges in distribution to chi^2(1)
under H0: theta = theta_0. So a (1 - alpha) CI is

  { theta : -2 log EL(theta) <= chi^2_{1, 1-alpha} }.

Advantages over t-based CIs:
  * DATA-DRIVEN shape (skew, kurtosis) reflected in CI.
  * No sd or normality assumption.
  * Rangeni-preserving (never crosses [ymin, ymax]).

Here we implement -2 log EL for E[Y] via Lagrange-multiplier
Newton-Raphson, and compute an EL 95% CI on synthetic skewed data.
Compare to t-based CI.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _log_el_mean(theta, Y, tol=1e-8, max_iter=100):
    """Return -2 log EL for E[Y] = theta. Solve for lambda in
       sum_i (Y_i - theta) / (1 + lambda (Y_i - theta)) = 0."""
    z = Y - theta
    lam = 0.0
    for _ in range(max_iter):
        denom = 1 + lam * z
        # Guard: denom must stay > 1/n (Owen).
        if np.any(denom <= 1 / len(Y)):
            lam *= 0.5
            continue
        f = np.sum(z / denom)
        fp = -np.sum(z ** 2 / denom ** 2)
        step = f / fp
        lam_new = lam - step
        if abs(step) < tol:
            lam = lam_new
            break
        lam = lam_new
    denom = 1 + lam * z
    if np.any(denom <= 0):
        return np.inf
    # -2 log R(theta) = 2 * sum log(1 + lambda * z_i)   (Owen 1990 Thm 1)
    return 2.0 * np.sum(np.log(denom))


def el_ci_mean(Y, alpha=0.05, tol=1e-3):
    """Bisection root-finding on either side of the sample mean."""
    from scipy.stats import chi2 as _chi2
    q = _chi2.ppf(1 - alpha, df=1)
    m = float(Y.mean()); lo_bound, hi_bound = float(Y.min()), float(Y.max())

    def _bisect(a, b):
        for _ in range(40):
            mid = 0.5 * (a + b)
            if _log_el_mean(mid, Y) > q:
                a = mid
            else:
                b = mid
            if abs(b - a) < tol:
                break
        return 0.5 * (a + b)

    lo = _bisect(lo_bound + 1e-3, m)
    hi = _bisect(hi_bound - 1e-3, m)
    return (lo, hi)


def t_ci_mean(Y, alpha=0.05):
    from scipy.stats import t as _t
    n = len(Y)
    m = float(Y.mean()); s = float(Y.std(ddof=1))
    q = _t.ppf(1 - alpha / 2, df=n - 1)
    half = q * s / np.sqrt(n)
    return (m - half, m + half)


if __name__ == "__main__":
    print("=== Empirical likelihood CI for a mean (Owen 1988) ===\n")
    rng = np.random.default_rng(0)
    n = 60
    # Skewed data: log-normal
    Y = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    print(f"  n = {n}   sample mean = {Y.mean():.3f}   min = {Y.min():.3f}   max = {Y.max():.3f}")

    lo_el, hi_el = el_ci_mean(Y, alpha=0.05)
    lo_t, hi_t = t_ci_mean(Y, alpha=0.05)
    print(f"\n  EL  95% CI for E[Y]: [{lo_el:.3f}, {hi_el:.3f}]   width = {hi_el - lo_el:.3f}")
    print(f"  t   95% CI for E[Y]: [{lo_t:.3f}, {hi_t:.3f}]   width = {hi_t - lo_t:.3f}")

    # Coverage check across 500 trials
    n_trials = 200
    hits_el, hits_t = 0, 0
    true_mean = float(np.exp(0.0 + 0.5 * 0.8 ** 2))       # true E[log-normal]
    for _ in range(n_trials):
        Y = rng.lognormal(mean=0.0, sigma=0.8, size=n)
        lo, hi = el_ci_mean(Y)
        if not np.isnan(lo) and lo <= true_mean <= hi:
            hits_el += 1
        lo, hi = t_ci_mean(Y)
        if lo <= true_mean <= hi:
            hits_t += 1
    print(f"\n  Coverage over {n_trials} trials (target 0.95):"
          f"  EL={hits_el/n_trials:.3f}   t={hits_t/n_trials:.3f}\n")

    print("--- library cross-check (emplik R; empirical-likelihood pip pkg) ---")
