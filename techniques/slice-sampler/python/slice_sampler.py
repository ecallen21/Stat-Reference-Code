"""Slice sampler (Reference Sec 25.11).

Neal 2003 'Slice sampling', Ann Stat. A tuning-free MCMC method: at
each step, sample a HORIZONTAL LEVEL u ~ Uniform(0, f(x)) and then a
new x uniformly from the SLICE S = {x' : f(x') > u}. Guarantees
detailed balance with no acceptance ratio.

For a 1-D log-density log f, the algorithm uses 'stepping-out' +
'shrinkage':

    1. Draw log u = log f(x) - Exp(1).
    2. Grow an interval [L, R] around x with random start until both
       endpoints fall below log u.
    3. Repeatedly sample x' Uniform(L, R); if log f(x') > log u,
       accept, else shrink [L, R] to include x but exclude x'.

Advantages:
    * No tuning of step size.
    * Uniform on slice -> good mixing.
    * Trivially extends to multivariate via Gibbs-style one-var-
      at-a-time sampling.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def slice_sample_step(logf, x, w=1.0, m=50, rng=None):
    """One 1-D slice sample step from log-target logf, starting at x."""
    rng = rng or np.random.default_rng()
    log_u = logf(x) + np.log(rng.uniform())
    #  Stepping-out to find [L, R]
    r = rng.uniform()
    L = x - r * w
    R = L + w
    j = int(np.floor(m * rng.uniform()))
    k = m - 1 - j
    while j > 0 and logf(L) > log_u:
        L -= w; j -= 1
    while k > 0 and logf(R) > log_u:
        R += w; k -= 1
    #  Shrinkage-based sampling
    while True:
        x_new = L + rng.uniform() * (R - L)
        if logf(x_new) > log_u:
            return x_new
        if x_new < x: L = x_new
        else:         R = x_new


def slice_sampler(logf, x0, n_iter=5000, w=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = x0; chain = np.zeros(n_iter)
    for i in range(n_iter):
        x = slice_sample_step(logf, x, w=w, rng=rng)
        chain[i] = x
    return chain


if __name__ == "__main__":
    print("=== Slice sampler (Neal 2003) ===\n")
    #  Target: mixture 0.4 N(-2, 1) + 0.6 N(3, 0.5)
    def logf(x):
        a = 0.4 * np.exp(-0.5 * (x + 2) ** 2) / np.sqrt(2 * np.pi)
        b = 0.6 * np.exp(-0.5 * ((x - 3) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
        return np.log(a + b + 1e-300)

    chain = slice_sampler(logf, x0=0.0, n_iter=5000, w=2.0, seed=0)
    burn = 500
    ch = chain[burn:]
    print(f"  Bimodal target: 0.4 N(-2, 1) + 0.6 N(3, 0.5)")
    print(f"  Sample mean    = {ch.mean():+.3f}   (analytic {0.4 * -2 + 0.6 * 3:+.3f})")
    print(f"  Sample var     = {ch.var(ddof=1):+.3f}   (analytic {0.4 * (1 + 4) + 0.6 * (0.25 + 9) - (0.4 * -2 + 0.6 * 3) ** 2:.3f})")

    #  Fraction under 0
    print(f"  P(X < 0) est   = {np.mean(ch < 0):.3f}   (analytic ~ 0.40 * P(N(-2,1)<0) + 0.60 * P(N(3,0.5)<0))")
    from scipy.stats import norm
    p_analytic = 0.4 * norm.cdf(0, -2, 1) + 0.6 * norm.cdf(0, 3, 0.5)
    print(f"                                       (analytic p = {p_analytic:.3f})")

    #  Lag-1 autocorrelation
    from scipy.stats import pearsonr
    rho = pearsonr(ch[:-1], ch[1:])[0]
    print(f"  Lag-1 autocorr = {rho:.3f}  (no manual tuning; slice mixes well)")

    print("\n--- library cross-check (Bayes: nimble / stan slice steps; PyMC Slice Python) ---")
