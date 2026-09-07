"""Elicitation of priors (Reference Sec 23.20).

Kadane et al. 1980; O'Hagan et al. 2006 'Uncertain Judgements:
Eliciting Experts' Probabilities', Wiley. Structured methods for
quantifying an expert's belief as a probability distribution.

Common workflow:
    1. Ask expert for a small number of quantitative summaries
       (median, quartiles, tail probabilities, plausible range).
    2. Fit a parametric distribution whose implied summaries best
       match the elicited values.
    3. Feed as prior into a Bayesian analysis; combine across
       experts by pooling (linear / logarithmic / SHELF).

We implement two common fitters:
    * FIT-BETA from expert median + IQR (for proportions).
    * FIT-NORMAL from expert median + 5th / 95th percentile.
Plus a simple linear pool across experts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # parameter fitting
from scipy.stats import beta as beta_dist, norm


def fit_beta_from_median_iqr(median, iqr):
    """Find (alpha, beta) matching a given median and IQR."""
    q1 = median - iqr / 2
    q3 = median + iqr / 2

    def loss(params):
        a, b = np.exp(params)      # ensure positivity
        m_hat = beta_dist.ppf(0.5, a, b)
        q1_hat = beta_dist.ppf(0.25, a, b)
        q3_hat = beta_dist.ppf(0.75, a, b)
        return ((m_hat - median) ** 2 + (q1_hat - q1) ** 2 + (q3_hat - q3) ** 2)

    r = minimize(loss, np.log([2.0, 2.0]), method="Nelder-Mead")
    return {"alpha": float(np.exp(r.x[0])), "beta": float(np.exp(r.x[1]))}


def fit_normal_from_quantiles(median, q05, q95):
    """mu = median; sigma matches (q95 - q05) / (2 * 1.645)."""
    return {"mu": median, "sigma": (q95 - q05) / (2 * 1.6449)}


def linear_pool(distributions, weights=None, grid=None):
    """Linear opinion pool: p_pool(x) = sum w_i * p_i(x)."""
    if grid is None:
        grid = np.linspace(0, 1, 400)
    if weights is None:
        weights = np.ones(len(distributions)) / len(distributions)
    density = np.zeros_like(grid)
    for w, d in zip(weights, distributions):
        density += w * d(grid)
    return grid, density


if __name__ == "__main__":
    print("=== Elicitation of priors ===\n")

    #  Case A: two experts on a proportion (efficacy rate 0.4 vs 0.6)
    exp_A = fit_beta_from_median_iqr(median=0.4, iqr=0.2)
    exp_B = fit_beta_from_median_iqr(median=0.6, iqr=0.1)
    print("  Proportion elicitation (Beta priors)")
    print(f"    Expert A  (median 0.40, IQR 0.20)  -> Beta({exp_A['alpha']:.2f}, {exp_A['beta']:.2f})")
    print(f"    Expert B  (median 0.60, IQR 0.10)  -> Beta({exp_B['alpha']:.2f}, {exp_B['beta']:.2f})")

    #  Pool with equal weights
    from scipy.stats import beta as bd
    dA = lambda x: bd.pdf(x, exp_A['alpha'], exp_A['beta'])
    dB = lambda x: bd.pdf(x, exp_B['alpha'], exp_B['beta'])
    grid, pooled = linear_pool([dA, dB])
    #  Pooled mean by numerical integration
    dx = grid[1] - grid[0]
    pool_mean = float(np.sum(grid * pooled) * dx)
    print(f"    Linear pool  (equal weights)  mean = {pool_mean:.3f}")

    #  Case B: log-fold change elicitation (Normal)
    exp_LF = fit_normal_from_quantiles(median=0.5, q05=0.1, q95=0.9)
    print(f"\n  Log-fold-change elicitation (Normal prior)")
    print(f"    median 0.50, 5-95 pctile (0.1, 0.9) -> N(mu={exp_LF['mu']:.2f}, sigma={exp_LF['sigma']:.3f})")

    #  Sanity check on the Beta fits' implied quartiles
    print("\n  Sanity: expert A's Beta implied (median, IQR):")
    m = bd.ppf(0.5, exp_A['alpha'], exp_A['beta'])
    iqr = bd.ppf(0.75, exp_A['alpha'], exp_A['beta']) - bd.ppf(0.25, exp_A['alpha'], exp_A['beta'])
    print(f"    median = {m:.3f}, IQR = {iqr:.3f}   (target 0.40 / 0.20)")

    print("\n--- library cross-check (SHELF R, rriskDistributions R; sheffield-elicitation Python) ---")
