"""Proper Scoring Rules - CRPS (Reference Sec 47.255).

Gneiting & Raftery 2007 'Strictly Proper Scoring Rules,
Prediction, and Estimation', JASA. The CONTINUOUS RANKED
PROBABILITY SCORE assesses a predictive CDF F against an
observed value y:

    CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx

For an EMPIRICAL forecast ensemble {x_1, ..., x_n}:

    CRPS = (1/n) sum |x_i - y|  -  (1/(2 n^2)) sum_{i,j} |x_i - x_j|

Strictly proper: minimised only by the true distribution.
Standard skill score for weather, finance, epidemic forecasts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def crps_ensemble(y, forecast_samples):
    """CRPS for a single observation vs an ensemble forecast sample."""
    x = np.sort(np.asarray(forecast_samples))
    n = len(x)
    term1 = np.mean(np.abs(x - y))
    # Sorted trick: sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - n - 1) x_(i)
    i = np.arange(1, n + 1)
    term2 = np.sum((2 * i - n - 1) * x) / (n * n)
    return term1 - term2


def crps_gaussian(y, mu, sigma):
    """CRPS closed-form for Normal(mu, sigma) forecast."""
    from scipy.stats import norm
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


if __name__ == "__main__":
    print("=== Proper Scoring Rules - CRPS (Gneiting & Raftery 2007) ===\n")
    rng = np.random.default_rng(0)

    # True DGP: N(0, 1); observe 500 y's
    y_obs = rng.standard_normal(500)

    forecasts = [
        ("perfect N(0, 1)      ", 0.0, 1.0),
        ("biased N(0.5, 1)     ", 0.5, 1.0),
        ("over-dispersed N(0,2)", 0.0, 2.0),
        ("under-disp N(0, 0.5) ", 0.0, 0.5),
    ]
    print(f"  Predictive distribution        Gaussian CRPS   Ensemble CRPS (500 samples)")
    for name, mu, sig in forecasts:
        c_gauss = np.mean([crps_gaussian(y, mu, sig) for y in y_obs])
        # Empirical ensemble
        c_ens = np.mean([crps_ensemble(y, rng.normal(mu, sig, size=500)) for y in y_obs])
        print(f"  {name}   {c_gauss:>7.4f}         {c_ens:>7.4f}")

    print(f"\n  Perfect forecast has lowest CRPS (~0.23 = |N(0,1)| expected value / sqrt(pi)).")
    print(f"  Biased and mis-scale forecasts pay a proper-scoring penalty; propriety")
    print(f"  guarantees that reporting the true predictive distribution is optimal.")

    print("\n--- library cross-check (properscoring.crps_ensemble; scoringutils R) ---")
