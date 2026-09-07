"""Concentration inequalities (Reference Sec 46.14).

Boucheron, Lugosi & Massart 2013 'Concentration Inequalities: A
Nonasymptotic Theory of Independence', OUP. Deterministic bounds on
the probability that a random variable deviates from its mean by
more than t.

Classical inequalities for sum S_n = X_1 + ... + X_n of independent
bounded r.v.s in [a, b]:

  MARKOV      : P(X >= t)      <= E[X] / t
  CHEBYSHEV   : P(|X - mu| >= t) <= Var(X) / t^2
  HOEFFDING   : P(|S_n - E S_n| >= t) <= 2 exp(-2 t^2 / (n (b - a)^2))
  BERNSTEIN   : P(|S_n - E S_n| >= t) <= 2 exp(-t^2 / (2 sigma^2 + 2 M t / 3))
  MCDIARMID   : bounded-differences generalisation of Hoeffding

We evaluate empirical probabilities vs the bounds on a Bernoulli sum.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hoeffding_bound(n, t, a=0.0, b=1.0):
    return 2 * np.exp(-2 * t ** 2 / (n * (b - a) ** 2))


def bernstein_bound(n, t, sigma2, M):
    return 2 * np.exp(-t ** 2 / (2 * sigma2 * n + 2 * M * t / 3))


def chebyshev_bound(t, variance_sum):
    return variance_sum / t ** 2


def markov_bound(t, mean_pos):
    return mean_pos / t


def empirical_prob(rng, n, p, t, n_sim=20000):
    """Estimate P(|S_n / n - p| >= t / n)."""
    S = rng.binomial(n, p, size=n_sim)
    return float(np.mean(np.abs(S - n * p) >= t))


if __name__ == "__main__":
    print("=== Concentration inequalities -- Bernoulli(p) sum ===\n")
    rng = np.random.default_rng(0)
    n = 200
    p = 0.3
    variance = n * p * (1 - p)

    print(f"  Setup: X_i ~ Bern(p={p}), n={n}. sum S_n has mean {n * p:.1f}, var {variance:.2f}.\n")
    print(f"  {'t (deviation)':>15s}  {'empirical':>12s}  {'Hoeffding':>12s}  "
          f"{'Bernstein':>12s}  {'Chebyshev':>12s}")
    for t in [5, 10, 20, 30, 40]:
        emp = empirical_prob(rng, n, p, t)
        H = hoeffding_bound(n, t, 0.0, 1.0)
        B = bernstein_bound(n, t, p * (1 - p), 1.0)
        C = chebyshev_bound(t, variance)
        print(f"  {t:>15d}  {emp:>12.4f}  {min(H, 1):>12.4f}  {min(B, 1):>12.4f}  {min(C, 1):>12.4f}")

    print("\n  Observations:")
    print("  - Bernstein is tighter than Hoeffding for low-variance tails.")
    print("  - Chebyshev is tightest for small t but loosest for large t.")
    print("  - All are UPPER BOUNDS: empirical probability <= bound.")

    print("\n--- library cross-check (from-scratch in R and Python) ---")
