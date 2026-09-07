"""Rao-Blackwellization (Reference Sec 45.11).

Rao 1945; Blackwell 1947. Given an estimator T(X) and a sufficient
statistic S(X), the CONDITIONAL expectation

    T*(S) = E[T(X) | S(X)]

is at least as good in mean-square error:

    Var(T*(S))  <=  Var(T(X))                (RAO-BLACKWELL THEOREM)

Applications in Monte Carlo:
    * Instead of averaging noisy indicator functions I{X_i in A},
      average the CONDITIONAL probability p(A | Y_i) where Y_i is a
      hidden variable you can integrate out analytically.
    * Gibbs sampling: replace theta_i draws with the closed-form
      conditional mean given the other variables.
    * Importance sampling: use E[f | proposal density] instead of
      raw f(X_i).

We demonstrate with a mixture-model example: estimate P(X > 2) for
X = Z * A + (1 - Z) * B with Z ~ Bern(0.4), A ~ N(0, 1), B ~ N(3, 1)
by (i) plain MC via indicator, (ii) Rao-Blackwell replacing the
indicator with the analytic P(X > 2 | Z).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm


def plain_mc(n, rng):
    Z = rng.binomial(1, 0.4, size=n)
    X = np.where(Z == 1, rng.normal(0, 1, size=n),
                          rng.normal(3, 1, size=n))
    return float(np.mean(X > 2))


def rao_blackwell(n, rng):
    Z = rng.binomial(1, 0.4, size=n)
    #  Given Z, X | Z=1 ~ N(0, 1); X | Z=0 ~ N(3, 1)
    p_given_Z = np.where(Z == 1, 1 - norm.cdf(2, loc=0, scale=1),
                                   1 - norm.cdf(2, loc=3, scale=1))
    return float(np.mean(p_given_Z))


if __name__ == "__main__":
    print("=== Rao-Blackwellization -- variance reduction via conditioning ===\n")
    n = 1000
    reps = 500
    est_mc = np.zeros(reps); est_rb = np.zeros(reps)
    for k in range(reps):
        rng = np.random.default_rng(k)
        est_mc[k] = plain_mc(n, rng)
        est_rb[k] = rao_blackwell(n, rng)

    #  True value: 0.4 * P(N(0,1) > 2) + 0.6 * P(N(3,1) > 2)
    true_val = 0.4 * (1 - norm.cdf(2, 0, 1)) + 0.6 * (1 - norm.cdf(2, 3, 1))
    print(f"  True value      = {true_val:.5f}")
    print(f"  Plain MC        mean = {est_mc.mean():.5f}   sd = {est_mc.std(ddof=1):.5f}")
    print(f"  Rao-Blackwell   mean = {est_rb.mean():.5f}   sd = {est_rb.std(ddof=1):.5f}")
    print(f"\n  Variance ratio MC / RB = {est_mc.std(ddof=1) ** 2 / est_rb.std(ddof=1) ** 2:.1f}x")
    print(f"  (RB replaces noisy indicators with closed-form conditional probabilities.)")

    print("\n--- library cross-check (from-scratch in R and Python) ---")
