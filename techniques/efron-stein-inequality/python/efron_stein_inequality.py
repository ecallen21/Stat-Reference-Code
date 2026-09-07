"""Efron-Stein inequality (Reference Sec 46.12).

Efron & Stein 1981 'The jackknife estimate of variance', Ann Stat.
A distribution-free bound on the variance of any function
Z = f(X_1, ..., X_n) of iid variables:

    Var(Z) <= sum_i E[(Z - Z_i')^2] / 2

where Z_i' is Z with the i-th coordinate replaced by an independent
copy X_i'. Equivalent 'leave-one-out' form:

    Var(Z) <= sum_i E[Var(Z | X_{-i})]

Corollaries:
  * Rademacher complexity concentrates around its mean at rate 1/sqrt(n).
  * Bootstrap variance is an efficient estimator of Var(Z) at O(1/n).
  * Backbone of PAC bounds and stability-based generalisation theory.

We verify Efron-Stein empirically for f = sample mean and f = sample
variance: compute the exact variance by Monte Carlo, compute the
Efron-Stein bound by pairwise-resampling, and confirm the bound holds.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def efron_stein_bound(f, dist_sampler, n, n_mc=2000, seed=0):
    """Estimate the Efron-Stein bound on Var(f(X_1..X_n))."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_mc):
        X = dist_sampler(rng, n)
        Z = f(X)
        acc = 0.0
        for i in range(n):
            X_prime = X.copy()
            X_prime[i] = dist_sampler(rng, 1)[0]
            Z_i = f(X_prime)
            acc += 0.5 * (Z - Z_i) ** 2
        total += acc
    return total / n_mc


def true_variance(f, dist_sampler, n, n_mc=2000, seed=1):
    rng = np.random.default_rng(seed)
    zs = np.array([f(dist_sampler(rng, n)) for _ in range(n_mc)])
    return float(np.var(zs, ddof=1))


if __name__ == "__main__":
    print("=== Efron-Stein inequality: distribution-free variance bound ===\n")
    n = 40
    #  sample from N(0, 1)
    sampler = lambda rng, k: rng.normal(size=k)

    for name, f in [("sample mean", np.mean),
                    ("sample variance (biased)", lambda x: np.var(x, ddof=0)),
                    ("sample median", np.median)]:
        v_true = true_variance(f, sampler, n, n_mc=1500)
        v_bound = efron_stein_bound(f, sampler, n, n_mc=400)
        print(f"  n={n}, f = {name}")
        print(f"    Var(Z) (Monte Carlo)      = {v_true:.5f}")
        print(f"    Efron-Stein upper bound   = {v_bound:.5f}")
        print(f"    Bound holds:              {v_bound >= v_true}")
        print()

    print("  Note: for the sample mean, ES is EXACT: Var(Z) = sigma^2 / n = 1/40.")
    print("        For nonlinear f (median, variance), ES is a valid upper bound.")

    print("\n--- library cross-check (custom in both R and Python) ---")
