"""Quasi-Monte Carlo (QMC) with Sobol / Halton sequences (Reference Sec 45.9).

Niederreiter 1992; Sobol 1967; Owen 1997 (scrambling). Replaces IID
uniform draws with LOW-DISCREPANCY (quasi-random) sequences that
cover the unit hypercube more evenly. For d-dim integration:

    Standard MC error  ~ O(1 / sqrt(N))
    QMC error          ~ O((log N)^d / N)     (bounded-variation f)

Randomised QMC (Owen scrambling) gives an UNBIASED estimator whose
variance still decays faster than 1/N.

Applications:
    * Bayesian marginal likelihood.
    * Expected policy / option pricing.
    * Uncertainty quantification.
    * Deep neural network dropout replacements.

We compare MC vs Sobol on integrating a d-dimensional Gaussian
integral and a couple of test functions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import qmc    # scipy Sobol / Halton


def integrate_mc(f, d, n, rng):
    X = rng.uniform(0, 1, size=(n, d))
    return float(np.mean([f(x) for x in X]))


def integrate_sobol(f, d, n, scramble=True, seed=0):
    engine = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    #  Sobol works best when n is a power of 2
    exponent = int(np.ceil(np.log2(n)))
    X = engine.random_base2(m=exponent)[:n]
    return float(np.mean([f(x) for x in X]))


if __name__ == "__main__":
    print("=== Quasi-Monte Carlo -- Sobol vs standard MC ===\n")
    rng = np.random.default_rng(0)

    #  1. Integrate f(x) = prod(x^2)   on [0,1]^d  (true value = (1/3)^d)
    for d in [3, 5]:
        true_val = (1 / 3) ** d
        for n in [1024, 4096]:
            errs_mc = np.zeros(30); errs_sob = np.zeros(30)
            for k in range(30):
                errs_mc[k] = integrate_mc(lambda x: np.prod(x ** 2), d, n, np.random.default_rng(k))
                errs_sob[k] = integrate_sobol(lambda x: np.prod(x ** 2), d, n, seed=k)
            rmse_mc = float(np.sqrt(np.mean((errs_mc - true_val) ** 2)))
            rmse_sob = float(np.sqrt(np.mean((errs_sob - true_val) ** 2)))
            print(f"  d={d}, N={n}   true = {true_val:.4e}   "
                  f"MC RMSE = {rmse_mc:.2e}   Sobol RMSE = {rmse_sob:.2e}   "
                  f"ratio = {rmse_mc / rmse_sob:.2f}x")

    #  2. Integrate a smooth radial function; QMC should crush MC.
    print("\n  Smooth radial f(x) = exp(-|| x - 0.5 ||^2), d=4:")
    for n in [1024, 4096, 16384]:
        errs_mc = np.zeros(20); errs_sob = np.zeros(20)
        for k in range(20):
            errs_mc[k] = integrate_mc(lambda x: np.exp(-np.sum((x - 0.5) ** 2)), 4, n, np.random.default_rng(k))
            errs_sob[k] = integrate_sobol(lambda x: np.exp(-np.sum((x - 0.5) ** 2)), 4, n, seed=k)
        rmse_mc = float(np.std(errs_mc, ddof=1))
        rmse_sob = float(np.std(errs_sob, ddof=1))
        print(f"    N={n}   MC sd = {rmse_mc:.4e}   Sobol sd = {rmse_sob:.4e}   "
              f"gain = {rmse_mc / rmse_sob:.2f}x")

    print("\n--- library cross-check (randtoolbox / qrng R; scipy.stats.qmc / SALib Python) ---")
