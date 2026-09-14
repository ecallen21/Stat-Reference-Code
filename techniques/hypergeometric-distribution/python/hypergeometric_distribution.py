"""Hypergeometric distribution — sampling WITHOUT replacement.

X = number of successes in n draws from a finite population
of N with K successes.

PMF: P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)
Mean = n * K / N
Var  = n * K/N * (N-K)/N * (N-n)/(N-1)   (finite-population correction)
"""

import math    # comb
import numpy as np    # arrays + random


def hypergeom_pmf(k, N, K, n):
    if k < max(0, n + K - N) or k > min(K, n):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def demo():
    print("=== Hypergeometric distribution ===")
    rng = np.random.default_rng(2026)
    # Population N=50, K=20 successes, draw n=10
    N, K, n = 50, 20, 10
    x = rng.hypergeometric(K, N - K, n, size=20000)
    emp_mean, emp_var = x.mean(), x.var(ddof=0)
    tru_mean = n * K / N
    tru_var = n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
    print(f"  N={N}, K={K}, n={n}")
    print(f"  Empirical mean = {emp_mean:.3f}, theory = {tru_mean:.3f}")
    print(f"  Empirical var  = {emp_var:.3f},  theory = {tru_var:.3f}")
    for k in [0, 2, 4, 6, 8, 10]:
        emp_p = np.mean(x == k)
        tru_p = hypergeom_pmf(k, N, K, n)
        print(f"    P(X = {k:2d}): empirical = {emp_p:.4f}, exact = {tru_p:.4f}")

    print("\nSee also: fisher-exact (2x2 hypergeom test),")
    print("          capture-recapture, permutation-tests,")
    print("          binomial-test (with-replacement analogue).")


if __name__ == "__main__":
    demo()
