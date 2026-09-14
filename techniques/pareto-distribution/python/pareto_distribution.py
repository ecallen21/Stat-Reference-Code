"""Pareto distribution (Pareto 1897).

Type-I: f(x; alpha, x_m) = alpha x_m^alpha / x^(alpha + 1), x >= x_m
       "80-20 rule" of wealth, city sizes, word frequencies,
       file sizes on disk, network node degree.

E[X]   = alpha x_m / (alpha - 1),  alpha > 1
Var[X] = alpha x_m^2 / ((alpha - 1)^2 (alpha - 2)),  alpha > 2

MLE:  alpha_hat = n / sum(log(x_i / x_m)),  x_m = min(x).
"""

import numpy as np    # arrays + random


def sample_pareto(alpha, x_m, n, rng):
    """Inverse-CDF: x = x_m / (1 - U)^(1/alpha)."""
    u = rng.uniform(size=n)
    return x_m / (1 - u) ** (1 / alpha)


def mle_pareto(x):
    x_m = x.min()
    alpha = len(x) / np.sum(np.log(x / x_m))
    return alpha, x_m


def demo():
    print("=== Pareto distribution (Pareto 1897) ===")
    rng = np.random.default_rng(2026)
    for alpha, x_m in [(1.16, 1.0), (2.5, 3.0), (5.0, 0.5)]:    # 1.16 approx 80-20
        x = sample_pareto(alpha, x_m, n=5000, rng=rng)
        a_hat, xm_hat = mle_pareto(x)
        print(f"  true alpha={alpha}, x_m={x_m}: MLE alpha={a_hat:.3f}, x_m={xm_hat:.3f}")
    # 80-20 property: for alpha=log(5)/log(4)~1.16, 20% of pop holds 80%
    alpha = np.log(5) / np.log(4)
    x = sample_pareto(alpha, 1.0, n=100000, rng=rng)
    total = x.sum()
    top20 = np.sort(x)[-int(0.20 * len(x)):].sum()
    print(f"\n  alpha = ln5/ln4 ~ {alpha:.3f}: fraction held by top 20% "
          f"= {100 * top20 / total:.1f}%  (theoretical 80.0%)")

    print("\nSee also: extreme-value-theory (Pareto = generalised Pareto MDA),")
    print("          pareto-charts (frequency plot, not the distribution),")
    print("          cvar-expected-shortfall, gini-lorenz, power-law-networks (via degree).")


if __name__ == "__main__":
    demo()
