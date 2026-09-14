"""Probability integral transform + inverse-CDF sampling.

Two related identities that undergird copulas, calibration
checks, and virtually every random-variable simulator:

    1. PIT: if X has CDF F, then U = F(X) ~ Uniform(0, 1).
    2. Inverse-CDF: if U ~ Uniform(0, 1) and F is a CDF,
       then X = F^{-1}(U) has CDF F.

Applications:
    - Uniformity-of-PIT checks for calibration diagnostics.
    - Copula construction: MVN(0, R) -> Phi() -> Uniform ->
      inverse target CDF.
    - Universal generator: any distribution with a computable
      quantile function can be sampled without library help.
"""

import numpy as np    # arrays + random


def inverse_cdf_sample(inv_cdf, n, rng):
    """X = inv_cdf(U), U ~ Uniform(0, 1)."""
    return inv_cdf(rng.uniform(size=n))


def demo():
    print("=== Probability integral transform + inverse-CDF ===")
    rng = np.random.default_rng(2026)

    print("\n1. PIT check: PIT of X ~ Exp(1) should be Uniform(0, 1)")
    x = rng.exponential(1.0, size=10000)
    u = 1 - np.exp(-x)    # F(x) for Exp(1)
    print(f"   Empirical mean = {u.mean():.3f} (target 0.5), "
          f"std = {u.std():.3f} (target ~ 0.289)")
    # KS test against uniform (basic)
    p_uniform = np.mean(u < 0.5)
    print(f"   P(U < 0.5) empirical = {p_uniform:.3f} (target 0.500)")

    print("\n2. Inverse-CDF sampling for Weibull(k=1.5, lam=2)")
    def weibull_inv(u, k=1.5, lam=2.0):
        return lam * (-np.log1p(-u)) ** (1 / k)
    x_w = inverse_cdf_sample(weibull_inv, n=10000, rng=rng)
    print(f"   Sample mean = {x_w.mean():.3f}, sd = {x_w.std():.3f}")
    # Weibull mean = lam * Gamma(1 + 1/k)
    from math import gamma
    print(f"   Theory: mean = 2 * Gamma(1 + 1/1.5) = {2 * gamma(1 + 1/1.5):.3f}")

    print("\n3. Copula construction: MVN -> Phi -> Uniform -> arbitrary marginals")
    from scipy.special import ndtr, ndtri
    # rho = 0.6 correlation in the Gaussian copula
    rho = 0.6
    L = np.array([[1, 0], [rho, np.sqrt(1 - rho ** 2)]])
    Z = rng.standard_normal((10000, 2)) @ L.T
    U = ndtr(Z)    # Gaussian copula uniform margins
    # X1 ~ Exp(1); X2 ~ Weibull(1.5, 2)
    x1 = -np.log1p(-U[:, 0])
    x2 = weibull_inv(U[:, 1])
    emp_corr = np.corrcoef(x1, x2)[0, 1]
    print(f"   Gaussian copula rho = {rho}; empirical Kendall's-like corr = {emp_corr:.3f}")

    print("\nSee also: copulas, latin-hypercube-sampling,")
    print("          quasi-monte-carlo-sobol, rejection-sampling,")
    print("          calibration-plots (PIT is the theoretical backbone).")


if __name__ == "__main__":
    demo()
