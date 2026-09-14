"""Central Limit Theorem demonstrations (Laplace 1810;
Lyapunov 1901; Lindeberg 1922).

Classical CLT: if X_1, ..., X_n iid with finite mean mu and
variance sigma^2, then

    sqrt(n) (X_bar - mu) / sigma  -->  N(0, 1)

Also demonstrates:
    * Rate of convergence (Berry-Esseen O(1/sqrt(n)))
    * Failure under infinite variance (Cauchy) — Generalised
      CLT to stable laws.
    * Multivariate CLT.
"""

import numpy as np    # arrays + random


def demo():
    print("=== Central Limit Theorem demonstrations ===")
    rng = np.random.default_rng(2026)

    print("\n1. CLT for Exponential(rate = 1)  (mu = 1, sigma = 1):")
    for n in [1, 5, 30, 300]:
        # 20 000 sample means of size n
        sample_means = rng.exponential(1.0, size=(20000, n)).mean(axis=1)
        z = np.sqrt(n) * (sample_means - 1)
        # skewness of Z should tend to 0 (exponential's skew = 2 / sqrt(n))
        m3 = np.mean((z - z.mean()) ** 3)
        skew = m3 / z.std() ** 3
        print(f"   n = {n:3d}: E[Z] = {z.mean():+.3f}, sd(Z) = {z.std():.3f}, "
              f"skew = {skew:+.3f} (target 0)")

    print("\n2. Berry-Esseen bound:  sup|F_n(x) - Phi(x)| <= C rho / (sigma^3 sqrt(n)) ")
    print("   For Exponential(1): rho = E|X - 1|^3 approx 2.5, sigma^3 = 1")
    print("   Bound: 2.5 / sqrt(n) -> 0")

    print("\n3. CLT FAILURE for Cauchy:")
    for n in [10, 100, 1000]:
        sample_means = np.mean(rng.standard_cauchy(size=(1000, n)), axis=1)
        print(f"   n = {n:4d}: sample-mean range = "
              f"[{sample_means.min():.2f}, {sample_means.max():.2f}]  "
              f"(no convergence)")

    print("\n4. Multivariate CLT (bivariate skewed source -> bivariate Normal):")
    for n in [3, 30, 300]:
        # sum of n log-normal(0, 1) marginals -> multivariate CLT
        X = rng.lognormal(0.0, 1.0, size=(50000, n, 2))
        means = X.mean(axis=1)
        # Compare covariance to (Sigma / n) where Sigma is variance of lognormal
        var_lognormal = (np.e - 1) * np.e
        cov_theory = var_lognormal / n
        cov_emp = np.cov(means.T)[0, 0]
        print(f"   n = {n:3d}: empirical Var(mean) = {cov_emp:.4f}, theory = {cov_theory:.4f}")

    print("\nSee also: multivariate-normal-distribution, delta-method,")
    print("          bootstrap (bootstrap CLT), stable-distributions (generalised CLT),")
    print("          bayesian-linear-regression (Bernstein-von Mises), fisher-information.")


if __name__ == "__main__":
    demo()
