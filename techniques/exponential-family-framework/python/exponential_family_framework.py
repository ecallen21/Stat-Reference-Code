"""Exponential-family framework — unified view of many
distributions (Fisher 1934; Koopman 1936; Pitman 1936).

    f(x; theta) = h(x) exp(eta(theta).T T(x) - A(eta))

    T(x)     -- sufficient statistic
    eta      -- natural parameter
    A(eta)   -- log-partition (cumulant generating function)
    h(x)     -- base measure

Members: Normal (mu, sigma known), Bernoulli, Poisson, Gamma,
         Beta, Dirichlet, Categorical, Exponential, Chi-square, ...

MLE = matching sufficient statistic to expected sufficient
statistic. Fisher information = A''(eta). Bayesian conjugate
priors have closed form.
"""

import numpy as np    # arrays + linalg


def demo():
    print("=== Exponential-family framework ===")
    print("Verifying the mean-parameter / natural-parameter identity")
    print("for three canonical members.\n")

    rng = np.random.default_rng(2026)
    n = 10000

    # 1. Bernoulli / Binomial
    p = 0.3
    x = rng.binomial(1, p, size=n)
    eta = np.log(p / (1 - p))
    A_prime = 1 / (1 + np.exp(-eta))    # sigma
    print(f"  Bernoulli(p={p})")
    print(f"    eta = log(p/(1-p)) = {eta:.3f}, A'(eta) = sigmoid = {A_prime:.3f}, "
          f"empirical mean = {x.mean():.3f}")

    # 2. Poisson
    lam = 3.5
    x = rng.poisson(lam, size=n)
    eta = np.log(lam)
    A_prime = np.exp(eta)
    print(f"\n  Poisson(lam={lam})")
    print(f"    eta = log(lam) = {eta:.3f}, A'(eta) = exp(eta) = {A_prime:.3f}, "
          f"empirical mean = {x.mean():.3f}")

    # 3. Normal (fixed sigma^2 = 1)
    mu = 2.0
    x = rng.normal(mu, 1.0, size=n)
    eta = mu    # natural parameter = mu when sigma^2 known
    A_prime = eta
    print(f"\n  N(mu={mu}, sigma=1) (sigma known)")
    print(f"    eta = mu = {eta:.3f}, A'(eta) = eta, empirical mean = {x.mean():.3f}")

    # Fisher information for Bernoulli
    print("\n  Fisher information (Bernoulli): I(theta) = 1 / (p(1-p))")
    print(f"    At p=0.3: I = {1 / (0.3 * 0.7):.3f}  ")
    print("    Cramer-Rao: Var(MLE) >= 1 / (n I)")
    print(f"    So Var(p_hat) >= {1 / (n * 1 / (0.3 * 0.7)):.2e}, "
          f"empirical Var = {np.var(rng.binomial(1, 0.3, n)) / n:.2e}")

    print("\nSee also: bayesian-glms, glm-diagnostics, information-geometry,")
    print("          fisher-information, poisson-regression, tweedie-glm-regression,")
    print("          gamma-regression, dirichlet-regression, inverse-gaussian-glm,")
    print("          modified-poisson, beta-regression, gee, conjugate-priors.")


if __name__ == "__main__":
    demo()
