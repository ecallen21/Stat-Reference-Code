"""Log-normal distribution — X = exp(Y),  Y ~ N(mu, sigma^2).

PDF: f(x) = 1 / (x sigma sqrt(2 pi)) exp(-(log x - mu)^2 / (2 sigma^2))
Mean: exp(mu + sigma^2 / 2)
Var:  (exp(sigma^2) - 1) exp(2 mu + sigma^2)

Heavy-right-tailed; ubiquitous in income, particle size,
biological growth, environmental pollution.
"""

import numpy as np    # arrays + random


def sample_lognormal(mu, sigma, n, rng):
    return np.exp(rng.normal(mu, sigma, size=n))


def mle_lognormal(x):
    y = np.log(x)
    return y.mean(), y.std(ddof=0)


def demo():
    print("=== Log-normal distribution ===")
    rng = np.random.default_rng(2026)
    for mu, sigma in [(0, 0.5), (1, 1.0), (-0.5, 0.3)]:
        x = sample_lognormal(mu, sigma, n=5000, rng=rng)
        mu_hat, sigma_hat = mle_lognormal(x)
        true_mean = np.exp(mu + sigma ** 2 / 2)
        emp_mean = x.mean()
        print(f"  true (mu, sigma) = ({mu:+.2f}, {sigma:.2f}) -> "
              f"MLE ({mu_hat:+.3f}, {sigma_hat:.3f}), "
              f"E[X] true = {true_mean:.3f}, empirical = {emp_mean:.3f}")
    print("\nSee also: bayesian-linear-regression (log-transformed y),")
    print("          extreme-value-theory (lognormal is medium-tail),")
    print("          gamma-regression, tweedie-glm-regression, gamlss.")


if __name__ == "__main__":
    demo()
