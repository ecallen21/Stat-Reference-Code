"""Epanechnikov kernel density (Epanechnikov 1969).

The Epanechnikov kernel

    K(u) = 0.75 * (1 - u^2)      for |u| <= 1
         = 0                      otherwise

MINIMIZES asymptotic MISE among all non-negative kernels
(Epanechnikov's theorem). Efficiency vs Gaussian ~ 94% ->
Gaussian is nearly optimal, but Epanechnikov's compact
support makes it faster and easier to plot.

Kernel density estimator:
    f_h(x) = (1 / (n h)) * sum_i K((x - X_i) / h)
"""

import numpy as np    # arrays


def epanechnikov(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)


def gaussian(u):
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def kde(x_eval, data, h, kernel=epanechnikov):
    n = len(data)
    diffs = (x_eval[:, None] - data[None, :]) / h
    return np.mean(kernel(diffs), axis=1) / h


def silverman_bw(data):
    """Silverman rule-of-thumb bandwidth (Gaussian-optimal)."""
    n = len(data)
    sigma = min(np.std(data, ddof=1),
                (np.percentile(data, 75) - np.percentile(data, 25)) / 1.34)
    return 1.06 * sigma * n ** (-1 / 5)


def demo():
    print("=== Epanechnikov kernel density (Epanechnikov 1969) ===")
    rng = np.random.default_rng(2026)
    n = 200

    # bimodal mixture: 0.5 N(-2, 1) + 0.5 N(2, 1.5)
    z = rng.binomial(1, 0.5, size=n)
    data = np.where(z == 0, rng.normal(-2, 1, n), rng.normal(2, 1.5, n))
    x_eval = np.linspace(-6, 8, 300)

    # true density
    def true_pdf(x):
        return 0.5 * gaussian((x + 2)) + 0.5 * gaussian((x - 2) / 1.5) / 1.5

    truth = true_pdf(x_eval)

    h = silverman_bw(data)
    print(f"  Silverman bandwidth = {h:.3f}")
    for name, K in [("Epanechnikov", epanechnikov),
                    ("Gaussian    ", gaussian)]:
        f_hat = kde(x_eval, data, h, kernel=K)
        ise = np.trapezoid((f_hat - truth) ** 2, x_eval)
        # Approximate efficiency: Epa should be ~ same as Gaussian on this
        # sample (paper 94% MISE efficiency ratio).
        print(f"  {name}: integrated squared error = {ise:.5f}")


if __name__ == "__main__":
    demo()
