"""Stable / alpha-stable distributions (Levy 1925).

The unique family CLOSED UNDER LINEAR COMBINATIONS whose
characteristic function is

    phi(t) = exp(i mu t - |c t|^alpha (1 - i beta sign(t) omega))

Parameters: alpha in (0, 2] tail (2 = Gaussian, 1 = Cauchy),
            beta in [-1, 1] skew, c > 0 scale, mu location.

- alpha = 2: Gaussian
- alpha = 1, beta = 0: Cauchy
- alpha = 0.5, beta = 1, mu = 0: Levy
- alpha < 2: no finite variance (heavy tail)

Chambers-Mallows-Stuck (1976) sampler works for any alpha.
"""

import numpy as np    # arrays + random


def sample_stable(alpha, beta, c, mu, n, rng):
    """Chambers-Mallows-Stuck sampler."""
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    W = rng.exponential(1.0, size=n)
    if abs(alpha - 1) < 1e-12:
        zeta = 2 / np.pi * ((np.pi / 2 + beta * U) * np.tan(U)
                            - beta * np.log((np.pi / 2 * W * np.cos(U))
                                             / (np.pi / 2 + beta * U)))
        return c * zeta + (2 / np.pi) * beta * c * np.log(c) + mu
    xi = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
    factor = ((1 + (beta * np.tan(np.pi * alpha / 2)) ** 2) ** (1 / (2 * alpha)))
    zeta = factor * np.sin(alpha * (U + xi)) / (np.cos(U)) ** (1 / alpha) * \
           (np.cos(U - alpha * (U + xi)) / W) ** ((1 - alpha) / alpha)
    return c * zeta + mu


def demo():
    print("=== Alpha-stable distributions (Levy 1925) ===")
    rng = np.random.default_rng(2026)
    print("  Sample means and 95%-tail width for varying alpha (mu=0, c=1):")
    for alpha in [2.0, 1.5, 1.0, 0.5]:
        x = sample_stable(alpha, beta=0.0, c=1.0, mu=0.0, n=20000, rng=rng)
        med = np.median(x)
        p975 = np.percentile(x, 97.5)
        note = "Gaussian" if alpha == 2.0 else "Cauchy" if alpha == 1.0 else "Levy(shifted)" if alpha == 0.5 else "heavy-tailed"
        print(f"    alpha = {alpha}: median = {med:+.3f},  97.5%ile = {p975:>8.2f}  ({note})")
    print("\nSee also: cauchy-distribution (alpha=1), extreme-value-theory,")
    print("          garch, stochastic-volatility, cvar-expected-shortfall,")
    print("          robust-regression, tukey-biweight-m-estimator.")


if __name__ == "__main__":
    demo()
