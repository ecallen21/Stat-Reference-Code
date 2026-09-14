"""Cauchy distribution — heavy-tailed pathological case.

PDF: f(x; x0, gamma) = 1 / (pi gamma (1 + ((x - x0)/gamma)^2))

- No mean, no variance (infinite).
- Sample mean does NOT converge (violates LLN); use MEDIAN.
- Ratio of two independent N(0, 1) is Cauchy.

Foundational counterexample and stress test for robust
estimators.
"""

import numpy as np    # arrays + random


def sample_cauchy(x0, gamma, n, rng):
    """Inverse-CDF sampling: x = x0 + gamma * tan(pi (U - 0.5))."""
    u = rng.uniform(size=n)
    return x0 + gamma * np.tan(np.pi * (u - 0.5))


def cauchy_pdf(x, x0, gamma):
    return 1 / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))


def demo():
    print("=== Cauchy distribution (heavy-tailed) ===")
    rng = np.random.default_rng(2026)
    x0_true, gamma_true = 2.0, 1.5
    x = sample_cauchy(x0_true, gamma_true, n=10000, rng=rng)

    # Sample mean is NOT a good estimator; use median + IQR-based scale.
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    gamma_est = iqr / 2    # IQR of Cauchy = 2 gamma

    print(f"  true x0 = {x0_true}, gamma = {gamma_true}")
    print(f"  Sample mean       = {np.mean(x):+.3f}  (NOT a valid estimator; explodes)")
    print(f"  Median            = {med:+.3f}         (target {x0_true})")
    print(f"  Sample variance   = {np.var(x):.2e}   (theoretical: infinite)")
    print(f"  IQR-based gamma   = {gamma_est:.3f}    (target {gamma_true})")

    # LLN violation: sample mean doesn't stabilise
    running_mean = np.cumsum(x) / np.arange(1, len(x) + 1)
    print(f"\n  Running-mean range (n=100..10000): "
          f"[{running_mean[100:].min():.2f}, {running_mean[100:].max():.2f}]"
          "  (Gaussian would converge to +2 within +/- 0.1)")

    print("\nSee also: robust-location-scale, huber-m-estimator,")
    print("          tukey-biweight-m-estimator, hodges-lehmann,")
    print("          theil-sen-slope, quantile-regression, extreme-value-theory.")


if __name__ == "__main__":
    demo()
