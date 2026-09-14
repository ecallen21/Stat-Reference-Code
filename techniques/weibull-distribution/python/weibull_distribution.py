"""Weibull distribution — foundational (Weibull 1939, 1951).

PDF: f(x; k, lam) = (k/lam) (x/lam)^{k-1} exp(-(x/lam)^k),  x >= 0.

Shape k < 1  -> decreasing hazard    (infant mortality)
Shape k = 1  -> constant hazard      (exponential)
Shape k > 1  -> increasing hazard    (wear-out)

MLE via log-likelihood on shape k (closed-form given k for lam).
"""

import numpy as np    # arrays + random + linalg


def sample_weibull(k, lam, n, rng):
    """Inverse-CDF sampling: x = lam * (-log(1 - U))^(1/k)."""
    u = rng.uniform(size=n)
    return lam * (-np.log1p(-u)) ** (1 / k)


def weibull_pdf(x, k, lam):
    x = np.asarray(x)
    return (k / lam) * (x / lam) ** (k - 1) * np.exp(-((x / lam) ** k))


def mle_weibull(x, tol=1e-8, max_iter=100):
    """Newton on the profile log-likelihood in k, then closed-form lam."""
    log_x = np.log(x)
    k = 1.0
    for _ in range(max_iter):
        x_k = x ** k
        num = np.sum(x_k * log_x)
        den = np.sum(x_k)
        g = 1 / k + np.mean(log_x) - num / den
        # derivative
        num2 = np.sum(x_k * log_x ** 2)
        h = -1 / k ** 2 - (num2 * den - num ** 2) / den ** 2
        step = g / h
        k = k - step
        if abs(step) < tol:
            break
    lam = (np.mean(x ** k)) ** (1 / k)
    return k, lam


def demo():
    print("=== Weibull distribution (Weibull 1939, 1951) ===")
    rng = np.random.default_rng(2026)
    for k_true, lam_true in [(0.7, 2.0), (1.5, 3.0), (3.0, 4.0)]:
        x = sample_weibull(k_true, lam_true, n=1000, rng=rng)
        k_hat, lam_hat = mle_weibull(x)
        print(f"  true k={k_true}, lam={lam_true}: MLE k={k_hat:.3f}, lam={lam_hat:.3f}")
    print("\nSee also: parametric-survival, extreme-value-theory, buckley-james-aft,")
    print("          accelerated-failure-time, deep-survival-network.")


if __name__ == "__main__":
    demo()
