"""Rejection sampling (Reference Sec 47.45).

von Neumann 1951 'Various techniques used in connection with random
digits'. To sample from density f(x) known up to a constant, pick a
proposal g(x) and constant M such that f(x) <= M * g(x) for all x.
Iterate:

    1. Draw x ~ g.
    2. Accept with probability f(x) / (M * g(x)); else reject.

Accepted x has density f. Expected acceptance rate = 1 / M (using
normalised f). Efficient only when M is close to sup f/g.

Adaptive rejection sampling (ARS; Gilks & Wild 1992) refines the
envelope online for log-concave densities.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rejection_sample(target_pdf, proposal_sampler, proposal_pdf, M, n, rng=None):
    """Draw n samples from target_pdf using rejection with envelope M*proposal_pdf."""
    rng = rng or np.random.default_rng(0)
    out = []
    tries = 0
    while len(out) < n:
        batch = max(100, int(1.5 * M * (n - len(out))))
        x = proposal_sampler(batch, rng)
        u = rng.uniform(size=batch)
        acc = u < target_pdf(x) / (M * proposal_pdf(x))
        out.extend(x[acc].tolist())
        tries += batch
    x = np.array(out[:n])
    return {"samples": x, "acceptance": n / tries, "expected_acc": 1.0 / M}


if __name__ == "__main__":
    print("=== Rejection sampling (von Neumann 1951) ===\n")

    # Target: truncated standard normal on [0, 3] (up to normalising constant)
    def target(x):
        return np.where((x >= 0) & (x <= 3), np.exp(-0.5 * x ** 2), 0.0)

    # Proposal: exponential(1) restricted, unnormalised density exp(-x) on x>=0
    def proposal_pdf(x):
        return np.where(x >= 0, np.exp(-x), 0.0)

    def proposal_sampler(n, rng):
        return rng.exponential(size=n)

    # Bound sup f(x)/g(x) on [0,3]: max exp(-x^2/2)/exp(-x) = max exp(x - x^2/2)
    # derivative zero at x=1; value = exp(0.5). Include truncation safety.
    M = float(np.exp(0.5))

    n = 20_000
    res = rejection_sample(target, proposal_sampler, proposal_pdf, M, n)
    print(f"  Envelope constant M = {M:.4f}")
    print(f"  Observed acceptance rate = {res['acceptance']:.3f}")
    print(f"  Theoretical (unnormalised f) 1/M = {res['expected_acc']:.3f}")

    # Compare sample moments to truncated N(0,1) on [0,3]
    from scipy.stats import truncnorm    # analytic truncated normal
    a, b = 0, 3
    tn = truncnorm(a, b, loc=0, scale=1)
    print(f"\n  Sample mean = {res['samples'].mean():.4f}   truth = {tn.mean():.4f}")
    print(f"  Sample var  = {res['samples'].var():.4f}   truth = {tn.var():.4f}")
    print(f"  KS-stat vs truth = "
          f"{__import__('scipy.stats', fromlist=['ks_1samp']).ks_1samp(res['samples'], tn.cdf).statistic:.4f}")

    print("\n--- library cross-check (rejsampling / ars R; scipy.stats Python) ---")
