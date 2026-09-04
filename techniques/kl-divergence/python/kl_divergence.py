"""Kullback-Leibler divergence (Reference Sec 34.3).

Kullback & Leibler (1951).

  KL(p || q) = sum_x p(x) log(p(x) / q(x))          (discrete)
             = int p(x) log(p(x) / q(x)) dx         (continuous)

Properties:
  * KL(p || q) >= 0, = 0 iff p = q  (Gibbs inequality).
  * ASYMMETRIC: KL(p || q) != KL(q || p) in general.
  * Reverse KL:  KL(q || p) has different tail behaviour.
  * Jensen-Shannon symmetrisation:
       JS(p, q) = 0.5 KL(p || m) + 0.5 KL(q || m),  m = 0.5(p + q).
    JS is bounded in [0, log 2].

Here we compute KL, reverse KL, and JS between Gaussians analytically
+ from Monte-Carlo samples; sanity-check the closed forms.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kl_discrete(p, q, base=2, eps=1e-12):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))) / np.log(base))


def js_discrete(p, q, base=2):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_discrete(p, m, base) + 0.5 * kl_discrete(q, m, base)


def kl_gaussians(mu1, sig1, mu2, sig2):
    """KL(N(mu1, sig1^2) || N(mu2, sig2^2))  -- closed form in nats."""
    return float(np.log(sig2 / sig1) + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2) - 0.5)


def kl_mc(sampler_p, log_p, log_q, n=5000, rng=None):
    """KL(p || q) = E_p[log p - log q]. Monte-Carlo estimator."""
    rng = rng or np.random.default_rng(0)
    x = sampler_p(n, rng)
    return float(np.mean(log_p(x) - log_q(x)))


if __name__ == "__main__":
    print("=== KL divergence + reverse KL + Jensen-Shannon ===\n")
    p = [0.7, 0.2, 0.1]; q = [0.5, 0.3, 0.2]
    print(f"  KL(p || q) = {kl_discrete(p, q):.4f} bits")
    print(f"  KL(q || p) = {kl_discrete(q, p):.4f} bits   (asymmetric)")
    print(f"  JS(p, q)   = {js_discrete(p, q):.4f} bits   (symmetric, in [0, 1])\n")

    # Gaussian analytic
    print("  Gaussian KL(N(0, 1) || N(1, 2))")
    print(f"    closed form (nats): {kl_gaussians(0, 1, 1, 2):.4f}")
    rng = np.random.default_rng(0)
    kl_mc_val = kl_mc(
        sampler_p=lambda n, r: r.normal(0, 1, n),
        log_p=lambda x: -0.5 * x ** 2 - 0.5 * np.log(2 * np.pi),
        log_q=lambda x: -0.5 * ((x - 1) / 2) ** 2 - np.log(2) - 0.5 * np.log(2 * np.pi),
        n=20000, rng=rng,
    )
    print(f"    Monte-Carlo (nats): {kl_mc_val:.4f}\n")

    # Zero-forcing: KL(p || q) explodes if q(x)=0 where p(x)>0
    print("  Zero-forcing: KL(p || q) with q having zero support at a p-support point:")
    p1 = [0.5, 0.5]; q1 = [1.0, 0.0]
    print(f"    KL(p || q) = {kl_discrete(p1, q1):.4f} (should be large)\n")

    print("--- library cross-check (scipy.special.rel_entr; scipy.stats.entropy(p, q); R FNN) ---")
