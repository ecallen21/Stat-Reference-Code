"""Importance sampling (Reference Sec 45.6).

Estimate E_p[f(X)] using draws from a DIFFERENT proposal q(x):

    E_p[f(X)] = E_q[ f(X) * p(X)/q(X) ]  ~= (1/n) sum f(x_i) w(x_i)
    where w(x_i) = p(x_i) / q(x_i).

Self-normalised (SNIS): use  sum(f * w) / sum(w)  when p is known
only up to a normalising constant.

Effective sample size (Kong 1992):
    ESS = (sum w)^2 / sum(w^2).

Diagnoses proposal quality; ESS << n means the proposal is a poor
fit to the target.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm


def importance_sample(f, log_p, log_q, sampler_q, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = sampler_q(rng, n)
    log_w = log_p(x) - log_q(x)
    w = np.exp(log_w - log_w.max())        # normalise for stability
    ess = float((w.sum() ** 2) / (w ** 2).sum())
    est = float((f(x) * w).sum() / w.sum())
    return {"estimate": est, "ESS": ess, "n": n}


if __name__ == "__main__":
    print("=== Importance sampling: E_p[f(X)] with proposal q ===\n")
    # Target p = N(0, 1); f = X^4 (true E = 3).  Proposal q = N(0, 2).
    log_p = lambda x: norm.logpdf(x, 0, 1)
    log_q = lambda x: norm.logpdf(x, 0, 2)
    sampler_q = lambda rng, n: rng.normal(0, 2, n)

    for n in (1000, 10_000, 100_000):
        r = importance_sample(lambda x: x ** 4, log_p, log_q, sampler_q, n=n)
        print(f"  n = {n:>7d}   estimate = {r['estimate']:.4f}   ESS = {r['ESS']:.0f}   (true = 3)")

    # Bad proposal: q too narrow
    print("\n  Bad proposal q = N(0, 0.5) (too narrow):")
    log_q2 = lambda x: norm.logpdf(x, 0, 0.5)
    sampler_q2 = lambda rng, n: rng.normal(0, 0.5, n)
    r_bad = importance_sample(lambda x: x ** 4, log_p, log_q2, sampler_q2, n=10_000)
    print(f"    estimate = {r_bad['estimate']:.4f}   ESS = {r_bad['ESS']:.0f}   (proposal ignores tail mass)\n")

    print("--- library cross-check (R custom + rstan; Python numpy + scipy + arviz) ---")
