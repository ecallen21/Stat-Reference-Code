"""Credible intervals and ROPE (Reference §14.9, §14.23).

A CREDIBLE INTERVAL is a range containing (1 - alpha) posterior probability
mass.  Two common definitions:

Equal-tail interval (ETI)
    (theta_L, theta_U) with Pr(theta < theta_L) = Pr(theta > theta_U) = alpha / 2.
    Easy to compute from posterior draws (quantile function).

Highest Posterior Density (HPD)
    The shortest interval containing (1 - alpha) posterior mass.  For a
    skewed posterior HPD < ETI in width and is contained inside the mode.
    Sensitive to reparameterization -- (theta, log theta) HPDs differ.

ROPE (Region of Practical Equivalence, Kruschke 2018)
    A pre-specified range around a null value (e.g. |theta| < 0.05) that is
    considered "practically zero" for the decision at hand.
    Decision rule:
        - if 95% HDI (=HPD) is INSIDE the ROPE   -> accept the null value
        - if 95% HDI is OUTSIDE the ROPE          -> reject the null value
        - otherwise                                 -> withhold judgment
    Avoids the "arbitrary point-null" problem of classical hypothesis tests.

The HPD computation below sorts the draws and finds the shortest window
covering the target mass -- exact for unimodal posteriors; for multimodal
posteriors report the highest-density REGION as a union of intervals.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def equal_tail_interval(samples, alpha: float = 0.05) -> tuple:
    """Equal-tail (quantile-based) credible interval."""
    samples = np.asarray(samples, dtype=float)
    return tuple(np.quantile(samples, [alpha / 2, 1 - alpha / 2]))


def hpd_interval(samples, alpha: float = 0.05) -> tuple:
    """Highest-posterior-density interval assuming unimodal posterior."""
    samples = np.sort(np.asarray(samples, dtype=float))
    n = len(samples); k = int(math.floor((1 - alpha) * n))
    widths = samples[k:] - samples[:n - k]
    i = int(np.argmin(widths))
    return float(samples[i]), float(samples[i + k])


def rope_decision(samples, rope: tuple, cred: float = 0.95) -> dict:
    """ROPE decision on a posterior sample.

    rope : (lo, hi) tuple defining the practically-equivalent region.
    """
    hdi_lo, hdi_hi = hpd_interval(samples, alpha=1 - cred)
    r_lo, r_hi = rope
    if hdi_lo > r_hi or hdi_hi < r_lo:
        decision = "reject null (HDI outside ROPE)"
    elif hdi_lo >= r_lo and hdi_hi <= r_hi:
        decision = "accept null (HDI inside ROPE)"
    else:
        decision = "withhold judgment (HDI overlaps ROPE)"
    frac_in_rope = float(((samples >= r_lo) & (samples <= r_hi)).mean())
    return {"hdi": (hdi_lo, hdi_hi),
            "eti": equal_tail_interval(samples, alpha=1 - cred),
            "rope": rope,
            "fraction_in_rope": frac_in_rope,
            "decision": decision,
            "cred_level": cred,
            "method": "Bayesian ROPE / HDI decision (Kruschke)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Symmetric (Normal) posterior: ETI and HPD should agree ===")
    x = rng.normal(0.4, 0.15, 5000)
    print(f"  ETI: {equal_tail_interval(x)}")
    print(f"  HPD: {hpd_interval(x)}")

    print("\n=== Skewed (log-normal) posterior: HPD narrower than ETI ===")
    x = rng.lognormal(1.0, 0.7, 5000)
    e = equal_tail_interval(x); h = hpd_interval(x)
    print(f"  ETI: ({e[0]:.3f}, {e[1]:.3f}) width = {e[1] - e[0]:.3f}")
    print(f"  HPD: ({h[0]:.3f}, {h[1]:.3f}) width = {h[1] - h[0]:.3f}")

    print("\n=== ROPE decision rule on three effect sizes ===")
    # Same ROPE (-0.05, 0.05); simulate three posteriors
    scenarios = {
        "clearly positive effect":  rng.normal(0.30, 0.05, 5000),
        "null-ish effect":          rng.normal(0.01, 0.03, 5000),
        "ambiguous":                rng.normal(0.05, 0.08, 5000),
    }
    for name, samples in scenarios.items():
        r = rope_decision(samples, rope=(-0.05, 0.05), cred=0.95)
        print(f"  {name}:")
        print(f"    HDI = ({r['hdi'][0]:.3f}, {r['hdi'][1]:.3f}), % in ROPE = {r['fraction_in_rope']*100:.1f}%")
        print(f"    decision: {r['decision']}")
