"""Cost-effectiveness analysis (Reference Sec 44.16).

Drummond et al. 2015 'Methods for the Economic Evaluation of Health
Care Programmes'; Briggs, Claxton & Sculpher 2006. Compares a new
treatment vs a comparator on cost + effectiveness (QALYs / life-years
gained), producing:

    ICER = (C_new - C_ref) / (E_new - E_ref)          (deltaC / deltaE)
    NMB(lambda) = lambda * E - C                       (net monetary benefit)
    INMB(lambda) = lambda * (E_new - E_ref) - (C_new - C_ref)
    Prob(cost-effective at lambda) = P(INMB > 0)      (from bootstrap / PSA)

The COST-EFFECTIVENESS PLANE plots (deltaE, deltaC) with WTP threshold
lambda as a slope through the origin.

We simulate a trial with per-patient costs and QALYs under two arms,
compute point-estimate ICER + bootstrap CEAC (cost-effectiveness
acceptability curve).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def icer(cost_a, eff_a, cost_b, eff_b):
    dC = cost_a.mean() - cost_b.mean()
    dE = eff_a.mean() - eff_b.mean()
    return {"delta_C": float(dC), "delta_E": float(dE),
            "ICER": float(dC / dE) if abs(dE) > 1e-9 else float("nan")}


def ceac(cost_a, eff_a, cost_b, eff_b, lambdas, n_boot=1000, seed=0):
    """Cost-effectiveness acceptability curve via bootstrap."""
    rng = np.random.default_rng(seed)
    n_a = len(cost_a); n_b = len(cost_b)
    probs = np.zeros(len(lambdas))
    for _ in range(n_boot):
        ia = rng.integers(0, n_a, n_a); ib = rng.integers(0, n_b, n_b)
        dC_b = cost_a[ia].mean() - cost_b[ib].mean()
        dE_b = eff_a[ia].mean() - eff_b[ib].mean()
        for k, lam in enumerate(lambdas):
            if lam * dE_b - dC_b > 0:
                probs[k] += 1
    return probs / n_boot


if __name__ == "__main__":
    print("=== Cost-effectiveness analysis (Drummond) ===\n")
    rng = np.random.default_rng(0)
    n = 120       # smaller trial -> uncertainty visible in CEAC
    #  Ref arm cost = 5000, QALY = 3.0.
    #  New arm cost = 25000, QALY = 3.4 (ICER = 50000/QALY ballpark).
    cost_ref = 5000 + 3000 * rng.normal(size=n)
    eff_ref = 3.0 + 0.8 * rng.normal(size=n)
    cost_new = 25000 + 5000 * rng.normal(size=n)
    eff_new = 3.4 + 0.8 * rng.normal(size=n)

    r = icer(cost_new, eff_new, cost_ref, eff_ref)
    print(f"  Delta C = {r['delta_C']:.0f}   Delta E = {r['delta_E']:+.3f} QALY")
    print(f"  ICER    = {r['ICER']:.0f}  cost per QALY gained\n")

    lambdas = np.array([10000, 20000, 30000, 50000, 75000, 100000])
    probs = ceac(cost_new, eff_new, cost_ref, eff_ref, lambdas, n_boot=1000)
    print(f"  Cost-effectiveness acceptability curve:")
    for lam, p in zip(lambdas, probs):
        print(f"    WTP $/QALY = {lam:6d}   P(cost-effective) = {p:.3f}")

    print("\n  Typical thresholds: NICE UK £20-30 k/QALY; US ~ $50-100 k/QALY.")

    print("\n--- library cross-check (BCEA / heemod R; from-scratch numpy Python) ---")
