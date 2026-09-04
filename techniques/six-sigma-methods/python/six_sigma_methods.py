"""Six Sigma statistical methods (Reference Sec 37.9).

The Six Sigma quality framework combines DMAIC (define, measure,
analyse, improve, control) with several signature calculations:

  * DPMO (Defects Per Million Opportunities).
  * Sigma level = z-score at which the two-sided normal-tail equals
    the DPMO (with the 1.5-sigma 'shift' convention).
  * Yield = 1 - fraction defective.

Standard mapping (with 1.5-sigma shift):
  sigma_level     DPMO         Yield
     3            66,807       93.32%
     4            6,210        99.379%
     5            233          99.977%
     6            3.4          99.99966%
     7            0.019        99.9999981%

Here we implement DPMO <-> sigma <-> yield conversions + a compact
DMAIC checklist framework.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import norm as _norm


def dpmo_from_defects(defects, units, opportunities_per_unit):
    return defects / (units * opportunities_per_unit) * 1_000_000


def sigma_from_dpmo(dpmo, shift=1.5):
    """Motorola Six Sigma: sigma level = Phi^-1(1 - DPMO/1e6) + 1.5-sigma shift."""
    p = 1 - dpmo / 1_000_000
    p = min(max(p, 1e-12), 1 - 1e-12)
    z = float(_norm.ppf(p))
    return z + shift


def dpmo_from_sigma(sigma, shift=1.5):
    z = sigma - shift
    return (1 - _norm.cdf(z)) * 1_000_000


def yield_from_dpmo(dpmo):
    return 1 - dpmo / 1_000_000


DMAIC_CHECKLIST = {
    "Define":    ["Business case", "Voice of customer (VOC)", "Project charter", "SIPOC"],
    "Measure":   ["Data-collection plan", "Baseline capability (Cpk / DPMO)", "Measurement-system analysis (Gauge R&R)"],
    "Analyse":   ["Root-cause tools (fishbone, 5 Whys)", "Hypothesis testing", "Pareto / regression"],
    "Improve":   ["Design of experiments", "Pilot / test", "Cost-benefit"],
    "Control":   ["Control plan", "SPC charts", "Standard operating procedure", "Handover to process owner"],
}


if __name__ == "__main__":
    print("=== Six Sigma statistical methods ===\n")
    print("  Standard DPMO / sigma / yield table (with 1.5-sigma shift):")
    print(f"  {'sigma':>6}  {'DPMO':>12}  {'Yield':>12}")
    for s in (3, 4, 5, 6, 7):
        dpmo = dpmo_from_sigma(s)
        y = yield_from_dpmo(dpmo)
        print(f"  {s:>6}  {dpmo:>12,.1f}  {100*y:>11.5f}%")

    # Empirical example
    defects = 42; units = 3500; opp = 3
    dpmo = dpmo_from_defects(defects, units, opp)
    sigma = sigma_from_dpmo(dpmo)
    print(f"\n  Empirical: {defects} defects in {units} units, {opp} opportunities/unit")
    print(f"    DPMO      = {dpmo:.1f}")
    print(f"    sigma lvl = {sigma:.2f}")
    print(f"    yield     = {yield_from_dpmo(dpmo) * 100:.4f}%\n")

    print("  DMAIC checklist:")
    for phase, items in DMAIC_CHECKLIST.items():
        print(f"    {phase}:")
        for i in items:
            print(f"      - {i}")

    print("\n--- library cross-check (R SixSigma; Python sixsigma pip pkg) ---")
