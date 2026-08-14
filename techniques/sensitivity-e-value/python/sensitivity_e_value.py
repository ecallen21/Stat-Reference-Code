"""Sensitivity analysis for unmeasured confounding (Reference §15.14).

VanderWeele-Ding E-value (2017): minimum RISK RATIO (on the relevant
scale) that an unmeasured confounder would need to have with BOTH the
treatment AND the outcome, above and beyond measured confounders, to
fully explain away the observed association.

Formulas (for a risk ratio RR > 1)
    E-value(estimate) = RR + sqrt(RR (RR - 1))
    E-value(CI bound closer to null):
        for RR > 1: use lower CI bound (LL)
            if LL <= 1: E-value = 1
            else: E-value = LL + sqrt(LL (LL - 1))
        for RR < 1: invert first, then compute.

Interpretation: an E-value of 3.0 means an unmeasured confounder with an
RR of 3.0 with treatment AND 3.0 with outcome could explain away the
observation.

Extensions
    - Continuous outcomes: approximate RR = exp(0.91 * std_effect)
    - Odds ratios (common outcome): approximate RR = sqrt(OR)

The demo below implements the RR case; also shows the standard equivalent
for a hazard ratio (approximates RR under rare-event assumption).

Rosenbaum bounds
    Assess how strongly an unobserved confounder would affect the odds of
    treatment for a matched pair to change the sign / significance of the
    treatment effect.  Reported as Gamma; Gamma = 2 means the odds could
    double.  See sensitivitymv / sensitivitymw / rbounds in R.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _e_from_rr(rr: float) -> float:
    if rr <= 1: rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))


def e_value_rr(rr_est: float, ci_low: float = None, ci_high: float = None) -> dict:
    """E-value for a risk ratio and the CI bound closer to the null."""
    ev = _e_from_rr(rr_est)
    ev_ci = None
    if rr_est > 1 and ci_low is not None:
        if ci_low <= 1: ev_ci = 1.0
        else: ev_ci = _e_from_rr(ci_low)
    elif rr_est < 1 and ci_high is not None:
        if ci_high >= 1: ev_ci = 1.0
        else: ev_ci = _e_from_rr(ci_high)
    return {"RR": float(rr_est),
            "E_value_point": float(ev),
            "E_value_CI_bound": float(ev_ci) if ev_ci is not None else None,
            "method": "E-value (VanderWeele-Ding)"}


def e_value_continuous(std_effect: float, sd_of_y: float) -> dict:
    """Approximate E-value for a standardized-mean-difference effect."""
    # VanderWeele-Ding 2017 approximation
    rr_approx = math.exp(0.91 * std_effect / sd_of_y) if sd_of_y > 0 else math.exp(0.91 * std_effect)
    return {"approx_RR": float(rr_approx),
            "E_value_point": float(_e_from_rr(rr_approx)),
            "method": "E-value for continuous outcome (approx via exp(0.91 * d))"}


def e_value_or(or_est: float, common_outcome: bool = True) -> dict:
    """Approximate E-value for an OR (rare-outcome -> RR ~ OR; common outcome -> RR ~ sqrt(OR))."""
    if common_outcome:
        rr_approx = math.sqrt(or_est) if or_est >= 1 else 1 / math.sqrt(1 / or_est)
    else:
        rr_approx = or_est
    return {"approx_RR": float(rr_approx),
            "E_value_point": float(_e_from_rr(rr_approx)),
            "method": f"E-value for OR ({'common' if common_outcome else 'rare'} outcome)"}


if __name__ == "__main__":
    print("=== E-value examples ===")
    # Observed RR = 2.0 with 95% CI (1.4, 2.8) - moderate association
    r = e_value_rr(rr_est=2.0, ci_low=1.4)
    print(f"  RR = 2.0 (CI lower 1.4):  E-value point = {r['E_value_point']:.3f},  E-value CI = {r['E_value_CI_bound']:.3f}")
    r = e_value_rr(rr_est=1.5, ci_low=1.05)
    print(f"  RR = 1.5 (CI lower 1.05): E-value point = {r['E_value_point']:.3f},  E-value CI = {r['E_value_CI_bound']:.3f}")
    r = e_value_rr(rr_est=0.6, ci_high=0.85)
    print(f"  RR = 0.6 (CI upper 0.85): E-value point = {r['E_value_point']:.3f},  E-value CI = {r['E_value_CI_bound']:.3f}")

    print("\n=== E-value for a standardized mean difference ===")
    r = e_value_continuous(std_effect=0.5, sd_of_y=1.0)
    print(f"  d = 0.5:  approx RR = {r['approx_RR']:.3f},  E-value = {r['E_value_point']:.3f}")

    print("\n=== E-value for an OR = 2.5 with common outcome ===")
    r = e_value_or(or_est=2.5, common_outcome=True)
    print(f"  OR = 2.5 (common outcome): approx RR = {r['approx_RR']:.3f},  E-value = {r['E_value_point']:.3f}")

    print("\n--- library cross-check (EValue::evalue in R) ---")
    print("  R: EValue::evalue(EValue::RR(2.0), lo = 1.4)")
