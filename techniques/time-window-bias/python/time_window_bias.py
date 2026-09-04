"""Time-window bias (Reference Sec 43.15).

Suissa & Dell'Aniello 2012 (statins & lung cancer).  In case-control
studies, different observation windows for cases and controls create
spurious associations even when the exposure has no effect.

Classical mistake: 'ever-user' status ascertained from the entire
available history for cases (dating back many years), but only from
a limited window before index date for controls.  Cases mechanically
show more 'ever use' than controls -- an artefact, not an effect.

Fix: use a COMMON LOOK-BACK WINDOW for both cases and controls
(equivalent duration + calendar-aligned).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def simulate_case_control(n_cases, n_controls, follow_up_years, prescribed_p, seed=0):
    """Simulate: cases observed since study start; controls observed only in a 1-yr window."""
    rng = np.random.default_rng(seed)
    # Everyone has the same underlying prescribing probability per year.
    case_years = rng.uniform(1, follow_up_years, n_cases)
    ctrl_years = rng.uniform(0.5, 1.5, n_controls)              # short window for controls
    case_ever = (rng.random(n_cases) < 1 - (1 - prescribed_p) ** case_years).astype(int)
    ctrl_ever = (rng.random(n_controls) < 1 - (1 - prescribed_p) ** ctrl_years).astype(int)
    return case_ever, ctrl_ever, case_years, ctrl_years


def odds_ratio(a, b, c, d):
    return float((a * d) / max(b * c, 1e-12))


if __name__ == "__main__":
    print("=== Time-window bias: unequal look-back windows create spurious OR ===\n")
    n_cases, n_controls = 500, 500
    p_per_year = 0.10   # everyone has 10% annual chance of prescription

    case_ever, ctrl_ever, cy, ky = simulate_case_control(
        n_cases, n_controls, follow_up_years=10.0, prescribed_p=p_per_year)

    # 2x2: rows = case (yes/no); cols = ever exposed (yes/no)
    a = int(case_ever.sum()); b = n_cases - a
    c = int(ctrl_ever.sum()); d = n_controls - c
    or_naive = odds_ratio(a, b, c, d)
    print(f"  Cases   (avg lookback = {cy.mean():.1f} yr): ever-user = {a}/{n_cases}")
    print(f"  Controls(avg lookback = {ky.mean():.1f} yr): ever-user = {c}/{n_controls}")
    print(f"  Naive OR (biased, different windows) = {or_naive:.2f}\n")

    # Fix: common 1-year window for both
    case_ever_common = (np.random.default_rng(1).random(n_cases) < 1 - (1 - p_per_year) ** 1.0).astype(int)
    ctrl_ever_common = (np.random.default_rng(2).random(n_controls) < 1 - (1 - p_per_year) ** 1.0).astype(int)
    a2 = int(case_ever_common.sum()); b2 = n_cases - a2
    c2 = int(ctrl_ever_common.sum()); d2 = n_controls - c2
    or_fixed = odds_ratio(a2, b2, c2, d2)
    print(f"  Same 1-year window for both -> OR = {or_fixed:.2f}   (unbiased, ~1.0)\n")

    print("  Same underlying prescribing rate; the naive OR is inflated purely by longer\n"
          "  lookback in cases.  Always use a COMMON LOOK-BACK WINDOW.\n")
    print("--- library cross-check (R EHR; Python custom + zepid) ---")
