"""3+3 Dose Escalation (Reference Sec 47.262).

Classical rule-based phase-I dose-finding (Storer 1989 Biom;
attribution unclear, "3+3" folklore). Cohort of 3 patients per
dose:

    if 0/3 dose-limiting toxicities (DLTs): escalate one level
    if 1/3 DLTs: add 3 more; if 1/6 total, escalate; if >= 2/6, STOP
    if >= 2/3 DLTs: STOP

MTD is defined as the highest dose with fewer than 2/6 DLTs.
Simple, transparent, well-understood by regulators - but wasteful
of information and slow. CRM / BOIN are model-based alternatives.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def three_plus_three(p_true, rng):
    """Simulate one 3+3 trial; return (recommended_dose, n_treated, n_dlt_per_dose)."""
    n_doses = len(p_true)
    n_treated = np.zeros(n_doses, dtype=int)
    n_dlt = np.zeros(n_doses, dtype=int)
    d = 0                                                          # start at dose 1
    while True:
        # First cohort of 3
        dlt3 = int(rng.binomial(3, p_true[d]))
        n_treated[d] += 3; n_dlt[d] += dlt3
        if dlt3 == 0:
            if d == n_doses - 1: return d, n_treated, n_dlt
            d += 1
        elif dlt3 == 1:
            # Add 3 more
            dlt3b = int(rng.binomial(3, p_true[d]))
            n_treated[d] += 3; n_dlt[d] += dlt3b
            total_dlt = dlt3 + dlt3b
            if total_dlt >= 2:
                # MTD is previous dose (or none if d == 0)
                return d - 1, n_treated, n_dlt
            else:
                if d == n_doses - 1: return d, n_treated, n_dlt
                d += 1
        else:                                                     # dlt3 >= 2
            return d - 1, n_treated, n_dlt


if __name__ == "__main__":
    print("=== 3+3 Dose Escalation (Storer 1989; folklore) ===\n")
    rng = np.random.default_rng(0)

    p_true = [0.02, 0.08, 0.18, 0.40, 0.65]
    p_target = 0.25
    print(f"  True tox: {p_true}, target rate = {p_target}")
    print(f"  (True MTD by target = dose 3, tox 0.18 - closest below target)\n")

    # Single-trial run
    d, n_t, n_d = three_plus_three(p_true, rng)
    print(f"  Single run - recommended MTD = dose {d + 1}")
    print(f"    n_treated per dose: {n_t.tolist()}")
    print(f"    n_DLT     per dose: {n_d.tolist()}\n")

    # 1000-trial simulation
    n_sim = 1000
    picks = np.zeros(len(p_true) + 1, dtype=int)                   # +1 for "no dose"
    total_n = 0
    for _ in range(n_sim):
        d, n_t, _ = three_plus_three(p_true, rng)
        picks[d + 1] += 1                                          # d = -1 -> "no dose"
        total_n += n_t.sum()

    print(f"  {n_sim} simulated trials:")
    print(f"    {'dose':>5}  {'true tox':>10}  {'% picked':>10}")
    print(f"    {'none':>5}  {'-':>10}  {picks[0] / n_sim * 100:>9.1f}%")
    for i in range(len(p_true)):
        print(f"    {i + 1:>5}  {p_true[i]:>10.2f}  {picks[i + 1] / n_sim * 100:>9.1f}%")
    print(f"\n    mean patients per trial: {total_n / n_sim:.1f}")

    print("\n  3+3 tends to under-shoot the true MTD; CRM / BOIN concentrate")
    print("  cohorts closer to the target rate and use fewer patients on average.")

    print("\n--- library cross-check (dfcrm R; bcrm R; UBCRM Python) ---")
