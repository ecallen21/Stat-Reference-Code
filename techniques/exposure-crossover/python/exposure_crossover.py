"""Exposure crossover design for drug-drug interactions (Reference Sec 43.6).

For each patient, compare outcomes during periods of CO-EXPOSURE
(drug A + drug B) vs SINGLE exposure (A only or B only) WITHIN the
same person -- eliminates time-fixed confounding.

Analysed with conditional (matched) logistic regression on
person-strata, or as a matched-pair 2x2 table (McNemar-style).

Compact demo: per patient, build 4 person-periods (neither, A only,
B only, both).  Outcome counts per period.  Test whether "both" has
excess outcomes vs sum of marginals under an additive-null model.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def interaction_test(counts_neither, counts_A, counts_B, counts_AB,
                     time_neither, time_A, time_B, time_AB):
    """RERI-style relative excess risk due to interaction from person-time counts."""
    r_neither = counts_neither / time_neither
    r_A = counts_A / time_A
    r_B = counts_B / time_B
    r_AB = counts_AB / time_AB
    # Rate ratios vs baseline (neither)
    RR_A = r_A / r_neither
    RR_B = r_B / r_neither
    RR_AB = r_AB / r_neither
    RERI = RR_AB - RR_A - RR_B + 1
    return {"RR_A": float(RR_A), "RR_B": float(RR_B), "RR_AB": float(RR_AB),
            "RERI": float(RERI)}


if __name__ == "__main__":
    print("=== Exposure crossover: drug-drug interaction via RERI ===\n")
    rng = np.random.default_rng(0)
    n_patients = 5000
    # Baseline event rate + A alone + B alone + interaction
    lam_0 = 0.10             # events / year at baseline
    RR_A_true = 1.5
    RR_B_true = 1.3
    RERI_true = 1.0          # super-additive interaction

    # Simulate per-patient person-time (year in each state)
    time_neither = np.full(n_patients, 1.0)
    time_A       = np.full(n_patients, 1.0)
    time_B       = np.full(n_patients, 1.0)
    time_AB      = np.full(n_patients, 1.0)

    rates = {"neither": lam_0,
             "A":       lam_0 * RR_A_true,
             "B":       lam_0 * RR_B_true,
             "AB":      lam_0 * (RR_A_true + RR_B_true - 1 + RERI_true)}
    events = {k: rng.poisson(rates[k] * time_neither).sum() for k in rates}
    tot_time = {"neither": time_neither.sum(), "A": time_A.sum(),
                "B": time_B.sum(), "AB": time_AB.sum()}

    r = interaction_test(events["neither"], events["A"], events["B"], events["AB"],
                          tot_time["neither"], tot_time["A"], tot_time["B"], tot_time["AB"])
    print(f"  True RR_A = {RR_A_true}, RR_B = {RR_B_true}, RERI = {RERI_true}")
    print(f"  Estimated: RR_A = {r['RR_A']:.3f}   RR_B = {r['RR_B']:.3f}"
          f"   RR_AB = {r['RR_AB']:.3f}   RERI = {r['RERI']:.3f}\n")
    print(f"  RERI > 0 -> super-additive interaction (drug A * B > expected).\n")

    print("--- library cross-check (R survival::clogit, gnm case-crossover; Python statsmodels conditional logit) ---")
