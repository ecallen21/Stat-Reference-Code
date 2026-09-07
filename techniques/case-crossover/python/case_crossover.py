"""Case-crossover study (Reference Sec 15.36).

Maclure 1991 'The case-crossover design: a method for studying transient
effects on the risk of acute events', AJE. Each case serves as its own
control -- exposure during a short 'hazard period' immediately before
the event is compared with exposure during 'control periods' from the
same person at different times.

Key features:
  * Only cases are used; controls come from within.
  * All time-invariant confounders (genetics, chronic disease, SES) are
    intrinsically controlled -- they cancel within-subject.
  * Answers: is transient exposure associated with acute event onset?

Analysis via conditional logistic regression / Mantel-Haenszel odds
ratio stratified by subject. When there is 1 case period and M control
periods per case, and exposure is binary:

    OR_MH = sum(a * d / T) / sum(b * c / T)

with 2x2 tables (a, b, c, d) per stratum. For M=1 this simplifies to
the McNemar-style discordant-pair ratio.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def case_crossover_mh(case_exposure, control_exposure_avg):
    """Mantel-Haenszel OR for case-crossover with unequal-length controls.

    case_exposure       : 0/1 exposure in hazard window (length n cases)
    control_exposure_avg: mean exposure across control windows per case (in [0, 1])
    """
    n = len(case_exposure)
    #  Build 2x2 tables per case: within-subject
    #    exposed case-period: a = 1 if case_exp=1 else 0
    #    unexposed case-period: c = 1 - a
    #    exposed control-period fraction: p
    #    unexposed control fraction: 1 - p
    #  For each stratum size T=2 (one case-period + one 'summary' control):
    #    a = case_exp, b = 1 - case_exp
    #    c = p,        d = 1 - p
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        a = case_exposure[i]
        b = 1 - a
        c = control_exposure_avg[i]
        d = 1 - c
        T = 2.0
        numerator += a * d / T
        denominator += b * c / T
    return numerator / denominator if denominator > 0 else np.inf


def case_crossover_discordant(case_exp, ctrl_exp_binary):
    """Classical 1:1 matched McNemar ratio.

    ctrl_exp_binary : 0/1 in the (single) control window.
    Returns OR_hat = f_10 / f_01  (discordant-pair ratio).
    """
    #  Only discordant pairs contribute.
    f10 = np.sum((case_exp == 1) & (ctrl_exp_binary == 0))
    f01 = np.sum((case_exp == 0) & (ctrl_exp_binary == 1))
    return f10 / f01 if f01 > 0 else np.inf, f10, f01


if __name__ == "__main__":
    print("=== Case-crossover: transient exposure -> acute event ===\n")
    rng = np.random.default_rng(0)
    n_cases = 800

    #  True instantaneous OR for exposure = 2.5.
    #  Baseline exposure prevalence = 0.15 across all windows.
    #  Given event, hazard-window exposure is elevated to reflect OR.
    p_ctrl = 0.15
    OR_true = 2.5
    #  For control periods, sample independently.
    ctrl_exp = (rng.uniform(size=n_cases) < p_ctrl).astype(int)
    #  For hazard period, using logistic model with case-conditional prevalence
    #  P(exp=1 | case) = OR*p / (OR*p + (1-p))
    p_haz = OR_true * p_ctrl / (OR_true * p_ctrl + (1 - p_ctrl))
    case_exp = (rng.uniform(size=n_cases) < p_haz).astype(int)

    OR_disc, f10, f01 = case_crossover_discordant(case_exp, ctrl_exp)
    print(f"  1:1 matched design (McNemar-style):")
    print(f"    discordant pairs  10-only: {f10}, 01-only: {f01}")
    print(f"    OR_hat = {OR_disc:.3f}     (truth {OR_true:.2f})")

    #  With 3 control windows per case -> average control exposure per case
    M = 3
    ctrl_multi = (rng.uniform(size=(n_cases, M)) < p_ctrl).mean(axis=1)
    OR_mh = case_crossover_mh(case_exp, ctrl_multi)
    print(f"\n  1:{M} matched design (Mantel-Haenszel):")
    print(f"    OR_MH  = {OR_mh:.3f}     (truth {OR_true:.2f})")

    print("\n--- library cross-check (survival::clogit R; statsmodels ConditionalLogit Python) ---")
