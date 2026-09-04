"""Prescription sequence symmetry analysis (Reference Sec 43.5, 43.13).

Hallas 1996.  Within-person design: for two drugs A and B, count
how many patients started A before B vs B before A among those
who ever used both.  Under NO causal effect of A on the condition
treated by B, sequence is symmetric.  A significant asymmetry
suggests A causes an adverse effect treated by B.

  Adjusted Sequence Ratio (ASR):
    ASR = (n_AB / n_BA) / null ratio
  where the null ratio corrects for time trends in prescribing
  (e.g., if A prescribing is growing over the study period, more
  A -> B sequences will appear even under no causal effect).

Binomial test on n_AB / (n_AB + n_BA) tests the null of 0.5 (after
null-ratio correction).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def pssa(dates_A, dates_B, null_ratio=1.0):
    """dates_A, dates_B: dict {patient_id: first_prescription_date}.

    Return counts + crude and adjusted sequence ratios.
    """
    common = set(dates_A) & set(dates_B)
    n_AB = sum(1 for pid in common if dates_A[pid] < dates_B[pid])
    n_BA = sum(1 for pid in common if dates_A[pid] > dates_B[pid])
    n_total = n_AB + n_BA
    if n_total == 0:
        return {"n_AB": 0, "n_BA": 0, "crude_SR": None, "asr": None, "p_value": 1.0}
    crude_sr = n_AB / n_BA if n_BA > 0 else float("inf")
    asr = crude_sr / null_ratio
    # Binomial test on n_AB out of n_total with expected proportion
    # p_null = null_ratio / (1 + null_ratio)
    p_null = null_ratio / (1 + null_ratio)
    p_val = stats.binomtest(n_AB, n_total, p=p_null).pvalue
    return {"n_AB": n_AB, "n_BA": n_BA, "n_total": n_total,
            "crude_SR": float(crude_sr), "adjusted_SR": float(asr),
            "p_value": float(p_val)}


if __name__ == "__main__":
    print("=== Prescription sequence symmetry analysis (PSSA) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    # Suppose drug A (statin) causes muscle pain -> drug B (analgesic)
    # 70% of patients get A before B, 30% B before A among common users
    dates_A = {}; dates_B = {}
    for pid in range(n):
        a_first = rng.random() < 0.70
        t0 = rng.uniform(0, 5); gap = rng.exponential(1)
        if a_first:
            dates_A[pid] = t0; dates_B[pid] = t0 + gap
        else:
            dates_B[pid] = t0; dates_A[pid] = t0 + gap

    # Assume null_ratio = 1 (no time-trend difference)
    r = pssa(dates_A, dates_B)
    print(f"  Common users: {r['n_total']}")
    print(f"  A -> B: {r['n_AB']}   B -> A: {r['n_BA']}")
    print(f"  Crude sequence ratio     = {r['crude_SR']:.3f}")
    print(f"  Adjusted (null_ratio 1)  = {r['adjusted_SR']:.3f}")
    print(f"  Binomial p-value         = {r['p_value']:.3e}")
    print()
    if r["adjusted_SR"] > 1.2 and r["p_value"] < 0.05:
        print(f"  ASR > 1.2 AND p < 0.05 -> signal: A may precede B, suggesting")
        print(f"  A causes the condition treated by B.\n")
    print("--- library cross-check (R custom via survival/lubridate; Python custom) ---")
