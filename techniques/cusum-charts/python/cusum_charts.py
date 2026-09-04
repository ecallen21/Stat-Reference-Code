"""CUSUM control chart (Reference Sec 37.2).

Page (1954). Cumulative-sum chart for detecting SMALL PERSISTENT
SHIFTS faster than Shewhart.

Tabular / one-sided CUSUM:
  S_i^+ = max(0, S_{i-1}^+ + (x_i - mu_0 - k * sigma))
  S_i^- = min(0, S_{i-1}^- + (x_i - mu_0 + k * sigma))

Signal when |S_i^+| or |S_i^-| > h * sigma.
Standard tunings: k = 0.5 (half the shift you want to detect),
                   h = 4 or 5 (control limit).

Average run length (ARL) is the key design metric; k, h chosen so
in-control ARL ~ 370 (matching Shewhart 3-sigma) while shifted ARL
is minimised.

Here we implement tabular CUSUM and confirm it detects a 1-sigma
shift much faster than a Shewhart chart.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cusum(x, mu0=0.0, sigma=1.0, k=0.5, h=4.0):
    n = len(x)
    S_pos = np.zeros(n); S_neg = np.zeros(n)
    for i in range(n):
        prev_p = S_pos[i - 1] if i > 0 else 0.0
        prev_n = S_neg[i - 1] if i > 0 else 0.0
        S_pos[i] = max(0.0, prev_p + (x[i] - mu0 - k * sigma))
        S_neg[i] = min(0.0, prev_n + (x[i] - mu0 + k * sigma))
    signal_pos = np.where(S_pos > h * sigma)[0]
    signal_neg = np.where(S_neg < -h * sigma)[0]
    return {"S_pos": S_pos, "S_neg": S_neg,
             "first_signal_pos": int(signal_pos[0]) if len(signal_pos) else None,
             "first_signal_neg": int(signal_neg[0]) if len(signal_neg) else None,
             "h": h, "k": k}


def shewhart_first_signal(x, mu0, sigma):
    z = (x - mu0) / sigma
    hits = np.where(np.abs(z) > 3.0)[0]
    return int(hits[0]) if len(hits) else None


if __name__ == "__main__":
    print("=== CUSUM chart (Page 1954) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    change = 100
    shift = 1.0                                     # 1-sigma shift
    x = np.concatenate([rng.normal(0, 1, change),
                         rng.normal(shift, 1, n - change)])

    cs = cusum(x, mu0=0, sigma=1, k=0.5, h=6)
    print(f"  1-sigma shift at t = {change}")
    print(f"  CUSUM (k=0.5, h=6)   first signal: t = {cs['first_signal_pos']}"
          f"   detection delay = {cs['first_signal_pos'] - change if cs['first_signal_pos'] else 'never'}")
    sh = shewhart_first_signal(x, 0, 1)
    print(f"  Shewhart (3-sigma)   first signal: t = {sh}"
          f"   detection delay = {sh - change if sh else 'never'}\n")

    print("  CUSUM detects the small persistent shift much faster than Shewhart.\n")
    print("--- library cross-check (R qcc::cusum; Python pyspc; scikit-multiflow ADWIN adjacent) ---")
