"""Drug utilization + adherence measures (Reference Sec 43.7).

Two standard adherence metrics from pharmacy dispensing records:

  MPR  (Medication Possession Ratio):
        sum(days_supplied) / (observation_window_days)
        Historically simplest; can exceed 1.

  PDC  (Proportion of Days Covered):
        # days with any supply on hand / observation_window_days
        Preferred (CMS Star Ratings) because it caps at 1 and handles
        overlap correctly.

DDD  = defined daily dose (WHO ATC/DDD system).  Total quantity
       dispensed / DDD -> equivalent days of therapy on a defined
       standard dose.

Persistence = time from initiation to first gap > threshold
  (e.g., 30 days).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mpr(fills, obs_days):
    """fills: list of (start_day, days_supplied)."""
    total = sum(d for _, d in fills)
    return float(total / obs_days)


def pdc(fills, obs_days):
    covered = np.zeros(obs_days, dtype=bool)
    for start, days in fills:
        s = max(0, start); e = min(obs_days, start + days)
        covered[s:e] = True
    return float(covered.sum() / obs_days)


def persistence(fills, obs_days, gap_threshold=30):
    """Days to first gap > threshold; obs_days if never discontinued."""
    events = sorted(fills)
    covered = np.zeros(obs_days, dtype=bool)
    for start, days in events:
        s = max(0, start); e = min(obs_days, start + days)
        covered[s:e] = True
    idx = np.where(~covered)[0]
    if len(idx) == 0:
        return obs_days
    # find first run of >= gap_threshold uncovered days
    run = 0; start_gap = None
    for i in range(len(covered)):
        if not covered[i]:
            if run == 0: start_gap = i
            run += 1
            if run >= gap_threshold:
                return start_gap
        else:
            run = 0
    return obs_days


if __name__ == "__main__":
    print("=== Drug utilization: MPR, PDC, persistence ===\n")
    obs = 365
    # Perfect adherence: monthly refills 30 days each
    perfect = [(30 * i, 30) for i in range(12)]
    # Sporadic: overlaps + gaps
    sporadic = [(0, 30), (25, 30), (100, 30), (200, 60), (330, 30)]
    for name, fills in [("perfect (12 monthly refills)", perfect),
                        ("sporadic (5 refills, gaps)", sporadic)]:
        print(f"  {name}")
        print(f"    MPR  = {mpr(fills, obs):.3f}")
        print(f"    PDC  = {pdc(fills, obs):.3f}")
        print(f"    Persistence (gap>30d, days) = {persistence(fills, obs, 30)}")
        print()

    ddd = 20      # mg of the WHO DDD for this drug
    dispensed_total = 1200   # mg total across all fills
    print(f"  DDD example: total mg dispensed = {dispensed_total}, DDD = {ddd} mg/day"
          f"   -> {dispensed_total / ddd:.1f} days of therapy on defined dose.\n")
    print("--- library cross-check (R AdhereR::CMA; Python custom + pandas) ---")
