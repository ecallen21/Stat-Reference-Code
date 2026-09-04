"""New-user, active-comparator design (Reference Sec 43.14).

Lund et al. (2015).  Standard pharmacoepi design:

  NEW USER      -- include only patients initiating the study drug
                   for the first time (define via washout period,
                   typically 6-12 months of no prior use).
  ACTIVE COMP.  -- comparator is a DIFFERENT drug indicated for the
                   same condition, not "no drug".  This aligns
                   indication and calendar time and greatly reduces
                   confounding by indication.

Together they avoid PREVALENT-USER BIAS and CONFOUNDING BY
INDICATION.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays


def build_new_user_cohort(prescriptions, study_start, washout_days=180):
    """Keep patients whose FIRST fill on or after `study_start` has NO earlier fill
    within the previous `washout_days`.  Time zero = that first study-window fill."""
    by_patient = {}
    for p, d, t in sorted(prescriptions, key=lambda r: (r[0], r[2])):
        by_patient.setdefault(p, []).append((d, t))
    cohort = []
    for p, hist in by_patient.items():
        study_fills = [(d, t) for d, t in hist if t >= study_start]
        if not study_fills:
            continue
        first_drug, first_t = study_fills[0]
        # Any prior fill within [first_t - washout_days, first_t) disqualifies
        prior = [t for d, t in hist if first_t - washout_days <= t < first_t]
        if not prior:
            cohort.append({"patient": p, "drug": first_drug, "time_zero": first_t})
    return cohort


def cohort_summary(cohort):
    from collections import Counter
    cnt = Counter(x["drug"] for x in cohort)
    return dict(cnt)


if __name__ == "__main__":
    print("=== New-user, active-comparator design ===\n")
    rng = np.random.default_rng(0)
    # 500 patients, some are prevalent users (had earlier fills)
    prescriptions = []
    for pid in range(500):
        # 30% of patients have prior use before day 365 -> prevalent user
        if rng.random() < 0.30:
            prescriptions.append((pid, "A", rng.integers(0, 300)))
        # First study-window fill of either A or B after day 365
        drug = "A" if rng.random() < 0.6 else "B"
        prescriptions.append((pid, drug, rng.integers(400, 700)))

    cohort_all = build_new_user_cohort(prescriptions, study_start=400, washout_days=365)
    print(f"  Cohort size after study-start=day400 + 365-day washout: {len(cohort_all)}")
    print(f"  (drops prevalent users who had prior fills in the year before their first study fill)")
    print(f"  Drug distribution in new-user cohort: {cohort_summary(cohort_all)}")

    # Active-comparator restriction: keep only patients whose first-fill was A or B
    active = [x for x in cohort_all if x["drug"] in ("A", "B")]
    print(f"\n  Active-comparator cohort (A vs B new users): n = {len(active)}")
    print(f"  Distribution: {cohort_summary(active)}")

    print("\n--- library cross-check (R MatchIt/WeightIt/CohortMethod OHDSI; Python zepid/pandas) ---")
