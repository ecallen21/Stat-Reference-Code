"""Modern staggered Difference-in-Differences (Reference Sec 35.19).

Callaway & Sant'Anna (2021) 'Difference-in-differences with multiple
time periods.'

Classical two-way FE staggered DiD gives biased estimates when
treatment timing varies AND treatment effects are heterogeneous
(Goodman-Bacon 2021).

CS 2021 estimator:  compute a group-time treatment effect

  ATT(g, t)  =  E[Y_t(g) - Y_t(0) | G_i = g]

for each (treated-group g, time t) using NEVER-TREATED or NOT-YET-
TREATED units as controls; aggregate across (g, t) via user-specified
weights.

Here we implement group-time ATTs + a simple event-time aggregation
on synthetic staggered data with a known heterogeneous effect.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def group_time_att(y, unit, t, group, group_val, time_val):
    """CS ATT(g, t): E[Y_t - Y_{g-1} | G=g]  -  E[Y_t - Y_{g-1} | never-treated]."""
    # Treated group g observed at times g-1 (pre) and time_val (post)
    treated = group == group_val
    control = group == 0                             # never treated
    # Compute per-unit pre-post differences
    def diff(mask):
        u_ids = np.unique(unit[mask])
        vals = []
        for u in u_ids:
            pre = y[(unit == u) & (t == group_val - 1)]
            post = y[(unit == u) & (t == time_val)]
            if len(pre) > 0 and len(post) > 0:
                vals.append(float(post[0]) - float(pre[0]))
        return np.array(vals)
    d_t = diff(treated)
    d_c = diff(control)
    if len(d_t) == 0 or len(d_c) == 0:
        return np.nan
    return float(d_t.mean() - d_c.mean())


def event_time_aggregation(y, unit, t, group, min_e=-3, max_e=3):
    """Average ATT(g, g + e) over cohorts g for each event time e."""
    unique_groups = [g for g in np.unique(group) if g > 0]
    ests = []
    for e in range(min_e, max_e + 1):
        vals = []
        for g in unique_groups:
            t_target = g + e
            if t_target < 0 or t_target > t.max(): continue
            if t_target == g - 1: continue           # skip normalisation period
            att = group_time_att(y, unit, t, group, g, t_target)
            if not np.isnan(att): vals.append(att)
        ests.append((e, float(np.mean(vals)) if vals else np.nan))
    return ests


if __name__ == "__main__":
    print("=== Callaway-Sant'Anna staggered DiD (2021) ===\n")
    rng = np.random.default_rng(0)
    n_units = 90
    T = 10
    # Assign each unit a treatment cohort: never-treated (0), cohort at t=3, 5, 7
    groups = rng.choice([0, 3, 5, 7], size=n_units, p=[0.4, 0.2, 0.2, 0.2])
    y = np.zeros(n_units * T)
    unit = np.zeros(n_units * T, dtype=int)
    t = np.zeros(n_units * T, dtype=int)
    group_col = np.zeros(n_units * T, dtype=int)
    for i in range(n_units):
        alpha_i = rng.normal(0, 1)
        for tt in range(T):
            idx = i * T + tt
            unit[idx] = i; t[idx] = tt; group_col[idx] = groups[i]
            base = alpha_i + 0.05 * tt
            # Treatment effect: 0.3 * (post-treatment periods) for treated cohorts
            eff = 0.3 * max(0, tt - groups[i] + 1) if groups[i] > 0 else 0.0
            y[idx] = base + eff + rng.normal(0, 0.3)

    ests = event_time_aggregation(y, unit, t, group_col, min_e=-3, max_e=3)
    print(f"  {'event_time':>10}  {'ATT_hat':>8}")
    for e, att in ests:
        marker = " <- should be near 0 (pre)" if e < 0 else (" <- should be positive (post)" if e >= 0 else "")
        print(f"  {e:>10}  {att:>8.3f}{marker}")

    print("\n  Positive post-treatment ATTs recover the treatment effect;\n"
          "  pre-treatment ATTs near 0 confirm parallel trends.\n")
    print("--- library cross-check (R did (Callaway-SantAnna); Python differences pip pkg) ---")
