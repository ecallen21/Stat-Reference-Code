"""Event-study design (Reference Sec 35.11).

Regress the outcome on LEAD and LAG treatment dummies:

  y_it = alpha_i + gamma_t + sum_{k in K} beta_k D^k_it + eps_it

where D^k_it = 1{t - g_i = k}  (event time k relative to treatment
adoption g_i).  Omit k = -1 as the reference to make coefficients
interpretable.

Pre-treatment coefficients (k <= -2) test parallel trends; post-
treatment coefficients (k >= 0) trace dynamic treatment effects.

Modern warning: with staggered adoption + heterogeneous effects,
TWFE event studies are contaminated (see staggered-did / Sun-Abraham
2021).  Here we implement the classical version + note the pitfall.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def event_study(y, unit, t, group, k_range=range(-3, 4), ref_k=-1):
    """Regress y on unit + time FE + event-time dummies (excluding ref_k)."""
    n = len(y)
    n_units = int(unit.max() + 1)
    n_periods = int(t.max() + 1)
    # Unit FE columns
    U = np.eye(n_units)[unit]
    # Time FE columns
    Tm = np.eye(n_periods)[t]
    # Event-time dummies
    ev_cols = []
    ev_labels = []
    e_arr = np.array([t[i] - group[i] if group[i] > 0 else -999 for i in range(n)])
    for k in k_range:
        if k == ref_k: continue
        ev_cols.append((e_arr == k).astype(float))
        ev_labels.append(k)
    E = np.stack(ev_cols, axis=1)
    # Drop one unit and one time column to avoid the dummy-variable trap.
    X = np.hstack([U[:, 1:], Tm[:, 1:], E, np.ones((n, 1))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ev_coefs = beta[(n_units - 1) + (n_periods - 1):-1]
    return {"labels": ev_labels, "coefs": ev_coefs}


if __name__ == "__main__":
    print("=== Event-study design ===\n")
    rng = np.random.default_rng(0)
    n_units, T = 80, 12
    groups = rng.choice([0, 4, 7], size=n_units, p=[0.4, 0.3, 0.3])
    y = np.zeros(n_units * T)
    unit = np.zeros(n_units * T, dtype=int); t = np.zeros(n_units * T, dtype=int)
    group_col = np.zeros(n_units * T, dtype=int)
    for i in range(n_units):
        alpha_i = rng.normal(0, 1)
        for tt in range(T):
            idx = i * T + tt
            unit[idx] = i; t[idx] = tt; group_col[idx] = groups[i]
            eff = 0.5 * max(0, tt - groups[i] + 1) if groups[i] > 0 else 0.0
            y[idx] = alpha_i + 0.05 * tt + eff + rng.normal(0, 0.2)

    res = event_study(y, unit, t, group_col)
    print(f"  {'event k':>7}  {'beta_hat':>9}   (true 0 before k=0, then cumulative 0.5 * (k+1))")
    for k, b in zip(res["labels"], res["coefs"]):
        true_val = 0.0 if k < 0 else 0.5 * (k + 1)
        print(f"  {k:>7}  {b:>9.3f}    truth = {true_val:>.3f}")

    print("\n  With STAGGERED adoption + heterogeneous effects, TWFE event-studies are")
    print("  known to be contaminated (Sun-Abraham 2021, Goodman-Bacon 2021):")
    print("  pre-treatment coefficients need not be zero even under parallel trends,")
    print("  and post-treatment coefficients mix effects across cohorts.")
    print("  For unbiased estimates use `staggered-did` (Callaway-Sant'Anna).\n")
    print("--- library cross-check (R fixest::feols i(event, ref = -1); Python pyfixest) ---")
