"""Manski bounds -- partial identification (Reference Sec 15.34).

Manski (1990, 2003). When you refuse to invoke unconfoundedness /
exclusion / monotonicity, you cannot point-identify the ATE. What
you CAN do is put logical bounds on it using only the data and the
outcome's support [y_min, y_max].

For a bounded outcome Y in [y_min, y_max] and binary T:

    E[Y(1)] = E[Y|T=1] * P(T=1) + E[Y(1)|T=0] * P(T=0)

The unknown counterfactual E[Y(1)|T=0] lies in [y_min, y_max], so:

    LB(E[Y(1)]) = E[Y|T=1] * P(T=1) + y_min * P(T=0)
    UB(E[Y(1)]) = E[Y|T=1] * P(T=1) + y_max * P(T=0)

Symmetric bounds for E[Y(0)]. The ATE bound is:

    LB(ATE) = LB(E[Y(1)]) - UB(E[Y(0)])
    UB(ATE) = UB(E[Y(1)]) - LB(E[Y(0)])

Width = (y_max - y_min). No assumptions -> wide bounds. Adding MTR
(monotone treatment response: Y(1) >= Y(0)) or MTS (monotone treatment
selection: those who take T tend to have higher Y anyway) tightens.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def worst_case_bounds(y, t, y_min, y_max):
    """Manski no-assumption bounds on E[Y(1)], E[Y(0)], ATE."""
    p1 = np.mean(t == 1)
    p0 = 1 - p1
    mu1_obs = np.mean(y[t == 1]) if p1 > 0 else 0.0
    mu0_obs = np.mean(y[t == 0]) if p0 > 0 else 0.0

    LB_ey1 = mu1_obs * p1 + y_min * p0
    UB_ey1 = mu1_obs * p1 + y_max * p0
    LB_ey0 = mu0_obs * p0 + y_min * p1
    UB_ey0 = mu0_obs * p0 + y_max * p1

    return {
        "E_Y1": (LB_ey1, UB_ey1),
        "E_Y0": (LB_ey0, UB_ey0),
        "ATE": (LB_ey1 - UB_ey0, UB_ey1 - LB_ey0),
    }


def mtr_bounds(y, t, y_min, y_max):
    """Monotone treatment response: Y(1) >= Y(0) for every unit.

    Under MTR, ATE >= 0, so the lower bound becomes max(0, worst-case LB).
    Upper bound is unchanged.
    """
    w = worst_case_bounds(y, t, y_min, y_max)
    lb, ub = w["ATE"]
    return {"ATE": (max(0.0, lb), ub)}


def mts_bounds(y, t):
    """Monotone treatment selection: E[Y(t)|T=1] >= E[Y(t)|T=0].

    Under MTS, the observed treated mean over-estimates E[Y(1)|T=0], so
    E[Y(1)] <= E[Y|T=1] and E[Y(0)] >= E[Y|T=0]:
       ATE <= E[Y|T=1] - E[Y|T=0]
    (The lower bound requires combining with MTR / trimming; here just
    the upper bound.)
    """
    return {"UB_ATE_MTS": np.mean(y[t == 1]) - np.mean(y[t == 0])}


if __name__ == "__main__":
    print("=== Manski bounds -- partial identification of ATE ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    #  Outcome bounded in [0, 10]. True ATE = 2.
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-(0.8 * x)))   # confounding
    t = (rng.uniform(size=n) < e).astype(int)
    y0 = 3 + x + rng.normal(scale=1, size=n)
    y1 = y0 + 2.0
    y = np.where(t == 1, y1, y0)
    y = np.clip(y, 0, 10)
    y_min, y_max = 0.0, 10.0

    naive = np.mean(y[t == 1]) - np.mean(y[t == 0])
    print(f"  Naive OLS diff (biased)  = {naive:+.3f}   (truth 2.00)")

    w = worst_case_bounds(y, t, y_min, y_max)
    print(f"\n  Worst-case Manski (no assumptions):")
    for k, (lo, hi) in w.items():
        print(f"    {k:6s}  [{lo:+.2f}, {hi:+.2f}]   width {hi - lo:.2f}")

    m = mtr_bounds(y, t, y_min, y_max)
    print(f"\n  + MTR (Y(1) >= Y(0)):")
    print(f"    ATE   [{m['ATE'][0]:+.2f}, {m['ATE'][1]:+.2f}]  "
          f"width {m['ATE'][1] - m['ATE'][0]:.2f}")

    s = mts_bounds(y, t)
    print(f"\n  + MTS (treated select on higher Y):")
    print(f"    UB(ATE | MTS) = {s['UB_ATE_MTS']:+.3f}")

    print("\n--- library cross-check (Rchoice / SizePar R; from-scratch Python) ---")
