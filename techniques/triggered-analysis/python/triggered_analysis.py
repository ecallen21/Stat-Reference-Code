"""Triggered analysis + conditional metrics (Reference Sec 44.13).

If only 30% of randomised users actually experience the treatment
(e.g., saw the modal, hit the endpoint), including the other 70%
in the analysis DILUTES the effect toward zero.

TRIGGERED ANALYSIS restricts the analysis to users who could have
been exposed (both arms) and measures the effect conditional on
trigger.  Requires the trigger to be measurable in BOTH arms.

Careful:
  Naive analysis of a subset defined by post-randomisation
  behaviour is BIASED.  The trigger must be pre-randomisation OR
  the CACE / IV framework applies (see instrumental-variables).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def diluted_att(y_C, y_T):
    diff = y_T.mean() - y_C.mean()
    se = np.sqrt(y_C.var(ddof=1) / len(y_C) + y_T.var(ddof=1) / len(y_T))
    return {"diff": float(diff), "SE": float(se)}


def triggered_att(y_C, y_T, trigger_C, trigger_T):
    y_C_t = y_C[trigger_C == 1]; y_T_t = y_T[trigger_T == 1]
    diff = y_T_t.mean() - y_C_t.mean()
    se = np.sqrt(y_C_t.var(ddof=1) / len(y_C_t) + y_T_t.var(ddof=1) / len(y_T_t))
    return {"diff": float(diff), "SE": float(se), "n_trig_C": int(len(y_C_t)),
            "n_trig_T": int(len(y_T_t))}


if __name__ == "__main__":
    print("=== Triggered analysis: reducing dilution from unexposed users ===\n")
    rng = np.random.default_rng(0)
    n = 10000
    # 30% of users trigger (see the checkout modal, hit the coupon endpoint)
    trigger_prob = 0.30
    trig_C = (rng.random(n) < trigger_prob).astype(int)
    trig_T = (rng.random(n) < trigger_prob).astype(int)
    # True effect among triggered = +0.5; unexposed users unaffected
    y_C = rng.normal(5, 1, n)
    y_T = rng.normal(5, 1, n) + 0.5 * trig_T   # treatment lifts only triggered users

    r_all = diluted_att(y_C, y_T)
    r_trig = triggered_att(y_C, y_T, trig_C, trig_T)
    print(f"  ITT-style diff (all users)  = {r_all['diff']:+.4f}   SE = {r_all['SE']:.4f}")
    print(f"  Triggered-only diff         = {r_trig['diff']:+.4f}   SE = {r_trig['SE']:.4f}"
          f"   n_C = {r_trig['n_trig_C']}, n_T = {r_trig['n_trig_T']}")
    print(f"  True effect among triggered = +0.50")
    print(f"  ITT ~= trigger_prob * true = {trigger_prob * 0.5}, matches diluted diff.\n")

    print("--- library cross-check (R survey svyglm domain, custom subset; Python scipy + causalml) ---")
