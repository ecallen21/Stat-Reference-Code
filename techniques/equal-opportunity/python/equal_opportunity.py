"""Equal opportunity (Reference Ch 31 Fairness).

Hardt, Price & Srebro (2016) 'Equality of Opportunity in Supervised
Learning' -- the WEAKER cousin of equalized odds:

  P( Y_hat = 1 | Y = 1, A = a )    equal for every group a.

Only TRUE POSITIVE RATE (recall / sensitivity) must be equal; false-
positive rates are allowed to differ.

Rationale: 'giving qualified individuals a fair chance' is often the
harm we most want to eliminate (loan approval, hiring, medical
treatment). Enforcing equal FPR on top can be overkill when the harm
is asymmetric.

Headline summaries:

  EOpp difference = max_a TPR_a - min_a TPR_a
  EOpp ratio      = min_a TPR_a / max_a TPR_a

Enforced at deployment via GROUP-SPECIFIC THRESHOLDS that hit the same
TPR (Hardt 2016).

Here we compute per-group TPR + EOpp summaries, then find group-specific
thresholds that equalise TPR at a target level.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def per_group_tpr(y_true, y_hat, groups):
    out = {}
    for a in np.unique(groups):
        m = (groups == a) & (y_true == 1)
        out[int(a)] = float((y_hat[m] == 1).mean()) if m.any() else float("nan")
    return out


def equal_opportunity_diff(y_true, y_hat, groups):
    t = list(per_group_tpr(y_true, y_hat, groups).values())
    return max(t) - min(t)


def equal_opportunity_ratio(y_true, y_hat, groups):
    t = list(per_group_tpr(y_true, y_hat, groups).values())
    return min(t) / max(t) if max(t) > 0 else float("nan")


def group_thresholds_for_tpr(scores, y_true, groups, target_tpr):
    """Group-specific threshold s.t. TPR_a = target_tpr for each group."""
    thresholds = {}
    for a in np.unique(groups):
        m = (groups == a) & (y_true == 1)
        s = scores[m]
        thresholds[int(a)] = float(np.quantile(s, 1 - target_tpr))
    return thresholds


def apply_group_thresholds(scores, groups, thresholds):
    y_hat = np.zeros_like(scores, dtype=int)
    for a, t in thresholds.items():
        m = groups == a
        y_hat[m] = (scores[m] >= t).astype(int)
    return y_hat


if __name__ == "__main__":
    print("=== Equal opportunity (Hardt 2016) ===\n")
    rng = np.random.default_rng(0)
    n_per = 600
    y0 = (rng.random(n_per) < 0.45).astype(int)
    y1 = (rng.random(n_per) < 0.20).astype(int)
    s0 = y0 + rng.normal(0, 0.5, n_per)
    s1 = y1 + rng.normal(0, 1.0, n_per)
    scores = np.concatenate([s0, s1])
    y = np.concatenate([y0, y1])
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    # 1. Single fixed threshold
    for t in (0.3, 0.5, 0.7):
        y_hat = (scores >= t).astype(int)
        tpr = per_group_tpr(y, y_hat, groups)
        print(f"  single thr={t:.2f}   TPR_0={tpr[0]:.3f}   TPR_1={tpr[1]:.3f}"
              f"   diff={equal_opportunity_diff(y, y_hat, groups):.3f}"
              f"   ratio={equal_opportunity_ratio(y, y_hat, groups):.3f}")
    print()

    # 2. Group-specific thresholds hitting target TPR
    for target in (0.6, 0.75, 0.9):
        thr = group_thresholds_for_tpr(scores, y, groups, target)
        y_hat = apply_group_thresholds(scores, groups, thr)
        tpr = per_group_tpr(y, y_hat, groups)
        print(f"  target TPR={target:.2f}   thr_0={thr[0]:.3f}   thr_1={thr[1]:.3f}"
              f"   TPR_0={tpr[0]:.3f}   TPR_1={tpr[1]:.3f}"
              f"   diff={equal_opportunity_diff(y, y_hat, groups):.3f}")

    print("\n  Group-specific thresholds achieve equal TPR by construction.")
    print("  FPRs are NOT constrained -- they typically diverge.\n")
    print("--- library cross-check (fairlearn.metrics.true_positive_rate_difference; aif360 EqOpp) ---")
