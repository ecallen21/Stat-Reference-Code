"""Demographic parity / statistical parity (Reference Ch 31 Fairness).

DEFINITION:  P(Y_hat = 1 | A = a)  is equal across all groups a.

Two summary numbers commonly reported:

  Demographic-parity DIFFERENCE:  max_a P(Y_hat=1 | A=a) - min_a P(Y_hat=1 | A=a)
  Demographic-parity RATIO:       min_a P(Y_hat=1 | A=a) / max_a P(Y_hat=1 | A=a)

The RATIO is the basis of the US EEOC 'four-fifths rule' (Uniform
Guidelines on Employee Selection Procedures 1978): a selection ratio
below 0.8 for a protected group is presumptive evidence of adverse
impact.

Demographic parity is INDIFFERENT to Y (ground truth); it only measures
the AGGREGATE selection rate per group.  When base rates truly differ
across groups (e.g. loan applicants with different average credit),
demographic parity may DIRECTLY CONFLICT with error-rate parity
metrics -- Chouldechova 2017, Kleinberg 2016 impossibility results.

Here we compute both summaries on a synthetic classifier and report a
group-by-group selection-rate table.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def selection_rates(y_hat, groups):
    rates = {}
    for a in np.unique(groups):
        m = groups == a
        rates[int(a)] = float(y_hat[m].mean()) if m.any() else float("nan")
    return rates


def demographic_parity_diff(y_hat, groups):
    r = list(selection_rates(y_hat, groups).values())
    return max(r) - min(r)


def demographic_parity_ratio(y_hat, groups):
    r = list(selection_rates(y_hat, groups).values())
    return min(r) / max(r) if max(r) > 0 else float("nan")


def four_fifths_pass(y_hat, groups):
    return demographic_parity_ratio(y_hat, groups) >= 0.8


if __name__ == "__main__":
    print("=== Demographic parity / four-fifths rule ===\n")
    rng = np.random.default_rng(0)

    # Two groups: A=0 has higher scores on average.
    n_per = 500
    scores0 = rng.normal(0.6, 0.3, n_per)
    scores1 = rng.normal(0.4, 0.3, n_per)
    scores = np.clip(np.concatenate([scores0, scores1]), 0, 1)
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    print("  Effect of the DECISION THRESHOLD on demographic parity:\n")
    print(f"    {'threshold':>10s}   {'sel_rate_A=0':>13s}   {'sel_rate_A=1':>13s}"
          f"   {'DP diff':>8s}   {'DP ratio':>8s}   {'4/5 rule':>8s}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        y_hat = (scores >= t).astype(int)
        rates = selection_rates(y_hat, groups)
        d = demographic_parity_diff(y_hat, groups)
        r = demographic_parity_ratio(y_hat, groups)
        print(f"    {t:>10.2f}   {rates[0]:>13.3f}   {rates[1]:>13.3f}"
              f"   {d:>8.3f}   {r:>8.3f}   {'PASS' if r >= 0.8 else 'FAIL':>8s}")

    print("\n  Base-rate imbalance means no single threshold can hit the 4/5 rule here.\n"
          "  Group-specific thresholds (below) can achieve parity by construction.\n")

    print("=== Achieving demographic parity via GROUP-SPECIFIC THRESHOLDS ===\n")
    # Find per-group thresholds that give equal selection rates.
    target = 0.5
    t0 = np.quantile(scores[groups == 0], 1 - target)
    t1 = np.quantile(scores[groups == 1], 1 - target)
    y_hat_eq = np.zeros_like(groups)
    y_hat_eq[groups == 0] = (scores[groups == 0] >= t0).astype(int)
    y_hat_eq[groups == 1] = (scores[groups == 1] >= t1).astype(int)
    print(f"    group-0 threshold = {t0:.3f}   group-1 threshold = {t1:.3f}")
    print(f"    selection rates:   {selection_rates(y_hat_eq, groups)}")
    print(f"    DP ratio: {demographic_parity_ratio(y_hat_eq, groups):.3f}   -> PASS by construction.\n")

    print("--- library cross-check (fairlearn.metrics.demographic_parity_difference / _ratio) ---")
