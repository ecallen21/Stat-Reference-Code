"""Shewhart control charts (Reference Sec 37.1).

Shewhart (1931). The X-bar / R (or S) chart tracks subgroup MEANS and
RANGES over time with 3-sigma control limits derived from in-control
subgroup statistics.

  X-bar chart:  UCL/LCL = X_double_bar +/- A_2 * R_bar
  R chart:      UCL = D_4 * R_bar,  LCL = D_3 * R_bar

with constants A_2, D_3, D_4 tabulated as functions of subgroup size n
(Montgomery Table VI).

Western Electric run rules flag runs / trends beyond the classical
3-sigma alarm.

Here we implement X-bar + R + Western-Electric rules on a stream that
develops a small mean shift.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


# Montgomery constants for subgroup n = 2..10
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419,
        8: 0.373, 9: 0.337, 10: 0.308}
_D3 = {2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076,
        8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.575, 4: 2.282, 5: 2.115, 6: 2.004, 7: 1.924,
        8: 1.864, 9: 1.816, 10: 1.777}


def xbar_r_chart(subgroups):
    """subgroups: (n_subgroups, n) array."""
    means = subgroups.mean(axis=1)
    ranges = subgroups.max(axis=1) - subgroups.min(axis=1)
    x_dbar = float(means.mean()); r_bar = float(ranges.mean())
    n = subgroups.shape[1]
    return {"means": means, "ranges": ranges, "x_dbar": x_dbar, "r_bar": r_bar,
             "ucl_x": x_dbar + _A2[n] * r_bar, "lcl_x": x_dbar - _A2[n] * r_bar,
             "ucl_r": _D4[n] * r_bar, "lcl_r": _D3[n] * r_bar}


def western_electric_flags(means, x_dbar, sigma_hat):
    """Detect the four standard Western Electric run rules."""
    z = (means - x_dbar) / sigma_hat
    n = len(z)
    flags = []
    for i in range(n):
        if abs(z[i]) > 3: flags.append((i, "rule 1: > 3-sigma"))
        # 2 of 3 consecutive beyond 2-sigma (same side)
        if i >= 2 and np.sum((z[i-2:i+1] > 2) | (z[i-2:i+1] < -2)) >= 2:
            side = z[i-2:i+1]
            if np.sum(side > 2) >= 2 or np.sum(side < -2) >= 2:
                flags.append((i, "rule 2: 2 of 3 > 2-sigma same side"))
        # 4 of 5 beyond 1-sigma
        if i >= 4 and (np.sum(z[i-4:i+1] > 1) >= 4 or np.sum(z[i-4:i+1] < -1) >= 4):
            flags.append((i, "rule 3: 4 of 5 > 1-sigma same side"))
        # 8 consecutive on one side
        if i >= 7 and (np.all(z[i-7:i+1] > 0) or np.all(z[i-7:i+1] < 0)):
            flags.append((i, "rule 4: 8 consecutive one side"))
    return flags


if __name__ == "__main__":
    print("=== Shewhart X-bar / R control chart ===\n")
    rng = np.random.default_rng(0)
    n_sub, n = 40, 5
    # Baseline in-control for first 25 subgroups, then 0.5-sigma shift.
    data = []
    for k in range(n_sub):
        mu = 10.0 + (1.5 if k >= 25 else 0.0)
        data.append(rng.normal(mu, 1.0, n))
    subgroups = np.stack(data)

    # Standard SPC practice: fit limits from Phase I baseline, then monitor.
    baseline = subgroups[:20]
    chart = xbar_r_chart(baseline)
    print(f"  X-bar center = {chart['x_dbar']:.3f}   UCL = {chart['ucl_x']:.3f}"
          f"   LCL = {chart['lcl_x']:.3f}")
    print(f"  R   center = {chart['r_bar']:.3f}   UCL = {chart['ucl_r']:.3f}"
          f"   LCL = {chart['lcl_r']:.3f}\n")

    # Approx sigma_x_bar from R-bar / d_2 / sqrt(n), d_2(n=5) = 2.326
    sigma_hat = chart['r_bar'] / 2.326 / np.sqrt(n)
    means_all = subgroups.mean(axis=1)
    flags = western_electric_flags(means_all, chart["x_dbar"], sigma_hat)
    print("  Western-Electric flags (subgroup index, rule):")
    for idx, rule in flags[:8]:
        print(f"    subgroup {idx:>3} (after shift = {idx - 25 >= 0}): {rule}")
    if len(flags) > 8:
        print(f"    ... and {len(flags) - 8} more.\n")

    print("--- library cross-check (R qcc; Python pyspc / statsmodels ---")
