"""Calibration parity / predictive parity (Reference Ch 31 Fairness).

Chouldechova (2017) 'Fair Prediction with Disparate Impact' -- the
key ProPublica-COMPAS-debate metric.

DEFINITIONS:

  PREDICTIVE PARITY (Chouldechova):  P(Y=1 | Y_hat=1, A=a)  equal across groups.
     -- positive-predictive-value parity at the ADOPTED THRESHOLD.

  CALIBRATION BY GROUP (Kleinberg 2016):  P(Y=1 | Y_hat=s, A=a) = s  for all s.
     -- the SCORE itself is a well-calibrated probability WITHIN each group.

Calibration parity is compatible with equalized odds only if the group
BASE RATES are equal (Chouldechova / Kleinberg impossibility).

Here we compute:
  1. Per-group PPV at fixed thresholds
     -> predictive parity difference.
  2. Per-group RELIABILITY DIAGRAM bins
     -> per-group ECE + calibration-by-group diagnostic.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def per_group_ppv(y_true, y_hat, groups):
    out = {}
    for a in np.unique(groups):
        m = (groups == a) & (y_hat == 1)
        out[int(a)] = float((y_true[m] == 1).mean()) if m.any() else float("nan")
    return out


def predictive_parity_diff(y_true, y_hat, groups):
    v = list(per_group_ppv(y_true, y_hat, groups).values())
    return max(v) - min(v)


def per_group_calibration(y_true, scores, groups, n_bins=5):
    """Return {group: list of (bin_mid, mean_score, mean_y, n) rows}."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = {}
    for a in np.unique(groups):
        m = groups == a
        s_a = scores[m]; y_a = y_true[m]
        rows = []
        for b in range(n_bins):
            in_bin = (s_a > edges[b]) & (s_a <= edges[b + 1])
            if in_bin.any():
                rows.append(((edges[b] + edges[b + 1]) / 2,
                              float(s_a[in_bin].mean()),
                              float(y_a[in_bin].mean()),
                              int(in_bin.sum())))
        out[int(a)] = rows
    return out


def per_group_ece(y_true, scores, groups, n_bins=10):
    ece_out = {}
    for a in np.unique(groups):
        m = groups == a
        s_a = scores[m]; y_a = y_true[m]
        n = m.sum()
        edges = np.linspace(0, 1, n_bins + 1)
        e = 0.0
        for b in range(n_bins):
            in_bin = (s_a > edges[b]) & (s_a <= edges[b + 1])
            if in_bin.any():
                e += in_bin.sum() / n * abs(s_a[in_bin].mean() - y_a[in_bin].mean())
        ece_out[int(a)] = float(e)
    return ece_out


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    print("=== Calibration / predictive parity (Chouldechova 2017) ===\n")
    rng = np.random.default_rng(0)
    n_per = 800
    # Group 0 base rate 0.45; group 1 base rate 0.20.
    # Well-calibrated by construction: draw scores from Beta with different
    # mean per group, then y_i | s_i ~ Bernoulli(s_i).  Different SCORE
    # DISTRIBUTIONS across groups will drive base-rate asymmetry and
    # eventually predictive-parity failure at fixed thresholds.
    p0 = rng.beta(4, 3, n_per)       # mean ~ 0.57
    p1 = rng.beta(2, 5, n_per)       # mean ~ 0.29
    y0 = (rng.random(n_per) < p0).astype(int)
    y1 = (rng.random(n_per) < p1).astype(int)
    scores = np.concatenate([p0, p1])
    y = np.concatenate([y0, y1])
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    print("  Predictive-parity at fixed thresholds (P(Y=1|Y_hat=1, A)):\n")
    print(f"    {'thr':>4}  {'PPV_0':>6}  {'PPV_1':>6}  {'PP diff':>7}")
    for t in (0.3, 0.5, 0.7):
        y_hat = (scores >= t).astype(int)
        ppv = per_group_ppv(y, y_hat, groups)
        print(f"    {t:>4.2f}  {ppv[0]:>6.3f}  {ppv[1]:>6.3f}"
              f"  {predictive_parity_diff(y, y_hat, groups):>7.3f}")

    print("\n  Reliability diagram bins per group (5 bins):\n")
    cal = per_group_calibration(y, scores, groups, n_bins=5)
    for a, rows in cal.items():
        print(f"    Group {a}:")
        for mid, s_mean, y_mean, n in rows:
            print(f"      bin center {mid:.2f}   mean_score={s_mean:.3f}"
                  f"   mean_y={y_mean:.3f}   n={n}")

    ece = per_group_ece(y, scores, groups)
    print(f"\n  Per-group ECE: {ece}     <- both low = calibration-by-group holds.")
    print("  Yet predictive parity FAILS at any fixed threshold -- the impossibility.\n")

    print("--- library cross-check (fairlearn.metrics.selection_rate + calibration_curve; aif360 Bin) ---")
