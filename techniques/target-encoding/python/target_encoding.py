"""Target / mean encoding (Reference Sec 41.11).

Micci-Barreca (2001).  For a high-cardinality categorical feature x
and a target y, replace each level with a shrunken estimate of
E[y | x = level]:

  enc(level) = (n_level * ybar_level + k * ybar_overall) / (n_level + k)

The smoothing constant k pulls low-count levels toward the grand
mean.  Leave-one-out variants further reduce leakage.  Weight of
Evidence (WOE) is the binary-target log-odds analogue.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def target_encode(x, y, k=10.0):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    y_mean = y.mean()
    enc = {}
    for lv in np.unique(x):
        m = x == lv
        n_l = m.sum()
        y_l = y[m].mean() if n_l > 0 else y_mean
        enc[lv] = (n_l * y_l + k * y_mean) / (n_l + k)
    return np.array([enc[v] for v in x]), enc


def loo_target_encode(x, y):
    """Leave-one-out target encoding (unshrunken)."""
    x = np.asarray(x); y = np.asarray(y, dtype=float)
    sum_by = {lv: y[x == lv].sum() for lv in np.unique(x)}
    cnt_by = {lv: (x == lv).sum() for lv in np.unique(x)}
    out = np.zeros(len(x))
    for i, (xi, yi) in enumerate(zip(x, y)):
        s = sum_by[xi] - yi
        c = cnt_by[xi] - 1
        out[i] = s / c if c > 0 else y.mean()
    return out


def woe_encode(x, y):
    """Weight of Evidence for binary y."""
    y = np.asarray(y); x = np.asarray(x)
    pos = y.sum(); neg = len(y) - pos
    enc = {}
    for lv in np.unique(x):
        m = x == lv
        p = max(y[m].sum() / pos, 1e-6)
        q = max((1 - y[m]).sum() / neg, 1e-6)
        enc[lv] = np.log(p / q)
    return np.array([enc[v] for v in x]), enc


if __name__ == "__main__":
    print("=== Target encoding + LOO + WOE ===\n")
    rng = np.random.default_rng(0)
    x = np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 20 + ["D"] * 8 + ["E"] * 2)
    y = rng.normal(0, 1, len(x)) + np.array([2.0, 1.0, 0.0, -1.0, -1.5])[np.array([0]*40 + [1]*30 + [2]*20 + [3]*8 + [4]*2)]

    xe, enc = target_encode(x, y, k=10)
    print(f"  Smoothed target encoding (k=10):")
    for lv, v in enc.items():
        print(f"    {lv}: {v:+.3f}   (mean(y|x={lv}) = {y[x == lv].mean():+.3f}, n = {(x == lv).sum()})")

    xe_loo = loo_target_encode(x, y)
    print(f"\n  LOO encoding first 5 rows: {np.round(xe_loo[:5], 3)}\n")

    y_bin = (y > 0).astype(int)
    xe_woe, woe = woe_encode(x, y_bin)
    print(f"  WOE (binary y):")
    for lv, v in woe.items():
        print(f"    {lv}: WOE = {v:+.3f}")

    print("\n--- library cross-check (R vtreat/recipes step_lencode_*; Python category_encoders) ---")
