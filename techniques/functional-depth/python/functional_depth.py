"""Functional depth + outlier detection (Reference Sec 31.9).

Lopez-Pintado & Romo (2009) 'On the concept of depth for functional
data' (band + modified band depth).

DEPTH ranks each curve by how CENTRAL it lies relative to the sample:
median-like curves get depth ~ 1; extreme curves get depth ~ 0.

MODIFIED BAND DEPTH (MBD):
  For every pair (x_a, x_b) in the sample, the 'band' is
    B(t; a, b) = [min(x_a(t), x_b(t)),  max(x_a(t), x_b(t))].
  MBD(x_i) = C(n, 2)^{-1} sum_{a<b} (1/T) sum_t I(x_i(t) in B(t; a, b)).

Depth-based FUNCTIONAL BOXPLOT: shade central 50% band (curves with
top-50% depth); flag curves outside 1.5x band as outliers.

Here we compute MBD on synthetic curves with one clear outlier and
verify it gets the lowest depth.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def modified_band_depth(X):
    """MBD (Lopez-Pintado - Romo, order-2 bands)."""
    n, T = X.shape
    depth = np.zeros(n)
    total_pairs = 0
    # Fraction of t where x_i lies inside the [min, max] envelope of (a, b).
    for a in range(n):
        for b in range(a + 1, n):
            lo = np.minimum(X[a], X[b])
            hi = np.maximum(X[a], X[b])
            inside = (X >= lo[None]) & (X <= hi[None])   # (n, T)
            depth += inside.mean(axis=1)
            total_pairs += 1
    return depth / total_pairs


def functional_boxplot_flags(depth, factor=1.5):
    """Simple outlier flag: curves whose depth is below (median - factor * IQR)."""
    med = float(np.median(depth))
    q1, q3 = np.quantile(depth, [0.25, 0.75])
    iqr = q3 - q1
    lo = med - factor * iqr
    return depth < lo, lo


if __name__ == "__main__":
    print("=== Modified band depth + functional outlier flagging ===\n")
    rng = np.random.default_rng(0)
    T = 60
    t = np.linspace(0, 1, T)
    n = 30
    # 29 "normal" curves + 1 outlier (huge amplitude).
    X = np.array([np.sin(2 * np.pi * t) + 0.2 * rng.normal(0, 1, T) for _ in range(n - 1)])
    X_out = 5 * np.sin(2 * np.pi * t) + 0.2 * rng.normal(0, 1, T)
    X = np.vstack([X, X_out[None]])

    depth = modified_band_depth(X)
    order = np.argsort(-depth)
    print(f"  top-3 deepest curves: {order[:3].tolist()}   depths {depth[order[:3]].round(3).tolist()}")
    print(f"  bottom-3 shallowest:  {order[-3:].tolist()}  depths {depth[order[-3:]].round(3).tolist()}")
    print(f"  the injected outlier is curve #{n - 1}; its depth rank = {int(np.where(order == n-1)[0][0]) + 1}/{n}")

    flags, lo = functional_boxplot_flags(depth, factor=1.5)
    n_flags = int(flags.sum())
    print(f"\n  functional-boxplot outliers flagged: {n_flags}"
          f"   flagged curve indices: {np.where(flags)[0].tolist()}   (threshold depth < {lo:.3f})\n")

    print("--- library cross-check (R fda.usc::depth.mode; roahd::MBD) ---")
