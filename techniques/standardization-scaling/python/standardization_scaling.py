"""Standardization, centering, and scaling (Reference Sec 41.4).

Four common scalings for numeric features:

  Z-SCORE       : (x - mean) / sd                             (StandardScaler)
  MIN-MAX       : (x - min) / (max - min)   -> [0, 1]        (MinMaxScaler)
  ROBUST        : (x - median) / IQR                          (RobustScaler)
  GELMAN /2SD   : (x - mean) / (2 * sd)     -- comparable to
                  binary predictors on the [-1, +1] scale.

Group-mean centering matters in multilevel models (Enders-Tofighi 2007):
  * Grand-mean centering keeps between+within variation confounded.
  * Group-mean centering separates within-cluster (level-1) from
    between-cluster (level-2) effects.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def robust(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return (x - med) / max(q3 - q1, 1e-12)

def gelman_2sd(x):
    return (x - x.mean()) / (2 * x.std(ddof=0))

def group_center(x, group):
    """Group-mean center (within-cluster deviations)."""
    x = np.asarray(x, dtype=float)
    means = np.array([x[group == g].mean() for g in np.unique(group)])
    lookup = {g: m for g, m in zip(np.unique(group), means)}
    return x - np.array([lookup[g] for g in group])


if __name__ == "__main__":
    print("=== Standardisation / scaling comparison ===\n")
    rng = np.random.default_rng(0)
    n = 200
    x = rng.gamma(shape=2, scale=3, size=n)          # skewed + outliers
    x = np.concatenate([x, [200, 220]])              # inject outliers

    for name, fn in [("z-score", zscore), ("min-max", minmax),
                     ("robust", robust), ("gelman /2SD", gelman_2sd)]:
        xt = fn(x)
        print(f"  {name:>14s}  mean={xt.mean():+.3f}  sd={xt.std():.3f}"
              f"  min={xt.min():+.3f}  max={xt.max():+.3f}")

    print("\n  Group-mean centering (2 clusters):")
    group = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([10, 12, 11, 20, 22, 21], dtype=float)
    xc = group_center(x, group)
    print(f"    raw    : {x}")
    print(f"    within : {xc}")
    print(f"    Between-cluster mean difference collapsed to 0 by within-centering.\n")
    print("--- library cross-check (R base::scale, recipes::step_normalize; Python sklearn scalers) ---")
