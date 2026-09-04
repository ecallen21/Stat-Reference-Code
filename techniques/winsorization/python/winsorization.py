"""Winsorization and truncation (Reference Sec 41.5).

WINSORIZATION : replace values below the q_lo-th percentile with the
                q_lo-th percentile value; likewise for q_hi.  Sample
                size is preserved.

TRUNCATION    : REMOVE values outside the [q_lo, q_hi] percentile
                range.  Sample size is REDUCED.

Both reduce the influence of extreme observations.  Winsorization
has smaller variance than trimming at the cost of a small bias;
Wilcox recommends symmetric 10-20 % Winsorization for robust means.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def winsorize(x, lower=0.05, upper=0.05):
    x = np.asarray(x, dtype=float)
    lo = np.quantile(x, lower)
    hi = np.quantile(x, 1 - upper)
    return np.clip(x, lo, hi), lo, hi


def trim(x, lower=0.05, upper=0.05):
    x = np.asarray(x, dtype=float)
    lo = np.quantile(x, lower)
    hi = np.quantile(x, 1 - upper)
    return x[(x >= lo) & (x <= hi)], lo, hi


if __name__ == "__main__":
    print("=== Winsorization vs truncation: robust mean estimation ===\n")
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(50, 10, n)
    # Inject a few extreme outliers
    x[:5] = [500, 550, -200, -180, 700]

    print(f"  raw mean   = {x.mean():.2f}   sd = {x.std():.2f}   n = {len(x)}")
    for pct in (0.05, 0.10, 0.20):
        w, lo, hi = winsorize(x, lower=pct, upper=pct)
        t, _, _ = trim(x, lower=pct, upper=pct)
        print(f"  {int(pct * 100):>2d}%   winsor mean = {w.mean():.2f}  sd = {w.std():.2f}  n = {len(w):3d}"
              f"     trim mean = {t.mean():.2f}  sd = {t.std():.2f}  n = {len(t):3d}")

    print("\n  Winsorization preserves n at some bias cost; trimming reduces n but is unbiased.\n")
    print("--- library cross-check (R DescTools::Winsorize; Python scipy.stats.mstats.winsorize) ---")
