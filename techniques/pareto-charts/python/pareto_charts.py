"""Pareto charts (Reference Sec 37.14).

Juran popularised the 80/20 rule: for many quality problems ~80 % of
the effect is driven by ~20 % of causes.  A Pareto chart sorts causes
by frequency (or cost) descending and overlays the cumulative percent
line so the "vital few" are visible at a glance.

Uses:
  * Prioritise defect-reduction effort.
  * Root-cause analysis after 5-Whys / Fishbone.
  * Focus DOE factors on the vital few.

Here we produce the tabular Pareto (sorted counts + cumulative %) and
identify the k-smallest set of categories that jointly cover >= 80 %.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pareto_table(categories, counts, threshold=80.0):
    """Return a sorted Pareto table (desc) + which categories cover the threshold."""
    order = np.argsort(-np.asarray(counts, dtype=float))
    cats = [categories[i] for i in order]
    cnts = np.asarray(counts, dtype=float)[order]
    total = cnts.sum()
    pct = 100.0 * cnts / max(total, 1e-9)
    cum = np.cumsum(pct)
    vital = int(np.searchsorted(cum, threshold) + 1)
    rows = [{"category": c, "count": float(n), "pct": float(p), "cum_pct": float(cp)}
            for c, n, p, cp in zip(cats, cnts, pct, cum)]
    return {"rows": rows, "vital_few": cats[:vital], "vital_count": vital, "total": float(total)}


if __name__ == "__main__":
    print("=== Pareto chart: sorted defects + cumulative percent ===\n")
    categories = ["Solder", "Wire", "Coating", "Alignment", "Marking",
                  "Contamination", "Warp", "Missing part", "Wrong colour", "Other"]
    counts = [143, 87, 62, 41, 24, 17, 12, 9, 4, 2]
    res = pareto_table(categories, counts, threshold=80.0)
    print(f"  Total defects: {int(res['total'])}")
    print(f"  {'Category':<15} {'Count':>7} {'Pct':>7} {'CumPct':>8}")
    for r in res["rows"]:
        print(f"  {r['category']:<15} {int(r['count']):>7d} {r['pct']:>6.1f}% {r['cum_pct']:>7.1f}%")
    print(f"\n  Vital few (>= 80 % of total): {res['vital_count']} categories -> "
          f"{res['vital_few']}")
    print("  --> focus improvement work here.\n")
    print("--- library cross-check (R qcc::pareto.chart; Python custom) ---")
