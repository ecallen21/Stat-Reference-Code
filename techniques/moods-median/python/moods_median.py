"""Mood's Median Test (Reference §6.8).

A simple test for equality of medians across k groups.

Algorithm
---------
1. Compute the OVERALL median M from the pooled data.
2. Build a 2 x k contingency table:
       row 1: # values > M in each group
       row 2: # values <= M in each group
3. Test independence with the chi-square test (or Fisher's exact for small n).

Less powerful than Kruskal-Wallis (which uses ranks) but very robust -- only
needs the data to be ordinal / comparable. K-W can be sensitive to scale
differences when groups have different shapes; the median test is purely about
location and is less affected.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def moods_median_test(groups: Sequence[Sequence[float]]) -> dict:
    """Mood's median test on a list of groups."""
    all_vals = [v for g in groups for v in g]
    M = float(np.median(all_vals))
    table = []
    for g in groups:
        above = sum(1 for v in g if v > M)
        below_or_eq = sum(1 for v in g if v <= M)
        table.append([above, below_or_eq])
    table = list(map(list, zip(*table)))   # transpose to 2 x k
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return {"median": M, "table_above_below": table,
            "chi_square": float(chi2), "df": int(dof), "p_value": float(p),
            "expected": expected.tolist(), "method": "Mood's median"}


def library_versions(*groups):
    res = stats.median_test(*groups)
    return {"scipy.stats.median_test (chi2, p, median, table)":
            (float(res[0]), float(res[1]), float(res[2]), res[3].tolist())}


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    a = rng.normal(50, 8, 30).tolist()
    b = rng.normal(55, 9, 28).tolist()
    c = rng.normal(60, 12, 32).tolist()

    print("=== Mood's median test ===")
    for k, v in moods_median_test([a, b, c]).items():
        print(f"  {k:18s}: {v}")

    print("\n--- library ---")
    for k, v in library_versions(a, b, c).items():
        print(f"  {k}: {v}")
