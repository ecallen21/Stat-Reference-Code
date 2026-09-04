"""Tidy data + long/wide reshaping (Reference Sec 41.15).

Wickham 2014 tidy data rules:
  1. Each variable is a column.
  2. Each observation is a row.
  3. Each type of observational unit is a table.

Reshape operations:
  * WIDE  -> LONG (melt / pivot_longer): one row per measurement,
    add id columns identifying which variable that row measures.
  * LONG -> WIDE (pivot / pivot_wider): one row per unit, one column
    per variable value.

Long form is preferred for ggplot2, lme4, tidymodels; wide form is
often more compact and matches classical tabular layouts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def melt(rows, id_cols, value_cols, var_name="variable", value_name="value"):
    """Wide -> long."""
    out = []
    for r in rows:
        for c in value_cols:
            out.append({**{k: r[k] for k in id_cols}, var_name: c, value_name: r[c]})
    return out


def pivot(rows, index, columns, values):
    """Long -> wide."""
    seen_idx = []
    seen_cols = []
    for r in rows:
        i = r[index]; c = r[columns]
        if i not in seen_idx: seen_idx.append(i)
        if c not in seen_cols: seen_cols.append(c)
    result = {i: {index: i} for i in seen_idx}
    for r in rows:
        result[r[index]][r[columns]] = r[values]
    return [result[i] for i in seen_idx]


if __name__ == "__main__":
    print("=== Tidy data: wide -> long -> wide ===\n")
    wide = [
        {"patient": "A", "week1": 120, "week2": 118, "week3": 115},
        {"patient": "B", "week1": 130, "week2": 125, "week3": 120},
        {"patient": "C", "week1": 128, "week2": 130, "week3": 127},
    ]
    print("  Wide form:")
    for r in wide: print(f"    {r}")

    long_ = melt(wide, id_cols=["patient"], value_cols=["week1", "week2", "week3"],
                 var_name="week", value_name="SBP")
    print("\n  Long form (melted):")
    for r in long_[:6]: print(f"    {r}")
    print(f"    ...  (total {len(long_)} rows)")

    wide_back = pivot(long_, index="patient", columns="week", values="SBP")
    print("\n  Back to wide:")
    for r in wide_back: print(f"    {r}")

    print("\n--- library cross-check (R tidyr pivot_longer/pivot_wider, data.table melt/dcast; Python pandas melt/pivot) ---")
