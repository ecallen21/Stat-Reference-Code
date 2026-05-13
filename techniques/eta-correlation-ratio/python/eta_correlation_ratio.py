"""Eta correlation ratio (Reference §4.13).

The "correlation" between a CATEGORICAL grouping variable X (k levels) and a
CONTINUOUS variable Y. Defined as the proportion of variance in Y explained by
the grouping:

    eta^2 = SS_between / SS_total
    eta   = sqrt(eta^2)         in [0, 1]

Equivalently, eta is the multiple correlation between Y and the indicator
encoding of X (= the R from a one-way ANOVA on Y by X). For k = 2 groups,
eta equals |point-biserial r| (see techniques/point-biserial-correlation).

Unlike Pearson's r, eta:
  - has no sign (one variable is categorical -- no "direction")
  - captures any difference in conditional means (not just linear trend on a coded version of X)
  - equals |r| when the within-group means lie on a perfectly linear contrast and X is coded as a single continuous variable

The same eta^2 appears as the ANOVA effect size in techniques/effect-sizes (Ch 1
§1.6, §1.25); this file re-frames it as a correlation measure.

Significance test: identical to the F-test for one-way ANOVA on Y by X
    F = (eta^2 / (k - 1)) / ((1 - eta^2) / (N - k))   ~  F(k - 1, N - k)
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _mean(x): return sum(x) / len(x)


def eta_squared(y_by_group: Sequence[Sequence[float]]) -> dict:
    """Eta^2 (and eta) for a continuous Y split into k groups.

    Parameters
    ----------
    y_by_group : list of per-group y-values (one sub-sample per category of X).
    """
    all_y = [v for g in y_by_group for v in g]
    N = len(all_y); k = len(y_by_group)
    grand = _mean(all_y)
    ss_between = sum(len(g) * (_mean(g) - grand) ** 2 for g in y_by_group)
    ss_total = sum((v - grand) ** 2 for v in all_y)
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
    eta = math.sqrt(eta2)
    # F-test (same as one-way ANOVA)
    df1, df2 = k - 1, N - k
    F = (eta2 / df1) / ((1 - eta2) / df2) if 0 < eta2 < 1 else float("inf") if eta2 >= 1 else 0.0
    p = float(stats.f.sf(F, df1, df2))
    return {"eta": eta, "eta_squared": eta2,
            "F": F, "df1": df1, "df2": df2, "p_value": p,
            "k_groups": k, "N": N}


def eta_from_columns(x: Sequence, y: Sequence[float]) -> dict:
    """Convenience wrapper when the data are in two parallel columns.

    ``x`` is a vector of categorical labels (any hashable), ``y`` is continuous.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    groups = {}
    for xi, yi in zip(x, y):
        groups.setdefault(xi, []).append(yi)
    return eta_squared(list(groups.values()))


def library_versions(x, y):
    try:
        import pingouin as pg
        import pandas as pd
        df = pd.DataFrame({"x": x, "y": y})
        a = pg.anova(data=df, dv="y", between="x", detailed=True)
        return {"pingouin.anova np2 (= eta^2)":
                float(a.loc[0, "np2"]) if "np2" in a.columns else "see anova table",
                "pingouin.anova full": a}
    except ImportError:
        return {"note": "install 'pingouin' for pg.anova"}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(14)
    g1 = rng.normal(50, 10, 30).tolist()
    g2 = rng.normal(55, 10, 28).tolist()
    g3 = rng.normal(65, 10, 32).tolist()
    print("=== Eta correlation ratio: Y split by 3 groups ===")
    for k, v in eta_squared([g1, g2, g3]).items():
        print(f"  {k:14s}: {v}")

    print("\n=== Same data as parallel (x, y) columns ===")
    x = (["A"] * 30 + ["B"] * 28 + ["C"] * 32)
    y = g1 + g2 + g3
    for k, v in eta_from_columns(x, y).items():
        print(f"  {k:14s}: {v}")

    print("\n--- library ---")
    res = library_versions(x, y)
    for k, v in res.items():
        print(f"  {k}: {v}")
