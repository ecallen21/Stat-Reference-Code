"""Gini coefficient and Lorenz curve (Reference §1.23).

Measures of inequality / concentration in a non-negative distribution (income,
wealth, healthcare utilization, biodiversity evenness, ...).

Lorenz curve
  Sort values ascending. Plot cumulative population share p_i = i/n on the x-axis
  against cumulative value share L_i = (sum of the smallest i values) / (sum of all)
  on the y-axis. The 45-degree line is perfect equality.

Gini coefficient
  G = 2 * (area between the equality line and the Lorenz curve)
    = 1 - sum_i (p_i - p_{i-1}) * (L_i + L_{i-1})         (trapezoidal form)
  Equivalent "mean absolute difference" form for a sample x_1..x_n:
    G = sum_i sum_j |x_i - x_j| / (2 * n^2 * x_bar)
  Range: 0 (perfect equality) .. 1 (one unit holds everything).
  Small-sample bias-corrected version multiplies by n / (n - 1).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import NamedTuple, Sequence    # stdlib: type hints (Sequence = indexable iterable; NamedTuple = tuple with named fields)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


class LorenzPoints(NamedTuple):
    """Lorenz curve coordinates. Unpacks like a tuple: ``p, L = lorenz_curve(x)``."""
    population_share: np.ndarray   # cumulative fraction of units (x-axis), 0 .. 1
    value_share: np.ndarray        # cumulative fraction of total value (y-axis), 0 .. 1


def lorenz_curve(x: Sequence[float]):
    """Compute the Lorenz curve points.

    Parameters
    ----------
    x : a non-negative sample (income, wealth, utilization, etc.).

    Returns
    -------
    LorenzPoints named tuple with fields ``population_share`` (x-axis) and
    ``value_share`` (y-axis). ``population_share[i] = i / n``; ``value_share[i]``
    is the share of the total held by the smallest i units. Both arrays start at
    ``(0, 0)`` and end at ``(1, 1)``; length is ``n + 1``.
    """
    arr = np.sort(np.asarray(x, dtype=float))
    if np.any(arr < 0):
        raise ValueError("Lorenz curve / Gini require non-negative values")
    n = arr.size
    cum = np.concatenate([[0.0], np.cumsum(arr)])
    L = cum / cum[-1]
    p = np.arange(0, n + 1) / n
    return LorenzPoints(population_share=p, value_share=L)


def gini_trapezoid(x: Sequence[float]) -> float:
    """Gini coefficient from the Lorenz curve via the trapezoidal area rule. ``x`` is a non-negative sample."""
    p, L = lorenz_curve(x)
    # area under the Lorenz curve by the trapezoid rule
    # (np.trapezoid in numpy >= 2.0, np.trapz before that)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area_under = trap(L, p)
    return 1.0 - 2.0 * area_under


def gini_mean_difference(x: Sequence[float], bias_corrected: bool = False) -> float:
    """Gini from the mean-absolute-difference definition (O(n log n) sorted form).

    Parameters
    ----------
    x : non-negative sample.
    bias_corrected : if True, multiply by ``n/(n-1)`` for the small-sample correction.
    """
    arr = np.sort(np.asarray(x, dtype=float))
    if np.any(arr < 0):
        raise ValueError("Gini requires non-negative values")
    n = arr.size
    # G = (2 * sum_i i*x_(i) ) / (n * sum x) - (n + 1) / n        with i = 1..n on sorted data
    idx = np.arange(1, n + 1)
    g = (2.0 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1.0) / n
    if bias_corrected:
        g *= n / (n - 1.0)
    return float(g)


def library_versions(x: Sequence[float]):
    # No Gini in numpy/scipy core; show the canonical numpy one-liner as the "library" form.
    arr = np.sort(np.asarray(x, dtype=float))
    n = arr.size
    g_numpy = (2.0 * np.sum(np.arange(1, n + 1) * arr) / (n * np.sum(arr))) - (n + 1.0) / n
    return {"numpy canonical Gini": float(g_numpy)}


if __name__ == "__main__":
    incomes = [10, 12, 15, 18, 20, 25, 30, 40, 60, 220]  # one big earner
    print("incomes:", incomes, "\n")
    p, L = lorenz_curve(incomes)
    print("Lorenz curve points (pop share -> value share):")
    for pi, li in zip(p, L):
        print(f"  {pi:4.2f} -> {li:6.3f}")
    print()
    print("Gini (trapezoid)            :", round(gini_trapezoid(incomes), 4))
    print("Gini (mean-difference form) :", round(gini_mean_difference(incomes), 4))
    print("Gini (bias-corrected)       :", round(gini_mean_difference(incomes, bias_corrected=True), 4))

    print("\nSanity checks:")
    print("  perfectly equal [5,5,5,5]    -> Gini =", round(gini_mean_difference([5, 5, 5, 5]), 4))
    print("  maximally unequal [0,0,0,100]-> Gini =", round(gini_mean_difference([0, 0, 0, 100]), 4),
          "(approaches 1 - 1/n)")

    print("\n--- library ---")
    for k, v in library_versions(incomes).items():
        print(f"{k:28s}: {v}")
