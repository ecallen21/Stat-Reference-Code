"""Hodges-Lehmann Estimator (Reference §6.29).

A robust nonparametric estimator of location, paired with the rank tests.

  ONE-SAMPLE (or paired differences):
      HL = median of all  (x_i + x_j) / 2  with i <= j  (Walsh averages)
      The natural location estimator paired with the Wilcoxon signed-rank test.

  TWO-SAMPLE shift:
      HL_2 = median of all x_i - y_j  (cross differences)
      The natural shift estimator paired with Mann-Whitney U.

  CI (paired with the rank test): take the (k_alpha, n_total - k_alpha)-th order
      statistics of the Walsh / cross differences, with k_alpha derived from
      the test's critical value.

Properties
  - Median-of-pairs is much more efficient than the sample median at the normal
    (95.5% asymptotic efficiency vs. mean) yet has 29% breakdown.
  - Always works when the median does; gives a different (often more sensible)
    answer when the distribution is skewed.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def hodges_lehmann_one_sample(x: Sequence[float]) -> float:
    """One-sample HL: median of all Walsh averages (x_i + x_j) / 2, i <= j."""
    x = np.asarray(x, dtype=float); n = len(x)
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((x[i] + x[j]) / 2)
    return float(np.median(walsh))


def hodges_lehmann_two_sample(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sample HL shift: median of all (x_i - y_j) cross differences."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    diffs = np.subtract.outer(x, y).reshape(-1)
    return float(np.median(diffs))


def hodges_lehmann_ci_two_sample(x, y, conf: float = 0.95) -> dict:
    """Confidence interval for the two-sample shift, derived from Mann-Whitney critical values.

    Take the k-th smallest and the k-th largest of the n1*n2 cross differences,
    where k is chosen so that the corresponding U falls inside the central
    1 - alpha region. Uses the normal approximation for the cutoff.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    diffs = np.sort(np.subtract.outer(x, y).reshape(-1))
    mu = n1 * n2 / 2
    var = n1 * n2 * (n1 + n2 + 1) / 12
    z = stats.norm.ppf(0.5 + conf / 2)
    half = z * math.sqrt(var)
    # Indices: U = (number of x_i - y_j > some threshold); the CI of U has half-width 'half'.
    k = max(0, int(math.floor(mu - half)))
    return {"estimate": float(np.median(diffs)),
            "ci_lower": float(diffs[k]),
            "ci_upper": float(diffs[-(k + 1)]),
            "conf": conf, "method": "Hodges-Lehmann two-sample"}


def library_versions(x, y=None):
    out = {}
    if y is None:
        # one-sample
        out["scipy median of x"] = float(np.median(x))
        out["scipy.stats.hmean unrelated"] = None
    else:
        # statsmodels has Hodges-Lehmann via mannwhitneyu return etc.; we use median(outer diffs)
        out["scipy median of cross-diffs"] = float(np.median(np.subtract.outer(x, y).reshape(-1)))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    x = rng.normal(5, 2, 25).tolist()

    print("=== One-sample HL (vs sample median) ===")
    print(f"  sample median  = {np.median(x):.4f}")
    print(f"  Hodges-Lehmann = {hodges_lehmann_one_sample(x):.4f}")

    y = rng.normal(6.5, 2.5, 30).tolist()
    print("\n=== Two-sample HL shift (X - Y) ===")
    print(f"  mean(x) - mean(y) = {np.mean(x) - np.mean(y):+.4f}")
    print(f"  HL shift estimate = {hodges_lehmann_two_sample(x, y):+.4f}")

    print("\n=== 95% CI for the shift ===")
    ci = hodges_lehmann_ci_two_sample(x, y)
    for k, v in ci.items():
        print(f"  {k:14s}: {v}")
