"""Measures of distribution shape: skewness and kurtosis (Reference §1.4).

Definitions
-----------
Population moments (about the mean):  m_k = E[(X - mu)^k]
  skewness  g1 = m3 / m2^(3/2)            > 0 right tail, < 0 left tail, 0 symmetric
  kurtosis  g2 = m4 / m2^2 - 3            "excess" kurtosis; normal = 0
                                          > 0 leptokurtic (heavy tails / peaked)
                                          < 0 platykurtic (light tails / flat)

Sample estimators
  - "biased" (method-of-moments) g1, g2 using the sample central moments.
  - "bias-corrected" Fisher-Pearson G1 (what most software reports) and G2:
        G1 = g1 * sqrt(n(n-1)) / (n-2)
        G2 = ((n+1) g2 + 6) * (n-1) / ((n-2)(n-3))

L-moment ratios (robust, see techniques/l-moments) are preferable for small or
heavy-tailed samples; we expose L-skewness / L-kurtosis there.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _central_moment(x, k):
    m = sum(x) / len(x)
    return sum((xi - m) ** k for xi in x) / len(x)


def skewness(x: Sequence[float], bias: bool = True) -> float:
    x = list(x)
    n = len(x)
    m2 = _central_moment(x, 2)
    m3 = _central_moment(x, 3)
    g1 = m3 / m2 ** 1.5
    if bias:
        return g1
    return g1 * (n * (n - 1)) ** 0.5 / (n - 2)


def kurtosis(x: Sequence[float], bias: bool = True, excess: bool = True) -> float:
    x = list(x)
    n = len(x)
    m2 = _central_moment(x, 2)
    m4 = _central_moment(x, 4)
    g2 = m4 / m2 ** 2 - 3.0  # excess
    if bias:
        return g2 if excess else g2 + 3.0
    G2 = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
    return G2 if excess else G2 + 3.0


def pearson_second_skewness(x: Sequence[float]) -> float:
    """Pearson's second skewness coefficient: 3 (mean - median) / sd."""
    arr = list(x)
    n = len(arr)
    mean = sum(arr) / n
    med = sorted(arr)[n // 2] if n % 2 else (sorted(arr)[n // 2 - 1] + sorted(arr)[n // 2]) / 2
    sd = (sum((v - mean) ** 2 for v in arr) / (n - 1)) ** 0.5
    return 3 * (mean - med) / sd


def library_versions(x: Sequence[float]):
    from scipy import stats

    arr = np.asarray(x, dtype=float)
    return {
        "skew biased (scipy)": float(stats.skew(arr, bias=True)),
        "skew bias-corrected G1 (scipy)": float(stats.skew(arr, bias=False)),
        "excess kurtosis biased (scipy)": float(stats.kurtosis(arr, fisher=True, bias=True)),
        "excess kurtosis bias-corrected G2 (scipy)": float(stats.kurtosis(arr, fisher=True, bias=False)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = np.round(rng.lognormal(mean=0.0, sigma=0.6, size=200), 3).tolist()  # right-skewed
    print(f"n = {len(data)}  (right-skewed lognormal sample)\n")
    print("--- from scratch ---")
    print("skewness (biased)        :", skewness(data, bias=True))
    print("skewness (G1, corrected) :", skewness(data, bias=False))
    print("excess kurtosis (biased) :", kurtosis(data, bias=True))
    print("excess kurtosis (G2)     :", kurtosis(data, bias=False))
    print("Pearson 2nd skewness     :", pearson_second_skewness(data))
    print("\n--- library ---")
    for k, v in library_versions(data).items():
        print(f"{k:42s}: {v}")
