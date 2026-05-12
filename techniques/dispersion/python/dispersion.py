"""Measures of dispersion / spread (Reference §1.2).

From-scratch implementations plus numpy / scipy equivalents.

Estimators
----------
- range              : max - min
- variance           : sum((x - mean)^2) / (n - ddof)   (ddof=1 -> sample variance)
- standard deviation : sqrt(variance)
- IQR                : Q3 - Q1
- coefficient of var : sd / mean              (unit-free; see techniques/coefficient-of-variation)
- mean abs deviation : mean(|x - center|)     (center = mean or median)
- median abs dev MAD : median(|x - median|)   (robust; *1.4826 to estimate sigma under normality)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def value_range(x: Sequence[float]) -> float:
    return max(x) - min(x)


def variance(x: Sequence[float], ddof: int = 1) -> float:
    x = list(x)
    n = len(x)
    if n - ddof <= 0:
        raise ValueError("not enough data for the requested ddof")
    m = sum(x) / n
    return sum((xi - m) ** 2 for xi in x) / (n - ddof)


def stddev(x: Sequence[float], ddof: int = 1) -> float:
    return variance(x, ddof) ** 0.5


def quantile(x: Sequence[float], p: float) -> float:
    """Linear-interpolation quantile (numpy's default / Hyndman-Fan type 7)."""
    s = sorted(x)
    n = len(s)
    if n == 1:
        return float(s[0])
    h = (n - 1) * p
    lo = int(h)
    frac = h - lo
    hi = min(lo + 1, n - 1)
    return s[lo] + frac * (s[hi] - s[lo])


def iqr(x: Sequence[float]) -> float:
    return quantile(x, 0.75) - quantile(x, 0.25)


def coefficient_of_variation(x: Sequence[float], ddof: int = 1) -> float:
    m = sum(x) / len(x)
    if m == 0:
        raise ValueError("CV is undefined when the mean is zero")
    return stddev(x, ddof) / m


def mean_abs_deviation(x: Sequence[float], center: str = "mean") -> float:
    x = list(x)
    c = (sum(x) / len(x)) if center == "mean" else _median(x)
    return sum(abs(xi - c) for xi in x) / len(x)


def median_abs_deviation(x: Sequence[float], scale: float = 1.4826) -> float:
    """MAD; multiply by 1.4826 for a consistent estimator of sigma under normality."""
    x = list(x)
    med = _median(x)
    return _median([abs(xi - med) for xi in x]) * scale


def _median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def library_versions(x: Sequence[float]):
    from scipy import stats

    arr = np.asarray(x, dtype=float)
    return {
        "range (np.ptp)": float(np.ptp(arr)),
        "variance ddof=1 (np.var)": float(np.var(arr, ddof=1)),
        "std ddof=1 (np.std)": float(np.std(arr, ddof=1)),
        "IQR (scipy.stats.iqr)": float(stats.iqr(arr)),
        "CV (scipy.stats.variation, ddof=1)": float(stats.variation(arr, ddof=1)),
        "MAD scaled (scipy.stats.median_abs_deviation)": float(
            stats.median_abs_deviation(arr, scale="normal")
        ),
    }


if __name__ == "__main__":
    data = [4, 8, 6, 5, 3, 9, 7, 11, 6, 100]
    print("data:", data, "\n")
    print("--- from scratch ---")
    print("range            :", value_range(data))
    print("variance (n-1)   :", variance(data))
    print("std dev  (n-1)   :", stddev(data))
    print("IQR              :", iqr(data))
    print("CV               :", coefficient_of_variation(data))
    print("MAD about mean   :", mean_abs_deviation(data, "mean"))
    print("MAD about median :", mean_abs_deviation(data, "median"))
    print("median abs dev   :", median_abs_deviation(data))
    print("\n--- library ---")
    for k, v in library_versions(data).items():
        print(f"{k:46s}: {v}")
