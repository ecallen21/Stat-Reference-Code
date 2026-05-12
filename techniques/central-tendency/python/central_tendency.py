"""Measures of central tendency (Reference §1.1).

From-scratch implementations of the common "center" summaries plus the
equivalent calls from numpy / scipy. Run as a script for a worked example.

Estimators
----------
- arithmetic mean : sum(x) / n                       (sensitive to outliers)
- median          : middle order statistic           (robust)
- mode            : most frequent value(s)           (only option for nominal data)
- trimmed mean    : drop k% from each tail, average the rest
- winsorized mean : replace each tail with its boundary value, then average
- weighted mean   : sum(w*x) / sum(w)
- geometric mean  : exp(mean(log(x)))                (multiplicative processes; x > 0)
- harmonic mean   : n / sum(1/x)                     (rates/ratios; x > 0)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# From-scratch implementations
# --------------------------------------------------------------------------
def arithmetic_mean(x: Sequence[float]) -> float:
    x = list(x)
    return sum(x) / len(x)


def median(x: Sequence[float]) -> float:
    s = sorted(x)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def mode(x: Sequence) -> list:
    """All values tied for the highest frequency (a list, since the mode need not be unique)."""
    counts = Counter(x)
    top = max(counts.values())
    return [v for v, c in counts.items() if c == top]


def trimmed_mean(x: Sequence[float], proportion: float = 0.1) -> float:
    """Mean after discarding ``proportion`` of the data from *each* tail."""
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    s = sorted(x)
    n = len(s)
    k = int(math.floor(n * proportion))
    kept = s[k:n - k] if k else s
    return sum(kept) / len(kept)


def winsorized_mean(x: Sequence[float], proportion: float = 0.1) -> float:
    """Mean after clamping each tail to the value at the ``proportion`` cut point."""
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    s = sorted(x)
    n = len(s)
    k = int(math.floor(n * proportion))
    if k:
        lo, hi = s[k], s[n - k - 1]
        s = [lo] * k + s[k:n - k] + [hi] * k
    return sum(s) / n


def weighted_mean(x: Sequence[float], w: Sequence[float]) -> float:
    if len(x) != len(w):
        raise ValueError("x and w must have the same length")
    num = sum(xi * wi for xi, wi in zip(x, w))
    den = sum(w)
    if den == 0:
        raise ValueError("weights sum to zero")
    return num / den


def geometric_mean(x: Sequence[float]) -> float:
    x = list(x)
    if any(v <= 0 for v in x):
        raise ValueError("geometric mean requires strictly positive values")
    return math.exp(sum(math.log(v) for v in x) / len(x))


def harmonic_mean(x: Sequence[float]) -> float:
    x = list(x)
    if any(v <= 0 for v in x):
        raise ValueError("harmonic mean requires strictly positive values")
    return len(x) / sum(1.0 / v for v in x)


# --------------------------------------------------------------------------
# Library equivalents (what you'd typically use in practice)
# --------------------------------------------------------------------------
def library_versions(x: Sequence[float]):
    from scipy import stats

    arr = np.asarray(x, dtype=float)
    return {
        "mean (numpy)": float(np.mean(arr)),
        "median (numpy)": float(np.median(arr)),
        "mode (scipy)": stats.mode(arr, keepdims=False).mode,
        "trimmed_mean 20% (scipy)": float(stats.trim_mean(arr, 0.2)),
        "geometric_mean (scipy)": float(stats.gmean(arr)),
        "harmonic_mean (scipy)": float(stats.hmean(arr)),
        "weighted_mean (numpy)": float(np.average(arr, weights=np.arange(1, len(arr) + 1))),
    }


if __name__ == "__main__":
    data = [2, 4, 4, 4, 5, 5, 7, 9, 100]  # note the outlier at the end
    print("data:", data)
    print()
    print("--- from scratch ---")
    print("arithmetic mean :", arithmetic_mean(data))
    print("median          :", median(data))
    print("mode            :", mode(data))
    print("trimmed mean 20%:", trimmed_mean(data, 0.2))
    print("winsor. mean 20%:", winsorized_mean(data, 0.2))
    print("weighted mean   :", weighted_mean(data, w=range(1, len(data) + 1)))
    print("geometric mean  :", geometric_mean(data))
    print("harmonic mean   :", harmonic_mean(data))
    print()
    print("--- library ---")
    for k, v in library_versions(data).items():
        print(f"{k:28s}: {v}")
