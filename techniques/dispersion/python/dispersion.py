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
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def value_range(x: Sequence[float]) -> float:
    """Range = max(x) - min(x).  ``x`` is a sample."""
    return max(x) - min(x)


def variance(x: Sequence[float], ddof: int = 1) -> float:
    """Sample variance = sum((x - x_bar)^2) / (n - ddof).

    Parameters
    ----------
    x : sample.
    ddof : "delta degrees of freedom" -- divisor is ``n - ddof``. ``1`` (default) gives
        the unbiased sample variance; ``0`` gives the population variance.
    """
    x = list(x)
    n = len(x)
    if n - ddof <= 0:
        raise ValueError("not enough data for the requested ddof")
    m = sum(x) / n
    return sum((xi - m) ** 2 for xi in x) / (n - ddof)


def stddev(x: Sequence[float], ddof: int = 1) -> float:
    """Standard deviation = sqrt(variance(x, ddof)). See ``variance`` for ``ddof``."""
    return variance(x, ddof) ** 0.5


def quantile(x: Sequence[float], p: float) -> float:
    """Linear-interpolation quantile (numpy's default / Hyndman-Fan type 7).

    Parameters
    ----------
    x : sample.
    p : probability in ``[0, 1]`` (e.g. ``0.5`` for the median, ``0.25`` for Q1).
    """
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
    """Interquartile range = Q3 - Q1. Robust spread of the middle 50%. ``x`` is a sample."""
    return quantile(x, 0.75) - quantile(x, 0.25)


def coefficient_of_variation(x: Sequence[float], ddof: int = 1) -> float:
    """CV = SD(x) / mean(x). Unit-free relative spread; undefined when the mean is zero.

    ``ddof`` is passed through to ``stddev``.
    """
    m = sum(x) / len(x)
    if m == 0:
        raise ValueError("CV is undefined when the mean is zero")
    return stddev(x, ddof) / m


def mean_abs_deviation(x: Sequence[float], center: str = "mean") -> float:
    """Mean absolute deviation about ``center`` ("mean" or "median"). ``x`` is a sample."""
    x = list(x)
    c = (sum(x) / len(x)) if center == "mean" else _median(x)
    return sum(abs(xi - c) for xi in x) / len(x)


def median_abs_deviation(x: Sequence[float], scale: float = 1.4826) -> float:
    """Median absolute deviation = median(|x - median(x)|) * scale.

    Parameters
    ----------
    x : sample.
    scale : multiplier; ``1.4826`` makes the MAD a consistent estimator of sigma
        under normality. Use ``1.0`` for the raw MAD.
    """
    x = list(x)
    med = _median(x)
    return _median([abs(xi - med) for xi in x]) * scale


def _median(x):
    """Internal helper -- median without re-importing other modules."""
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def library_versions(x: Sequence[float]):
    """numpy / scipy implementations of the same measures, for cross-checking."""
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
