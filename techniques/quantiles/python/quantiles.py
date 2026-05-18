"""Sample quantiles, percentiles and order statistics (Reference §1.5).

There is no single "the" sample quantile -- Hyndman & Fan (1996) catalogue nine
definitions. The two you meet most often:

  - Type 7  : default in R's quantile(), numpy, pandas. h = (n-1)p, linear interp.
  - Type 6  : "Weibull"; used by Minitab/SPSS. h = (n+1)p, linear interp.
  - Type 1  : inverse of the empirical CDF (no interpolation); a step function.

We implement those three plus the five-number summary, percentile ranks, and the
empirical CDF helper. (More on the ECDF in techniques/ecdf.)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def quantile(x: Sequence[float], p: float, kind: int = 7) -> float:
    """Sample quantile at probability ``p`` using Hyndman-Fan type ``kind`` (1, 6, or 7)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    s = sorted(x)
    n = len(s)
    if n == 0:
        raise ValueError("empty input")
    if n == 1:
        return float(s[0])

    if kind == 1:  # inverse empirical CDF, no interpolation
        if p == 0.0:
            return float(s[0])
        h = n * p
        j = math.ceil(h) - 1
        return float(s[max(0, min(j, n - 1))])

    if kind == 7:
        h = (n - 1) * p
    elif kind == 6:
        h = (n + 1) * p - 1
    else:
        raise ValueError("kind must be 1, 6, or 7")

    h = max(0.0, min(h, n - 1))
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def quantiles(x: Sequence[float], ps: Sequence[float], kind: int = 7) -> dict:
    """Multiple quantiles at once. ``ps`` is an iterable of probabilities; returns ``{p: q(p)}``."""
    return {p: quantile(x, p, kind) for p in ps}


def five_number_summary(x: Sequence[float], kind: int = 7) -> dict:
    """Tukey's five-number summary: min, Q1, median, Q3, max.

    ``x`` is the sample; ``kind`` is the Hyndman-Fan quantile type (default 7).
    """
    return {
        "min": min(x),
        "Q1": quantile(x, 0.25, kind),
        "median": quantile(x, 0.50, kind),
        "Q3": quantile(x, 0.75, kind),
        "max": max(x),
    }


def percentile_rank(x: Sequence[float], value: float) -> float:
    """Percentile rank of ``value`` within sample ``x`` (%).

    Defined as 100 * (#{x_i < value} + 0.5 * #{x_i == value}) / n -- the "mean" rule
    that handles ties symmetrically (matches ``scipy.stats.percentileofscore(kind='mean')``).
    """
    s = list(x)
    n = len(s)
    below = sum(1 for v in s if v < value)
    equal = sum(1 for v in s if v == value)
    return 100.0 * (below + 0.5 * equal) / n


def ecdf(x: Sequence[float]):
    """Empirical CDF as two parallel arrays.

    Returns
    -------
    (xs, ys) -- ``xs`` are the sorted *unique* values; ``ys[i]`` is the cumulative
    probability F_n(xs[i]) (jumps where there are ties).
    """
    s = sorted(x)
    n = len(s)
    xs, ys = [], []
    for i, v in enumerate(s, start=1):
        if not xs or v != xs[-1]:
            xs.append(v)
            ys.append(i / n)
        else:
            ys[-1] = i / n
    return xs, ys


def library_versions(x: Sequence[float]):
    from scipy import stats

    arr = np.asarray(x, dtype=float)
    return {
        "median (np, type7)": float(np.quantile(arr, 0.5)),
        "Q1/Q3 (np, type7)": (float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))),
        "p90 (np 'inverted_cdf' = HF type1)": float(np.quantile(arr, 0.9, method="inverted_cdf")),
        "p90 (np 'weibull' = HF type6)": float(np.quantile(arr, 0.9, method="weibull")),
        "scoreatpercentile 90 (scipy)": float(stats.scoreatpercentile(arr, 90)),
        "percentileofscore(value=median) (scipy)": float(stats.percentileofscore(arr, np.median(arr), kind="mean")),
    }


if __name__ == "__main__":
    data = [3, 7, 8, 5, 12, 14, 21, 13, 18]
    print("data:", sorted(data), "\n")
    print("--- from scratch ---")
    for k in (1, 6, 7):
        print(f"p=0.9, HF type {k}: {quantile(data, 0.9, k)}")
    print("five-number summary (type 7):", five_number_summary(data))
    print("percentile rank of 13       :", percentile_rank(data, 13))
    xs, ys = ecdf(data)
    print("ECDF points                 :", list(zip(xs, [round(y, 3) for y in ys])))
    print("\n--- library ---")
    for k, v in library_versions(data).items():
        print(f"{k:42s}: {v}")
