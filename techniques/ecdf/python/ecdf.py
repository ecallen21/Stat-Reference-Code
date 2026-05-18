"""Empirical Cumulative Distribution Function (Reference §1.13).

The ECDF is  F_n(t) = (# observations <= t) / n  -- a right-continuous step
function that jumps by 1/n at each data point (more if there are ties). It is the
nonparametric MLE of the true CDF, underlies the Kolmogorov-Smirnov test, P-P
plots, and bootstrap resampling.

We also include the Dvoretzky-Kiefer-Wolfowitz (DKW) simultaneous confidence
band:  F_n(t) +/- eps  with  eps = sqrt(ln(2/alpha) / (2n)), clipped to [0, 1].
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import bisect    # stdlib: binary search on a sorted list (O(log n) insertion-point)
import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import NamedTuple, Sequence    # stdlib: type hints (Sequence = indexable iterable; NamedTuple = tuple with named fields)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


class DKWBand(NamedTuple):
    """Return type of ``ECDF.dkw_band``. Unpacks like a tuple: ``xs, lo, hi = band``."""
    xs: list
    lower: np.ndarray
    upper: np.ndarray


class ECDF:
    """Callable empirical CDF: ``F = ECDF(data); F(t)`` returns P_n(X <= t)."""

    def __init__(self, data: Sequence[float]):
        """Build the ECDF from a sample.

        Parameters
        ----------
        data : numeric sample (sequence of values).
        """
        self.x = sorted(float(v) for v in data)
        self.n = len(self.x)
        if self.n == 0:
            raise ValueError("empty data")

    def __call__(self, t):
        """Evaluate F_n(t). ``t`` may be a scalar or array-like; the return type matches."""
        scalar = np.isscalar(t)
        ts = np.atleast_1d(np.asarray(t, dtype=float))
        # number of x_i <= t  ==  insertion point on the right
        counts = np.array([bisect.bisect_right(self.x, v) for v in ts])
        out = counts / self.n
        return float(out[0]) if scalar else out

    def quantile(self, p: float) -> float:
        """Inverse ECDF (Hyndman-Fan type 1): smallest x with F_n(x) >= p. ``p`` in (0, 1]."""
        if not 0.0 < p <= 1.0:
            if p == 0.0:
                return self.x[0]
            raise ValueError("p must be in (0, 1]")
        k = math.ceil(p * self.n)
        return self.x[min(k, self.n) - 1]

    def dkw_band(self, alpha: float = 0.05):
        """Dvoretzky-Kiefer-Wolfowitz simultaneous confidence band for F.

        Parameters
        ----------
        alpha : 1 - confidence level (``0.05`` -> 95% band).

        Returns
        -------
        DKWBand named tuple with fields ``xs``, ``lower``, ``upper`` -- xs are the
        sorted data points; ``lower[i]`` and ``upper[i]`` bracket F at xs[i]. The
        band covers the *whole* curve at the chosen level (unlike a pointwise CI).
        Unpacks like a tuple: ``xs, lo, hi = F.dkw_band()``.
        """
        eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * self.n))
        xs = self.x
        fhat = np.array([self(v) for v in xs])
        lower = np.clip(fhat - eps, 0.0, 1.0)
        upper = np.clip(fhat + eps, 0.0, 1.0)
        return DKWBand(xs=xs, lower=lower, upper=upper)


def library_versions(data: Sequence[float], grid):
    from statsmodels.distributions.empirical_distribution import ECDF as SMECDF

    sm = SMECDF(data)
    out = {f"statsmodels ECDF({g})": float(sm(g)) for g in grid}
    try:  # scipy >= 1.11
        from scipy import stats

        res = stats.ecdf(data)
        out.update({f"scipy ecdf.cdf.evaluate({g})": float(res.cdf.evaluate(g)) for g in grid})
    except (ImportError, AttributeError):
        pass
    return out


if __name__ == "__main__":
    data = [2, 3, 3, 5, 8, 13, 21]
    F = ECDF(data)
    grid = [1, 3, 4, 8, 25]
    print("data:", data)
    print("\n--- from scratch ---")
    for t in grid:
        print(f"F_n({t:>2}) = {F(t):.4f}")
    print("inverse ECDF p=0.5 :", F.quantile(0.5))
    print("inverse ECDF p=0.9 :", F.quantile(0.9))
    xs, lo, hi = F.dkw_band(alpha=0.05)
    print("95% DKW band at the data points:")
    for x, l, h in zip(xs, lo, hi):
        print(f"  x={x:>2}: F_n={F(x):.3f}  band=[{l:.3f}, {h:.3f}]")
    print("\n--- library ---")
    for k, v in library_versions(data, grid).items():
        print(f"{k:36s}: {v}")
