"""Robust location and scale estimators (Reference §1.3 and §1.26).

Estimators that resist outliers and heavy tails:

Location
  - median
  - trimmed mean        : discard k% from each tail, average the rest
  - Winsorized mean     : clamp each tail to its boundary value, then average
  - Huber M-estimator   : iteratively-reweighted mean; quadratic loss near the
                          center, linear in the tails (tuning constant k, default 1.345)

Scale
  - IQR
  - MAD                 : median(|x - median|); ×1.4826 -> consistent estimate of sigma
  - Winsorized variance : variance of the Winsorized sample (used by Yuen's test)

Two-sample
  - Yuen's trimmed t    : Welch-style t-test on trimmed means with Winsorized
                          variances; robust alternative to the two-sample t-test
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def median(x: Sequence[float]) -> float:
    """Sample median. ``x`` is a numeric sample."""
    return _median(list(x))


def mad(x: Sequence[float], scale: float = 1.4826) -> float:
    """Median absolute deviation about the median.

    Parameters
    ----------
    x : sample.
    scale : multiplier (``1.4826`` -> consistent estimator of sigma at the normal;
        pass ``1.0`` for the raw MAD).
    """
    x = list(x)
    m = _median(x)
    return _median([abs(v - m) for v in x]) * scale


def _winsorize(x: Sequence[float], proportion: float):
    """Internal: return (Winsorized-sorted-sample, k_per_tail). ``proportion`` in [0, 0.5)."""
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    s = sorted(x)
    n = len(s)
    k = int(math.floor(n * proportion))
    if k:
        lo, hi = s[k], s[n - k - 1]
        s = [lo] * k + s[k:n - k] + [hi] * k
    return s, k


def trimmed_mean(x: Sequence[float], proportion: float = 0.2) -> float:
    """Mean after dropping ``proportion`` from each tail of ``x``. See top-level README."""
    s = sorted(x)
    n = len(s)
    k = int(math.floor(n * proportion))
    kept = s[k:n - k] if k else s
    return sum(kept) / len(kept)


def winsorized_mean(x: Sequence[float], proportion: float = 0.2) -> float:
    """Mean of the Winsorized sample (tails clamped to their boundary values)."""
    s, _ = _winsorize(x, proportion)
    return sum(s) / len(s)


def winsorized_variance(x: Sequence[float], proportion: float = 0.2, ddof: int = 1) -> float:
    """Variance of the Winsorized sample. Pairs with the trimmed mean in Yuen's test.

    Parameters
    ----------
    x : sample.
    proportion : Winsorizing fraction per tail.
    ddof : divisor is ``n - ddof`` (``1`` for the unbiased sample variance).
    """
    s, _ = _winsorize(x, proportion)
    n = len(s)
    m = sum(s) / n
    return sum((v - m) ** 2 for v in s) / (n - ddof)


def huber_location(x: Sequence[float], k: float = 1.345, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Huber M-estimator of location via iteratively reweighted least squares.

    Parameters
    ----------
    x : sample.
    k : Huber tuning constant in standardized residual units. ``1.345`` gives ~95%
        efficiency at the normal distribution; smaller ``k`` -> more robust, less efficient.
    tol : relative-change convergence tolerance.
    max_iter : safety cap on the IRLS iterations.
    """
    x = list(x)
    mu = _median(x)
    s = mad(x) or 1.0  # scale; guard against zero MAD
    for _ in range(max_iter):
        r = [(v - mu) / s for v in x]
        # Huber weight: 1 inside [-k, k], else k/|r|
        w = [1.0 if abs(ri) <= k else k / abs(ri) for ri in r]
        new_mu = sum(wi * vi for wi, vi in zip(w, x)) / sum(w)
        if abs(new_mu - mu) < tol * max(1.0, abs(mu)):
            return new_mu
        mu = new_mu
    return mu


def yuen_trimmed_t(x: Sequence[float], y: Sequence[float], proportion: float = 0.2):
    """Yuen's two-sample trimmed-mean t-test (Welch-style).

    Parameters
    ----------
    x, y : independent samples for the two groups.
    proportion : symmetric trimming fraction per tail applied to both groups.

    Returns
    -------
    dict with the two trimmed means, their difference, standard error, t-statistic,
    Welch-style df, and the two-sided p-value.
    """
    from scipy import stats

    def parts(z):
        s = sorted(z)
        n = len(s)
        g = int(math.floor(n * proportion))
        h = n - 2 * g  # effective sample size
        sw, _ = _winsorize(z, proportion)
        m = sum(sw) / n
        ssd = sum((v - m) ** 2 for v in sw)  # Winsorized sum of squares
        return n, g, h, trimmed_mean(z, proportion), ssd

    n1, g1, h1, t1, ss1 = parts(x)
    n2, g2, h2, t2, ss2 = parts(y)
    d1 = ss1 / (h1 * (h1 - 1))
    d2 = ss2 / (h2 * (h2 - 1))
    se = math.sqrt(d1 + d2)
    tstat = (t1 - t2) / se
    df = (d1 + d2) ** 2 / (d1 ** 2 / (h1 - 1) + d2 ** 2 / (h2 - 1))
    pval = 2 * stats.t.sf(abs(tstat), df)
    return {
        "trimmed_mean_x": t1, "trimmed_mean_y": t2,
        "difference": t1 - t2, "se": se, "t": tstat, "df": df, "p_value": pval,
    }


def library_versions(x: Sequence[float]):
    from scipy import stats
    from statsmodels.robust.scale import huber as sm_huber

    arr = np.asarray(x, dtype=float)
    loc, scale = sm_huber(arr)
    return {
        "median (numpy)": float(np.median(arr)),
        "MAD scaled (scipy)": float(stats.median_abs_deviation(arr, scale="normal")),
        "trimmed_mean 20% (scipy)": float(stats.trim_mean(arr, 0.2)),
        "Huber location (statsmodels)": float(loc),
        "Huber scale (statsmodels)": float(scale),
    }


if __name__ == "__main__":
    data = [4, 8, 6, 5, 3, 9, 7, 11, 6, 100]
    print("data:", data, "\n")
    print("--- from scratch ---")
    print("median             :", median(data))
    print("MAD (scaled sigma) :", mad(data))
    print("IQR                :", float(np.quantile(data, 0.75) - np.quantile(data, 0.25)))
    print("trimmed mean 20%   :", trimmed_mean(data, 0.2))
    print("winsorized mean 20%:", winsorized_mean(data, 0.2))
    print("winsorized var 20% :", winsorized_variance(data, 0.2))
    print("Huber location     :", huber_location(data))
    print("\n--- library ---")
    for k, v in library_versions(data).items():
        print(f"{k:30s}: {v}")

    print("\n--- Yuen's trimmed t-test (robust two-sample) ---")
    g1 = [10, 12, 11, 9, 13, 10, 11, 50]    # the 50 is an outlier
    g2 = [14, 15, 13, 16, 14, 15, 14, 13]
    for k, v in yuen_trimmed_t(g1, g2, 0.2).items():
        print(f"{k:16s}: {v}")
