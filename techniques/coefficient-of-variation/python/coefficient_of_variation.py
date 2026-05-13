"""Coefficient of variation (Reference §1.22 and §1.33).

CV = standard deviation / mean -- a unit-free measure of *relative* variability,
useful for comparing spread across variables on different scales or with
different means. Often reported as a percentage.

Variants
  - sample CV          : s / x̄            (optionally x 100%)
  - geometric CV       : for log-normal data, GCV = sqrt(exp(s_log^2) - 1)
                         where s_log = SD of log(x); approx equals CV for small CV
  - within-subject CV  : CV_w = sqrt(mean of within-subject variances) / overall mean,
                         from repeated measurements per subject (assay precision,
                         biological variation studies)

Confidence interval (McKay's chi-squared approximation), valid for CV < ~0.33:
  with v = n - 1,  CV in [ CV / sqrt((chi2_{1-a/2,v}/v - 1)*CV^2 + chi2_{1-a/2,v}/v),
                           CV / sqrt((chi2_{a/2,v}/v   - 1)*CV^2 + chi2_{a/2,v}/v) ]

Caveat: CV is meaningless when the mean is near zero or the scale has an
arbitrary zero (e.g. temperature in Celsius), and undefined for data with mixed
signs.
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _mean(x):
    return sum(x) / len(x)


def _sd(x, ddof=1):
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - ddof))


def cv(x: Sequence[float], ddof: int = 1, as_percent: bool = False) -> float:
    """Coefficient of variation = sd(x) / mean(x).

    Parameters
    ----------
    x : sample.
    ddof : passed to the SD (``1`` -> sample SD).
    as_percent : if True, return 100 * SD/mean.

    Raises ``ValueError`` if the mean is zero.
    """
    m = _mean(x)
    if m == 0:
        raise ValueError("CV is undefined when the mean is zero")
    val = _sd(x, ddof) / m
    return val * 100.0 if as_percent else val


def geometric_cv(x: Sequence[float], ddof: int = 1, as_percent: bool = False) -> float:
    """Geometric CV for (assumed) log-normal data: sqrt(exp(sd(log x)**2) - 1).

    ``x`` must be strictly positive. ``ddof`` is passed to the SD of log(x).
    """
    if any(v <= 0 for v in x):
        raise ValueError("geometric CV requires strictly positive values")
    s_log = _sd([math.log(v) for v in x], ddof)
    val = math.sqrt(math.exp(s_log ** 2) - 1.0)
    return val * 100.0 if as_percent else val


def within_subject_cv(subjects: Sequence[Sequence[float]], as_percent: bool = False) -> float:
    """Within-subject CV from repeated measurements (assay precision / biological variation).

    Parameters
    ----------
    subjects : list of per-subject vectors of replicate measurements
        (e.g. ``[[10.1, 10.3, 9.8], [12.0, 11.7, 12.2], ...]``).
        Subjects with fewer than 2 measurements are ignored.
    as_percent : if True, return 100 * CV_w.

    Formula: ``sqrt(mean of within-subject variances) / overall mean``.
    """
    var_w = [
        sum((v - _mean(s)) ** 2 for v in s) / (len(s) - 1)
        for s in subjects if len(s) >= 2
    ]
    overall_mean = _mean([v for s in subjects for v in s])
    val = math.sqrt(sum(var_w) / len(var_w)) / overall_mean
    return val * 100.0 if as_percent else val


def cv_ci_mckay(x: Sequence[float], conf: float = 0.95):
    """McKay's approximate chi-squared CI for the (population) CV.

    Reasonable for CV < ~0.33. Returns a (lower, upper) tuple at confidence level ``conf``.
    """
    n = len(x)
    v = n - 1
    k = cv(x)  # sample CV (proportion)
    a = 1 - conf
    chi_hi = stats.chi2.ppf(1 - a / 2, v) / v
    chi_lo = stats.chi2.ppf(a / 2, v) / v
    lo = k / math.sqrt((chi_hi - 1) * k * k + chi_hi)
    hi = k / math.sqrt((chi_lo - 1) * k * k + chi_lo)
    return lo, hi


def library_versions(x: Sequence[float]):
    import numpy as np

    arr = np.asarray(x, dtype=float)
    return {
        "scipy.stats.variation (ddof=1)": float(stats.variation(arr, ddof=1)),
        "numpy std/mean (ddof=1)": float(np.std(arr, ddof=1) / np.mean(arr)),
    }


if __name__ == "__main__":
    # Two assays measuring the same thing on different scales:
    assay_a = [98.2, 101.4, 99.7, 100.1, 102.3, 97.9, 100.8]      # arbitrary units ~100
    assay_b = [4.91, 5.07, 4.98, 5.00, 5.12, 4.90, 5.04]          # mg/dL ~5
    print("assay A: mean=%.2f sd=%.3f  -> CV = %.2f%%" % (_mean(assay_a), _sd(assay_a), cv(assay_a, as_percent=True)))
    print("assay B: mean=%.2f sd=%.3f  -> CV = %.2f%%" % (_mean(assay_b), _sd(assay_b), cv(assay_b, as_percent=True)))
    print("(absolute SDs differ ~20x, but the relative variability is similar)\n")

    print("geometric CV of assay A      :", round(geometric_cv(assay_a, as_percent=True), 3), "%")
    lo, hi = cv_ci_mckay(assay_a)
    print("McKay 95%% CI for CV(A)       : (%.4f, %.4f)" % (lo, hi))

    # Within-subject CV: 4 subjects, 3 replicate measurements each
    reps = [[10.1, 10.3, 9.8], [12.0, 11.7, 12.2], [9.5, 9.7, 9.4], [11.2, 11.0, 11.5]]
    print("within-subject CV            :", round(within_subject_cv(reps, as_percent=True), 3), "%")

    print("\n--- library ---")
    for k, v in library_versions(assay_a).items():
        print(f"{k:34s}: {v}")
