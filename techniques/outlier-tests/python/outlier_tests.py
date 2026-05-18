"""Formal tests for univariate outliers (Reference §3.25).

These test "is the extreme value in this sample farther from the mean than we
would expect under a normal distribution?" They assume the *rest* of the data
are normal -- on heavily skewed or heavy-tailed data they over-flag.

Tests
-----
- Grubbs (one-outlier)        : (max |x - mean|) / sd, single most extreme value
                                exact small-sample critical value via the t dist.
- Generalized ESD (Rosner)    : iteratively masks the most extreme point and
                                re-tests up to ``r`` times; designed for *multiple*
                                outliers.
- Dixon's Q                   : (gap to nearest neighbor) / range, for very small n
                                (3 <= n <= ~30). Uses tabulated critical values
                                (we ship the common 90 / 95 / 99% values).
- IQR rule                    : non-test heuristic: outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                                (Tukey's fences). Not a hypothesis test; included
                                because it's the most-used "what's an outlier?" rule.

Important caveat
----------------
"Outlier" is *not* synonymous with "error." Before deleting flagged points:
investigate them, transform the data (log for right-skewed), or use a robust
method (``techniques/robust-location-scale``). Throwing away inconvenient data
is a frequent source of false-positive results in the published literature.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(x): return sum(x) / len(x)


def _sd(x, ddof=1):
    m = _mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - ddof))


def grubbs(x: Sequence[float], alpha: float = 0.05, alternative: str = "two-sided") -> dict:
    """Grubbs' test for a single outlier.

    Parameters
    ----------
    x : numeric sample, assumed approximately normal under H0.
    alpha : significance level for the critical value.
    alternative : "two-sided" / "max" (only the maximum) / "min" (only the minimum).
    """
    n = len(x); m = _mean(x); s = _sd(x)
    if s == 0:
        return {"G": float("nan"), "p_value": 1.0, "critical": float("nan"),
                "candidate": None, "reject_normal": False}
    devs = [(v - m) / s for v in x]                # studentized deviations
    if alternative == "max":
        i = max(range(n), key=lambda i: devs[i]); G = devs[i]
    elif alternative == "min":
        i = min(range(n), key=lambda i: devs[i]); G = -devs[i]
    else:
        i = max(range(n), key=lambda i: abs(devs[i])); G = abs(devs[i])

    # Critical value (two-sided form): G_crit = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2+t^2))
    # with t = quantile of t_{n-2}. For one-sided, halve the tail probability.
    p_t = alpha / (2 * n) if alternative == "two-sided" else alpha / n
    t_crit = stats.t.ppf(1 - p_t, n - 2)
    G_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))

    # Approximate two-sided p-value via the same Bonferroni inversion
    t_obs = math.sqrt((n - 2) * G ** 2 / max(1e-300, (n - 1) ** 2 / n - G ** 2))
    p_tail = float(stats.t.sf(t_obs, n - 2))
    p_value = min(1.0, p_tail * (2 * n if alternative == "two-sided" else n))
    return {"G": G, "G_critical": G_crit, "p_value": p_value,
            "candidate_index": i, "candidate_value": x[i],
            "reject_normal": G > G_crit, "alternative": alternative}


def generalized_esd(x: Sequence[float], r: int = 3, alpha: float = 0.05) -> dict:
    """Rosner's Generalized ESD: detect up to ``r`` outliers in a single pass.

    Returns the indices flagged as outliers (largest set of consecutive steps whose
    test statistic exceeded its critical value).
    """
    data = list(x)
    n0 = len(data)
    indices = list(range(n0))                  # indices into the ORIGINAL x
    R = []        # statistics
    crit = []     # critical values
    removed = []  # (index_in_original, value) tuples for the candidate at each step

    for i in range(1, r + 1):
        n = len(data)
        if n <= 2: break
        m = _mean(data); s = _sd(data)
        if s == 0: break
        devs = [(v - m) / s for v in data]
        j = max(range(n), key=lambda k: abs(devs[k]))
        Ri = abs(devs[j])

        # Critical value (Rosner 1983)
        p = 1 - alpha / (2 * (n - i + 1))
        t_p = stats.t.ppf(p, n - i - 1)
        lam = ((n - i) * t_p) / math.sqrt((n - i - 1 + t_p ** 2) * (n - i + 1))

        R.append(Ri); crit.append(lam)
        removed.append((indices[j], data[j]))
        del data[j]; del indices[j]

    # The largest L such that R_L > crit_L is the number of outliers
    L = 0
    for k in range(len(R) - 1, -1, -1):
        if R[k] > crit[k]:
            L = k + 1
            break
    outlier_indices = [idx for idx, _ in removed[:L]]
    return {"R": R, "critical": crit, "L_detected": L,
            "outlier_indices": outlier_indices,
            "outlier_values": [removed[k][1] for k in range(L)]}


# Dixon's Q critical values (tabulated; subset of Rorabacher 1991)
# rows: n = 3..10; columns: alpha = 0.10, 0.05, 0.01
_DIXON_Q_CRIT = {
    3: (0.941, 0.970, 0.994),
    4: (0.765, 0.829, 0.926),
    5: (0.642, 0.710, 0.821),
    6: (0.560, 0.625, 0.740),
    7: (0.507, 0.568, 0.680),
    8: (0.468, 0.526, 0.634),
    9: (0.437, 0.493, 0.598),
    10: (0.412, 0.466, 0.568),
}


def dixons_q(x: Sequence[float], alpha: float = 0.05) -> dict:
    """Dixon's Q test for a single outlier, 3 <= n <= 10 here.

    Uses the simple Q = gap / range form. Critical values from Rorabacher (1991).
    """
    n = len(x)
    if n < 3 or n > 10:
        raise ValueError("this implementation tabulates n in [3, 10]")
    s = sorted(x); rng = s[-1] - s[0]
    if rng == 0:
        return {"Q": float("nan"), "candidate": None, "reject_normal": False}
    Q_max = (s[-1] - s[-2]) / rng
    Q_min = (s[1] - s[0]) / rng
    if Q_max >= Q_min:
        Q = Q_max; cand = s[-1]
    else:
        Q = Q_min; cand = s[0]
    col = {0.10: 0, 0.05: 1, 0.01: 2}.get(alpha)
    if col is None:
        raise ValueError("alpha must be 0.10, 0.05, or 0.01")
    Q_crit = _DIXON_Q_CRIT[n][col]
    return {"Q": Q, "Q_critical": Q_crit, "candidate": cand,
            "reject_normal": Q > Q_crit, "alpha": alpha}


def iqr_rule(x: Sequence[float], k: float = 1.5) -> dict:
    """Tukey's IQR rule: flag points outside [Q1 - k*IQR, Q3 + k*IQR].

    Not a hypothesis test; just a robust description. ``k = 1.5`` -> "outliers";
    ``k = 3.0`` -> "far out" points (Tukey 1977).
    """
    from statistics import median
    s = sorted(x)
    q1 = _quantile(s, 0.25); q3 = _quantile(s, 0.75); iqr_ = q3 - q1
    lo, hi = q1 - k * iqr_, q3 + k * iqr_
    flagged = [(i, v) for i, v in enumerate(x) if v < lo or v > hi]
    return {"Q1": q1, "Q3": q3, "IQR": iqr_, "lower_fence": lo, "upper_fence": hi,
            "flagged": flagged}


def _quantile(s, p):
    n = len(s)
    if n == 1: return s[0]
    h = (n - 1) * p; lo = int(h); frac = h - lo
    hi = min(lo + 1, n - 1)
    return s[lo] + frac * (s[hi] - s[lo])


if __name__ == "__main__":
    x = [10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 80]   # 80 is the candidate

    print("=== Grubbs (two-sided) ===")
    print(grubbs(x))

    print("\n=== Generalized ESD (r = 3) ===")
    print(generalized_esd(x, r=3))

    print("\n=== Dixon's Q (n = 8 sub-sample) ===")
    small = [10, 12, 14, 15, 16, 17, 19, 50]
    print(dixons_q(small))

    print("\n=== Tukey IQR rule ===")
    print(iqr_rule(x))
