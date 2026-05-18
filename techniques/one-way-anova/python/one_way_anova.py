"""One-way ANOVA: classic F, Welch's, and Brown-Forsythe (Reference §3.8, §3.9).

Compare the means of k >= 2 groups. The "classic" F-test assumes equal variances
across groups; Welch's and Brown-Forsythe relax that.

Notation
  k    : number of groups
  N    : total sample size
  n_i, x_bar_i, s_i^2 : per-group n / mean / sample variance

Classic ANOVA
  SS_between = sum n_i (x_bar_i - x_grand)^2
  SS_within  = sum (n_i - 1) s_i^2
  SS_total   = SS_between + SS_within
  F = (SS_between / (k - 1)) / (SS_within / (N - k))     ~  F(k-1, N-k) under H0

Welch's ANOVA (Welch 1951): F-statistic uses weights w_i = n_i / s_i^2; df_2 is
estimated like Welch-Satterthwaite. Better Type I error under unequal variances.

Brown-Forsythe F* (Brown & Forsythe 1974): uses the same SS_between but
"normalizes" the denominator by (1 - n_i/N) s_i^2; df_2 also Satterthwaite.

Effect sizes (see techniques/effect-sizes):
  eta^2     = SS_between / SS_total
  omega^2   = (SS_b - (k-1) MS_w) / (SS_total + MS_w)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import NamedTuple, Sequence    # stdlib: type hints (Sequence = indexable iterable; NamedTuple = tuple with named fields)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


class GroupStats(NamedTuple):
    """Per-group summary. Unpacks like a tuple: ``n, m, s2 = _stats(g)``."""
    n: int       # sample size
    mean: float  # sample mean
    var: float   # sample variance (ddof=1)


def _stats(g):
    n = len(g)
    m = sum(g) / n
    s2 = sum((v - m) ** 2 for v in g) / (n - 1)
    return GroupStats(n=n, mean=m, var=s2)


def classic_anova(groups: Sequence[Sequence[float]]) -> dict:
    """Classic one-way ANOVA assuming equal variances across groups.

    ``groups`` is a list of per-group samples.
    """
    stats_ = [_stats(g) for g in groups]
    ns, means, vars_ = zip(*stats_)
    N = sum(ns); k = len(ns)
    grand = sum(n * m for n, m in zip(ns, means)) / N
    ss_b = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    ss_w = sum((n - 1) * v for n, v in zip(ns, vars_))
    df_b, df_w = k - 1, N - k
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    F = ms_b / ms_w
    p = float(stats.f.sf(F, df_b, df_w))
    eta2 = ss_b / (ss_b + ss_w)
    omega2 = (ss_b - df_b * ms_w) / (ss_b + ss_w + ms_w)
    return {"k": k, "N": N, "ss_between": ss_b, "ss_within": ss_w,
            "df1": df_b, "df2": df_w, "ms_between": ms_b, "ms_within": ms_w,
            "F": F, "p_value": p, "eta_squared": eta2, "omega_squared": omega2}


def welch_anova(groups: Sequence[Sequence[float]]) -> dict:
    """Welch's ANOVA -- robust to unequal variances. Recommended default."""
    stats_ = [_stats(g) for g in groups]
    ns, means, vars_ = zip(*stats_)
    k = len(ns)
    w = [n / s2 for n, s2 in zip(ns, vars_)]
    W = sum(w)
    grand = sum(wi * m for wi, m in zip(w, means)) / W
    num = sum(wi * (m - grand) ** 2 for wi, m in zip(w, means)) / (k - 1)
    denom = 1 + (2 * (k - 2) / (k ** 2 - 1)) * sum(
        (1 - wi / W) ** 2 / (n - 1) for wi, n in zip(w, ns)
    )
    F = num / denom
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * sum((1 - wi / W) ** 2 / (n - 1) for wi, n in zip(w, ns)))
    p = float(stats.f.sf(F, df1, df2))
    return {"F": F, "df1": df1, "df2": df2, "p_value": p, "method": "Welch"}


def brown_forsythe_anova(groups: Sequence[Sequence[float]]) -> dict:
    """Brown-Forsythe F* test (1974) -- another unequal-variances ANOVA."""
    stats_ = [_stats(g) for g in groups]
    ns, means, vars_ = zip(*stats_)
    N = sum(ns); k = len(ns)
    grand = sum(n * m for n, m in zip(ns, means)) / N
    num = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    denom = sum((1 - n / N) * v for n, v in zip(ns, vars_))
    F = num / denom
    df1 = k - 1
    df2 = denom ** 2 / sum(((1 - n / N) * v) ** 2 / (n - 1)
                           for n, v in zip(ns, vars_))
    p = float(stats.f.sf(F, df1, df2))
    return {"F": F, "df1": df1, "df2": df2, "p_value": p, "method": "Brown-Forsythe"}


def library_versions(*groups):
    return {
        "scipy.stats.f_oneway": stats.f_oneway(*groups),
        # scipy >= 1.10 has Welch via alexandergovern; older versions need pingouin
    }


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(2)
    # Three groups; group C has a larger variance (illustrates Welch vs classic)
    a = rng.normal(50, 10, 30).tolist()
    b = rng.normal(55, 10, 28).tolist()
    c = rng.normal(60, 18, 32).tolist()

    print("=== Classic one-way ANOVA ===")
    for k, v in classic_anova([a, b, c]).items():
        print(f"  {k:15s}: {v}")

    print("\n=== Welch's ANOVA ===")
    for k, v in welch_anova([a, b, c]).items():
        print(f"  {k:15s}: {v}")

    print("\n=== Brown-Forsythe F* ===")
    for k, v in brown_forsythe_anova([a, b, c]).items():
        print(f"  {k:15s}: {v}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(a, b, c).items():
        print(f"  {k}: {v}")
