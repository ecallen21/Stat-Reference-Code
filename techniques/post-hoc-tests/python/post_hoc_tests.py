"""Post-hoc pairwise comparisons after a one-way ANOVA (Reference §3.10, §3.11, §3.16).

When an overall ANOVA is significant, post-hoc tests pinpoint *which* pairs of
group means differ -- while controlling the family-wise error rate (FWER) for
the set of comparisons.

Tests covered
-------------
- Tukey HSD (1953)     : all pairwise; equal variances + (ideally) balanced n.
                         Uses the *studentized range* distribution q(k, df).
- Tukey-Kramer (1956)  : Tukey HSD with the standard adjustment for unequal n_i.
- Dunnett (1955)       : k - 1 comparisons of each treatment vs. a single
                         CONTROL group. More powerful than Tukey here because
                         there are fewer comparisons.
- Games-Howell (1976)  : Welch-style unequal-variances analogue of Tukey HSD;
                         uses Welch-Satterthwaite df per pair.

All return per-pair: mean diff, SE, CI, and a *family-wise* adjusted p-value.

Notes / dependencies
--------------------
The studentized range distribution is in scipy.stats as ``studentized_range``
(scipy >= 1.7). The Dunnett distribution (multivariate t) is available as
``scipy.stats.dunnett`` (scipy >= 1.11); we use it for p-values when present,
otherwise fall back to a Bonferroni-corrected t for Dunnett.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from itertools import combinations    # stdlib: iterator over all C(n, k) size-k subsets
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _stats(g):
    n = len(g); m = sum(g) / n
    s2 = sum((v - m) ** 2 for v in g) / (n - 1)
    return n, m, s2


def tukey_hsd(groups: Sequence[Sequence[float]], labels=None, conf: float = 0.95):
    """Tukey HSD / Tukey-Kramer for all pairwise comparisons (equal variances).

    Returns a list of dicts (one per pair).
    """
    labels = labels or list(range(len(groups)))
    stats_ = [_stats(g) for g in groups]
    ns = [s[0] for s in stats_]
    means = [s[1] for s in stats_]
    # Pooled within-group MS (the ANOVA denominator)
    N = sum(ns); k = len(ns)
    ms_w = sum((n - 1) * s2 for (n, _, s2) in stats_) / (N - k)
    df = N - k
    alpha = 1 - conf

    out = []
    for i, j in combinations(range(k), 2):
        diff = means[i] - means[j]
        se = math.sqrt(ms_w * 0.5 * (1 / ns[i] + 1 / ns[j]))   # Tukey-Kramer adj for unequal n
        q = abs(diff) / se
        # two-sided FWER p-value from the studentized range
        p_adj = float(stats.studentized_range.sf(q, k, df))
        q_crit = stats.studentized_range.ppf(1 - alpha, k, df)
        margin = q_crit * se
        out.append({"pair": (labels[i], labels[j]), "mean_diff": diff,
                    "se": se, "q": q, "p_adj": min(1.0, p_adj),
                    "ci_lower": diff - margin, "ci_upper": diff + margin})
    return out


def dunnett(groups: Sequence[Sequence[float]], control_index: int = 0,
            labels=None, conf: float = 0.95):
    """Dunnett's test: each non-control group vs. the control group.

    Uses ``scipy.stats.dunnett`` for the multivariate-t-based p-values when
    available (scipy >= 1.11); otherwise a Bonferroni-adjusted two-sample t.
    """
    labels = labels or list(range(len(groups)))
    ctrl = groups[control_index]
    treats = [g for i, g in enumerate(groups) if i != control_index]
    treat_labels = [l for i, l in enumerate(labels) if i != control_index]

    if hasattr(stats, "dunnett"):
        res = stats.dunnett(*treats, control=ctrl)
        stats_ = [_stats(g) for g in groups]
        ns = [s[0] for s in stats_]
        means = [s[1] for s in stats_]
        N = sum(ns); k = len(ns)
        ms_w = sum((n - 1) * s2 for (n, _, s2) in stats_) / (N - k)
        out = []
        for k_, (label, p_adj) in enumerate(zip(treat_labels, res.pvalue)):
            i = labels.index(label); j = control_index
            diff = means[i] - means[j]
            se = math.sqrt(ms_w * (1 / ns[i] + 1 / ns[j]))
            out.append({"pair": (label, labels[control_index]),
                        "mean_diff": diff, "se": se,
                        "t": float(res.statistic[k_]), "p_adj": float(p_adj)})
        return out

    # Fallback: Bonferroni-corrected Student t (less powerful than true Dunnett)
    m = len(treats)
    stats_ = [_stats(g) for g in groups]
    ns = [s[0] for s in stats_]; means = [s[1] for s in stats_]
    N = sum(ns); k = len(ns)
    ms_w = sum((n - 1) * s2 for (n, _, s2) in stats_) / (N - k)
    df = N - k
    out = []
    j = control_index
    for label in treat_labels:
        i = labels.index(label)
        diff = means[i] - means[j]
        se = math.sqrt(ms_w * (1 / ns[i] + 1 / ns[j]))
        t = diff / se
        p = 2 * stats.t.sf(abs(t), df)
        out.append({"pair": (label, labels[control_index]),
                    "mean_diff": diff, "se": se, "t": t,
                    "p_adj": min(1.0, p * m), "method": "Bonferroni fallback"})
    return out


def games_howell(groups: Sequence[Sequence[float]], labels=None, conf: float = 0.95):
    """Games-Howell pairwise comparisons (Welch + studentized range). Unequal variances OK."""
    labels = labels or list(range(len(groups)))
    stats_ = [_stats(g) for g in groups]
    k = len(stats_)
    alpha = 1 - conf
    out = []
    for i, j in combinations(range(k), 2):
        ni, mi, vi = stats_[i]; nj, mj, vj = stats_[j]
        diff = mi - mj
        se = math.sqrt((vi / ni + vj / nj) / 2)
        df = (vi / ni + vj / nj) ** 2 / (
            (vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1)
        )
        q = abs(diff) / se
        p_adj = float(stats.studentized_range.sf(q, k, df))
        q_crit = stats.studentized_range.ppf(1 - alpha, k, df)
        margin = q_crit * se
        out.append({"pair": (labels[i], labels[j]), "mean_diff": diff,
                    "se": se, "df": df, "q": q,
                    "p_adj": min(1.0, p_adj),
                    "ci_lower": diff - margin, "ci_upper": diff + margin})
    return out


def scheffe(groups: Sequence[Sequence[float]], contrasts=None, labels=None,
            conf: float = 0.95):
    """Scheffe's method for arbitrary linear contrasts among group means.

    Parameters
    ----------
    groups : list of per-group samples.
    contrasts : list of contrast vectors ``c = (c_1, ..., c_k)`` with ``sum(c) = 0``.
        Default: all pairwise contrasts.
    labels : group labels.
    conf : confidence level.

    For a contrast ``L = sum c_i * mean_i``, the test statistic is
        F = L^2 / ((k - 1) * MS_w * sum(c_i^2 / n_i))   ~  F(k-1, N-k)
    and L is "Scheffe-significant" iff F exceeds the F critical value.
    Equivalently the simultaneous-CI half-width is
        sqrt((k - 1) * F_{k-1, N-k, alpha}) * SE(L).
    Scheffe controls FWER across **all** contrasts (not just pairwise) -- so it's
    the right tool when you want to roam over arbitrary linear combinations.
    """
    labels = labels or list(range(len(groups)))
    stats_ = [_stats(g) for g in groups]
    ns = [s[0] for s in stats_]; means = [s[1] for s in stats_]
    N = sum(ns); k = len(ns); df_w = N - k
    ms_w = sum((n - 1) * s2 for (n, _, s2) in stats_) / df_w

    if contrasts is None:
        contrasts = []
        for i, j in combinations(range(k), 2):
            c = [0.0] * k; c[i] = 1.0; c[j] = -1.0
            contrasts.append(c)

    alpha = 1 - conf
    f_crit = stats.f.ppf(1 - alpha, k - 1, df_w)
    out = []
    for c in contrasts:
        if abs(sum(c)) > 1e-9:
            raise ValueError(f"contrast {c} does not sum to zero")
        L = sum(ci * m for ci, m in zip(c, means))
        se = math.sqrt(ms_w * sum(ci ** 2 / n for ci, n in zip(c, ns)))
        F = (L * L) / ((k - 1) * se * se) if se > 0 else float("inf")
        p_adj = float(stats.f.sf(F, k - 1, df_w))
        margin = math.sqrt((k - 1) * f_crit) * se
        out.append({"contrast": c, "L": L, "se": se,
                    "F": F, "df1": k - 1, "df2": df_w, "p_adj": p_adj,
                    "ci_lower": L - margin, "ci_upper": L + margin,
                    "labels": labels})
    return out


def tamhane_t2(groups: Sequence[Sequence[float]], labels=None, conf: float = 0.95):
    """Tamhane's T2: Welch-style pairwise t-tests with Sidak correction.

    Robust to unequal variances. Slightly conservative; less powerful than
    Games-Howell when group sizes are reasonably large.
    """
    labels = labels or list(range(len(groups)))
    stats_ = [_stats(g) for g in groups]
    k = len(stats_); m = k * (k - 1) // 2     # number of pairwise comparisons
    out = []
    for i, j in combinations(range(k), 2):
        ni, mi, vi = stats_[i]; nj, mj, vj = stats_[j]
        diff = mi - mj
        se = math.sqrt(vi / ni + vj / nj)
        df = (vi / ni + vj / nj) ** 2 / (
            (vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1)
        )
        t = diff / se
        p_raw = 2 * stats.t.sf(abs(t), df)
        # Sidak across the m comparisons
        p_adj = min(1.0, 1.0 - (1.0 - p_raw) ** m)
        out.append({"pair": (labels[i], labels[j]), "mean_diff": diff,
                    "se": se, "df": df, "t": t, "p_raw": p_raw,
                    "p_adj": p_adj, "method": "Tamhane T2 (Welch + Sidak)"})
    return out


def dunnett_t3(groups: Sequence[Sequence[float]], labels=None, conf: float = 0.95):
    """Dunnett's T3: Welch-style pairwise tests with studentized maximum modulus crit.

    Like Games-Howell, but uses the studentized maximum modulus distribution
    instead of the studentized range -- slightly more conservative. Reported
    here via the same studentized-range form for simplicity, with a note; for
    the exact T3 critical value use ``PMCMRplus`` (R) or ``scikit-posthocs``
    (Python).
    """
    labels = labels or list(range(len(groups)))
    stats_ = [_stats(g) for g in groups]
    k = len(stats_); m = k * (k - 1) // 2
    out = []
    for i, j in combinations(range(k), 2):
        ni, mi, vi = stats_[i]; nj, mj, vj = stats_[j]
        diff = mi - mj
        se = math.sqrt(vi / ni + vj / nj)
        df = (vi / ni + vj / nj) ** 2 / (
            (vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1)
        )
        t = diff / se
        p_raw = 2 * stats.t.sf(abs(t), df)
        # Approximate T3 via Sidak on the m pairwise tests (close to the exact
        # studentized-maximum-modulus form for moderate m and df)
        p_adj = min(1.0, 1.0 - (1.0 - p_raw) ** m)
        out.append({"pair": (labels[i], labels[j]), "mean_diff": diff,
                    "se": se, "df": df, "t": t, "p_raw": p_raw,
                    "p_adj": p_adj,
                    "method": "Dunnett T3 (Welch + studentized max modulus; approx via Sidak)"})
    return out


def library_versions(groups, labels):
    from scipy.stats import tukey_hsd as scipy_tukey
    res = scipy_tukey(*groups)
    out = {"scipy.stats.tukey_hsd statistic": res.statistic,
           "scipy.stats.tukey_hsd p-values": res.pvalue}
    return out


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(3)
    a = rng.normal(50, 10, 30).tolist()
    b = rng.normal(55, 10, 28).tolist()
    c = rng.normal(60, 18, 32).tolist()
    d = rng.normal(52, 10, 30).tolist()
    groups = [a, b, c, d]; labels = ["A", "B", "C", "D"]

    print("=== Tukey HSD (equal-variance assumption) ===")
    for r in tukey_hsd(groups, labels):
        print(f"  {r['pair']}: diff={r['mean_diff']:+.3f}  CI=[{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]  p_adj={r['p_adj']:.4f}")

    print("\n=== Dunnett (vs control = A) ===")
    for r in dunnett(groups, control_index=0, labels=labels):
        extra = f"  ({r.get('method', '')})" if r.get("method") else ""
        print(f"  {r['pair']}: diff={r['mean_diff']:+.3f}  p_adj={r['p_adj']:.4f}{extra}")

    print("\n=== Games-Howell (unequal-variance) ===")
    for r in games_howell(groups, labels):
        print(f"  {r['pair']}: diff={r['mean_diff']:+.3f}  df={r['df']:.1f}  p_adj={r['p_adj']:.4f}")

    print("\n=== Scheffe (default = all pairwise) ===")
    for r in scheffe(groups, labels=labels):
        active = [(labels[i], c) for i, c in enumerate(r["contrast"]) if c != 0]
        print(f"  {active}: L={r['L']:+.3f}  F={r['F']:.3f}  p_adj={r['p_adj']:.4f}")
    print("  Scheffe also handles arbitrary contrasts, e.g. (A+B)/2 vs (C+D)/2:")
    custom = [[0.5, 0.5, -0.5, -0.5]]
    for r in scheffe(groups, contrasts=custom, labels=labels):
        print(f"  contrast {r['contrast']}: L={r['L']:+.3f}  F={r['F']:.3f}  p_adj={r['p_adj']:.4f}")

    print("\n=== Tamhane T2 (Welch + Sidak) ===")
    for r in tamhane_t2(groups, labels):
        print(f"  {r['pair']}: diff={r['mean_diff']:+.3f}  p_raw={r['p_raw']:.4f}  p_adj={r['p_adj']:.4f}")

    print("\n=== Dunnett T3 (Welch + studentized max modulus, approx) ===")
    for r in dunnett_t3(groups, labels):
        print(f"  {r['pair']}: diff={r['mean_diff']:+.3f}  p_raw={r['p_raw']:.4f}  p_adj={r['p_adj']:.4f}")

    print("\n--- library (scipy.stats.tukey_hsd) ---")
    for k, v in library_versions(groups, labels).items():
        print(f"  {k}: {v}")
