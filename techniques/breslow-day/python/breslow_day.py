"""Breslow-Day test for homogeneity of odds ratios across strata (Reference §8.6).

Given K 2x2 tables (one per stratum), tests

    H0: OR_1 = OR_2 = ... = OR_K = OR_common

The test statistic is a chi-square on (K - 1) df:

    BD = sum_k (a_k - E_k(OR_MH))^2 / Var_k(a_k | OR_MH)

where E_k and Var_k come from the noncentral hypergeometric with common odds
ratio equal to the Mantel-Haenszel estimate OR_MH. That E and Var need a
one-dimensional root-find per stratum (`a_k` is the unique real root of the
quadratic derived from a_k * d_k / (b_k * c_k) = OR_MH).

Tarone's correction improves the chi-square approximation:
    BD_Tarone = BD - (sum_k (a_k - E_k))^2 / sum_k Var_k

Use Tarone-corrected version by default; both are provided.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: root-find;  stats: distributions/tests


def _mh_or(tables) -> float:
    R = S = 0.0
    for tbl in tables:
        a, b = tbl[0]; c, d = tbl[1]
        N = a + b + c + d
        if N == 0: continue
        R += a * d / N
        S += b * c / N
    if S == 0: return float("inf")
    return R / S


def _expected_a_given_or(n1, m1, N, psi):
    """Solve for E[a] under noncentral hypergeometric with odds ratio psi.

    a d / b c = psi  =>  a (a - n1 - m1 + N) / ((n1 - a)(m1 - a)) = psi
    Solve the resulting quadratic for a in [max(0, n1+m1-N), min(n1, m1)].
    """
    lo = max(0, n1 + m1 - N)
    hi = min(n1, m1)
    if psi == 1.0:
        return n1 * m1 / N
    # Quadratic:  (psi - 1) a^2 - [psi (n1 + m1) + (N - n1 - m1)] a + psi n1 m1 = 0
    A = psi - 1
    B = -(psi * (n1 + m1) + (N - n1 - m1))
    C = psi * n1 * m1
    if abs(A) < 1e-12:
        return -C / B
    disc = B * B - 4 * A * C
    if disc < 0: disc = 0
    r1 = (-B + math.sqrt(disc)) / (2 * A)
    r2 = (-B - math.sqrt(disc)) / (2 * A)
    # pick the root inside [lo, hi]
    for r in (r1, r2):
        if lo - 1e-9 <= r <= hi + 1e-9:
            return min(max(r, lo), hi)
    # numerical fallback: bounded root
    return min(max(r1, lo), hi)


def _var_a(E_a, n1, m1, N):
    """Hypergeometric variance of a given E[a] under noncentral hypergeometric."""
    n0 = N - n1; m0 = N - m1
    # 1 / (1/E_a + 1/(n1 - E_a) + 1/(m1 - E_a) + 1/(N - n1 - m1 + E_a))
    parts = [E_a, n1 - E_a, m1 - E_a, n0 - m1 + E_a]
    if any(v <= 0 for v in parts):
        return 0.0
    return 1.0 / sum(1.0 / v for v in parts)


def breslow_day(tables, tarone: bool = True) -> dict:
    """Breslow-Day test for OR homogeneity across K 2x2 tables.

    ``tarone=True`` applies the Tarone correction (recommended).
    """
    or_mh = _mh_or(tables)
    if not math.isfinite(or_mh) or or_mh == 0:
        return {"chi_square": float("nan"), "df": max(0, len(tables) - 1),
                "p_value": float("nan"),
                "note": "MH OR undefined; homogeneity test cannot be computed"}
    stat = 0.0
    numer_sum = 0.0
    denom_sum = 0.0
    per_stratum = []
    for k, tbl in enumerate(tables):
        a, b = tbl[0]; c, d = tbl[1]
        n1 = a + b; m1 = a + c; N = a + b + c + d
        if N == 0: continue
        E_a = _expected_a_given_or(n1, m1, N, or_mh)
        V_a = _var_a(E_a, n1, m1, N)
        if V_a == 0:
            per_stratum.append({"stratum": k, "skipped": True})
            continue
        contrib = (a - E_a) ** 2 / V_a
        stat += contrib
        numer_sum += a - E_a
        denom_sum += V_a
        per_stratum.append({"stratum": k, "a": a, "E_a": E_a, "Var_a": V_a,
                            "contribution": contrib})
    K_eff = sum(1 for s in per_stratum if not s.get("skipped"))
    df = max(0, K_eff - 1)
    if tarone and denom_sum > 0:
        stat -= (numer_sum ** 2) / denom_sum
        stat = max(stat, 0.0)
        method = "Breslow-Day-Tarone"
    else:
        method = "Breslow-Day"
    p = float(stats.chi2.sf(stat, df)) if df > 0 else float("nan")
    return {"chi_square": float(stat), "df": df, "p_value": p,
            "OR_MH_used": or_mh, "K_strata": len(tables),
            "per_stratum": per_stratum,
            "method": method}


def library_versions(tables):
    from statsmodels.stats.contingency_tables import StratifiedTable
    arr = np.array(tables)
    st = StratifiedTable(np.transpose(arr, (1, 2, 0)))
    return {"statsmodels test_equal_odds (Breslow-Day)":
            {"stat": float(st.test_equal_odds().statistic),
             "p": float(st.test_equal_odds().pvalue)}}


if __name__ == "__main__":
    # Case 1: homogeneous (OR ~ 2 in each stratum) -- should NOT reject
    homo = [
        [[36, 44], [24, 56]],
        [[30, 50], [22, 58]],
        [[40, 30], [28, 52]],
    ]
    print("=== Homogeneous strata ===")
    print("  ", breslow_day(homo, tarone=True))
    print("  ", library_versions(homo))

    # Case 2: heterogeneous (OR flips across strata) -- should reject
    hetero = [
        [[40, 20], [10, 30]],    # OR ~ 6
        [[20, 40], [40, 20]],    # OR ~ 0.25
        [[35, 35], [30, 30]],    # OR ~ 1
    ]
    print("\n=== Heterogeneous strata ===")
    print("  ", breslow_day(hetero, tarone=True))
    print("  ", library_versions(hetero))
