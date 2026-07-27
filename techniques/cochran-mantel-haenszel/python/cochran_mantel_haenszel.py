"""Cochran-Mantel-Haenszel test and Mantel-Haenszel OR estimator (Reference §8.3, §8.16).

Setting
-------
K independent strata, each a 2x2 table of (exposure) x (outcome):

              outcome=1   outcome=0
    expo=1     a_k          b_k       n1_k
    expo=0     c_k          d_k       n0_k
              m1_k         m0_k       N_k

Under the null of *conditional independence* (no association within any stratum),
the CMH statistic is chi^2_1-distributed:

    X_CMH^2 = ( sum_k [a_k - E(a_k)] )^2 / sum_k Var(a_k)

with E(a_k) = n1_k * m1_k / N_k
     Var(a_k) = (n1_k * n0_k * m1_k * m0_k) / (N_k^2 * (N_k - 1))    (hypergeometric)

Mantel-Haenszel common OR estimator (efficient under homogeneity):
    OR_MH = sum_k a_k * d_k / N_k  /  sum_k b_k * c_k / N_k

Robins-Breslow-Greenland (RBG) SE for log OR_MH -- the standard software SE:
    Var(log OR_MH) = sum(P R) / (2 (sum R)^2)
                   + sum(P S + Q R) / (2 sum R sum S)
                   + sum(Q S) / (2 (sum S)^2)
    with P = (a+d)/N, Q = (b+c)/N, R = a*d/N, S = b*c/N.

Woolf's test is a simpler heterogeneity test than Breslow-Day (see the
`breslow-day` technique for the standard homogeneity test).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _unpack(table):
    """Take a 2x2 [[a, b], [c, d]] and return (a, b, c, d, N, n1, n0, m1, m0)."""
    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])
    N = a + b + c + d
    n1 = a + b; n0 = c + d
    m1 = a + c; m0 = b + d
    return a, b, c, d, N, n1, n0, m1, m0


def cmh_test(tables, continuity: bool = False) -> dict:
    """Cochran-Mantel-Haenszel test across K strata.

    Parameters
    ----------
    tables : sequence of 2x2 tables ``[[a_k, b_k], [c_k, d_k]]``, one per stratum.
    continuity : if True, subtract 0.5 from |numerator| before squaring
        (Mantel-Haenszel continuity correction). Off by default.

    Returns
    -------
    dict with the CMH statistic, df, p-value, and per-stratum contributions.
    """
    numer = 0.0
    denom = 0.0
    per_stratum = []
    for k, tbl in enumerate(tables):
        a, b, c, d, N, n1, n0, m1, m0 = _unpack(tbl)
        if N <= 1:
            per_stratum.append({"stratum": k, "N": N, "skipped": True})
            continue
        e_a = n1 * m1 / N
        var_a = (n1 * n0 * m1 * m0) / (N * N * (N - 1))
        numer += a - e_a
        denom += var_a
        per_stratum.append({"stratum": k, "N": N, "a": a,
                            "E(a)": e_a, "Var(a)": var_a})
    if denom == 0:
        return {"chi_square": 0.0, "df": 1, "p_value": 1.0,
                "note": "zero total variance; test undefined",
                "per_stratum": per_stratum}
    num2 = (abs(numer) - (0.5 if continuity else 0.0))
    num2 = max(0.0, num2)
    x2 = num2 * num2 / denom
    return {"chi_square": x2, "df": 1,
            "p_value": float(stats.chi2.sf(x2, 1)),
            "continuity": continuity,
            "sum_(a - E)": numer, "sum_var": denom,
            "K_strata": len(tables),
            "per_stratum": per_stratum,
            "method": "Cochran-Mantel-Haenszel"}


def mh_common_odds_ratio(tables) -> dict:
    """Mantel-Haenszel common odds ratio + Robins-Breslow-Greenland log-CI."""
    R = S = 0.0
    sum_PR = sum_PS_QR = sum_QS = 0.0
    for tbl in tables:
        a, b, c, d, N, *_ = _unpack(tbl)
        if N == 0: continue
        R_k = a * d / N
        S_k = b * c / N
        P_k = (a + d) / N
        Q_k = (b + c) / N
        R += R_k; S += S_k
        sum_PR += P_k * R_k
        sum_PS_QR += P_k * S_k + Q_k * R_k
        sum_QS += Q_k * S_k
    if S == 0 or R == 0:
        return {"OR_MH": float("inf") if R > 0 else 0.0, "SE_log_OR": float("nan"),
                "note": "zero denominator or numerator across all strata"}
    or_mh = R / S
    var_log = sum_PR / (2 * R * R) + sum_PS_QR / (2 * R * S) + sum_QS / (2 * S * S)
    se_log = math.sqrt(var_log)
    z = stats.norm.ppf(0.975)
    lo = math.exp(math.log(or_mh) - z * se_log)
    hi = math.exp(math.log(or_mh) + z * se_log)
    return {"OR_MH": or_mh, "log_OR_SE_RBG": se_log,
            "CI95_lower": lo, "CI95_upper": hi,
            "method": "Mantel-Haenszel OR + Robins-Breslow-Greenland SE"}


def woolf_homogeneity(tables) -> dict:
    """Woolf's test for homogeneity of stratum-specific ORs.

    Weighted-least-squares chi-square on log(OR) around the inverse-variance-weighted
    mean. See ``breslow-day`` for the more common Breslow-Day test.
    """
    log_ors = []
    weights = []
    for tbl in tables:
        a, b, c, d, N, *_ = _unpack(tbl)
        # Continuity: add 0.5 to every cell only if any cell is zero
        if 0 in (a, b, c, d):
            a += 0.5; b += 0.5; c += 0.5; d += 0.5
        or_k = (a * d) / (b * c)
        var_k = 1 / a + 1 / b + 1 / c + 1 / d
        log_ors.append(math.log(or_k))
        weights.append(1 / var_k)
    log_ors = np.array(log_ors); weights = np.array(weights)
    log_or_bar = float((weights * log_ors).sum() / weights.sum())
    x2 = float((weights * (log_ors - log_or_bar) ** 2).sum())
    K = len(tables)
    return {"chi_square": x2, "df": K - 1,
            "p_value": float(stats.chi2.sf(x2, K - 1)),
            "log_OR_pooled": log_or_bar,
            "OR_pooled_Woolf": math.exp(log_or_bar),
            "method": "Woolf's homogeneity test"}


def run_all(tables) -> dict:
    return {
        "cmh_test": cmh_test(tables),
        "cmh_test_continuity": cmh_test(tables, continuity=True),
        "mh_odds_ratio": mh_common_odds_ratio(tables),
        "woolf_homogeneity": woolf_homogeneity(tables),
    }


def library_versions(tables):
    from statsmodels.stats.contingency_tables import StratifiedTable
    arr = np.array(tables)          # shape (K, 2, 2) -- statsmodels wants (2, 2, K)
    st = StratifiedTable(np.transpose(arr, (1, 2, 0)))
    return {
        "statsmodels CMH (Bhapkar)": {
            "stat": float(st.test_null_odds().statistic),
            "p": float(st.test_null_odds().pvalue),
        },
        "statsmodels OR_MH": float(st.oddsratio_pooled),
        "statsmodels OR_MH log-SE": float(st.logodds_pooled_se),
        "statsmodels test_equal_odds (Breslow-Day)": {
            "stat": float(st.test_equal_odds().statistic),
            "p": float(st.test_equal_odds().pvalue),
        },
    }


if __name__ == "__main__":
    # 3 strata (e.g. age groups): mild positive association in each
    tables = [
        [[36, 44], [24, 56]],   # stratum 1
        [[30, 50], [22, 58]],   # stratum 2
        [[40, 30], [28, 52]],   # stratum 3
    ]
    print("=== 3 strata ===")
    for k, t in enumerate(tables):
        print(f"  stratum {k}: {t}")

    print("\n=== From-scratch ===")
    for k, v in run_all(tables).items():
        if k == "cmh_test" or k == "cmh_test_continuity":
            print(f"  {k}: chi2={v['chi_square']:.4f}, p={v['p_value']:.4g}")
        else:
            print(f"  {k}: {v}")

    print("\n--- library (statsmodels StratifiedTable) ---")
    for k, v in library_versions(tables).items():
        print(f"  {k}: {v}")
