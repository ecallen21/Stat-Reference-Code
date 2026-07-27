"""McNemar's test for paired binary outcomes (Reference §8.2, §8.18).

Setup
-----
Each subject contributes ONE pair of binary responses (e.g. before/after,
or matched case/control). The 2x2 table of PAIRS looks like:

                       After
                   +  |  -  |
                +--+-----+
          Before | a |  b  |
                +--+-----+
                | c |  d  |
                +--+-----+

McNemar's H0 says the marginals of the paired table are equal, which reduces
to: b == c (the two OFF-DIAGONAL / DISCORDANT cells match). Concordant pairs
(a, d) carry no information about a treatment effect.

Statistics computed
-------------------
1. **Asymptotic chi-square**            X2 = (b - c)^2 / (b + c)     ~ chi2_1
2. **Continuity-corrected (Edwards)**   X2 = (|b - c| - 1)^2 / (b + c)
3. **Exact (binomial on discordants)**  b ~ Binomial(b+c, 0.5) under H0
4. **Mid-p** (half of the point-probability added; better calibration)
5. **McNemar odds ratio**  OR = b / c   (ratio of discordant pairs)
6. **Newcombe CI** for the difference of proportions (paired, Wilson-based)

Use exact/mid-p when b + c is small (< ~25). For large discordant totals the
asymptotic chi-square (without continuity correction) is accurate.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def build_paired_table(before: Sequence[int], after: Sequence[int]) -> list:
    """Build the 2x2 paired table from two parallel 0/1 sequences.

    Parameters
    ----------
    before, after : 0/1 sequences of equal length; ``before[i]`` and ``after[i]``
        are the two responses from subject i.

    Returns
    -------
    ``[[a, b], [c, d]]`` where a = both +, b = before+/after-, c = before-/after+,
    d = both -.
    """
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")
    a = b = c = d = 0
    for x, y in zip(before, after):
        x = int(x); y = int(y)
        if x == 1 and y == 1: a += 1
        elif x == 1 and y == 0: b += 1
        elif x == 0 and y == 1: c += 1
        else: d += 1
    return [[a, b], [c, d]]


def mcnemar(table, continuity: bool = False) -> dict:
    """Asymptotic McNemar chi-square on a 2x2 paired table.

    Parameters
    ----------
    table : ``[[a, b], [c, d]]`` (as returned by :func:`build_paired_table`).
    continuity : if True, use Edwards' continuity correction. Recommended off
        unless the discordant total b+c is small AND you can't use the exact test.
    """
    b, c = int(table[0][1]), int(table[1][0])
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "chi_square": 0.0,
                "df": 1, "p_value": 1.0, "continuity": continuity,
                "note": "all pairs concordant; test is undefined"}
    diff = abs(b - c) - (1 if continuity else 0)
    if diff < 0: diff = 0
    x2 = diff * diff / n_disc
    p = float(stats.chi2.sf(x2, 1))
    return {"b": b, "c": c, "n_discordant": n_disc,
            "chi_square": x2, "df": 1, "p_value": p,
            "continuity": continuity,
            "method": "McNemar (asymptotic)" + (" with continuity correction" if continuity else "")}


def mcnemar_exact(table, mid_p: bool = False) -> dict:
    """Exact McNemar test: two-sided binomial(b+c, 0.5) test on b.

    ``mid_p=True`` uses the mid-p adjustment (subtract half the point probability
    of the observed count) -- less conservative and better calibrated.
    """
    b, c = int(table[0][1]), int(table[1][0])
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0,
                "mid_p": mid_p, "method": "McNemar exact"}
    k = min(b, c)
    # two-sided exact: 2 * P(X <= k) capped at 1
    p_tail = float(stats.binom.cdf(k, n, 0.5))
    p_two = min(1.0, 2.0 * p_tail)
    if mid_p:
        # subtract half the point mass at k from EACH tail (symmetric)
        pmf_k = float(stats.binom.pmf(k, n, 0.5))
        p_two = min(1.0, max(0.0, p_two - pmf_k))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_two,
            "mid_p": mid_p,
            "method": "McNemar exact" + (" (mid-p)" if mid_p else "")}


def mcnemar_odds_ratio(table) -> dict:
    """OR = b / c and Wald log-CI. Guards against zeros with 0.5 continuity."""
    b, c = int(table[0][1]), int(table[1][0])
    b_adj, c_adj = (b if b else 0.5), (c if c else 0.5)
    or_hat = b_adj / c_adj
    se_log = math.sqrt(1.0 / b_adj + 1.0 / c_adj)
    z = stats.norm.ppf(0.975)
    lo = math.exp(math.log(or_hat) - z * se_log)
    hi = math.exp(math.log(or_hat) + z * se_log)
    return {"OR": or_hat, "log_OR_SE": se_log,
            "CI95_lower": lo, "CI95_upper": hi,
            "note": "0.5 continuity added if b or c is 0"}


def newcombe_paired_diff_ci(table, conf: float = 0.95) -> dict:
    """Newcombe's method 10 for the difference of paired proportions p1 - p2,
    where p1 = (a + b) / n  and  p2 = (a + c) / n  (marginal +ve rates).

    This is the paired-CI companion to McNemar's test -- it gives the size of
    the marginal shift, not just its significance.
    """
    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])
    n = a + b + c + d
    if n == 0:
        raise ValueError("empty table")
    p1 = (a + b) / n
    p2 = (a + c) / n
    z = stats.norm.ppf(0.5 + conf / 2)

    def wilson(x, nn):
        if nn == 0: return 0.0, 0.0
        p = x / nn
        z2 = z * z
        denom = 1 + z2 / nn
        center = (p + z2 / (2 * nn)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / nn + z2 / (4 * nn * nn))
        return max(0.0, center - half), min(1.0, center + half)

    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    # Correlation between paired proportions
    if 0 < p1 < 1 and 0 < p2 < 1 and n > 0:
        phi = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d)) \
              if (a + b) * (c + d) * (a + c) * (b + d) > 0 else 0.0
    else:
        phi = 0.0
    delta = p1 - p2
    lower = delta - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    upper = delta + math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return {"p1_marg": p1, "p2_marg": p2, "diff": delta,
            "CI_lower": max(-1.0, lower), "CI_upper": min(1.0, upper),
            "conf": conf, "method": "Newcombe method 10 (paired)"}


def run_all(before, after) -> dict:
    table = build_paired_table(before, after)
    return {
        "table": table,
        "asymptotic": mcnemar(table, continuity=False),
        "continuity_corrected": mcnemar(table, continuity=True),
        "exact": mcnemar_exact(table, mid_p=False),
        "mid_p": mcnemar_exact(table, mid_p=True),
        "odds_ratio": mcnemar_odds_ratio(table),
        "newcombe_diff_CI95": newcombe_paired_diff_ci(table, 0.95),
    }


def library_versions(before, after):
    import numpy as np
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    table = np.array(build_paired_table(before, after))
    res_asym = sm_mcnemar(table, exact=False, correction=False)
    res_cc = sm_mcnemar(table, exact=False, correction=True)
    res_exact = sm_mcnemar(table, exact=True)
    return {
        "statsmodels asymptotic (no CC)": {"stat": float(res_asym.statistic), "p": float(res_asym.pvalue)},
        "statsmodels asymptotic (Yates CC)": {"stat": float(res_cc.statistic), "p": float(res_cc.pvalue)},
        "statsmodels exact": {"stat": float(res_exact.statistic), "p": float(res_exact.pvalue)},
    }


if __name__ == "__main__":
    import random
    random.seed(7)
    # 300 subjects; the treatment nudges the marginal + rate up by ~10 points
    before = [1 if random.random() < 0.35 else 0 for _ in range(300)]
    after = []
    for x in before:
        # someone who was + tends to stay + with prob 0.9; someone who was - tends to become + with prob 0.15
        p_pos = 0.90 if x == 1 else 0.15
        after.append(1 if random.random() < p_pos else 0)

    print("=== McNemar suite ===")
    out = run_all(before, after)
    for k, v in out.items():
        print(f"  {k:22s}: {v}")

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(before, after).items():
        print(f"  {k:34s}: {v}")
