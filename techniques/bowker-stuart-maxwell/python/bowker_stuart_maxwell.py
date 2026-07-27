"""Bowker's test of symmetry and Stuart-Maxwell test of marginal homogeneity
(Reference §8.7, §8.15).

Both operate on a paired K x K table (same categories on rows and columns:
before/after, rater1/rater2, ...). They are the natural extensions of
McNemar's test to K > 2 categories.

Bowker's symmetry test
----------------------
H0: p_ij = p_ji for all i != j (the paired table is symmetric).
    X2 = sum_{i < j} (n_ij - n_ji)^2 / (n_ij + n_ji)      ~ chi^2 with K(K-1)/2 df.

Stuart-Maxwell marginal homogeneity
-----------------------------------
H0: row marginals == column marginals (each category has the same total probability
    before and after). Symmetry implies marginal homogeneity but not vice versa.
    Let d = (row_i - col_i)_{i=1..K-1} (drop the last, since they sum to 0).
    Let V be the (K-1) x (K-1) covariance matrix with
        V_ii = row_i + col_i - 2 n_ii
        V_ij = -(n_ij + n_ji)     for i != j
    X2 = d^T V^{-1} d      ~ chi^2 with (K - 1) df.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def build_paired_table(rater1, rater2, categories=None) -> tuple:
    """Build a K x K paired table from two parallel sequences of category labels."""
    if len(rater1) != len(rater2):
        raise ValueError("rater1 and rater2 must have equal length")
    if categories is None:
        categories = sorted(set(rater1) | set(rater2))
    idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    m = np.zeros((K, K), dtype=int)
    for a, b in zip(rater1, rater2):
        m[idx[a], idx[b]] += 1
    return categories, m


def bowker_test(table) -> dict:
    """Bowker's test of symmetry on a K x K paired table.

    Reduces to McNemar's chi-square (without CC) when K = 2.
    """
    m = np.asarray(table, dtype=int)
    K = m.shape[0]
    if K < 2 or m.shape[1] != K:
        raise ValueError("table must be square (same categories rows and cols)")
    stat = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            n_ij = m[i, j]; n_ji = m[j, i]
            if n_ij + n_ji == 0:
                # 0/0 -> conventionally 0 contribution; df unchanged
                continue
            stat += (n_ij - n_ji) ** 2 / (n_ij + n_ji)
    df = K * (K - 1) // 2
    p = float(stats.chi2.sf(stat, df))
    return {"chi_square": float(stat), "df": df, "p_value": p,
            "K": K, "method": "Bowker's test of symmetry"}


def stuart_maxwell_test(table) -> dict:
    """Stuart-Maxwell test of marginal homogeneity on a K x K paired table."""
    m = np.asarray(table, dtype=float)
    K = m.shape[0]
    if K < 2 or m.shape[1] != K:
        raise ValueError("table must be square")
    row = m.sum(axis=1); col = m.sum(axis=0)
    # If all marginals equal, statistic is 0
    d_full = row - col                       # length K, sums to 0
    if np.allclose(d_full, 0):
        return {"chi_square": 0.0, "df": K - 1, "p_value": 1.0,
                "K": K, "method": "Stuart-Maxwell marginal homogeneity"}
    d = d_full[:K - 1]                       # drop the last
    V = np.zeros((K - 1, K - 1))
    for i in range(K - 1):
        V[i, i] = row[i] + col[i] - 2 * m[i, i]
        for j in range(K - 1):
            if i != j:
                V[i, j] = -(m[i, j] + m[j, i])
    try:
        stat = float(d @ np.linalg.solve(V, d))
    except np.linalg.LinAlgError:
        # singular V -> reduce categories that never disagree; fall back to pinv
        stat = float(d @ np.linalg.pinv(V) @ d)
    df = K - 1
    p = float(stats.chi2.sf(stat, df))
    return {"chi_square": stat, "df": df, "p_value": p,
            "K": K, "method": "Stuart-Maxwell marginal homogeneity"}


def run_all(rater1, rater2):
    cats, m = build_paired_table(rater1, rater2)
    return {"categories": cats,
            "paired_table": m.tolist(),
            "bowker": bowker_test(m),
            "stuart_maxwell": stuart_maxwell_test(m)}


def library_versions(rater1, rater2):
    from statsmodels.stats.contingency_tables import SquareTable
    cats, m = build_paired_table(rater1, rater2)
    # shift_zeros=False -> pure textbook formula (matches our from-scratch);
    # statsmodels' default (shift_zeros=True) adds +0.5 to zero cells.
    sq_raw = SquareTable(m, shift_zeros=False)
    sq_shift = SquareTable(m, shift_zeros=True)
    return {"statsmodels symmetry (Bowker)":
                {"stat": float(sq_raw.symmetry().statistic),
                 "df": int(sq_raw.symmetry().df),
                 "p": float(sq_raw.symmetry().pvalue)},
            "statsmodels SM (shift_zeros=False, matches from-scratch)":
                {"stat": float(sq_raw.homogeneity().statistic),
                 "df": int(sq_raw.homogeneity().df),
                 "p": float(sq_raw.homogeneity().pvalue)},
            "statsmodels SM (shift_zeros=True, statsmodels default)":
                {"stat": float(sq_shift.homogeneity().statistic),
                 "p": float(sq_shift.homogeneity().pvalue)}}


if __name__ == "__main__":
    import random
    random.seed(5)
    cats = ["improved", "stable", "worsened"]
    n = 300
    # Simulate before/after with a mild "improve" shift
    before = [random.choice(cats) for _ in range(n)]
    idx = {c: i for i, c in enumerate(cats)}
    def transition(c):
        j = idx[c]
        # 60% stay, 30% improve, 10% worsen
        r = random.random()
        if r < 0.60: return c
        if r < 0.90 and j > 0: return cats[j - 1]
        if j < len(cats) - 1: return cats[j + 1]
        return c
    after = [transition(c) for c in before]

    out = run_all(before, after)
    print("=== Paired table ===")
    print("  categories:", out["categories"])
    for row in out["paired_table"]:
        print(" ", row)

    print("\n=== Bowker's symmetry ===")
    for k, v in out["bowker"].items(): print(f"  {k:16s}: {v}")

    print("\n=== Stuart-Maxwell marginal homogeneity ===")
    for k, v in out["stuart_maxwell"].items(): print(f"  {k:16s}: {v}")

    print("\n--- library (statsmodels SquareTable) ---")
    for k, v in library_versions(before, after).items():
        print(f"  {k}: {v}")
