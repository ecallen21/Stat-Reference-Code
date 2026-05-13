"""Goodman-Kruskal gamma, Somers' D, and Kendall's tau-b on contingency tables
(Reference §4.11, §4.12).

All three are association measures for ORDINAL-by-ORDINAL contingency tables,
defined in terms of concordant and discordant *pairs* of observations:

  C = #concordant pairs    -- both rankings agree on which obs is higher
  D = #discordant pairs    -- rankings disagree
  T_r = #pairs tied in row (X) only
  T_c = #pairs tied in col (Y) only
  T_rc = #pairs tied in both

  Goodman-Kruskal gamma   = (C - D) / (C + D)          ignores all ties
  Somers' D(y | x)        = (C - D) / (C + D + T_c)    asymmetric: predict y FROM x
  Somers' D(x | y)        = (C - D) / (C + D + T_r)    asymmetric the other way
  Kendall's tau-b         = (C - D) / sqrt((C + D + T_r)(C + D + T_c))

All four are in [-1, 1]. Gamma is the most "optimistic" (ties hurt none of its
denominator); tau-b is the most "pessimistic"; Somers' D sits between them.

Counting C, D, T from a contingency table is O((rc)^2). Closed form:

  C = sum_{i, j} n_{ij} * (sum_{r > i, s > j} n_{rs})   # cells strictly upper-right
  D = sum_{i, j} n_{ij} * (sum_{r > i, s < j} n_{rs})   # cells strictly lower-right
  T_r = sum_{i, j} n_{ij} * (sum_{r > i, s = j} n_{rs}) # same column, lower row
  T_c = sum_{i, j} n_{ij} * (sum_{r = i, s > j} n_{rs}) # same row, later column
"""
from __future__ import annotations

import numpy as np


def _ordinal_pair_counts(table):
    """Concordant C, Discordant D, row-only ties T_r, column-only ties T_c."""
    obs = np.asarray(table, dtype=int)
    r, c = obs.shape
    C = D = Tr = Tc = 0
    for i in range(r):
        for j in range(c):
            nij = int(obs[i, j])
            if nij == 0: continue
            # strictly greater on both axes -> concordant
            if i + 1 < r and j + 1 < c:
                C += nij * int(obs[i + 1:, j + 1:].sum())
            # strictly greater on row, strictly less on col -> discordant
            if i + 1 < r and j > 0:
                D += nij * int(obs[i + 1:, :j].sum())
            # same row, later columns -> tie on row only (T_c contributes T_c, hmm)
            # Wait: same row index (i == r), j' > j -> ties in ROW (x), differ in col (y)
            # So these are "T_r"-style ties? Let me re-check convention.
            # Convention used here:
            #   T_r = ties on ROW variable X only (=> same row index, different col)
            #   T_c = ties on COL variable Y only (=> different row, same col index)
            pass

    # cleaner second pass for the tie counts so the bookkeeping is obvious
    for i in range(r):
        # ties on X (same row), differing on Y => same row, j != j2
        for j in range(c):
            for j2 in range(j + 1, c):
                Tr += int(obs[i, j]) * int(obs[i, j2])
    for j in range(c):
        for i in range(r):
            for i2 in range(i + 1, r):
                Tc += int(obs[i, j]) * int(obs[i2, j])
    return C, D, Tr, Tc


def goodman_kruskal_gamma(table) -> float:
    """gamma = (C - D) / (C + D). Ignores all ties."""
    C, D, _, _ = _ordinal_pair_counts(table)
    return (C - D) / (C + D) if (C + D) > 0 else float("nan")


def somers_d_y_given_x(table) -> float:
    """D(y | x) = (C - D) / (C + D + T_y), where T_y counts pairs tied on Y only.

    In this convention T_y corresponds to pairs sharing the same column index --
    these are "Tc" in ``_ordinal_pair_counts``.
    """
    C, D, _, Tc = _ordinal_pair_counts(table)
    denom = C + D + Tc
    return (C - D) / denom if denom > 0 else float("nan")


def somers_d_x_given_y(table) -> float:
    """D(x | y) = (C - D) / (C + D + T_x), where T_x counts pairs tied on X only."""
    C, D, Tr, _ = _ordinal_pair_counts(table)
    denom = C + D + Tr
    return (C - D) / denom if denom > 0 else float("nan")


def kendall_tau_b_table(table) -> float:
    """tau-b on a contingency table = (C - D) / sqrt((C+D+T_x)(C+D+T_y))."""
    import math
    C, D, Tr, Tc = _ordinal_pair_counts(table)
    denom = math.sqrt((C + D + Tr) * (C + D + Tc))
    return (C - D) / denom if denom > 0 else float("nan")


def all_ordinal_associations(table) -> dict:
    C, D, Tr, Tc = _ordinal_pair_counts(table)
    return {"C": C, "D": D, "T_x_only": Tr, "T_y_only": Tc,
            "gamma": goodman_kruskal_gamma(table),
            "somers_d_y_given_x": somers_d_y_given_x(table),
            "somers_d_x_given_y": somers_d_y_given_x(table.T if hasattr(table, 'T')
                                                    else list(map(list, zip(*table)))),
            "kendall_tau_b": kendall_tau_b_table(table)}


if __name__ == "__main__":
    # Strong positive ordinal association: rows = education, cols = income tier
    table = np.array([
        [40,  20,  5],
        [15,  30, 20],
        [ 5,  15, 50],
    ])
    print("=== Ordinal-by-ordinal table ===")
    print(table)
    for k, v in all_ordinal_associations(table).items():
        print(f"  {k:24s}: {v}")
