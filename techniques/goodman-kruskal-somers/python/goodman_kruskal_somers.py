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
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import NamedTuple    # stdlib: NamedTuple = tuple with named fields (prints as Name(field=...))

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


class PairCounts(NamedTuple):
    """Pair-count summary used by all ordinal-association measures.

    Unpacks like a 4-tuple, so existing callers (``C, D, _, _ = _ordinal_pair_counts(t)``)
    keep working.
    """
    concordant: int       # C  (both X and Y strictly greater)
    discordant: int       # D  (X greater, Y smaller)
    ties_x_only: int      # T_x: same row, different column
    ties_y_only: int      # T_y: different row, same column


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
    return PairCounts(concordant=C, discordant=D, ties_x_only=Tr, ties_y_only=Tc)


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


# ===========================================================================
# Goodman-Kruskal lambda and Goodman-Kruskal tau (NOMINAL measures)
#
# These are proportional-reduction-in-error (PRE) measures: by how much does
# knowing X reduce the error in predicting Y? They are NOT ordinal -- you can
# permute rows/columns and the value doesn't change.
#
# Lambda(y | x)   "mode-based" PRE
#   P_e        = 1 - max_y(n_{.y}/N)                  prob of error without X
#   P_{e|X}    = (N - sum_x max_y n_{xy}) / N         prob of error given X
#   lambda     = (P_e - P_{e|X}) / P_e
#
# Tau(y | x)      "variance-based" PRE (Goodman-Kruskal's *own* tau)
#   V_y        = 1 - sum_y (n_{.y}/N)^2               Gini variance of Y
#   V_{y|X}    = 1 - sum_x (n_{x.}/N) sum_y (n_{xy}/n_{x.})^2
#   tau        = (V_y - V_{y|X}) / V_y
#
# Both are asymmetric (direction matters). Symmetric versions exist; we don't
# need them here. Note: G-K tau is *unrelated* to Kendall's tau!
# ===========================================================================
def gk_lambda(table, predict: str = "y_given_x") -> float:
    """Goodman-Kruskal lambda: PRE measure for nominal categorical data.

    ``predict``:
      "y_given_x"  -- reduction in error predicting column from row
      "x_given_y"  -- reduction in error predicting row from column
    """
    obs = np.asarray(table, dtype=float)
    if predict == "x_given_y":
        obs = obs.T
    elif predict != "y_given_x":
        raise ValueError("predict must be 'y_given_x' or 'x_given_y'")
    N = obs.sum()
    col_tot = obs.sum(axis=0)
    P_e = 1.0 - col_tot.max() / N
    # error given X: for each row x, the modal column predicts; misses = n_x. - max_y n_xy
    P_e_given_x = (N - obs.max(axis=1).sum()) / N
    return (P_e - P_e_given_x) / P_e if P_e > 0 else float("nan")


def gk_tau(table, predict: str = "y_given_x") -> float:
    """Goodman-Kruskal tau: variance-based PRE for nominal data.

    Independent of Kendall's tau-b despite the unfortunate name overlap.
    """
    obs = np.asarray(table, dtype=float)
    if predict == "x_given_y":
        obs = obs.T
    elif predict != "y_given_x":
        raise ValueError("predict must be 'y_given_x' or 'x_given_y'")
    N = obs.sum()
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    V_y = 1 - float(np.sum((col_tot / N) ** 2))
    # V_{y|X}
    V_y_given_x = 0.0
    for i, ri in enumerate(row_tot):
        if ri == 0: continue
        V_y_given_x += (ri / N) * (1 - np.sum((obs[i, :] / ri) ** 2))
    return (V_y - V_y_given_x) / V_y if V_y > 0 else float("nan")


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

    print("\n=== Nominal PRE measures on the SAME table ===")
    print(f"  G-K lambda  (y|x): {gk_lambda(table, 'y_given_x'):+.4f}")
    print(f"  G-K lambda  (x|y): {gk_lambda(table, 'x_given_y'):+.4f}")
    print(f"  G-K tau     (y|x): {gk_tau(table, 'y_given_x'):+.4f}")
    print(f"  G-K tau     (x|y): {gk_tau(table, 'x_given_y'):+.4f}")
    print("  (G-K tau != Kendall's tau-b -- it's a PRE / variance reduction measure.)")
