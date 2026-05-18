"""Chi-square tests (Reference §3.5).

Two classical uses of Pearson's X^2 statistic on counts:

  - Goodness-of-fit (GOF) : do observed category counts match a hypothesized
                            distribution?  df = k - 1 - (#params estimated).
  - Test of independence  : in an r x c contingency table, are the row and
                            column variables independent?  df = (r-1)(c-1).
                            Expected counts under independence:
                              E_ij = (row_i * col_j) / total

Both reduce to:
        X^2 = sum  (O - E)^2 / E
and compare to a chi-square distribution.

Yates' continuity correction (2x2 only): use |O - E| - 0.5 in the numerator.
Standardized (Pearson) residuals: r_ij = (O_ij - E_ij) / sqrt(E_ij);
``|r_ij| > 2`` flags cells that drive the result.

Rule of thumb: chi-square is approximate -- want E_ij >= 5 for most cells.
Below that, prefer Fisher's exact test (``techniques/fisher-exact``).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def goodness_of_fit(observed: Sequence[int], expected_probs: Sequence[float] | None = None,
                    ddof: int = 0) -> dict:
    """Pearson goodness-of-fit chi-square.

    Parameters
    ----------
    observed : observed counts per category (length k).
    expected_probs : null-hypothesis probabilities (sum to 1). Default: uniform.
    ddof : number of parameters estimated from the data (subtract from df).

    Returns dict with the expected counts, X^2, df, and p-value.
    """
    observed = list(observed)
    n = sum(observed)
    k = len(observed)
    if expected_probs is None:
        expected_probs = [1 / k] * k
    if abs(sum(expected_probs) - 1.0) > 1e-9:
        raise ValueError("expected_probs must sum to 1")
    expected = [n * p for p in expected_probs]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = k - 1 - ddof
    return {"observed": observed, "expected": expected, "chi_square": chi2,
            "df": df, "p_value": float(stats.chi2.sf(chi2, df))}


def independence(table: Sequence[Sequence[int]], correction: bool = False) -> dict:
    """Chi-square test of independence on an r x c contingency table.

    Parameters
    ----------
    table : 2D list/array of counts (rows x cols).
    correction : Yates' continuity correction (2x2 tables only).

    Returns
    -------
    dict with row/col totals, expected counts, X^2, df, p-value, Cramer's V
    (effect size for independence), and the Pearson standardized residuals.
    """
    r = len(table)
    c = len(table[0])
    row_tot = [sum(row) for row in table]
    col_tot = [sum(table[i][j] for i in range(r)) for j in range(c)]
    n = sum(row_tot)
    expected = [[row_tot[i] * col_tot[j] / n for j in range(c)] for i in range(r)]
    if correction and (r, c) == (2, 2):
        chi2 = sum(max(0.0, abs(table[i][j] - expected[i][j]) - 0.5) ** 2
                   / expected[i][j]
                   for i in range(r) for j in range(c))
    else:
        chi2 = sum((table[i][j] - expected[i][j]) ** 2 / expected[i][j]
                   for i in range(r) for j in range(c))
    df = (r - 1) * (c - 1)
    p = float(stats.chi2.sf(chi2, df))
    cramers_v = math.sqrt(chi2 / (n * min(r - 1, c - 1)))
    residuals = [[(table[i][j] - expected[i][j]) / math.sqrt(expected[i][j])
                  for j in range(c)] for i in range(r)]
    return {"row_totals": row_tot, "col_totals": col_tot, "n": n,
            "expected": expected, "chi_square": chi2, "df": df, "p_value": p,
            "cramers_v": cramers_v, "residuals": residuals,
            "warning_small_expected": any(e < 5 for row in expected for e in row)}


def library_versions(table):
    from scipy.stats import chi2_contingency, chisquare
    obs = [r[:] for r in table]
    chi2, p, df, exp = chi2_contingency(obs, correction=False)
    out = {"chi2_contingency (no correction)": (chi2, p, df)}
    if len(obs) == 2 and len(obs[0]) == 2:
        chi2c, pc, dfc, _ = chi2_contingency(obs, correction=True)
        out["chi2_contingency (Yates)"] = (chi2c, pc, dfc)
    # GOF example
    gof = chisquare(f_obs=[18, 22, 19, 21])
    out["chisquare(equal probs)"] = (float(gof.statistic), float(gof.pvalue))
    return out


if __name__ == "__main__":
    print("=== Goodness-of-fit: die fairness ===")
    rolls = [18, 22, 19, 16, 23, 22]   # 120 rolls of a six-sided die
    res = goodness_of_fit(rolls)
    for k, v in res.items():
        print(f"  {k:14s}: {v}")

    print("\n=== Independence: region x outcome ===")
    table = [[30, 10], [25, 15], [10, 20]]   # rows = North/South/East, cols = Yes/No
    res = independence(table)
    for k, v in res.items():
        if k == "residuals":
            print(f"  {k:22s}:")
            for row in v:
                print("   ", [round(x, 2) for x in row])
        else:
            print(f"  {k:22s}: {v}")

    print("\n=== 2x2 with Yates correction ===")
    twobytwo = [[8, 12], [15, 5]]
    for k, v in independence(twobytwo, correction=True).items():
        if k == "residuals":
            print(f"  {k:22s}: {[[round(x, 2) for x in r] for r in v]}")
        else:
            print(f"  {k:22s}: {v}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(table).items():
        print(f"  {k}: {v}")
