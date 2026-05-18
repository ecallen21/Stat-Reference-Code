"""Fisher's exact test (Reference §3.6).

For a 2x2 contingency table

           col 1   col 2   row totals
    row 1    a       b        a + b
    row 2    c       d        c + d
    totals  a+c    b+d      n = a+b+c+d

Fisher's exact test conditions on the row AND column margins and computes the
exact probability (under H0: independence / OR = 1) of seeing a table at least
as extreme as the observed one, via the hypergeometric distribution:

    P(X = a | margins) = C(a+b, a) * C(c+d, c) / C(n, a+c)

Use it instead of chi-square when expected counts are small (any E_ij < ~5).
Generalizes to r x c via "Fisher-Freeman-Halton" (scipy / R both implement it).

Effect size: the sample odds ratio  OR = (a*d) / (b*c).
A common CI for log(OR) is the Wald interval based on
    SE(log OR) = sqrt(1/a + 1/b + 1/c + 1/d)
(with the Haldane-Anscombe 0.5 correction for any zero cell). The "conditional
exact" CI inverts Fisher's test on OR and is available via
``scipy.stats.fisher_exact`` returning a confidence interval (1.11+).
"""
from __future__ import annotations

import math


def _hypergeom_pmf(a: int, n1: int, n2: int, k: int) -> float:
    """P(X = a) under H_0 for the 2x2 with row totals n1, n2 and col total k."""
    return (math.comb(n1, a) * math.comb(n2, k - a)) / math.comb(n1 + n2, k)


def fisher_exact_2x2(a: int, b: int, c: int, d: int,
                     alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Fisher's exact test for a 2x2 table.

    Parameters
    ----------
    a, b, c, d : the four cell counts (row-major: a=top-left, d=bottom-right).
    alternative : "two-sided" / "greater" (OR > 1) / "less" (OR < 1).
    conf : confidence level for the Wald log-OR interval.

    Returns the observed odds ratio, the table-probability-based p-value
    (two-sided uses the standard "sum of probabilities <= observed" definition,
    matching scipy.stats.fisher_exact), and a Wald log-OR CI on the OR.
    """
    n1, n2 = a + b, c + d              # row totals
    k = a + c                          # column 1 total
    nmax = n1 + n2

    # All admissible values of a given the margins:
    a_min = max(0, k - n2)
    a_max = min(n1, k)
    pmf = {ai: _hypergeom_pmf(ai, n1, n2, k) for ai in range(a_min, a_max + 1)}
    p_obs = pmf[a]

    if alternative == "two-sided":
        p_value = sum(p for p in pmf.values() if p <= p_obs + 1e-12)
    elif alternative == "greater":
        p_value = sum(p for ai, p in pmf.items() if ai >= a)
    elif alternative == "less":
        p_value = sum(p for ai, p in pmf.items() if ai <= a)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Odds ratio + Wald log-OR CI (Haldane-Anscombe correction on zero cells)
    if 0 in (a, b, c, d):
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d
    or_hat = (a_ * d_) / (b_ * c_)
    log_or = math.log(or_hat)
    se = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    from scipy import stats
    z = stats.norm.ppf(0.5 + conf / 2)
    ci_lower = math.exp(log_or - z * se)
    ci_upper = math.exp(log_or + z * se)
    return {"table": [[a, b], [c, d]], "p_observed": p_obs,
            "p_value": min(1.0, p_value), "odds_ratio": or_hat,
            "log_or_se": se, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "alternative": alternative}


def library_versions(a, b, c, d):
    from scipy import stats
    odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
    return {"scipy.stats.fisher_exact": (float(odds_ratio), float(p))}


if __name__ == "__main__":
    # Classic small-table example: cured vs not, by treatment
    # treatment: 8 cured, 2 not   |   control: 3 cured, 7 not
    a, b, c, d = 8, 2, 3, 7
    print(f"=== 2x2 = [[{a}, {b}], [{c}, {d}]] ===")
    for k, v in fisher_exact_2x2(a, b, c, d).items():
        print(f"  {k:14s}: {v}")
    print("\n--- library (scipy.stats.fisher_exact) ---")
    for k, v in library_versions(a, b, c, d).items():
        print(f"  {k}: {v}")

    # One-sided alternatives
    print("\n=== one-sided 'greater' (OR > 1) ===")
    print(fisher_exact_2x2(a, b, c, d, alternative="greater"))

    # Edge case with a zero cell
    print("\n=== with a zero cell [[0, 9], [6, 3]] ===")
    for k, v in fisher_exact_2x2(0, 9, 6, 3).items():
        print(f"  {k:14s}: {v}")
