"""Fisher's exact and Barnard's exact tests for 2x2 tables (Reference §8.4).

Both test independence in a 2x2 contingency table without the chi-square
large-sample approximation.  They differ in how the null distribution is
constructed.

Fisher's exact (conditional)
    CONDITIONS on both row totals and column totals -- the marginals are
    treated as ancillary.  Under H_0, the count in the (1,1) cell is
    hypergeometric.
    Two-sided p-value: sum of hypergeometric probabilities of tables at
    least as extreme (by probability, following R's convention).

Barnard's exact (unconditional)
    Conditions only on the sample sizes n1, n2 (the two row totals).  The
    common success probability p under H_0 is a NUISANCE parameter --
    Barnard maximizes the p-value over p in [0, 1] (Barnard 1945).
    Uniformly more powerful than Fisher's on average (Berger & Boos 1994),
    but requires a supremum over p; the standard grid is 100-500 points.

Practical note
    Fisher's is more conservative and universally reported; Barnard's
    delivers modest power gains for small tables and is the recommended
    default in some modern references (Andres-Sanchez 1994).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def fisher_exact_2x2(table) -> dict:
    """Fisher's exact test (two-sided by probability-based rule)."""
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
    n = a + b + c + d; r1 = a + b; r2 = c + d
    c1 = a + c; c2 = b + d
    obs_logp = stats.hypergeom.logpmf(a, n, r1, c1)
    # Sum probability of tables with pmf <= observed
    p_two = 0.0
    for k in range(max(0, r1 - c2), min(r1, c1) + 1):
        lp = stats.hypergeom.logpmf(k, n, r1, c1)
        if lp <= obs_logp + 1e-12: p_two += math.exp(lp)
    # One-sided (right tail: k >= a)
    p_right = float(stats.hypergeom.sf(a - 1, n, r1, c1))
    # Odds ratio (with 0.5 continuity)
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_hat = (aa * dd) / (bb * cc)
    return {"table": [[a, b], [c, d]],
            "odds_ratio": float(or_hat),
            "p_two_sided": float(min(p_two, 1.0)),
            "p_one_sided_right": p_right,
            "method": "Fisher's exact test (conditional)"}


def _cnk(n, k): return math.comb(n, k)


def barnard_exact_2x2(table, n_grid: int = 200) -> dict:
    """Barnard's unconditional exact test.

    Grid-search over p in (0, 1) to maximize the p-value under H_0.
    Uses the score-like statistic T = (p1_hat - p2_hat) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    with pooled p_pool.  Two-sided p = sup_p Pr(|T| >= |T_obs| | H_0, p).
    """
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
    n1 = a + b; n2 = c + d
    if n1 == 0 or n2 == 0:
        return {"p_two_sided": 1.0, "table": [[a, b], [c, d]],
                "method": "Barnard's exact test"}

    def T_stat(x, y):
        # x successes out of n1, y out of n2
        p1 = x / n1; p2 = y / n2
        p_pool = (x + y) / (n1 + n2)
        denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if 0 < p_pool < 1 else 0.0
        return 0.0 if denom == 0 else (p1 - p2) / denom

    T_obs = abs(T_stat(a, c))
    # Enumerate all (x, y) tables and compute their probability grid
    ps = np.linspace(1e-4, 1 - 1e-4, n_grid)
    max_p = 0.0
    for p in ps:
        # Table probability P(X=x, Y=y | p) = Binom(n1, p, x) * Binom(n2, p, y)
        pmf1 = np.array([_cnk(n1, x) * p ** x * (1 - p) ** (n1 - x) for x in range(n1 + 1)])
        pmf2 = np.array([_cnk(n2, y) * p ** y * (1 - p) ** (n2 - y) for y in range(n2 + 1)])
        total = 0.0
        for x in range(n1 + 1):
            for y in range(n2 + 1):
                if abs(T_stat(x, y)) >= T_obs - 1e-12:
                    total += pmf1[x] * pmf2[y]
        if total > max_p: max_p = total
    return {"table": [[a, b], [c, d]],
            "p_two_sided": float(min(max_p, 1.0)),
            "T_stat_obs": float(T_obs),
            "n_grid": int(n_grid),
            "method": "Barnard's exact test (unconditional)"}


if __name__ == "__main__":
    print("=== Small imbalanced 2x2 table ===")
    tbl = [[3, 1], [1, 5]]  # 4-4 marginal-ish
    print(f"  table: {tbl}")
    r_f = fisher_exact_2x2(tbl)
    print(f"  Fisher OR = {r_f['odds_ratio']:.3f}, two-sided p = {r_f['p_two_sided']:.4f}")
    r_b = barnard_exact_2x2(tbl, n_grid=100)
    print(f"  Barnard two-sided p = {r_b['p_two_sided']:.4f}")

    print("\n=== Larger 2x2 table ===")
    tbl = [[8, 2], [3, 10]]
    print(f"  table: {tbl}")
    r_f = fisher_exact_2x2(tbl)
    print(f"  Fisher OR = {r_f['odds_ratio']:.3f}, two-sided p = {r_f['p_two_sided']:.4f}")
    r_b = barnard_exact_2x2(tbl, n_grid=100)
    print(f"  Barnard two-sided p = {r_b['p_two_sided']:.4f}")

    print("\n--- library cross-check (scipy.stats.fisher_exact / barnard_exact) ---")
    try:
        from scipy.stats import fisher_exact, barnard_exact
        odds, p_f = fisher_exact([[8, 2], [3, 10]])
        print(f"  scipy fisher_exact: OR = {odds:.3f}, p = {p_f:.4f}")
        bres = barnard_exact([[8, 2], [3, 10]])
        print(f"  scipy barnard_exact: statistic = {bres.statistic:.3f}, p = {bres.pvalue:.4f}")
    except Exception as ex:
        print(f"  (scipy exact tests unavailable: {ex})")
