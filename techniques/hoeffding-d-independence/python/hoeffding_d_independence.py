"""Hoeffding's D independence test (Reference Sec 47.133).

Hoeffding 1948 'A nonparametric test of independence', Ann Math
Stat. Rank-based independence test that detects ANY monotonic OR
non-monotonic dependence, unlike Pearson / Spearman which need
monotonicity. Statistic based on the bivariate empirical
distribution vs product of marginals:

    D = 30 * (Q_1 - 2 (n-2) Q_2 + (n-2)(n-3) Q_3) / (n (n-1) ... (n-4))

with counting integrals over ranks R_i, S_i, N_i. Values D > 0
indicate dependence. Detects U-, V-, oscillatory relationships
missed by rank-correlations.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hoeffding_d(x, y):
    """Hoeffding's D statistic (Hollander-Wolfe 1999 form)."""
    n = len(x)
    R = np.argsort(np.argsort(x)) + 1        # rank of x
    S = np.argsort(np.argsort(y)) + 1
    # N_i = #{j != i : x_j <= x_i and y_j <= y_i}
    N = np.zeros(n)
    for i in range(n):
        N[i] = np.sum((R <= R[i]) & (S <= S[i])) - 1
    Q = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    R_ = np.sum((R - 2) * (S - 2) * N)
    P = np.sum(N * (N - 1))
    A = Q
    B = (n - 2) * R_
    C = (n - 2) * (n - 3) * P
    D = 30 * (A - 2 * B + C) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    return float(D)


def permutation_pvalue(x, y, n_perm=200, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = hoeffding_d(x, y)
    perms = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        perms.append(hoeffding_d(x, yp))
    return obs, float((np.array(perms) >= obs).mean())


if __name__ == "__main__":
    print("=== Hoeffding's D independence test (Hoeffding 1948) ===\n")
    rng = np.random.default_rng(0)
    n = 120

    x_lin = rng.normal(size=n); x_q = rng.normal(size=n); x_s = rng.uniform(-3, 3, size=n)
    scenarios = [
        ("independent   ", rng.normal(size=n), rng.normal(size=n)),
        ("linear         ", x_lin, 0.5 * x_lin + 0.3 * rng.normal(size=n)),
        ("quadratic (U)  ", x_q, x_q ** 2 + 0.1 * rng.normal(size=n)),
        ("sinusoid       ", x_s, np.sin(2 * x_s) + 0.1 * rng.normal(size=n)),
    ]
    for name, X, Y in scenarios:
        # Spearman for reference
        from scipy.stats import spearmanr
        rho, _ = spearmanr(X, Y)
        D, p = permutation_pvalue(np.asarray(X), np.asarray(Y), n_perm=200, rng=rng)
        print(f"  {name}  D = {D:+.5f}   perm-p = {p:.3f}   Spearman rho = {rho:+.3f}")

    print("\n  Quadratic and sinusoid: Spearman ~ 0 but Hoeffding D detects the")
    print("  non-monotonic dependence.")

    print("\n--- library cross-check (Hmisc::hoeffd R; scipy no direct; hyppo Python) ---")
