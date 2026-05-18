"""Jonckheere-Terpstra Test (Reference §6.10).

A K-W-like test for ORDERED alternatives: H_A says the group distributions
are stochastically ordered  F_1 <= F_2 <= ... <= F_k  (or the reverse).
More powerful than K-W when the order is real.

Statistic
---------
J = sum over all i < j of U_ij,
    where U_ij = # { (x, y) : x in group i, y in group j, x < y }
               + 0.5 * # { (x, y) : x = y }                       (Mann-Whitney U_ij)

Under H0 (all groups same):
    E[J] = (1/4) * (N^2 - sum_i n_i^2)
    Var[J] = (1/72) * ( N^2 (2 N + 3) - sum_i n_i^2 (2 n_i + 3) )
    z = (J - E[J]) / sqrt(Var[J])

Two-sided / one-sided as you'd expect.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def jonckheere_terpstra(groups: Sequence[Sequence[float]],
                       alternative: str = "two-sided") -> dict:
    """Jonckheere-Terpstra test.

    ``groups`` should be in the assumed order (group 1 lowest, group k highest under H_A).
    ``alternative``: "two-sided" / "increasing" / "decreasing".
    """
    k = len(groups)
    sizes = [len(g) for g in groups]
    N = sum(sizes)
    # Compute U_ij = Mann-Whitney U for group i vs group j, i < j
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            gi = groups[i]; gj = groups[j]
            U_ij = 0
            for x in gi:
                for y in gj:
                    if x < y: U_ij += 1
                    elif x == y: U_ij += 0.5
            J += U_ij
    mu = (N ** 2 - sum(n ** 2 for n in sizes)) / 4
    var = (N ** 2 * (2 * N + 3) - sum(n ** 2 * (2 * n + 3) for n in sizes)) / 72
    z = (J - mu) / math.sqrt(var) if var > 0 else 0.0
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "increasing":
        p = float(stats.norm.sf(z))
    elif alternative == "decreasing":
        p = float(stats.norm.cdf(z))
    else:
        raise ValueError("alternative must be 'two-sided', 'increasing', or 'decreasing'")
    return {"J": J, "expected_J": mu, "var_J": var,
            "z": z, "p_value": float(p),
            "n_per_group": sizes, "alternative": alternative,
            "method": "Jonckheere-Terpstra"}


def library_versions(*groups, alternative="two-sided"):
    out = {}
    try:
        from scipy.stats import jonckheere
        res = jonckheere(*groups, alternative=alternative)
        out["scipy.stats.jonckheere"] = (float(res.statistic), float(res.pvalue))
    except Exception as exc:
        out["scipy"] = f"not available ({exc}); see DescrTab2 / clinfun in R"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    g1 = rng.normal(50, 5, 25).tolist()
    g2 = rng.normal(53, 5, 25).tolist()
    g3 = rng.normal(57, 5, 25).tolist()
    g4 = rng.normal(62, 5, 25).tolist()

    print("=== Jonckheere-Terpstra (ordered alternative) ===")
    for k, v in jonckheere_terpstra([g1, g2, g3, g4], alternative="increasing").items():
        print(f"  {k:14s}: {v}")

    print("\n--- library ---")
    for k, v in library_versions(g1, g2, g3, g4, alternative="increasing").items():
        print(f"  {k}: {v}")
