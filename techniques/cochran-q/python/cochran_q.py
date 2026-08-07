"""Cochran's Q test for repeated-measures binary outcomes (Reference §8.10).

Generalizes McNemar's test to k >= 2 related samples of binary outcomes.
Each subject measured under k conditions -> binary matrix (n x k).

    H_0: probability of success is the same under all k conditions
    H_a: at least one condition differs

Statistic (Cochran 1950):
    Q = (k - 1) * (k * sum_j C_j^2 - (sum_j C_j)^2) /
                    (k * sum_i R_i - sum_i R_i^2)
    where R_i = row sum (number of successes for subject i),
          C_j = column sum (number of successes in condition j).

Under H_0, Q ~ chi-square(k - 1) approximately when n is large.

Post-hoc: pairwise McNemar tests + multiple-comparison correction
(Bonferroni / BH).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def cochran_q(data) -> dict:
    """Cochran's Q for an n x k binary matrix."""
    X = np.asarray(data, dtype=int)
    n, k = X.shape
    R = X.sum(1); C = X.sum(0)
    numerator = (k - 1) * (k * np.sum(C ** 2) - np.sum(C) ** 2)
    denominator = k * np.sum(R) - np.sum(R ** 2)
    if denominator == 0:
        return {"Q": float("nan"), "note": "denominator zero (all row sums are 0 or k)"}
    Q = float(numerator / denominator)
    return {"Q": Q, "df": int(k - 1),
            "p_value": float(stats.chi2.sf(Q, k - 1)),
            "n": int(n), "k_conditions": int(k),
            "condition_success_rates": (C / n).tolist(),
            "method": "Cochran's Q test"}


def pairwise_mcnemar(data, adjust: str = "bonferroni") -> list:
    """Post-hoc pairwise McNemar tests across all condition pairs."""
    X = np.asarray(data, dtype=int); n, k = X.shape
    rows = []
    for j1 in range(k):
        for j2 in range(j1 + 1, k):
            b = int(np.sum((X[:, j1] == 1) & (X[:, j2] == 0)))
            c = int(np.sum((X[:, j1] == 0) & (X[:, j2] == 1)))
            if b + c == 0:
                p = 1.0
            else:
                # Exact binomial
                p = float(stats.binomtest(min(b, c), b + c, p=0.5).pvalue)
            rows.append({"cond_i": j1, "cond_j": j2, "b": b, "c": c, "p_raw": p})
    m = len(rows)
    if adjust == "bonferroni":
        for r in rows: r["p_adj"] = min(m * r["p_raw"], 1.0)
    else:
        for r in rows: r["p_adj"] = r["p_raw"]
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 30 subjects, 4 conditions with success probs 0.3, 0.35, 0.55, 0.60
    n = 30; k = 4
    p_true = np.array([0.30, 0.35, 0.55, 0.60])
    # correlated observations via a subject-level latent
    u = rng.normal(0, 0.6, n)[:, None]
    p_ij = 1 / (1 + np.exp(-(np.log(p_true / (1 - p_true))[None, :] + u)))
    X = (rng.uniform(size=(n, k)) < p_ij).astype(int)

    print(f"=== Cochran's Q (n = {n}, k = {k} conditions) ===")
    r = cochran_q(X)
    for kk, v in r.items():
        if isinstance(v, float): print(f"  {kk}: {v:.4f}")
        else: print(f"  {kk}: {v}")

    print("\n=== Pairwise McNemar (Bonferroni-adjusted) ===")
    for row in pairwise_mcnemar(X):
        marker = "*" if row["p_adj"] < 0.05 else " "
        print(f"  {marker} conditions {row['cond_i']} vs {row['cond_j']}: "
              f"b = {row['b']}, c = {row['c']}, p_adj = {row['p_adj']:.4f}")

    print("\n--- library cross-check (statsmodels cochrans_q) ---")
    try:
        from statsmodels.stats.contingency_tables import cochrans_q
        r = cochrans_q(X)
        print(f"  statsmodels Q = {r.statistic:.4f}, p = {r.pvalue:.4f}")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
