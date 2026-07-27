"""Fleiss' kappa: agreement among m >= 3 raters (Reference §8.4).

Input matrix: n rows (items) x K columns (categories). Each cell counts how many
of the m raters assigned that item to that category. Rows sum to m.

    P_i    = (1 / (m(m-1))) * sum_j (n_ij (n_ij - 1))    (per-item agreement)
    P_bar  = mean_i P_i
    p_j    = sum_i n_ij / (n * m)                         (marginal cat rate)
    P_e    = sum_j p_j^2
    kappa  = (P_bar - P_e) / (1 - P_e)

Per-category kappa_j and ASE for the overall kappa are from Fleiss (1971).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fleiss_matrix(ratings: Sequence[Sequence], categories=None) -> tuple:
    """Build the n x K matrix of category counts per item.

    ``ratings`` is an n x m matrix of labels (each row = one item, each column
    = one rater). Returns (categories, count_matrix).
    """
    ratings = np.asarray(ratings)
    if categories is None:
        categories = sorted(set(ratings.flatten().tolist()))
    idx = {c: j for j, c in enumerate(categories)}
    n, m = ratings.shape
    M = np.zeros((n, len(categories)), dtype=int)
    for i in range(n):
        for j in range(m):
            M[i, idx[ratings[i, j]]] += 1
    return categories, M


def fleiss_kappa(count_matrix) -> dict:
    """Fleiss' kappa + per-category kappa + Fleiss (1971) ASE and z-test."""
    M = np.asarray(count_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("count_matrix must be 2D (n items x K categories)")
    n, K = M.shape
    m = M.sum(axis=1)
    if not np.allclose(m, m[0]):
        raise ValueError("all rows must sum to the same m (number of raters)")
    m = float(m[0])
    if m < 2:
        raise ValueError("need >= 2 raters")
    P_i = (M * (M - 1)).sum(axis=1) / (m * (m - 1))
    P_bar = float(P_i.mean())
    p_j = M.sum(axis=0) / (n * m)
    P_e = float((p_j ** 2).sum())
    if P_e == 1.0:
        kappa = 1.0 if P_bar == 1.0 else 0.0
        se = float("nan")
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
        # Fleiss (1971) ASE
        term1 = 2.0 / (n * m * (m - 1))
        num = (p_j * (1 - p_j)).sum() ** 2 - (p_j * (1 - p_j) * (1 - 2 * p_j)).sum()
        denom = ((p_j * (1 - p_j)).sum()) ** 2
        se = math.sqrt(term1 * num / denom) if denom > 0 else float("nan")
    z = kappa / se if se and se > 0 else float("inf")
    p_two = float(2 * stats.norm.sf(abs(z))) if math.isfinite(z) else 0.0
    # Per-category kappa: kappa_j = 1 - p_bar_j / (p_j * (1 - p_j))
    # with p_bar_j = sum_i n_ij (m - n_ij) / (n * m * (m - 1))
    kappa_j = {}
    for j in range(K):
        p_bar_j = ((M[:, j] * (m - M[:, j])).sum()) / (n * m * (m - 1))
        denom_j = p_j[j] * (1 - p_j[j])
        kappa_j[str(list(range(K))[j])] = float(1 - p_bar_j / denom_j) if denom_j > 0 else float("nan")
    return {"kappa": float(kappa), "P_bar": P_bar, "P_expected": P_e,
            "ASE": float(se), "z": float(z), "p_value": p_two,
            "kappa_per_category_by_index": kappa_j,
            "n_items": int(n), "m_raters": int(m), "K_categories": K,
            "method": "Fleiss' kappa (Fleiss 1971 ASE)"}


def library_versions(ratings):
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss, aggregate_raters
    cats, M = fleiss_matrix(ratings)
    return {"statsmodels.fleiss_kappa": float(sm_fleiss(np.asarray(M)))}


if __name__ == "__main__":
    import random
    random.seed(9)
    cats = ["low", "medium", "high"]
    n, m = 30, 5
    # generate an item-level "truth" then each rater agrees w/p 0.7
    truth = [random.choice(cats) for _ in range(n)]
    ratings = [
        [t if random.random() < 0.7 else random.choice([c for c in cats if c != t])
         for _ in range(m)]
        for t in truth
    ]
    cats_out, M = fleiss_matrix(ratings, cats)
    print("=== Item x Category count matrix (first 5 rows) ===")
    for row in M[:5]:
        print(" ", row.tolist())

    print("\n=== Fleiss' kappa ===")
    out = fleiss_kappa(M)
    for k, v in out.items():
        print(f"  {k:26s}: {v}")

    print("\n--- library ---")
    for k, v in library_versions(ratings).items():
        print(f"  {k}: {v}")
