"""Friedman Test (Reference §6.5).

Nonparametric REPEATED-MEASURES one-way ANOVA: H0 says the k treatments have
the same distribution within each subject (block). Each row of the data matrix
is a subject; each column a treatment.

Algorithm
---------
Given n subjects (rows) and k treatments (cols):
1. Rank treatments WITHIN each row (average ranks at ties).
2. R_j = column sum of ranks. R_bar_j = R_j / n.
3. Friedman chi^2:
     F = (12 / (n k (k + 1))) * sum_j R_j^2 - 3 n (k + 1)
4. Tie correction: divide by  1 - sum_t (t^3 - t) / (n (k^3 - k))
5. Under H0: F ~ chi^2_{k-1}.

Kendall's W = F / (n (k - 1)) is the related coefficient of concordance.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from collections import Counter    # stdlib: dict subclass that counts occurrences (Counter([1,1,2]) -> {1:2, 2:1})
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def friedman_test(data) -> dict:
    """Friedman test on a 2D array (rows = subjects, cols = treatments)."""
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    # Rank within each row
    ranks = np.zeros_like(data)
    tie_terms = 0.0
    for i in range(n):
        row = data[i]
        order = np.argsort(row, kind="stable")
        # average-rank ties
        rk = np.empty(k); j = 0
        while j < k:
            jj = j
            while jj + 1 < k and row[order[jj + 1]] == row[order[j]]:
                jj += 1
            avg = (j + jj) / 2 + 1
            for m in range(j, jj + 1):
                rk[order[m]] = avg
            t = jj - j + 1
            if t > 1:
                tie_terms += t ** 3 - t
            j = jj + 1
        ranks[i] = rk
    R = ranks.sum(axis=0)
    F = (12 / (n * k * (k + 1))) * float((R ** 2).sum()) - 3 * n * (k + 1)
    denom = 1 - tie_terms / (n * (k ** 3 - k)) if (k ** 3 - k) > 0 else 1.0
    F_corr = F / denom if denom != 0 else F
    p = float(stats.chi2.sf(F_corr, df=k - 1))
    kendall_w = F_corr / (n * (k - 1)) if n * (k - 1) > 0 else float("nan")
    return {"F": F, "F_corrected": F_corr, "df": k - 1, "p_value": p,
            "rank_sums": R.tolist(), "kendalls_W": kendall_w,
            "n_subjects": n, "k_treatments": k, "method": "Friedman"}


def library_versions(data):
    cols = [np.asarray(data)[:, j] for j in range(np.shape(data)[1])]
    return {"scipy.stats.friedmanchisquare": stats.friedmanchisquare(*cols)}


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    n = 20; k = 4
    # n subjects measured under 4 treatments with consistent effects + subject noise
    subj = rng.normal(50, 8, n).reshape(-1, 1)
    treatment_effects = np.array([0, 2, 5, 1])
    data = subj + treatment_effects + rng.normal(0, 2, (n, k))

    print("=== Friedman test (4 treatments, 20 subjects) ===")
    for k_, v in friedman_test(data).items():
        print(f"  {k_:14s}: {v}")

    print("\n--- library ---")
    for k_, v in library_versions(data).items():
        print(f"  {k_}: {v}")
