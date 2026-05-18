"""Kruskal-Wallis Test (Reference §6.4).

Nonparametric one-way ANOVA. H0: all k groups come from the same distribution.

Algorithm
---------
1. Combine all groups and rank with average ranks at ties.
2. R_i = sum of ranks in group i. R_bar_i = R_i / n_i.
3. H = (12 / (N (N + 1))) * sum_i n_i (R_bar_i - (N + 1)/2)^2
     = (12 / (N (N + 1))) * sum_i R_i^2 / n_i - 3 (N + 1)
4. Tie-correct: H_corrected = H / (1 - sum(t^3 - t) / (N^3 - N))
5. Under H0 (no ties, k >= 3, n_i >= ~5):  H ~ chi^2_{k-1}
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from collections import Counter    # stdlib: dict subclass that counts occurrences (Counter([1,1,2]) -> {1:2, 2:1})
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _avg_ranks(values):
    pairs = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and values[pairs[j + 1]] == values[pairs[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k]] = avg
        i = j + 1
    return ranks


def kruskal_wallis(groups: Sequence[Sequence[float]]) -> dict:
    """Kruskal-Wallis H test on a list of groups."""
    all_vals = [v for g in groups for v in g]
    sizes = [len(g) for g in groups]
    N = len(all_vals); k = len(groups)
    ranks = _avg_ranks(all_vals)
    # Slice ranks back into groups
    rank_sums = []
    idx = 0
    for n in sizes:
        rank_sums.append(sum(ranks[idx:idx + n]))
        idx += n
    H = (12 / (N * (N + 1))) * sum(rs ** 2 / n for rs, n in zip(rank_sums, sizes)) - 3 * (N + 1)
    # Tie correction
    counts = Counter(all_vals).values()
    tie_sum = sum(t ** 3 - t for t in counts if t > 1)
    H_corr = H / (1 - tie_sum / (N ** 3 - N)) if N ** 3 - N > 0 else H
    p = float(stats.chi2.sf(H_corr, df=k - 1))
    return {"H": H, "H_corrected": H_corr, "df": k - 1,
            "p_value": p, "n_per_group": sizes,
            "rank_sums": rank_sums, "method": "Kruskal-Wallis"}


def library_versions(*groups):
    return {"scipy.stats.kruskal": stats.kruskal(*groups)}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(3)
    a = rng.normal(50, 8, 30).tolist()
    b = rng.normal(55, 9, 28).tolist()
    c = rng.normal(60, 10, 32).tolist()

    print("=== Kruskal-Wallis ===")
    for k, v in kruskal_wallis([a, b, c]).items():
        print(f"  {k:14s}: {v}")
    print("\n--- library ---")
    for k, v in library_versions(a, b, c).items():
        print(f"  {k}: {v}")
