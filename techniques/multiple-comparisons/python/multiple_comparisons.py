"""Multiple-comparison corrections for a family of p-values (Reference §3.13, §3.14).

Given m raw p-values from m hypothesis tests, these procedures return the
adjusted p-values that control either:

  - the family-wise error rate (FWER) -- prob >= 1 false discovery
      * Bonferroni  : p_i * m, capped at 1
      * Sidak       : 1 - (1 - p_i)^m
      * Holm        : step-down Bonferroni; uniformly more powerful than plain Bonferroni
      * Hochberg    : step-up Bonferroni-type; assumes positively-dependent or
                      independent tests
  - the false discovery rate (FDR) -- E[#false rejections / #rejections]
      * Benjamini-Hochberg (BH)   : assumes independent or positively-correlated tests
      * Benjamini-Yekutieli (BY)  : works under any dependence (more conservative)

For each procedure we return adjusted p-values aligned with the original order
of the input. Reject H0_i at level alpha iff adjusted_p_i <= alpha.

When to pick which
------------------
- A few comparisons (< ~10), strict FWER control desired -> Holm.
- All-pairs ANOVA post-hoc -> see techniques/post-hoc-tests instead (Tukey HSD
  bakes the correction into the critical value).
- Many tests (genomics, neuroimaging, etc.), some false discoveries acceptable
  -> Benjamini-Hochberg.
- Tests that may be negatively correlated -> Benjamini-Yekutieli.
"""
from __future__ import annotations

import math
from typing import Sequence


def bonferroni(p: Sequence[float]) -> list:
    """Bonferroni: p_adj = min(1, m * p_i)."""
    m = len(p)
    return [min(1.0, m * pi) for pi in p]


def sidak(p: Sequence[float]) -> list:
    """Sidak: p_adj = 1 - (1 - p_i)^m. Slightly less conservative than Bonferroni."""
    m = len(p)
    return [min(1.0, 1.0 - (1.0 - pi) ** m) for pi in p]


def holm(p: Sequence[float]) -> list:
    """Holm-Bonferroni (step-down). Uniformly more powerful than plain Bonferroni."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])     # ascending
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):               # rank 0..m-1
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adj[idx] = min(1.0, running_max)
    return adj


def hochberg(p: Sequence[float]) -> list:
    """Hochberg (step-up). Assumes independence or positive dependence; more powerful than Holm."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i], reverse=True)   # descending
    adj = [0.0] * m
    running_min = 1.0
    for rank_from_top, idx in enumerate(order):                  # 0 -> largest p
        i = m - rank_from_top                                    # rank 1..m
        candidate = (m - i + 1) * p[idx]
        running_min = min(running_min, candidate)
        adj[idx] = min(1.0, running_min)
    return adj


def benjamini_hochberg(p: Sequence[float]) -> list:
    """Benjamini-Hochberg FDR (step-up). Independent / positively correlated tests."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i], reverse=True)   # descending
    adj = [0.0] * m
    running_min = 1.0
    for rank_from_top, idx in enumerate(order):
        i = m - rank_from_top                                    # rank 1..m
        candidate = (m / i) * p[idx]
        running_min = min(running_min, candidate)
        adj[idx] = min(1.0, running_min)
    return adj


def benjamini_yekutieli(p: Sequence[float]) -> list:
    """Benjamini-Yekutieli FDR -- valid under any dependence structure."""
    m = len(p)
    cm = sum(1.0 / k for k in range(1, m + 1))                   # harmonic number H_m
    raw_bh = benjamini_hochberg(p)
    return [min(1.0, q * cm) for q in raw_bh]


def adjust(p: Sequence[float], method: str = "holm") -> list:
    """Dispatch on method name.

    ``method`` in {"bonferroni", "sidak", "holm", "hochberg", "bh", "fdr", "by"}.
    """
    table = {"bonferroni": bonferroni, "sidak": sidak,
             "holm": holm, "hochberg": hochberg,
             "bh": benjamini_hochberg, "fdr": benjamini_hochberg,
             "by": benjamini_yekutieli}
    if method not in table:
        raise ValueError(f"unknown method {method!r}; use one of {list(table)}")
    return table[method](p)


def library_versions(p):
    from statsmodels.stats.multitest import multipletests

    out = {}
    for sm_name in ("bonferroni", "sidak", "holm", "simes-hochberg",
                    "fdr_bh", "fdr_by"):
        _, adj, _, _ = multipletests(p, method=sm_name)
        out[f"statsmodels {sm_name}"] = [round(a, 4) for a in adj]
    return out


if __name__ == "__main__":
    raw_p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.30, 0.45, 0.78]
    print(f"raw p-values ({len(raw_p)} tests):")
    for i, p in enumerate(raw_p, 1):
        print(f"  H0_{i}: {p}")
    print()
    for m in ("bonferroni", "sidak", "holm", "hochberg", "bh", "by"):
        adj = adjust(raw_p, m)
        rej = [i + 1 for i, q in enumerate(adj) if q <= 0.05]
        print(f"{m:11s}: adj_p = {[round(q, 4) for q in adj]}  reject at 0.05: {rej}")

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(raw_p).items():
        print(f"  {k:24s}: {v}")
