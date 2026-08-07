"""Multiple-testing corrections (Reference §3.30, §4.24).

Given m p-values, control either
    - FAMILY-WISE ERROR RATE (FWER): probability of at least one false positive.
    - FALSE DISCOVERY RATE (FDR): expected proportion of rejections that are false.

FWER (very conservative)
    Bonferroni : reject if p_i <= alpha / m.
    Sidak      : reject if p_i <= 1 - (1 - alpha)^(1/m).
    Holm       : step-down on sorted p; reject in ascending order while
                 p_(k) <= alpha / (m - k + 1).
    Hochberg   : step-up.

FDR (less conservative)
    Benjamini-Hochberg (BH, 1995): reject the first K where p_(K) <= K/m * alpha
                                    on the sorted p-values.  Assumes independence
                                    or positive regression dependence (PRDS).
    Benjamini-Yekutieli (BY, 2001): BH with a log(m) inflation factor -- valid
                                     under arbitrary dependence.

Storey q-value (Storey 2002)
    Adaptive version of BH that estimates the fraction pi_0 of TRUE nulls
    and uses (pi_0 * m / K) alpha instead of (m / K) alpha.  More power
    when pi_0 < 1.

Adjusted p-values
    p_adj = smallest alpha at which that hypothesis is rejected.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def bonferroni(p): return np.clip(np.asarray(p, dtype=float) * len(p), 0, 1)


def sidak(p):
    p = np.asarray(p, dtype=float); m = len(p)
    return np.clip(1 - (1 - p) ** m, 0, 1)


def holm(p):
    p = np.asarray(p, dtype=float); m = len(p)
    order = np.argsort(p); p_sorted = p[order]
    adj = np.empty(m)
    for k in range(m):
        adj[k] = min(1.0, (m - k) * p_sorted[k])
    # Enforce monotonicity (running max)
    adj = np.maximum.accumulate(adj)
    out = np.empty(m); out[order] = adj
    return out


def hochberg(p):
    p = np.asarray(p, dtype=float); m = len(p)
    order = np.argsort(p); p_sorted = p[order]
    adj = np.empty(m)
    for k in range(m):
        adj[k] = min(1.0, (m - k) * p_sorted[k])
    # step-up: enforce reverse monotone min
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m); out[order] = adj
    return out


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float); m = len(p)
    order = np.argsort(p); p_sorted = p[order]
    adj = np.empty(m)
    for k in range(m):
        adj[k] = min(1.0, m * p_sorted[k] / (k + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m); out[order] = adj
    return out


def benjamini_yekutieli(p):
    p = np.asarray(p, dtype=float); m = len(p)
    cm = np.sum(1 / np.arange(1, m + 1))
    order = np.argsort(p); p_sorted = p[order]
    adj = np.empty(m)
    for k in range(m):
        adj[k] = min(1.0, m * cm * p_sorted[k] / (k + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m); out[order] = adj
    return out


def storey_q(p, lambda_: float = 0.5):
    """Storey q-values with fixed lambda (default 0.5)."""
    p = np.asarray(p, dtype=float); m = len(p)
    pi0 = np.mean(p >= lambda_) / (1 - lambda_)
    pi0 = min(pi0, 1.0)
    order = np.argsort(p); p_sorted = p[order]
    q = np.empty(m)
    for k in range(m):
        q[k] = min(1.0, pi0 * m * p_sorted[k] / (k + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[order] = q
    return out, float(pi0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 50 tests, 40 truly null (uniform p), 10 with effect (small p)
    p_null = rng.uniform(0, 1, 40)
    p_alt = rng.beta(0.3, 20, 10)    # skewed toward 0
    p = np.concatenate([p_null, p_alt])
    truth = np.array([False] * 40 + [True] * 10)

    print("=== 50 tests, 10 true effects, alpha = 0.05 ===")
    print("  method              rejects  true positives  false positives")
    for name, fn in [("raw p <= 0.05", lambda a: a),
                      ("Bonferroni", bonferroni),
                      ("Sidak", sidak),
                      ("Holm", holm),
                      ("Hochberg", hochberg),
                      ("Benjamini-Hochberg (FDR)", benjamini_hochberg),
                      ("Benjamini-Yekutieli", benjamini_yekutieli)]:
        adj = fn(p)
        rej = adj < 0.05
        tp = int((rej & truth).sum()); fp = int((rej & ~truth).sum())
        print(f"  {name:30s}   {int(rej.sum()):3d}         {tp:3d}             {fp:3d}")

    q, pi0 = storey_q(p, lambda_=0.5)
    rej = q < 0.05
    tp = int((rej & truth).sum()); fp = int((rej & ~truth).sum())
    print(f"  {'Storey q (pi0 = ' + f'{pi0:.2f})':30s}   {int(rej.sum()):3d}         {tp:3d}             {fp:3d}")

    print("\n--- library cross-check (statsmodels multipletests) ---")
    try:
        from statsmodels.stats.multitest import multipletests
        for m in ("bonferroni", "holm", "hochberg", "fdr_bh", "fdr_by"):
            adj = multipletests(p, alpha=0.05, method=m)[1]
            print(f"  {m:12s}: rejects = {int((adj < 0.05).sum())}")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
