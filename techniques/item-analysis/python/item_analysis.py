"""Item analysis: difficulty + discrimination + point-biserial + distractor
(Reference §22.2).

Classical Test Theory (CTT) item statistics for a K-item test scored 0/1.

Difficulty (proportion correct)
    p_j = fraction of examinees getting item j correct.
    Usable range 0.2 - 0.8; too easy (p > 0.95) or too hard (p < 0.05) gives
    little discrimination information.

Discrimination
    Simple: proportion correct in top-27% examinees minus in bottom-27%.
    Point-biserial correlation r_pb between item score and TOTAL SCORE
    (or REST SCORE = total minus this item to avoid part-whole bias).
    Rule of thumb: r_pb > 0.3 = good, > 0.2 = acceptable, < 0.15 = review.

Distractor analysis (multiple-choice)
    For each incorrect option: fraction choosing it, average total score
    of those choosers.  A "good" distractor is chosen more by low scorers
    than by high scorers.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def item_analysis(Y) -> list:
    """CTT item analysis for a K-item 0/1 matrix.  Returns per-item dict."""
    Y = np.asarray(Y, dtype=float); n, K = Y.shape
    total = Y.sum(axis=1)
    # Upper/lower 27% by total score
    cutoff = int(round(0.27 * n))
    order = np.argsort(-total)                       # descending
    upper = order[:cutoff]; lower = order[-cutoff:]
    rows = []
    for j in range(K):
        p = float(Y[:, j].mean())
        d_ul = float(Y[upper, j].mean() - Y[lower, j].mean())
        # Rest score (total minus this item) to avoid part-whole bias
        rest = total - Y[:, j]
        if Y[:, j].std() > 0 and rest.std() > 0:
            r_pb = float(np.corrcoef(Y[:, j], rest)[0, 1])
        else:
            r_pb = float("nan")
        rows.append({"item": j,
                     "difficulty_p": p,
                     "discrimination_UL": d_ul,
                     "r_pointbiserial_rest": r_pb,
                     "flag": ("too easy" if p > 0.95 else
                              "too hard" if p < 0.05 else
                              "low discrim" if r_pb < 0.15 else "ok")})
    return rows


def _print(rows, cols):
    def fmt(v):
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)
    w = {c: max(len(c), max(len(fmt(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    print("  " + "  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(fmt(r[c]).ljust(w[c]) for c in cols))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, K = 500, 10
    theta = rng.normal(0, 1, n)
    # Items span easy (b = -2) to hard (b = 2); one has zero discrimination
    b = np.linspace(-2, 2, K)
    a = np.ones(K); a[3] = 0.05     # item 3 barely discriminates
    P = 1 / (1 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    rows = item_analysis(Y)
    print("=== Item analysis (K = 10, item 3 near-zero discrimination) ===")
    _print(rows, ["item", "difficulty_p", "discrimination_UL", "r_pointbiserial_rest", "flag"])

    print("\n--- library cross-check (R psych::score.items / difR) ---")
