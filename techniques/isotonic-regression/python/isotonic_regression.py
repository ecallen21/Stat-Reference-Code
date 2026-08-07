"""Isotonic (monotone) regression via Pool-Adjacent-Violators (Reference §5.29).

Fit a MONOTONE (nondecreasing or nonincreasing) function y_hat(x) that
minimizes weighted squared error:

    minimize  sum_i w_i (y_i - y_hat_i)^2
    subject to y_hat_1 <= y_hat_2 <= ... <= y_hat_n     (if x sorted ascending)

Pool-Adjacent-Violators Algorithm (PAVA, Ayer et al. 1955):
    1. Sort observations by x.
    2. Scan left-to-right; whenever consecutive block violates the monotone
       constraint (block_i mean > block_{i+1} mean for nondecreasing), MERGE
       them into a single block with weighted mean.
    3. Continue until all blocks are non-decreasing.

Output: piecewise-constant monotone function.

Applications
    - CALIBRATION of classifier probabilities (Platt scaling alternative).
    - Dose-response with known monotonicity.
    - Order-restricted inference.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def pava(y, w=None, increasing: bool = True) -> np.ndarray:
    """Pool-Adjacent-Violators isotonic fit on already-x-sorted y (and optional weights)."""
    y = np.asarray(y, dtype=float); n = len(y)
    if w is None: w = np.ones(n)
    w = np.asarray(w, dtype=float)
    y_fit = y.copy()
    if not increasing: y_fit = -y_fit
    # Blocks tracked by parallel lists (mean, weight, count)
    means = list(y_fit); weights = list(w); counts = [1] * n
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1]:
            new_w = weights[i] + weights[i + 1]
            new_mean = (weights[i] * means[i] + weights[i + 1] * means[i + 1]) / new_w
            means[i] = new_mean; weights[i] = new_w
            counts[i] += counts[i + 1]
            del means[i + 1]; del weights[i + 1]; del counts[i + 1]
            if i > 0: i -= 1
        else:
            i += 1
    # Expand back to n
    out = np.zeros(n); pos = 0
    for m, c in zip(means, counts):
        out[pos:pos + c] = m
        pos += c
    return out if increasing else -out


def isotonic_regression(x, y, increasing: bool = True) -> dict:
    """Sort by x, run PAVA."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    order = np.argsort(x); x_sorted = x[order]; y_sorted = y[order]
    y_hat_sorted = pava(y_sorted, increasing=increasing)
    y_hat = np.empty_like(y_hat_sorted); y_hat[order] = y_hat_sorted
    return {"x_sorted": x_sorted, "y_hat_sorted": y_hat_sorted,
            "y_hat": y_hat,
            "rss": float(np.sum((y - y_hat) ** 2)),
            "increasing": bool(increasing),
            "method": "Pool-Adjacent-Violators isotonic regression"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 60
    x = np.linspace(0, 10, n)
    y_true = np.log1p(x)                     # true monotone function
    y = y_true + rng.normal(0, 0.4, n)

    r = isotonic_regression(x, y, increasing=True)
    print(f"=== Isotonic regression, n = {n}, monotone log(1+x) target ===")
    print(f"  RSS on original y  = {r['rss']:.3f}")
    print(f"  RSS to true y_true = {np.sum((r['y_hat'] - y_true) ** 2):.3f}")
    # Show a few fitted values
    for i in (0, 10, 20, 40, 59):
        print(f"  x = {x[i]:5.2f}   y = {y[i]:.3f}   y_hat = {r['y_hat'][i]:.3f}   true = {y_true[i]:.3f}")
    # Sanity: fitted should be nondecreasing
    print(f"\n  monotone? {'yes' if np.all(np.diff(r['y_hat'][np.argsort(x)]) >= -1e-9) else 'NO'}")

    print("\n--- library cross-check (sklearn IsotonicRegression) ---")
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(increasing=True).fit(x, y)
        yh = ir.predict(x)
        print(f"  sklearn RSS to y_true = {np.sum((yh - y_true) ** 2):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
