"""Test equating: mean, linear, equipercentile (Reference §22.12).

Two forms X and Y designed to measure the same construct.  Equating maps
a score on Y to the equivalent score on X so that examinees are not
advantaged / disadvantaged by which form they take.

Mean equating
    y_eq = y - mean(Y) + mean(X)                    shift only

Linear equating (Levine / Tucker)
    y_eq = mean(X) + (sd(X) / sd(Y)) (y - mean(Y))    shift + scale

Equipercentile equating (Braun-Holland / Kolen-Brennan)
    Match percentiles: y_eq = quantile function of X at F_Y(y).
    Nonparametric; usually smoothed to reduce noise.

The demo below assumes RANDOM-GROUPS design (each form given to a random
subset).  Anchor-test designs use common-item non-equivalent groups.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def mean_equating(y_scores, X_scores) -> dict:
    yb = float(np.mean(y_scores)); xb = float(np.mean(X_scores))
    def map_fn(y): return np.asarray(y) - yb + xb
    return {"map": map_fn, "shift": xb - yb,
            "method": "Mean equating"}


def linear_equating(y_scores, X_scores) -> dict:
    yb = float(np.mean(y_scores)); xb = float(np.mean(X_scores))
    ys = float(np.std(y_scores, ddof=1)); xs = float(np.std(X_scores, ddof=1))
    scale = xs / ys
    def map_fn(y): return xb + scale * (np.asarray(y) - yb)
    return {"map": map_fn, "scale": scale, "shift": xb - scale * yb,
            "method": "Linear equating"}


def equipercentile_equating(y_scores, X_scores) -> dict:
    """Nonparametric equipercentile mapping via quantile matching."""
    y_sorted = np.sort(y_scores); x_sorted = np.sort(X_scores)
    def map_fn(y):
        # For each y value, find its empirical percentile on Y, then look up x at that percentile
        y = np.atleast_1d(y)
        pctl = np.array([np.mean(y_sorted <= yy) for yy in y])
        return np.array([np.quantile(x_sorted, p) for p in pctl])
    return {"map": map_fn, "method": "Equipercentile equating"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Form Y is slightly harder than X: shift by -3 and small scale change
    X = rng.normal(70, 10, 1000)                          # form X scores
    Y = rng.normal(67, 12, 1000)                          # form Y scores

    print("=== Mean equating ===")
    e = mean_equating(Y, X)
    print(f"  shift = {e['shift']:.3f}")
    print(f"  map of Y = 60: {float(e['map'](np.array([60]))[0]):.3f}")

    print("\n=== Linear equating ===")
    e = linear_equating(Y, X)
    print(f"  scale = {e['scale']:.4f}, shift = {e['shift']:.3f}")
    print(f"  map of Y = 60: {float(e['map'](np.array([60]))[0]):.3f}")

    print("\n=== Equipercentile equating (nonparametric) ===")
    e = equipercentile_equating(Y, X)
    y_pts = np.array([50, 60, 70, 80, 90])
    x_eq = e["map"](y_pts)
    print("  y_score   x_equivalent")
    for yv, xv in zip(y_pts, x_eq):
        print(f"    {yv}         {xv:.3f}")

    print("\n--- library cross-check (R equate::equate / kequate) ---")
