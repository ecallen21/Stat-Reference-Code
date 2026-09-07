"""Partial dependence + ICE plots (Reference Sec 47.34).

Friedman 2001 (PDP); Goldstein et al. 2015 (ICE). Global / per-
instance visualisation of a feature's marginal effect on a black-box
model prediction.

PARTIAL DEPENDENCE:
    PD_j(v) = (1/n) sum_i  f(x_i with x_ij replaced by v)

ICE (individual conditional expectation):
    ICE_j^{(i)}(v) = f(x_i with x_ij replaced by v)

PDP is the AVERAGE of ICE curves.  Discrepancies (fanning ICE) reveal
INTERACTIONS the PDP hides.

Centered ICE (c-ICE) subtracts f(x_i at v_min) from each curve so
they start at 0 -> shape comparison.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pdp_ice(f, X, feature, grid):
    """Return (PDP curve, ICE matrix n x |grid|)."""
    n = len(X); G = len(grid)
    ICE = np.zeros((n, G))
    for k, v in enumerate(grid):
        X_new = X.copy()
        X_new[:, feature] = v
        ICE[:, k] = np.array([f(x) for x in X_new])
    return {"pdp": ICE.mean(axis=0), "ice": ICE, "grid": grid}


if __name__ == "__main__":
    print("=== Partial dependence + ICE plots (Friedman / Goldstein) ===\n")
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 3))

    #  Model with INTERACTION x0 * x1: f(x) = 0.5 * x0 * x1 + 0.3 * x2
    def f(x):
        return float(0.5 * x[0] * x[1] + 0.3 * x[2])

    grid = np.linspace(-2, 2, 9)
    r = pdp_ice(f, X, feature=0, grid=grid)

    print(f"  Feature j = 0.  Grid = {grid.round(2).tolist()}")
    print(f"  PDP:   {r['pdp'].round(3).tolist()}")
    print(f"  Notice PDP is ~ 0 across the grid (avg over x1 kills the interaction).")
    print(f"\n  ICE curves for x1 quartiles (low, med, high):")
    x1_sorted = np.argsort(X[:, 1])
    for label, idx in [("low  x1 (0.10 quantile)", x1_sorted[10]),
                        ("mid  x1 (0.50 quantile)", x1_sorted[50]),
                        ("high x1 (0.90 quantile)", x1_sorted[90])]:
        print(f"    {label}  ICE = {r['ice'][idx].round(3).tolist()}")

    print("\n  Fanning ICE reveals the interaction with x1 the PDP averaged away.")

    print("\n--- library cross-check (iml / pdp / ICEbox R; sklearn.inspection.partial_dependence Python) ---")
