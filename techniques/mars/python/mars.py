"""Multivariate Adaptive Regression Splines - MARS (Reference §5.28; Friedman 1991).

Adaptive nonparametric regression that grows a piecewise-linear model by
searching over all HINGE FUNCTIONS max(0, x - c) and their reflections
max(0, c - x).  Automatically selects variables and knot locations.

Forward pass
    Start with y = beta_0.  At each step add a pair
        (max(0, x_j - c), max(0, c - x_j))
    that most reduces RSS.  Iterate until M_max terms or improvement stalls.

Backward pass
    Prune terms via generalized cross-validation (GCV):
        GCV(M) = RSS(M) / (n (1 - C(M) / n)^2)
    where C(M) = M + d * (M - 1) / 2 counts effective df (d ~ 3).

Contrast with splines / GAM
    - GAM: user picks smooth per variable; MARS discovers knots + interactions.
    - MARS naturally handles interactions via products of hinge functions.
    - Piecewise-linear only (not smooth like a spline).

The demo below implements a simplified forward pass (no backward pruning) on
1-D input for clarity; extend to multivariate + interactions for full MARS.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _add_hinge_pair(B, x_j, knot):
    """Add two hinge columns (x - c)_+ and (c - x)_+ to design matrix B."""
    h_plus = np.maximum(x_j - knot, 0)
    h_minus = np.maximum(knot - x_j, 0)
    return np.column_stack([B, h_plus, h_minus])


def mars_forward(X, y, max_terms: int = 20) -> dict:
    """Forward-pass MARS (variables + knot search) with simplified additive terms."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    B = np.ones((n, 1))    # intercept
    terms = [{"kind": "intercept"}]
    for _ in range(max_terms // 2):
        best = {"rss": np.inf, "j": -1, "knot": None}
        for j in range(p):
            candidates = np.quantile(X[:, j], np.linspace(0.05, 0.95, 15))
            for c in candidates:
                B_try = _add_hinge_pair(B, X[:, j], c)
                beta, res, *_ = np.linalg.lstsq(B_try, y, rcond=None)
                rss = float(np.sum((y - B_try @ beta) ** 2))
                if rss < best["rss"]:
                    best = {"rss": rss, "j": j, "knot": float(c)}
        if best["j"] < 0: break
        B = _add_hinge_pair(B, X[:, best["j"]], best["knot"])
        terms.append({"kind": "hinge", "var": best["j"], "knot": best["knot"]})
    beta, *_ = np.linalg.lstsq(B, y, rcond=None)
    y_hat = B @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    M = len(beta)
    gcv = rss / max((n * (1 - M / n) ** 2), 1e-8) if n > M else float("nan")
    return {"beta": beta, "terms": terms, "fitted": y_hat,
            "rss": rss, "gcv": gcv, "n_terms": int(M),
            "method": "MARS (forward pass, additive hinges)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    # Piecewise-linear + interaction target
    y_true = np.maximum(x1 - 0.5, 0) - 2 * np.maximum(-x1 - 1, 0) + 0.5 * np.abs(x2)
    y = y_true + rng.normal(0, 0.3, n)
    X = np.column_stack([x1, x2])

    r = mars_forward(X, y, max_terms=12)
    rmse = math.sqrt(np.mean((r["fitted"] - y_true) ** 2))
    print(f"=== MARS forward-pass fit ===")
    print(f"  n_terms = {r['n_terms']}, RSS = {r['rss']:.3f}, GCV = {r['gcv']:.4f}")
    print(f"  in-sample RMSE vs truth = {rmse:.3f}   (noise sd = 0.3)")
    print("\n  Terms:")
    for t in r["terms"]:
        if t["kind"] == "hinge":
            print(f"    hinge on x{t['var'] + 1} at knot = {t['knot']:.2f}")

    print("\n--- library cross-check (pyearth / R earth) ---")
    try:
        from pyearth import Earth
        e = Earth(max_terms=12).fit(X, y)
        print(f"  pyearth GCV: {e.gcv_}")
    except Exception as ex:
        print(f"  (pyearth unavailable: {ex})")
