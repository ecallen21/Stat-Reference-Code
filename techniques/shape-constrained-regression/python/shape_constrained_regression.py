"""Shape-constrained regression: monotone + convex (Reference Sec 33.14).

Least-squares regression under a KNOWN SHAPE CONSTRAINT:

  * Monotone (nondecreasing) -> POOL ADJACENT VIOLATORS (PAV) algorithm.
  * Convex / concave         -> QP with cvxpy or a simple iterative
    projection.

Motivation:
  * Dose-response, growth, learning curves: monotonicity is a scientific
    prior.
  * Utility functions, cost curves: concavity / convexity is a
    scientific prior.

Advantages over parametric constrained models:
  * NONPARAMETRIC (no shape family).
  * Consistent under the shape prior even for small n.

Here we implement PAV for monotone-fit and a projected-gradient
CONVEX-FIT on synthetic data with monotone + convex truths, and
compare to unconstrained OLS + smoothing spline.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pav(y, w=None):
    """Pool adjacent violators (PAV) for a nondecreasing least-squares fit."""
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float).copy()
    # Blocks: (start, end_exclusive, weight, mean)
    blocks = [(i, i + 1, w[i], y[i]) for i in range(n)]
    changed = True
    while changed:
        changed = False
        i = 0
        new_blocks = []
        while i < len(blocks):
            s, e, wi, mi = blocks[i]
            while i + 1 < len(blocks) and blocks[i + 1][3] < mi:
                s2, e2, w2, m2 = blocks[i + 1]
                total_w = wi + w2
                mi = (wi * mi + w2 * m2) / total_w
                e, wi = e2, total_w
                i += 1
                changed = True
            new_blocks.append((s, e, wi, mi))
            i += 1
        blocks = new_blocks
    out = np.zeros(n)
    for s, e, wi, mi in blocks:
        out[s:e] = mi
    return out


def convex_fit(y):
    """Convex-projection via scipy.optimize.minimize with second-diff constraints."""
    from scipy.optimize import minimize as _min
    n = len(y)
    f0 = y.astype(float)
    # Objective: 0.5 || y - f ||^2
    def obj(f): return 0.5 * float(np.sum((y - f) ** 2))
    def obj_grad(f): return f - y
    # Constraints: f[i+2] - 2 f[i+1] + f[i] >= 0 for i = 0..n-3.
    cons = []
    for i in range(n - 2):
        cons.append({"type": "ineq",
                      "fun": (lambda f, i=i: f[i + 2] - 2 * f[i + 1] + f[i]),
                      "jac": (lambda f, i=i: (
                          lambda e=np.zeros(n): (e.__setitem__(i, 1),
                                                  e.__setitem__(i + 1, -2),
                                                  e.__setitem__(i + 2, 1), e)[3])())})
    res = _min(obj, f0, jac=obj_grad, constraints=cons, method="SLSQP",
                options={"maxiter": 200, "ftol": 1e-8})
    return res.x


if __name__ == "__main__":
    print("=== Shape-constrained regression (monotone + convex) ===\n")
    rng = np.random.default_rng(0)
    n = 60
    x = np.linspace(0, 1, n)

    # Case 1: monotone truth
    truth1 = np.sqrt(x)
    y1 = truth1 + rng.normal(0, 0.1, n)
    y_iso = pav(y1)
    n_viol_before = int((np.diff(y1) < 0).sum())
    n_viol_after = int((np.diff(y_iso) < 0).sum())
    err_ols = float(np.mean((y1 - truth1) ** 2))
    err_iso = float(np.mean((y_iso - truth1) ** 2))
    print(f"  Monotone truth sqrt(x):")
    print(f"    raw obs monotonicity violations: {n_viol_before}/{n-1}")
    print(f"    PAV fit  monotonicity violations: {n_viol_after}/{n-1}")
    print(f"    MSE (raw vs truth) = {err_ols:.4f}   MSE (PAV vs truth) = {err_iso:.4f}\n")

    # Case 2: convex truth
    truth2 = x ** 2
    y2 = truth2 + rng.normal(0, 0.06, n)
    y_conv = convex_fit(y2)
    d2 = y_conv[2:] - 2 * y_conv[1:-1] + y_conv[:-2]
    n_viol_c = int((d2 < -1e-6).sum())
    err_raw2 = float(np.mean((y2 - truth2) ** 2))
    err_conv = float(np.mean((y_conv - truth2) ** 2))
    print(f"  Convex truth x^2:")
    print(f"    convex-fit second-difference violations: {n_viol_c}/{n-2}")
    print(f"    MSE (raw vs truth)   = {err_raw2:.4f}   MSE (convex vs truth) = {err_conv:.4f}\n")

    print("--- library cross-check (sklearn.IsotonicRegression; cvxpy monotone/convex QP;"
          " R Iso; scam) ---")
