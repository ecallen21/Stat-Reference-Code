"""Dantzig selector (Reference Sec 6.18).

Candes & Tao 2007 'The Dantzig selector: statistical estimation
when p is much larger than n', Ann Stat. Sparse linear-regression
estimator alternative to lasso:

    minimise ||beta||_1
    subject to  ||X'(y - X * beta)||_inf <= lambda

The dual constraint ensures the maximum absolute correlation of any
covariate with the residuals is bounded. Solved as a LINEAR
PROGRAM (LP) in polynomial time.

Contrast with lasso: min 0.5 ||y - Xb||^2 + lambda ||b||_1
    * Dantzig selector has similar theoretical guarantees under
      restricted-isometry.
    * Solution is often close but not identical to lasso.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import linprog    # LP solver


def dantzig_selector(X, y, lam):
    """Solve Dantzig selector via LP with beta = u - v, u, v >= 0.

    Objective: sum u + sum v
    Constraints: | X' (y - X (u - v)) |_inf <= lam
      -> X'X u - X'X v <= lam + X'y
         X'X v - X'X u <= lam - X'y  (i.e. -X'X u + X'X v <= lam - X'y)
    """
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    c = np.ones(2 * p)
    A_ub = np.vstack([np.hstack([XtX, -XtX]),
                        np.hstack([-XtX, XtX])])
    b_ub = np.concatenate([lam + Xty, lam - Xty])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (2 * p), method="highs")
    beta = res.x[:p] - res.x[p:]
    return {"beta": beta, "obj": float(res.fun)}


if __name__ == "__main__":
    print("=== Dantzig selector (Candes-Tao 2007) ===\n")
    rng = np.random.default_rng(0)
    n = 100; p = 40
    X = rng.normal(size=(n, p))
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    beta_true = np.zeros(p)
    beta_true[[0, 5, 12]] = [1.5, -1.0, 0.8]
    y = X @ beta_true + 0.05 * rng.normal(size=n)

    for lam in [0.02, 0.05, 0.1, 0.2]:
        r = dantzig_selector(X, y, lam)
        n_nonzero = int((np.abs(r["beta"]) > 1e-3).sum())
        top = np.argsort(np.abs(r["beta"]))[::-1][:5]
        print(f"  lam = {lam:.2f}   nonzeros = {n_nonzero:2d}   "
              f"top5 coefs = {[(int(i), round(r['beta'][i], 3)) for i in top]}")

    print("\n  True support = features {0, 5, 12} with coefs (1.5, -1.0, 0.8).")
    print("  Choose lambda by cross-validation in practice.")

    print("\n--- library cross-check (flare / hdi R; skglm / cvxpy Python) ---")
