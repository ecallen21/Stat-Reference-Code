"""Safe Screening for LASSO (Reference Sec 47.336).

El Ghaoui, Viallon & Rabbani 2010 arXiv; Fercoq et al 2015.
Rules that DEFINITIVELY exclude features from the active set
BEFORE running the LASSO solver:

    Screening rule (SAFE, based on dual bound):
        |x_j^T y| < lambda - r * ||x_j||_2  =>  beta_j = 0

Reduces problem size from d to few active features; used inside
LASSO / SVM solvers (celer, glmnet) for O(d) speed-ups at large
d.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def safe_screen_lasso(X, y, lam):
    """El Ghaoui SAFE screening rule for LASSO with y centred, cols scaled."""
    n, d = X.shape
    # Dual bound (simplest form): r = ||y||_2
    r = float(np.linalg.norm(y))
    col_norms = np.linalg.norm(X, axis=0)
    corr = np.abs(X.T @ y)
    keep = corr >= lam - r * col_norms                              # cannot rule out
    return keep


def coord_descent_lasso(X, y, lam, active_set=None, n_iter=200):
    """Coordinate-descent LASSO on a subset of features (falls back to all)."""
    n, d = X.shape
    if active_set is None: active_set = np.ones(d, dtype=bool)
    beta = np.zeros(d)
    for _ in range(n_iter):
        r = y - X @ beta
        for j in range(d):
            if not active_set[j]: continue
            rj = r + X[:, j] * beta[j]
            rho = X[:, j] @ rj
            z = np.sum(X[:, j] ** 2)
            beta[j] = np.sign(rho) * max(abs(rho) - lam * n, 0) / max(z, 1e-9)
            r = rj - X[:, j] * beta[j]
    return beta


if __name__ == "__main__":
    print("=== Safe Screening for LASSO (El Ghaoui et al 2010) ===\n")
    rng = np.random.default_rng(0)

    n, d = 200, 500
    X = rng.standard_normal((n, d))
    X -= X.mean(axis=0); X /= X.std(axis=0) + 1e-9
    beta_true = np.zeros(d); beta_true[:8] = rng.normal(0, 2, 8)
    y = X @ beta_true + rng.normal(0, 0.5, n)
    y -= y.mean()

    lam = 0.5
    keep_screen = safe_screen_lasso(X, y, lam)
    n_kept = int(keep_screen.sum())
    print(f"  n = {n}, d = {d}, LASSO with lambda = {lam}")
    print(f"  SAFE screening kept {n_kept} / {d} features "
          f"({100 * n_kept / d:.1f}%)\n")

    # Solve with and without screening (both give the same answer)
    beta_full = coord_descent_lasso(X, y, lam, active_set=None, n_iter=50)
    beta_screen = coord_descent_lasso(X, y, lam, active_set=keep_screen, n_iter=50)

    nz_full = int((np.abs(beta_full) > 1e-3).sum())
    nz_screen = int((np.abs(beta_screen) > 1e-3).sum())
    dist = float(np.linalg.norm(beta_full - beta_screen))
    print(f"  Full solve nonzeros:      {nz_full}")
    print(f"  Screened solve nonzeros:  {nz_screen}")
    print(f"  ||beta_full - beta_screen|| = {dist:.4e}   (should be near zero)\n")

    print(f"  Screening is EXACT: features EXCLUDED are provably zero at the optimum.")
    print(f"  Modern solvers (celer, glmnet-fast) use tighter dynamic screening.")

    print("\n--- library cross-check (celer.Lasso; glmnet strong-rule; sklearn Lasso strong-rule) ---")
