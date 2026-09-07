"""D-optimal design (Reference Sec 17.19).

Given a MODEL matrix parameterisation and a candidate set of
runs, choose n runs to maximise DET(X^T X) -- equivalently, minimise
the volume of the joint confidence ellipsoid of beta.

D-efficiency = (det(X^T X) / det(X^T X for best))^{1/p} * 100.

Fedorov exchange algorithm: greedy swap-in / swap-out heuristic
that improves det(X^T X) each iteration.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fedorov(X_all, n, max_iter=100, seed=0):
    """Choose n runs from X_all (rows) to maximise det(X^T X)."""
    rng = np.random.default_rng(seed)
    N, p = X_all.shape
    idx = list(rng.choice(N, size=n, replace=False))
    X = X_all[idx]
    best_det = np.linalg.det(X.T @ X)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for j in range(N):
                if j in idx: continue
                trial = list(idx)
                trial[i] = j
                Xt = X_all[trial]
                d = np.linalg.det(Xt.T @ Xt)
                if d > best_det * (1 + 1e-9):
                    idx = trial; best_det = d; improved = True
                    break
            if improved: break
        if not improved: break
    return idx, float(best_det)


if __name__ == "__main__":
    print("=== D-optimal design via Fedorov exchange ===\n")
    rng = np.random.default_rng(0)
    # Candidate set: 3^3 factorial (27 runs, 3 factors 3 levels)
    grid = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    # Model: intercept + 3 main effects + 3 pairwise interactions
    def model_row(x):
        return np.array([1, x[0], x[1], x[2], x[0] * x[1], x[0] * x[2], x[1] * x[2]])
    X_all = np.array([model_row(x) for x in grid])
    p = X_all.shape[1]

    for n in (7, 9, 12):
        idx, det = fedorov(X_all, n=n)
        print(f"  n = {n:>2d}   det(X'X) = {det:.4e}   selected candidate rows: {sorted(idx)}")

    # Full-design comparison
    X_full = X_all
    d_full = float(np.linalg.det(X_full.T @ X_full))
    print(f"\n  Full 27-run design det(X'X) = {d_full:.4e}")
    # D-efficiency of n=9 D-optimal vs full design
    idx9, det9 = fedorov(X_all, n=9)
    d_eff = 100 * (det9 / d_full) ** (1 / p) * (27 / 9)
    print(f"  D-efficiency of 9-run design vs 27-run full: {d_eff:.1f}%\n")

    print("--- library cross-check (R AlgDesign::optFederov, DoE.wrapper; Python pyDOE2 + custom) ---")
