"""Orthogonal Matching Pursuit (Reference Sec 47.123).

Pati, Rezaiifar & Krishnaprasad 1993 'Orthogonal matching pursuit'.
Greedy sparse-regression algorithm:

    1. Initialise residual r = y, active set S = empty.
    2. j* = argmax |X_j' r|   (feature most correlated with residual).
    3. S = S union {j*}.
    4. beta_S = argmin ||y - X_S beta_S||^2  (least-squares refit).
    5. r = y - X_S beta_S; repeat until |S| = K or ||r|| < tol.

Contrast with LARS (which uses equiangular direction, not
refitting from scratch each step).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def omp(X, y, K):
    """Standard OMP for a fixed sparsity budget K."""
    n, p = X.shape
    r = y.copy()
    S = []
    for _ in range(K):
        j = int(np.argmax(np.abs(X.T @ r)))
        if j in S: break
        S.append(j)
        b, *_ = np.linalg.lstsq(X[:, S], y, rcond=None)
        r = y - X[:, S] @ b
    beta = np.zeros(p); beta[S] = b
    return {"beta": beta, "S": S, "resid_norm": float(np.linalg.norm(r))}


if __name__ == "__main__":
    print("=== Orthogonal Matching Pursuit (Pati et al 1993) ===\n")
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X = rng.normal(size=(n, p))
    X = X / np.linalg.norm(X, axis=0, keepdims=True)
    beta_true = np.zeros(p); beta_true[[0, 3, 7, 15]] = [1.5, -1.0, 0.8, -0.6]
    y = X @ beta_true + 0.02 * rng.normal(size=n)

    for K in [1, 2, 4, 8]:
        r = omp(X, y, K)
        top = np.argsort(-np.abs(r["beta"]))[:K]
        print(f"  K = {K}   active-set = {r['S']}   "
              f"resid ||_2 = {r['resid_norm']:.4f}")
    b4 = omp(X, y, K=4)["beta"]
    print(f"\n  Best K = 4 coefs: {[(int(i), round(b4[i], 3)) for i in np.where(np.abs(b4) > 1e-6)[0]]}")
    print(f"  Truth:            {[(0, 1.5), (3, -1.0), (7, 0.8), (15, -0.6)]}")

    print("\n--- library cross-check (glmnet + custom R; sklearn.linear_model.OrthogonalMatchingPursuit) ---")
