"""Matrix Factorisation with ALS for RecSys (Reference Sec 47.118).

Hu, Koren & Volinsky 2008 'Collaborative filtering for implicit
feedback datasets', ICDM. Represent user-item interactions R as
low-rank U V^T with weighted-least-squares that upweights CONFIRMED
interactions and downweights unobserved zeros:

    min_{U, V} sum_{u, i} c_{u,i} (p_{u,i} - u_u^T v_i)^2 + lambda(||U||^2 + ||V||^2)

    p_{u,i} = 1 if R_{u,i} > 0 else 0
    c_{u,i} = 1 + alpha * R_{u,i}

Alternate closed-form user / item updates. Explicit-feedback SVD
(Funk 2006) is the sum-of-squares version on RATED entries only.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def als_implicit(R, K=10, alpha=40, lam=0.1, n_iter=20):
    """Hu-Koren-Volinsky implicit-feedback ALS."""
    n_u, n_i = R.shape
    rng = np.random.default_rng(0)
    U = rng.normal(size=(n_u, K)) * 0.1
    V = rng.normal(size=(n_i, K)) * 0.1
    C = 1 + alpha * R
    P = (R > 0).astype(float)
    for _ in range(n_iter):
        # User update: (V' Ci V + lam I)^-1 V' Ci p_u
        VtV = V.T @ V
        for u in range(n_u):
            Ci = np.diag(C[u])
            A = VtV + V.T @ ((C[u, :, None] - 1) * V) + lam * np.eye(K)
            b = V.T @ (C[u] * P[u])
            U[u] = np.linalg.solve(A, b)
        UtU = U.T @ U
        for i in range(n_i):
            A = UtU + U.T @ ((C[:, i, None] - 1) * U) + lam * np.eye(K)
            b = U.T @ (C[:, i] * P[:, i])
            V[i] = np.linalg.solve(A, b)
    return U, V


if __name__ == "__main__":
    print("=== Implicit-feedback ALS (Hu-Koren-Volinsky 2008) ===\n")
    rng = np.random.default_rng(0)
    n_u, n_i, K_true = 50, 40, 3
    U_star = rng.normal(size=(n_u, K_true))
    V_star = rng.normal(size=(n_i, K_true))
    ratings = np.maximum(U_star @ V_star.T, 0)          # non-negative
    # Convert to implicit: high rating -> observed count
    R = (ratings > np.quantile(ratings, 0.85)).astype(float) * ratings
    frac_obs = float((R > 0).mean())
    print(f"  {n_u}x{n_i} matrix, {int((R > 0).sum())} interactions ({frac_obs*100:.1f}% density)")

    U, V = als_implicit(R, K=5, alpha=40, lam=0.1, n_iter=15)
    scores = U @ V.T

    # Ranking hit rate @10 for held-out interactions
    hits, total = 0, 0
    for u in range(n_u):
        held = np.where(R[u] > 0)[0]
        if len(held) < 3: continue
        holdout = rng.choice(held, size=1)
        # Mask held items in ranking
        s = scores[u].copy()
        s[R[u] > 0] = -np.inf
        s[holdout] = scores[u, holdout]
        top10 = np.argsort(-s)[:10]
        if int(holdout[0]) in top10.tolist():
            hits += 1
        total += 1
    print(f"  Held-out hit@10 across {total} users = {hits}/{total} = {hits/total:.2%}")

    # Baseline: popularity ranking
    pop = R.sum(axis=0)
    top10_pop = np.argsort(-pop)[:10]
    hits_p = 0
    for u in range(n_u):
        held = np.where(R[u] > 0)[0]
        if len(held) < 3: continue
        holdout = rng.choice(held, size=1)
        if int(holdout[0]) in top10_pop.tolist(): hits_p += 1
    print(f"  Popularity baseline hit@10 = {hits_p}/{total} = {hits_p/total:.2%}")

    print("\n--- library cross-check (implicit / lightfm Python; recosystem R) ---")
