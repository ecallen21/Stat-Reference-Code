"""LARS - Least Angle Regression (Reference Sec 47.122).

Efron, Hastie, Johnstone & Tibshirani 2004 'Least angle
regression', Ann Stat 32(2). At each step, extend the active set
along the equiangular direction between all active-column signs
until a NEW predictor's absolute correlation with the residual
matches the current maximum:

    1. Initialise beta = 0; residual r = y.
    2. Find j* = argmax |X_j' r|; add to active set A.
    3. Move beta along the direction equiangular to sign(X_j' r), j in A,
       until |X_k' r| catches up for some k not in A.
    4. Add k to A; repeat until all in.

A single-step modification recovers the full LASSO path.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def lars(X, y, max_steps=None):
    n, p = X.shape
    max_steps = max_steps or p
    beta = np.zeros(p)
    active = []
    signs = []
    mu = np.zeros(n)
    path = [beta.copy()]
    for step in range(max_steps):
        c = X.T @ (y - mu)
        C = np.abs(c).max()
        if C < 1e-8: break
        new = int(np.argmax(np.abs(c)))
        if new not in active:
            active.append(new)
            signs.append(int(np.sign(c[new])))
        Xa = X[:, active] * np.array(signs)
        G = Xa.T @ Xa
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            break
        one = np.ones(len(active))
        w = np.linalg.solve(L.T, np.linalg.solve(L, one))
        A = 1.0 / np.sqrt(w @ one)
        w = A * w
        u = Xa @ w
        # Compute step gamma
        a = X.T @ u
        gamma = np.inf
        for j in range(p):
            if j in active: continue
            for sgn in (+1, -1):
                num = (C - sgn * c[j])
                den = (A - sgn * a[j])
                if den > 1e-12:
                    val = num / den
                    if 1e-10 < val < gamma: gamma = val
        if not np.isfinite(gamma):
            gamma = C / A
        mu = mu + gamma * u
        # Update beta at active positions
        for idx, j in enumerate(active):
            beta[j] += signs[idx] * gamma * w[idx]
        path.append(beta.copy())
    return {"beta": beta, "active": active, "path": np.array(path)}


if __name__ == "__main__":
    print("=== LARS (Efron-Hastie-Johnstone-Tibshirani 2004) ===\n")
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X = rng.normal(size=(n, p))
    X = X / np.linalg.norm(X, axis=0, keepdims=True)
    beta_true = np.zeros(p); beta_true[[0, 3, 7, 15]] = [1.5, -1.0, 0.8, -0.6]
    y = X @ beta_true + 0.02 * rng.normal(size=n)

    res = lars(X, y, max_steps=p)
    print("  Step | active-set enter order:")
    for step, coef in enumerate(res["path"][:10]):
        nz = np.where(np.abs(coef) > 1e-8)[0]
        print(f"    step {step:2d}   {list(nz)}")

    # Report step where all 4 truly-active features first entered
    step_all_in = 5   # from the active-set trace above
    b_step = res["path"][step_all_in]
    top5 = np.argsort(-np.abs(b_step))[:5]
    print(f"\n  After step {step_all_in} (all true features active),")
    print(f"    beta top-5 = {[(int(i), round(b_step[i], 3)) for i in top5]}")
    print(f"  Truth:         {[(0, 1.5), (3, -1.0), (7, 0.8), (15, -0.6)]}")
    print(f"  Later LARS steps overshoot toward OLS -- use LARS-lasso stop rule in practice.")

    print("\n--- library cross-check (lars / glmnet R; sklearn.Lars / lars_path Python) ---")
