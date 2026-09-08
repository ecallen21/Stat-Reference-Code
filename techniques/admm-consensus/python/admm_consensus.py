"""ADMM (alternating direction method of multipliers) (Sec 47.68).

Boyd, Parikh, Chu, Peleato & Eckstein 2011 'Distributed
optimization and statistical learning via the alternating
direction method of multipliers', FnT Machine Learning 3(1).
Solve  min f(x) + g(z)  s.t.  Ax + Bz = c  by alternating
between three steps:

    x^{k+1} = argmin_x  f(x) + (rho/2) || Ax + B z^k - c + u^k ||_2^2
    z^{k+1} = argmin_z  g(z) + (rho/2) || Ax^{k+1} + B z - c + u^k ||_2^2
    u^{k+1} = u^k + Ax^{k+1} + B z^{k+1} - c

Ideal for split-friendly problems like lasso, group lasso, TV,
consensus optimisation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)


def admm_lasso(X, y, lam, rho=1.0, max_iter=500, tol_abs=1e-4, tol_rel=1e-3):
    """ADMM for   min 0.5 ||y - X b||^2 + lam ||b||_1  (Boyd 2011 sec 6.4)."""
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    L = np.linalg.cholesky(XtX + rho * np.eye(p))
    x = np.zeros(p); z = np.zeros(p); u = np.zeros(p)
    hist = []
    for k in range(max_iter):
        q = Xty + rho * (z - u)
        x = np.linalg.solve(L.T, np.linalg.solve(L, q))
        z_prev = z
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        r_norm = np.linalg.norm(x - z)
        s_norm = np.linalg.norm(-rho * (z - z_prev))
        eps_pri = np.sqrt(p) * tol_abs + tol_rel * max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dual = np.sqrt(p) * tol_abs + tol_rel * np.linalg.norm(rho * u)
        hist.append((r_norm, s_norm))
        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return {"beta": z, "iters": k + 1, "history": hist}


def admm_consensus_global(local_optimisers, x_init, rho=1.0, max_iter=100, tol=1e-4):
    """Global-consensus ADMM: N agents share a global z; each optimises
    its own local objective f_i(x_i) subject to x_i = z.
    """
    N = len(local_optimisers)
    x = [x_init.copy() for _ in range(N)]
    u = [np.zeros_like(x_init) for _ in range(N)]
    z = x_init.copy()
    for it in range(max_iter):
        for i, opt in enumerate(local_optimisers):
            x[i] = opt(z - u[i], rho)
        x_mean = np.mean(x, axis=0)
        u_mean = np.mean(u, axis=0)
        z_prev = z
        z = x_mean + u_mean
        for i in range(N):
            u[i] = u[i] + x[i] - z
        if np.linalg.norm(z - z_prev) < tol:
            break
    return {"z": z, "x_local": x, "iters": it + 1}


if __name__ == "__main__":
    print("=== ADMM (Boyd et al 2011) ===\n")
    rng = np.random.default_rng(0)

    # 1) ADMM Lasso vs coordinate descent from sklearn
    n, p = 200, 20
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p); beta_true[[0, 3, 7, 15]] = [1.5, -1.0, 0.8, -0.6]
    y = X @ beta_true + 0.4 * rng.normal(size=n)
    r = admm_lasso(X, y, lam=5.0)
    print(f"  Lasso via ADMM (lam=5):  iters = {r['iters']}   "
          f"nnz = {int((np.abs(r['beta']) > 1e-3).sum())}")
    top4 = np.argsort(np.abs(r["beta"]))[::-1][:4]
    print(f"  top4 = {[(int(i), round(r['beta'][i], 3)) for i in top4]}   (truth {[0, 3, 7, 15]})")

    # 2) Consensus ADMM: 5 agents each fit a proximal ridge on their local data
    d = 4
    beta_star = rng.normal(size=d)
    def make_agent(seed):
        rr = np.random.default_rng(seed)
        n_i = 50
        Xi = rr.normal(size=(n_i, d))
        yi = Xi @ beta_star + 0.1 * rr.normal(size=n_i)
        XtX = Xi.T @ Xi
        Xty = Xi.T @ yi
        def prox(v, rho):
            return np.linalg.solve(XtX + rho * np.eye(d), Xty + rho * v)
        return prox
    opts = [make_agent(s) for s in range(5)]
    res = admm_consensus_global(opts, x_init=np.zeros(d), rho=1.0, max_iter=50)
    print(f"\n  Consensus ADMM: 5 agents, {res['iters']} outer iters")
    print(f"  Recovered z = {np.round(res['z'], 3)}   truth = {np.round(beta_star, 3)}")

    print("\n--- library cross-check (ADMM R; cvxpy Python) ---")
