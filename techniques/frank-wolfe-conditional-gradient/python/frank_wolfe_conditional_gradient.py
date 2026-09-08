"""Frank-Wolfe (conditional-gradient) method (Reference Sec 47.100).

Frank & Wolfe 1956 'An algorithm for quadratic programming', Naval
Res Logist. For a differentiable convex f over a compact convex
set D, at each iteration:

    1. Linearise: s_k = argmin_{s in D}  < grad f(x_k), s >.
    2. Line search / decaying step:  gamma_k in [0, 1].
    3. x_{k+1} = (1 - gamma_k) x_k + gamma_k s_k.

No projections needed -- s_k comes from a linear minimisation oracle
(LMO). Convergence O(1/k). Popular for sparse / low-rank optimisation
over the L1 or nuclear-norm ball, where LMO is a single vector.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fw_lasso(X, y, radius, max_iter=500, tol=1e-6):
    """Frank-Wolfe for  min 0.5 ||y - X beta||^2  s.t. ||beta||_1 <= radius.
    LMO over the L1 ball: s = -radius * sign(g_i) e_i where i = argmax |g|.
    """
    n, p = X.shape
    beta = np.zeros(p)
    history = []
    for k in range(max_iter):
        g = X.T @ (X @ beta - y)                        # gradient
        i = int(np.argmax(np.abs(g)))
        s = np.zeros(p); s[i] = -radius * np.sign(g[i])
        # exact line search on quadratic
        d = s - beta
        Xd = X @ d
        denom = float(Xd @ Xd)
        if denom < 1e-15:
            gamma = 0.0
        else:
            gamma = float(np.clip(-((X @ beta - y) @ Xd) / denom, 0, 1))
        beta_new = beta + gamma * d
        gap = float(g @ (beta - s))
        history.append((float(0.5 * ((X @ beta - y) ** 2).sum()), gap))
        if gap < tol:
            beta = beta_new; break
        beta = beta_new
    return {"beta": beta, "iters": k + 1, "final_gap": gap, "history": history}


if __name__ == "__main__":
    print("=== Frank-Wolfe conditional gradient (Frank-Wolfe 1956) ===\n")
    rng = np.random.default_rng(0)
    n, p = 200, 30
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p); beta_true[[0, 5, 12]] = [1.5, -1.0, 0.8]
    y = X @ beta_true + 0.5 * rng.normal(size=n)

    for r in [1.0, 2.5, 5.0, 10.0]:
        res = fw_lasso(X, y, radius=r, max_iter=500)
        top = np.argsort(np.abs(res["beta"]))[::-1][:5]
        print(f"  L1 ball radius = {r:4.1f}   iters = {res['iters']:3d}   "
              f"final gap = {res['final_gap']:.2e}   "
              f"||beta||_1 = {np.abs(res['beta']).sum():.3f}   "
              f"top5 = {[(int(i), round(res['beta'][i], 2)) for i in top]}")

    print("\n  True support {0, 5, 12} with coefs (1.5, -1.0, 0.8),")
    print("  L1 norm 3.3. FW iterate is sparse (one active coord per step).")
    print("\n--- library cross-check (CVXR / cvxr R; cvxpy / pymanopt / cyipopt Python) ---")
