"""Barzilai-Borwein step-size rule (Barzilai-Borwein 1988).

Quasi-Newton flavour of gradient descent using SPECTRAL step
sizes:

    s_k = x_k - x_{k-1}
    y_k = grad f(x_k) - grad f(x_{k-1})

    alpha_BB1 = (s_k.T s_k) / (s_k.T y_k)   (long)
    alpha_BB2 = (s_k.T y_k) / (y_k.T y_k)   (short)

The BB step-size makes plain gradient descent super-linearly
convergent on quadratic problems and highly competitive with
CG.  Cornerstone of `scipy.optimize.minimize(method='L-BFGS-B')`
and many modern proximal-gradient solvers.
"""

import numpy as np    # arrays + linalg


def bb_gd(f, grad, x0, max_iter=200, tol=1e-8, use_bb2=False):
    x_prev = np.array(x0, dtype=float)
    g_prev = grad(x_prev)
    alpha = 1.0 / (np.linalg.norm(g_prev) + 1e-10)
    x = x_prev - alpha * g_prev
    history = [f(x_prev), f(x)]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        s = x - x_prev
        y = g - g_prev
        sy = s @ y
        if sy <= 0:
            alpha = 1e-3
        elif use_bb2:
            alpha = sy / (y @ y + 1e-30)
        else:
            alpha = (s @ s) / sy
        x_prev, g_prev = x.copy(), g.copy()
        x = x - alpha * g
        history.append(f(x))
    return x, history


def gd_fixed(f, grad, x0, lr, max_iter=200):
    x = np.array(x0, dtype=float)
    hist = [f(x)]
    for _ in range(max_iter):
        x = x - lr * grad(x)
        hist.append(f(x))
    return x, hist


def demo():
    print("=== Barzilai-Borwein step (Barzilai-Borwein 1988) ===")
    rng = np.random.default_rng(2026)
    d = 50
    # ill-conditioned SPD quadratic: eigenvalues from 0.001 to 100
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = np.logspace(-1, 2, d)    # kappa = 1e3
    XtX = Q @ np.diag(eigs) @ Q.T
    Xtb = rng.standard_normal(d)
    L_max = eigs.max()

    def f(x):
        return 0.5 * x @ XtX @ x - Xtb @ x

    def grad(x):
        return XtX @ x - Xtb

    x_bb1, hist_bb1 = bb_gd(f, grad, np.zeros(d), max_iter=200, use_bb2=False)
    x_bb2, hist_bb2 = bb_gd(f, grad, np.zeros(d), max_iter=200, use_bb2=True)
    x_fx, hist_fx = gd_fixed(f, grad, np.zeros(d), lr=1.0 / L_max, max_iter=200)
    x_star = np.linalg.solve(XtX, Xtb)
    f_star = f(x_star)

    print(f"  200 iterations, d={d}, kappa = {eigs.max() / eigs.min():.0e}")
    print(f"  Fixed-step 1/L GD : f - f* = {hist_fx[-1] - f_star:.4e}")
    print(f"  BB1 (long)        : f - f* = {hist_bb1[-1] - f_star:.4e}")
    print(f"  BB2 (short)       : f - f* = {hist_bb2[-1] - f_star:.4e}")


if __name__ == "__main__":
    demo()
