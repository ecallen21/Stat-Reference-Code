"""Nelder-Mead simplex (Nelder & Mead 1965).

Derivative-free minimisation using a moving simplex of
n + 1 vertices. Each iteration reflects, expands, contracts,
or shrinks the worst vertex:

    reflect  x_r = x_bar + alpha (x_bar - x_worst)
    expand   x_e = x_bar + gamma (x_r    - x_bar)
    contract x_c = x_bar + rho   (x_worst - x_bar)
    shrink   x_i = x_best + sigma (x_i - x_best)

Standard coefficients: alpha=1, gamma=2, rho=0.5, sigma=0.5.
"""

import numpy as np    # arrays


def nelder_mead(f, x0, tol=1e-8, max_iter=500,
                alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        v = np.array(x0, dtype=float)
        v[i] = v[i] + (0.05 if v[i] != 0 else 0.00025)
        simplex.append(v)
    simplex = np.array(simplex)
    fx = np.array([f(v) for v in simplex])
    history = []
    for k in range(max_iter):
        order = np.argsort(fx)
        simplex, fx = simplex[order], fx[order]
        history.append(fx[0])
        if np.max(np.abs(simplex[1:] - simplex[0])) < tol:
            break
        x_bar = simplex[:-1].mean(axis=0)
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        fr = f(x_r)
        if fx[0] <= fr < fx[-2]:
            simplex[-1], fx[-1] = x_r, fr
            continue
        if fr < fx[0]:
            x_e = x_bar + gamma * (x_r - x_bar)
            fe = f(x_e)
            simplex[-1], fx[-1] = (x_e, fe) if fe < fr else (x_r, fr)
            continue
        x_c = x_bar + rho * (simplex[-1] - x_bar)
        fc = f(x_c)
        if fc < fx[-1]:
            simplex[-1], fx[-1] = x_c, fc
            continue
        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            fx[i] = f(simplex[i])
    return simplex[0], fx[0], k + 1, history


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
               for i in range(len(x) - 1))


def demo():
    print("=== Nelder-Mead simplex (Nelder-Mead 1965) ===")
    print("Rosenbrock d=2, minimum at [1, 1], f*=0")
    x0 = np.array([-1.2, 1.0])
    x_star, f_star, iters, _ = nelder_mead(rosenbrock, x0, tol=1e-9)
    print(f"  d=2: x* = {np.round(x_star, 4)}, f* = {f_star:.2e}, iters = {iters}")

    print("Rosenbrock d=5, minimum at [1,...,1]")
    x0 = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
    x_star, f_star, iters, _ = nelder_mead(rosenbrock, x0, tol=1e-9, max_iter=2000)
    print(f"  d=5: x* ~ {np.round(x_star, 3)}, f* = {f_star:.2e}, iters = {iters}")

    print("Compare with scipy:")
    try:
        from scipy.optimize import minimize
        res = minimize(rosenbrock, x0, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 2000})
        print(f"  scipy.optimize: x = {np.round(res.x, 3)}, f = {res.fun:.2e}, "
              f"iters = {res.nit}")
    except ImportError:
        pass


if __name__ == "__main__":
    demo()
