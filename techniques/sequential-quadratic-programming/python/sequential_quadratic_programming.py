"""Sequential Quadratic Programming (Wilson 1963; Han 1976;
Powell 1978).

For a nonlinearly-constrained problem
    min f(x)  s.t.  h(x) = 0

solve a sequence of QP sub-problems:

    min_p  grad f(x_k)^T p + 0.5 p^T H_k p
    s.t.   h(x_k) + J_h(x_k) p = 0

with `H_k` a (quasi-)Hessian of the Lagrangian.

Demo: Rosenbrock subject to x^2 + y^2 = 1 (min on unit circle).
"""

import numpy as np    # arrays + linalg


def sqp_equality(f, grad_f, h, jac_h, x0, max_iter=100, tol=1e-10, mu_pen=100.0):
    x = np.array(x0, dtype=float)
    n = len(x)
    B = np.eye(n)    # damped BFGS approximation of Hessian of Lagrangian
    lam = np.zeros(len(np.atleast_1d(h(x))))
    g_prev = grad_f(x) - jac_h(x).T @ lam
    for k in range(max_iter):
        g = grad_f(x)
        J = jac_h(x)
        hx = h(x)
        m = len(hx)
        K = np.block([[B, J.T], [J, np.zeros((m, m))]])
        rhs = np.concatenate([-g, -hx])
        sol = np.linalg.solve(K, rhs)
        p = sol[:n]
        lam_new = sol[n:]
        merit_old = f(x) + mu_pen * np.sum(np.abs(hx))
        alpha = 1.0
        for _ in range(50):
            x_new = x + alpha * p
            merit_new = f(x_new) + mu_pen * np.sum(np.abs(h(x_new)))
            if merit_new <= merit_old - 1e-4 * alpha * (g @ p) + 1e-12:
                break
            alpha = alpha * 0.5
        s = alpha * p
        x_new = x + s
        # damped BFGS update on Lagrangian gradient
        g_new = grad_f(x_new) - jac_h(x_new).T @ lam_new
        y = g_new - g_prev
        sBs = s @ B @ s
        sy = s @ y
        if sy < 0.2 * sBs:
            theta = 0.8 * sBs / (sBs - sy)
            r = theta * y + (1 - theta) * B @ s
        else:
            r = y
        Bs = B @ s
        denom_sr = s @ r
        if denom_sr > 1e-12 and sBs > 1e-12:
            B = B - np.outer(Bs, Bs) / sBs + np.outer(r, r) / denom_sr
        x = x_new
        lam = lam_new
        g_prev = g_new
        if alpha * np.linalg.norm(p) < tol:
            break
    return x, k + 1


def demo():
    print("=== Sequential Quadratic Programming (Wilson 1963; Powell 1978) ===")
    print("Min Rosenbrock  subject to  x^2 + y^2 = 1")
    print("Analytical minimum near (0.7864, 0.6177), f ~ 0.0457")

    def f(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def grad_f(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0] ** 2)
        ])

    def h(x):
        return np.array([x[0] ** 2 + x[1] ** 2 - 1])

    def jac_h(x):
        return np.array([[2 * x[0], 2 * x[1]]])

    x_star, iters = sqp_equality(f, grad_f, h, jac_h, x0=[0.0, 1.0], max_iter=100)
    print(f"  SQP result: x* = {np.round(x_star, 5)}, f = {f(x_star):.6f}, "
          f"h = {h(x_star)[0]:.2e}, iters = {iters}")

    try:
        from scipy.optimize import minimize
        con = {"type": "eq", "fun": lambda x: h(x)[0], "jac": lambda x: jac_h(x)[0]}
        res = minimize(f, [0.0, 1.0], jac=grad_f, method="SLSQP", constraints=con)
        print(f"  scipy SLSQP : x* = {np.round(res.x, 5)}, f = {res.fun:.6f}")
    except ImportError:
        pass


if __name__ == "__main__":
    demo()
