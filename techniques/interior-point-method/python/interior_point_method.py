"""Interior-point / barrier method (Karmarkar 1984;
Nesterov-Nemirovski 1994).

For inequality-constrained convex program
    min f(x)  subject to  A x <= b

Replace hard constraints by a LOG BARRIER:
    F(x, t) = t f(x) - sum_i log(b_i - a_i^T x)

Minimise F for increasing t; the barrier keeps iterates
STRICTLY FEASIBLE. Converges in O(sqrt(m) log(1/eps)) Newton
steps for self-concordant barriers.

Demo: LP min c^T x s.t. A x <= b, x >= 0
"""

import numpy as np    # arrays + linalg


def barrier_lp(c, A, b, x0, mu=10.0, t0=1.0, tol=1e-8, max_outer=30):
    """Log-barrier interior-point for LP: min c.T x s.t. A x <= b."""
    x = np.array(x0, dtype=float)
    t = t0
    m = A.shape[0]
    for _ in range(max_outer):
        # centering: Newton method on t c.T x - sum log(b - A x)
        for _ in range(50):
            s = b - A @ x
            g = t * c + A.T @ (1.0 / s)
            H = A.T @ np.diag(1.0 / s ** 2) @ A
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            # backtracking to stay feasible
            step = 1.0
            while np.any(b - A @ (x + step * dx) <= 0):
                step = step * 0.5
                if step < 1e-12:
                    break
            # Armijo
            f0 = t * c @ x - np.sum(np.log(s))
            for _ in range(50):
                x_new = x + step * dx
                s_new = b - A @ x_new
                if np.all(s_new > 0):
                    f_new = t * c @ x_new - np.sum(np.log(s_new))
                    if f_new < f0 + 0.1 * step * g @ dx:
                        break
                step = step * 0.5
            x = x + step * dx
            if step * np.linalg.norm(dx) < 1e-10:
                break
        if m / t < tol:
            break
        t = t * mu
    return x


def demo():
    print("=== Interior-point / log-barrier LP (Karmarkar 1984; NN 1994) ===")
    # min c^T x subject to x >= 0, and (extra) A x <= b
    # tiny classic LP
    c = np.array([-3.0, -5.0])
    A = np.array([[1, 0], [0, 2], [3, 2], [-1, 0], [0, -1]], dtype=float)
    b = np.array([4, 12, 18, 0, 0], dtype=float)
    # ordinary optimum at (2, 6), f* = -3*2 - 5*6 = -36

    x0 = np.array([1.0, 1.0])    # strictly feasible
    x_star = barrier_lp(c, A, b, x0, mu=10.0, t0=1.0)
    print(f"  Solved  x* = {np.round(x_star, 5)},   f* = {c @ x_star:.4f}")
    print(f"  Expected  x* ~ (2, 6),                 f* = -36.0000")

    # comparison to scipy.linprog
    try:
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A, b_ub=b, method="highs")
        print(f"  scipy.linprog highs: x = {np.round(res.x, 5)}, f = {res.fun:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    demo()
