"""GMRES — Generalized Minimum Residual (Saad & Schultz 1986).

Iteratively solve A x = b for non-symmetric A. Build an
orthonormal Krylov basis via Arnoldi:

    K_k(A, r0) = span{r0, A r0, A^2 r0, ...}

and choose the iterate that minimises ||b - A x||_2 over
x in x0 + K_k. Restart every m steps to bound memory.
"""

import numpy as np    # arrays + linalg


def gmres(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    max_iter = max_iter or n
    restart = restart or min(max_iter, n)
    x = x0.copy()
    residuals = []
    for outer in range(max_iter // restart + 1):
        r = b - A @ x
        beta = np.linalg.norm(r)
        residuals.append(beta)
        if beta < tol:
            return x, residuals
        V = np.zeros((n, restart + 1))
        V[:, 0] = r / beta
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1)
        g[0] = beta
        cs, sn = np.zeros(restart), np.zeros(restart)
        for j in range(restart):
            w = A @ V[:, j]
            for i in range(j + 1):
                H[i, j] = V[:, i] @ w
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-14:
                break
            V[:, j + 1] = w / H[j + 1, j]
            # apply prior Givens rotations
            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            # new Givens
            r_ = np.hypot(H[j, j], H[j + 1, j])
            cs[j] = H[j, j] / r_
            sn[j] = H[j + 1, j] / r_
            H[j, j] = r_
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            residuals.append(abs(g[j + 1]))
            if abs(g[j + 1]) < tol:
                break
        y = np.linalg.solve(H[:j + 1, :j + 1], g[:j + 1])
        x = x + V[:, :j + 1] @ y
    return x, residuals


def demo():
    print("=== GMRES — Generalized Minimum Residual (Saad-Schultz 1986) ===")
    rng = np.random.default_rng(2026)
    n = 200
    # non-symmetric matrix with clustered eigenvalues
    A = rng.standard_normal((n, n))
    A = A + n * np.eye(n)    # make it diagonally dominant
    b = rng.standard_normal(n)

    x_ref = np.linalg.solve(A, b)
    print(f"  Non-symmetric n={n}, non-sym asymmetry ‖A - A.T‖/‖A‖ = "
          f"{np.linalg.norm(A - A.T) / np.linalg.norm(A):.3f}")

    for restart in [10, 30, 60]:
        x, res = gmres(A, b, restart=restart, max_iter=200, tol=1e-10)
        err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        print(f"  restart = {restart:2d}: {len(res)} matvecs, "
              f"relative error = {err:.2e}, final res = {res[-1]:.2e}")


if __name__ == "__main__":
    demo()
