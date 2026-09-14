"""Preconditioned Conjugate Gradient (Hestenes-Stiefel 1952;
Concus-Golub-O'Leary 1976).

For SPD A and preconditioner M (approximating A^{-1}), solve
A x = b using CG on the preconditioned system M A x = M b:

    r = b - A x
    z = M r
    p = z
    for k in ...:
        alpha = (r.T z) / (p.T A p)
        x = x + alpha p
        r_new = r - alpha A p
        z_new = M r_new
        beta = (r_new.T z_new) / (r.T z)
        p = z_new + beta p

Convergence rate depends on the spectrum of M A, so a good
preconditioner turns O(sqrt(kappa)) into O(sqrt(kappa_M A)).
"""

import numpy as np    # arrays + linalg


def pcg(A, b, M_solve, x0=None, tol=1e-10, max_iter=None):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    r = b - A @ x
    z = M_solve(r)
    p = z.copy()
    rz_old = r @ z
    residuals = [np.linalg.norm(r)]
    max_iter = max_iter or n
    for k in range(max_iter):
        Ap = A @ p
        alpha = rz_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(np.linalg.norm(r))
        if residuals[-1] < tol:
            break
        z = M_solve(r)
        rz_new = r @ z
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, residuals


def demo():
    print("=== Preconditioned Conjugate Gradient (Hestenes-Stiefel 1952) ===")
    rng = np.random.default_rng(2026)
    n = 300
    # SPD matrix with poor conditioning (widely spread eigenvalues)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = np.linspace(1.0, 1000.0, n)    # kappa = 1000
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)

    # 1. Plain CG (M = I)
    x_cg, res_cg = pcg(A, b, M_solve=lambda r: r, tol=1e-10, max_iter=500)

    # 2. Jacobi preconditioner (M = diag(A)^{-1})
    diagA = np.diag(A)
    x_j, res_j = pcg(A, b, M_solve=lambda r: r / diagA, tol=1e-10, max_iter=500)

    # 3. Nearly-perfect preconditioner (Cholesky of A -> M = A^{-1})
    L = np.linalg.cholesky(A)
    def M_perfect(r):
        y = np.linalg.solve(L, r)
        return np.linalg.solve(L.T, y)
    x_p, res_p = pcg(A, b, M_solve=M_perfect, tol=1e-10, max_iter=500)

    print(f"  n = {n}, condition number kappa = {eigvals[-1] / eigvals[0]:.0f}")
    print(f"  Plain CG           : {len(res_cg) - 1:3d} iter, final ‖r‖ = {res_cg[-1]:.2e}")
    print(f"  Jacobi PCG         : {len(res_j) - 1:3d} iter, final ‖r‖ = {res_j[-1]:.2e}")
    print(f"  A^{{-1}} PCG          : {len(res_p) - 1:3d} iter, final ‖r‖ = {res_p[-1]:.2e}")


if __name__ == "__main__":
    demo()
