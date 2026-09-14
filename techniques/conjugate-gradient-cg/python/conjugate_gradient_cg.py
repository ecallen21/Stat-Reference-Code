"""Conjugate Gradient (Reference Sec 47.320).

Hestenes & Stiefel 1952 J Res NBS. Iterative method for solving
A x = b with A symmetric positive-definite. Iterates traverse
CONJUGATE (A-orthogonal) directions:

    r_0 = b - A x_0
    p_0 = r_0
    for k = 0, 1, ..., n-1:
        alpha_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + alpha_k p_k
        r_{k+1} = r_k - alpha_k A p_k
        beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + beta_k p_k

Converges in <= n iterations exactly; much faster if A has
clustered eigenvalues. Preferred for large sparse SPD systems.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=None):
    n = len(b)
    if x0 is None: x0 = np.zeros(n)
    if max_iter is None: max_iter = n * 2
    x = x0.copy(); r = b - A @ x; p = r.copy()
    rr = r @ r
    residuals = [np.sqrt(rr)]
    for k in range(max_iter):
        Ap = A @ p
        alpha = rr / (p @ Ap + 1e-16)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        residuals.append(np.sqrt(rr_new))
        if np.sqrt(rr_new) < tol: break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, residuals


if __name__ == "__main__":
    print("=== Conjugate Gradient (Hestenes & Stiefel 1952) ===\n")
    rng = np.random.default_rng(0)

    # Random SPD system
    n = 100
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)                                    # SPD via M M^T + shift
    b = rng.standard_normal(n)

    x, residuals = conjugate_gradient(A, b, tol=1e-10)
    x_direct = np.linalg.solve(A, b)

    print(f"  n = {n}, condition number = {np.linalg.cond(A):.2f}")
    print(f"  Converged in {len(residuals) - 1} iterations")
    print(f"  Final residual norm: {residuals[-1]:.2e}")
    print(f"  ||x_CG - x_direct||: {np.linalg.norm(x - x_direct):.2e}\n")

    print(f"  Residual decay:")
    for it in sorted({0, 5, 10, min(20, len(residuals) - 1), min(40, len(residuals) - 1), len(residuals) - 1}):
        print(f"    iter {it:>3}   ||r|| = {residuals[it]:.4e}")

    # Show the "clustered eigenvalue" speedup
    A_clustered = np.diag(np.concatenate([np.ones(90), 100 * np.ones(10)]))
    b2 = rng.standard_normal(n)
    _, res_clust = conjugate_gradient(A_clustered, b2, tol=1e-10)
    print(f"\n  A with 2 eigenvalue clusters: converges in {len(res_clust) - 1} iterations")
    print(f"  (theory: CG converges in as many iters as distinct eigenvalue clusters)")

    print("\n--- library cross-check (scipy.sparse.linalg.cg; numpy.linalg.solve for direct) ---")
