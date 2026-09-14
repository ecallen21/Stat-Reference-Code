"""Lanczos iteration (Lanczos 1950).

Compute the top eigenpairs of a large symmetric matrix by
building a Krylov subspace and reducing A to a small tri-
diagonal matrix T.

Recurrence:
    beta_j v_j = A v_{j-1} - alpha_{j-1} v_{j-1} - beta_{j-1} v_{j-2}
    alpha_j = v_j.T @ A @ v_j
The eigenvalues of T_{k x k} (Ritz values) converge to
extremal eigenvalues of A in O(k) matrix-vector products.
"""

import numpy as np    # arrays + linalg
import time    # timing


def lanczos(A, k, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)
    V = [v]
    alphas = []
    betas = [0.0]
    v_prev = np.zeros(n)
    for j in range(k):
        w = A @ V[-1]
        alpha = V[-1] @ w
        alphas.append(alpha)
        w = w - alpha * V[-1] - betas[-1] * v_prev
        # full re-orthogonalisation (numerical stability)
        for u in V:
            w = w - (u @ w) * u
        beta = np.linalg.norm(w)
        if beta < 1e-12:
            break
        v_prev = V[-1]
        V.append(w / beta)
        betas.append(beta)
    T = np.diag(alphas) + np.diag(betas[1:-1], 1) + np.diag(betas[1:-1], -1)
    ritz_vals, ritz_vecs = np.linalg.eigh(T)
    V_mat = np.array(V[:len(alphas)]).T
    approx_vecs = V_mat @ ritz_vecs
    return ritz_vals, approx_vecs


def demo():
    rng = np.random.default_rng(2026)
    n = 400
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2    # symmetric

    print("=== Lanczos Iteration (Lanczos 1950) ===")
    print(f"Symmetric matrix n = {n}, exact eigendecomposition...")
    t0 = time.time()
    lam_exact, _ = np.linalg.eigh(A)
    t_ex = time.time() - t0

    for k in [10, 30, 60]:
        t0 = time.time()
        ritz, _ = lanczos(A, k=k)
        t_l = time.time() - t0
        err_max = np.max(np.abs(ritz[-3:] - lam_exact[-3:]))
        err_min = np.max(np.abs(ritz[:3] - lam_exact[:3]))
        print(f"  k = {k:2d} steps ({t_l:.3f}s): top-3 err {err_max:.2e}, bot-3 err {err_min:.2e}")

    print(f"Reference    ({t_ex:.3f}s): top-3 exact = {np.round(lam_exact[-3:], 3)}")


if __name__ == "__main__":
    demo()
