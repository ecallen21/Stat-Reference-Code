"""Randomized SVD (Halko, Martinsson & Tropp 2011).

Approximate the top-k singular triplets of a large matrix by
projecting onto a small random subspace, then doing an exact
SVD of the compressed matrix.

Steps:
    1. Draw Omega ~ N(0, 1) of shape (n, k + p)
    2. Y = A @ Omega                    (n rows of A pushed to k+p cols)
    3. Q = orth(Y)                       (QR)
    4. B = Q.T @ A                       (small: (k+p) x n)
    5. U_b, S, Vt = svd(B, full=False)   (small SVD)
    6. U = Q @ U_b
Truncate to top k. Power iteration q > 0 sharpens singular-value decay.
"""

import numpy as np    # arrays + linear algebra
import time    # timing


def randomized_svd(A, k, oversample=10, n_iter=2, seed=0):
    """Randomized top-k SVD of A (m x n)."""
    rng = np.random.default_rng(seed)
    m, n = A.shape
    r = k + oversample
    Omega = rng.standard_normal((n, r))
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)    # power iteration for faster singular decay
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], S[:k], Vt[:k, :]


def demo():
    rng = np.random.default_rng(2026)
    m, n, k_true = 500, 300, 20
    L = rng.standard_normal((m, k_true))
    R = rng.standard_normal((k_true, n))
    A = L @ R + 0.01 * rng.standard_normal((m, n))    # rank-20 + noise

    print("=== Randomized SVD (Halko-Martinsson-Tropp 2011) ===")
    print(f"Matrix shape: {A.shape}, effective rank ~ {k_true}")

    t0 = time.time()
    U_ex, S_ex, Vt_ex = np.linalg.svd(A, full_matrices=False)
    t_ex = time.time() - t0

    t0 = time.time()
    U_r, S_r, Vt_r = randomized_svd(A, k=25, oversample=10, n_iter=2)
    t_r = time.time() - t0

    print(f"Exact SVD time    : {t_ex:.3f}s (top-25 sigmas)")
    print(f"Random. SVD time  : {t_r:.3f}s")
    print(f"|sigma_exact - sigma_rand| max = {np.max(np.abs(S_ex[:25] - S_r)):.2e}")

    A_rand = U_r @ np.diag(S_r) @ Vt_r
    A_exact = U_ex[:, :25] @ np.diag(S_ex[:25]) @ Vt_ex[:25, :]
    err_rand = np.linalg.norm(A - A_rand, "fro")
    err_exact = np.linalg.norm(A - A_exact, "fro")
    print(f"|A - A_rank25|_F  exact = {err_exact:.4f}, random = {err_rand:.4f}")


if __name__ == "__main__":
    demo()
