"""Nystrom kernel approximation (Williams & Seeger 2001).

Approximate an n x n kernel matrix K by sampling m << n
landmark columns, then reconstructing:

    K ~ C @ pinv(W) @ C.T

where C = K[:, S] (n x m) and W = K[S, S] (m x m).
Reduces cost from O(n^2) storage to O(nm) and turns downstream
kernel ridge / kernel PCA into low-rank problems.
"""

import numpy as np    # arrays
import time    # timing


def rbf_kernel(X, Y, gamma):
    D = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1) - 2 * X @ Y.T
    return np.exp(-gamma * D)


def nystrom(X, m, gamma, seed=0):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    Xm = X[idx]
    W = rbf_kernel(Xm, Xm, gamma)
    C = rbf_kernel(X, Xm, gamma)
    U, s, Vt = np.linalg.svd(W)
    s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
    W_pinv = Vt.T @ np.diag(s_inv) @ U.T
    K_hat = C @ W_pinv @ C.T
    return K_hat, C, W


def demo():
    rng = np.random.default_rng(2026)
    n, d = 800, 5
    X = rng.standard_normal((n, d))
    gamma = 0.1

    print("=== Nystrom Kernel Approximation (Williams-Seeger 2001) ===")
    print(f"n = {n}, d = {d}, RBF gamma = {gamma}")

    t0 = time.time()
    K = rbf_kernel(X, X, gamma)
    t_full = time.time() - t0
    print(f"Full kernel : shape {K.shape}, time {t_full:.3f}s, memory {K.nbytes / 1e6:.1f} MB")

    for m in [50, 100, 200]:
        t0 = time.time()
        K_hat, C, W = nystrom(X, m=m, gamma=gamma)
        t_ny = time.time() - t0
        err_rel = np.linalg.norm(K - K_hat, "fro") / np.linalg.norm(K, "fro")
        cost_mb = (C.nbytes + W.nbytes) / 1e6
        print(f"  m = {m:3d}: time {t_ny:.3f}s, storage {cost_mb:.2f} MB, "
              f"relative Frobenius err = {err_rel:.3f}")


if __name__ == "__main__":
    demo()
