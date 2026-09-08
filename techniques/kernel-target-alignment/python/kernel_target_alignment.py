"""Kernel Target Alignment (KTA) (Reference Sec 47.136).

Cristianini, Kandola, Elisseeff & Shawe-Taylor 2001 'On kernel-
target alignment', NeurIPS. Model-agnostic kernel selection:

    A(K, K*) = <K_c, K*_c>_F / (||K_c||_F ||K*_c||_F)

with K_c the CENTERED kernel matrix and K* = y y^T (ideal
kernel from labels). Higher = better; used to pick RBF bandwidth
or combine multiple kernels (MKL).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def target_alignment(K, y):
    Kc = center_kernel(K)
    Kt = np.outer(y, y)
    Ktc = center_kernel(Kt)
    return float(np.sum(Kc * Ktc) / (np.linalg.norm(Kc) * np.linalg.norm(Ktc) + 1e-12))


def rbf(X, sigma):
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-sq / (2 * sigma ** 2))


if __name__ == "__main__":
    print("=== Kernel Target Alignment (Cristianini et al 2001) ===\n")
    rng = np.random.default_rng(0)
    n, d = 200, 2
    # Half-moons style: y = sign(x_0)
    X = rng.normal(size=(n, d))
    y = np.sign(X[:, 0])
    y = np.where(y == 0, 1, y).astype(float)

    print("  Alignment A(K_RBF(sigma), y y^T) across sigma:\n")
    sigmas = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    for s in sigmas:
        A = target_alignment(rbf(X, s), y)
        print(f"    sigma = {s:5.2f}   A = {A:.4f}")

    # Linear kernel comparison
    K_lin = X @ X.T
    A_lin = target_alignment(K_lin, y)
    print(f"\n  Linear kernel:  A = {A_lin:.4f}   (best for a linearly separable target)")

    print("\n  Use the sigma with highest A as a data-driven bandwidth choice.")

    print("\n--- library cross-check (kernlab R; sklearn / MKL Python) ---")
