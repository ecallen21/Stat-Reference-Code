"""Random Fourier Features (Reference Sec 47.86).

Rahimi & Recht 2007 'Random features for large-scale kernel
machines', NeurIPS. Bochner's theorem: any shift-invariant kernel
k(x, y) = k(x - y) is a Fourier transform of a probability
measure. Draw D random frequencies w_j ~ p(w) and phases b_j ~
Unif(0, 2 pi); the feature map

    phi(x) = sqrt(2 / D) * [ cos(w_1' x + b_1), ..., cos(w_D' x + b_D) ]

satisfies  E[ phi(x)' phi(y) ] = k(x, y).  A linear model on
phi(X) approximates a kernel-ridge / kernel-SVM at O(nD) memory
instead of O(n^2).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rff_gaussian(X, D, sigma, seed=0):
    """Random Fourier feature map for RBF kernel exp(-|x-y|^2 / (2 sigma^2))."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.normal(size=(d, D)) / sigma
    b = rng.uniform(0, 2 * np.pi, size=D)
    return np.sqrt(2 / D) * np.cos(X @ W + b), W, b


def apply_rff(X, W, b):
    D = W.shape[1]
    return np.sqrt(2 / D) * np.cos(X @ W + b)


def rbf_kernel(X, Y, sigma):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-sq / (2 * sigma ** 2))


if __name__ == "__main__":
    print("=== Random Fourier Features (Rahimi-Recht 2007) ===\n")
    rng = np.random.default_rng(0)
    n = 500; d = 8
    X = rng.normal(size=(n, d))
    sigma = float(np.sqrt(d))                # median-ish heuristic

    Kex = rbf_kernel(X, X, sigma)
    for D in [50, 200, 800]:
        Phi, W, b = rff_gaussian(X, D, sigma)
        K_rff = Phi @ Phi.T
        rel_err = float(np.linalg.norm(Kex - K_rff, "fro") / np.linalg.norm(Kex, "fro"))
        print(f"  D = {D:4d}   ||K - Phi Phi'||_F / ||K||_F = {rel_err:.4f}")

    # Kernel ridge on RFF vs exact
    y = X[:, 0] ** 2 - X[:, 1] * X[:, 2] + 0.2 * rng.normal(size=n)
    lam = 1.0
    from time import time    # naive timing
    t0 = time()
    alpha = np.linalg.solve(Kex + lam * np.eye(n), y)
    yhat_ex = Kex @ alpha
    t_ex = time() - t0
    print(f"\n  Exact kernel ridge  MSE = {float(((y - yhat_ex) ** 2).mean()):.4f}   "
          f"(time {t_ex:.2f} s)")

    for D in [100, 500, 2000]:
        Phi, W, b = rff_gaussian(X, D, sigma)
        t0 = time()
        beta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(D), Phi.T @ y)
        yhat = Phi @ beta
        t_r = time() - t0
        mse = float(((y - yhat) ** 2).mean())
        print(f"  RFF ridge D = {D:4d}  MSE = {mse:.4f}   (time {t_r:.2f} s)")

    print("\n--- library cross-check (kernlab R; scikit-learn RBFSampler / Nystroem Python) ---")
