"""Kernel PCA (Reference §9.10; Scholkopf-Smola-Muller 1998).

Nonlinear dimensionality reduction: implicit-lift the data to a high-
dimensional feature space via a kernel k(x, y) = <phi(x), phi(y)>, then
run PCA there.  Never materialize phi(x) -- rely on the KERNEL TRICK.

Algorithm
    1. Build the (n x n) kernel Gram matrix K_ij = k(x_i, x_j).
    2. CENTER K in feature space:
        K_c = K - 1_n K - K 1_n + 1_n K 1_n
        where 1_n is the (n x n) matrix of 1/n.
    3. Eigen-decompose K_c: K_c v_k = lambda_k v_k.
    4. k-th principal component of x_i is (1/sqrt(lambda_k)) K_c[i, :] v_k.

Common kernels
    - Radial basis function: exp(-gamma ||x - y||^2)  (default)
    - Polynomial:           (x^T y + c)^d
    - Sigmoid:              tanh(a x^T y + b)

Contrast with PCA: linear PCA cannot separate concentric rings; kernel PCA
with RBF can.  Also related to spectral clustering / diffusion maps.

Out-of-sample projection needs the kernel between the new point and all
training points; add to the code below when you need it.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def rbf_gram(X, gamma: float = 1.0):
    X = np.asarray(X, dtype=float)
    d2 = np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :] - 2 * X @ X.T
    return np.exp(-gamma * d2)


def polynomial_gram(X, degree: int = 3, coef0: float = 1.0):
    X = np.asarray(X, dtype=float)
    return (X @ X.T + coef0) ** degree


def kernel_pca(X, n_components: int = 2, kernel: str = "rbf",
               gamma: float = 1.0, degree: int = 3, coef0: float = 1.0) -> dict:
    """Kernel PCA on X (n x p).  Returns embedding (n x n_components)."""
    X = np.asarray(X, dtype=float); n = X.shape[0]
    if kernel == "rbf": K = rbf_gram(X, gamma)
    elif kernel == "poly": K = polynomial_gram(X, degree, coef0)
    else: raise ValueError("kernel must be 'rbf' or 'poly'")
    # Centering
    one_n = np.ones((n, n)) / n
    K_c = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    # Eigen decomposition (symmetric)
    lam, V = np.linalg.eigh(K_c)
    # Sort descending
    idx = np.argsort(-lam); lam = lam[idx]; V = V[:, idx]
    # Keep positive eigenvalues
    pos = lam > 1e-8
    lam = lam[pos]; V = V[:, pos]
    Z = V[:, :n_components] * np.sqrt(lam[:n_components])
    return {"embedding": Z, "eigenvalues": lam[:n_components],
            "kernel": kernel,
            "gamma": gamma if kernel == "rbf" else None,
            "method": f"Kernel PCA ({kernel} kernel)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Two concentric rings: NOT separable by linear PCA
    n_per = 100
    r_inner = 1 + rng.normal(0, 0.1, n_per)
    theta_inner = rng.uniform(0, 2 * math.pi, n_per)
    r_outer = 3 + rng.normal(0, 0.1, n_per)
    theta_outer = rng.uniform(0, 2 * math.pi, n_per)
    X = np.vstack([np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]),
                    np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])])
    y = np.array([0] * n_per + [1] * n_per)

    print("=== Kernel PCA (RBF, gamma = 0.5) on concentric rings ===")
    r = kernel_pca(X, n_components=2, kernel="rbf", gamma=0.5)
    Z = r["embedding"]
    print(f"  embedding shape: {Z.shape}")
    print(f"  top eigenvalues: {r['eigenvalues'][:5].round(4)}")
    # Class separation on the first component: gap between mean per class
    sep = abs(Z[y == 0, 0].mean() - Z[y == 1, 0].mean()) / Z[:, 0].std()
    print(f"  class-mean separation (SD units) on PC1: {sep:.3f}")

    print("\n=== Linear PCA (for contrast) ===")
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Zlin = U * S
    sep = abs(Zlin[y == 0, 0].mean() - Zlin[y == 1, 0].mean()) / Zlin[:, 0].std()
    print(f"  class-mean separation on linear PC1: {sep:.3f}")

    print("\n--- library cross-check (sklearn KernelPCA) ---")
    try:
        from sklearn.decomposition import KernelPCA
        Zsk = KernelPCA(n_components=2, kernel="rbf", gamma=0.5).fit_transform(X)
        sep = abs(Zsk[y == 0, 0].mean() - Zsk[y == 1, 0].mean()) / Zsk[:, 0].std()
        print(f"  sklearn KernelPCA class-mean sep on PC1: {sep:.3f}")
    except Exception as ex:
        print(f"  (sklearn KernelPCA unavailable: {ex})")
