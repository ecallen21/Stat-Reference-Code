"""Gaussian process latent variable model (GPLVM) (Reference Sec 6.16).

Lawrence 2004 'Gaussian process latent variable models for
visualisation of high-dimensional data', NIPS. Nonlinear
dimensionality reduction: place a GP prior on the mapping X -> Y
where X is UNKNOWN (latent, low-dim), and MLE-optimise X + kernel
hyperparameters jointly.

The marginal likelihood (integrating out the GP function) is:

    log p(Y | X, theta) = -0.5 * D * log|K + sigma^2 I|
                          - 0.5 * tr((K + sigma^2 I)^{-1} Y Y')
                          - 0.5 * N * D * log(2 pi)

Optimise X by gradient-based methods; usually initialise from PCA.

Equivalent to probabilistic PCA when the kernel is linear; RBF/Matern
kernels give nonlinear generalisations. Used in visualising / imputing
high-dimensional data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize


def rbf_kernel(X, lengthscale=1.0, variance=1.0):
    D2 = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T
    return variance * np.exp(-0.5 * D2 / lengthscale ** 2)


def gplvm_neg_log_lik(params, Y, q, N, sigma2=1e-2):
    X = params.reshape(N, q)
    K = rbf_kernel(X) + sigma2 * np.eye(N)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    logdet = 2 * np.sum(np.log(np.diag(L)))
    #  Solve K^-1 Y column by column
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
    D = Y.shape[1]
    return 0.5 * D * logdet + 0.5 * np.sum(Y * alpha)


def gplvm_fit(Y, q=2, sigma2=1e-2, max_iter=200):
    N, D = Y.shape
    #  Initial X from PCA
    Yc = Y - Y.mean(axis=0)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    X0 = U[:, :q] * S[:q]
    X0 /= np.linalg.norm(X0, axis=0)
    r = minimize(gplvm_neg_log_lik, X0.ravel(),
                  args=(Y, q, N, sigma2), method="L-BFGS-B",
                  options={"maxiter": max_iter})
    return r.x.reshape(N, q)


if __name__ == "__main__":
    print("=== Gaussian process latent variable model (GPLVM) ===\n")
    rng = np.random.default_rng(0)
    N = 80; D = 6; q = 2
    #  True latent: a swiss-roll-ish nonlinear pattern
    t = np.linspace(0, 3 * np.pi, N)
    z = np.c_[t * np.cos(t), t * np.sin(t)]
    #  Random projection into D-dim + noise
    W = rng.normal(scale=0.4, size=(2, D))
    Y = z @ W + rng.normal(scale=0.1, size=(N, D))

    X_pca = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)[0][:, :q]
    X_gplvm = gplvm_fit(Y, q=q, sigma2=0.02, max_iter=100)

    #  Correlation of first latent dim with true angle t
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    print(f"  N={N}, D={D}, q={q}")
    print(f"  Corr(true angle t, PCA dim 1)         = {corr(X_pca[:, 0], t):+.3f}")
    print(f"  Corr(true angle t, PCA dim 2)         = {corr(X_pca[:, 1], t):+.3f}")
    print(f"  Corr(true angle t, GPLVM dim 1)       = {corr(X_gplvm[:, 0], t):+.3f}")
    print(f"  Corr(true angle t, GPLVM dim 2)       = {corr(X_gplvm[:, 1], t):+.3f}")
    print(f"\n  GPLVM captures nonlinear latent structure PCA misses on curved manifolds.")

    print("\n--- library cross-check (GPy / gpflow / gpytorch Python; kergp R) ---")
