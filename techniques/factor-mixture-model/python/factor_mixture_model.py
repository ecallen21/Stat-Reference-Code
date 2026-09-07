"""Factor mixture model (Reference Sec 36.10).

Yung 1997 / Lubke-Muthen 2005.  Combines factor analysis
(continuous latent factor within class) with a categorical latent
class:

    y_i | class = k, eta_i ~ N(mu_k + Lambda_k eta_i, Theta_k)
    eta_i                  ~ N(0, Psi_k)
    class                  ~ Categorical(pi_1, ..., pi_K)

Motivation: even within a class, indicators may share common
latent factors.  Compact demo: 2 classes with different factor
loadings.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.mixture import GaussianMixture


def factor_mixture_fit(X, K=2, n_factors=1, n_iter=50, seed=0):
    """Compact factor mixture via GaussianMixture with structured covariance approximation.

    Approximates each class covariance as Lambda_k Lambda_k^T + Psi_k (diagonal),
    then extracts loadings via SVD of the fitted covariance.
    """
    m = GaussianMixture(n_components=K, covariance_type="full",
                        n_init=5, random_state=seed).fit(X)
    loadings = []
    for k in range(K):
        C = m.covariances_[k]
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        L = U[:, :n_factors] * np.sqrt(s[:n_factors])
        loadings.append(L)
    return {"gmm": m, "loadings": loadings,
            "means": m.means_, "weights": m.weights_}


if __name__ == "__main__":
    print("=== Factor mixture model (2 classes, 1 factor each) ===\n")
    rng = np.random.default_rng(0)
    n = 800; p = 6
    # Two classes with different loadings + means
    L1 = np.array([0.8, 0.7, 0.6, 0.1, 0.1, 0.1])
    L2 = np.array([0.1, 0.1, 0.1, 0.7, 0.8, 0.6])
    mu1 = np.array([1, 1, 1, 3, 3, 3], dtype=float)
    mu2 = np.array([3, 3, 3, 1, 1, 1], dtype=float)
    Psi = 0.3

    z = rng.integers(0, 2, n)
    eta = rng.normal(0, 1, n)
    X = np.zeros((n, p))
    for i in range(n):
        L = L1 if z[i] == 0 else L2
        mu = mu1 if z[i] == 0 else mu2
        X[i] = mu + L * eta[i] + rng.normal(0, np.sqrt(Psi), p)

    r = factor_mixture_fit(X, K=2, n_factors=1)
    # Align classes by mean
    order = np.argsort(r["means"][:, 0])   # sort by first indicator mean
    print(f"  Estimated pi: {np.round(r['weights'][order], 3)}   (true 0.5 / 0.5)")
    for i, k in enumerate(order):
        L = r["loadings"][k].flatten()
        # Flip sign to align with truth
        target = L1 if i == 0 else L2
        if np.corrcoef(L, target)[0, 1] < 0: L = -L
        print(f"  Class {i}: loadings = {np.round(L, 2)}")
        print(f"           truth    = {np.round(target, 2)}")

    print("\n--- library cross-check (R OpenMx / Mplus; Python custom + sklearn.mixture) ---")
