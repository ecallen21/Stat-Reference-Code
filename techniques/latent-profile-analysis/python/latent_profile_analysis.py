"""Latent Profile Analysis (Reference Sec 36.2).

Continuous-indicator cousin of Latent Class Analysis: assume a
mixture of K multivariate Gaussians on continuous items.  Fit by
EM; select K by BIC / entropy / substantive plausibility.

  Y_i | class = k ~ MVN(mu_k, Sigma_k)
  class          ~ Categorical(pi_1, ..., pi_K)

Standard variants:
  * Diagonal Sigma  -> LPA-1 (independent items within class).
  * Full     Sigma  -> LPA with residual correlations.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.mixture import GaussianMixture


def fit_lpa(X, K, covariance_type="diag", n_init=5, seed=0):
    m = GaussianMixture(n_components=K, covariance_type=covariance_type,
                        n_init=n_init, random_state=seed).fit(X)
    return m


def entropy(gmm, X):
    P = gmm.predict_proba(X)
    ent = -(P * np.log(P + 1e-12)).sum() / len(X)
    E = 1 - ent / np.log(gmm.n_components)
    return float(E)


if __name__ == "__main__":
    print("=== Latent profile analysis: continuous mixture with K selection ===\n")
    rng = np.random.default_rng(0)
    n = 600
    # 3 profiles differ in 4 continuous indicators
    true_pi = [0.4, 0.4, 0.2]
    means = np.array([[1, 2, 3, 4],
                       [5, 4, 2, 1],
                       [3, 3, 5, 5]], dtype=float)
    z = rng.choice(3, size=n, p=true_pi)
    X = np.stack([rng.normal(means[zi], 0.7, 4) for zi in z], axis=0)

    print(f"  {'K':>3s}  {'BIC':>10s}  {'AIC':>10s}  {'entropy':>8s}")
    best = None
    for K in range(1, 6):
        m = fit_lpa(X, K)
        bic = m.bic(X); aic = m.aic(X); ent = entropy(m, X) if K > 1 else 1.0
        print(f"  {K:>3d}  {bic:>10.2f}  {aic:>10.2f}  {ent:>8.3f}")
        if best is None or bic < best[0]:
            best = (bic, K, m)

    print(f"\n  BIC-optimal K = {best[1]}")
    m = best[2]
    print(f"  Estimated class proportions: {np.round(m.weights_, 3)}")
    print(f"  Estimated means (rows = class):\n{np.round(m.means_, 2)}\n")

    print("--- library cross-check (R mclust, tidyLPA, poLCA (LCA); Python sklearn.mixture) ---")
