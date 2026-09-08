"""EM algorithm for finite mixtures (Reference Sec 47.77).

Dempster, Laird & Rubin 1977 'Maximum likelihood from incomplete
data via the EM algorithm', JRSS-B 39. Iterate between:

    E-step: gamma_ik = pi_k * f_k(x_i | theta_k)  /  sum_j pi_j f_j(x_i | theta_j)
    M-step: pi_k     = (1/n) sum_i gamma_ik
             theta_k  = weighted MLE with weights gamma_ik.

Guaranteed monotone increase in log-likelihood; converges to a
local optimum. Illustrated here on a 1D Gaussian mixture with
K components, closed-form M-step updates.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm    # Gaussian density


def em_gmm_1d(x, K, max_iter=200, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    # Initialise: random data means, common std, uniform pi
    mu = rng.choice(x, size=K, replace=False).astype(float)
    sigma = np.full(K, x.std())
    pi = np.full(K, 1.0 / K)

    ll_prev = -np.inf
    for it in range(max_iter):
        # E-step
        pdf = np.stack([pi[k] * norm.pdf(x, mu[k], sigma[k]) for k in range(K)], axis=1)
        pdf_sum = pdf.sum(axis=1)
        ll = np.sum(np.log(pdf_sum + 1e-300))
        gamma = pdf / (pdf_sum[:, None] + 1e-300)
        # M-step
        Nk = gamma.sum(axis=0)
        pi = Nk / n
        mu = (gamma * x[:, None]).sum(axis=0) / Nk
        sigma = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll
    return {"pi": pi, "mu": mu, "sigma": sigma, "loglik": float(ll), "iters": it + 1}


if __name__ == "__main__":
    print("=== EM algorithm for GMM (Dempster-Laird-Rubin 1977) ===\n")
    rng = np.random.default_rng(0)
    # True 3-component 1D mixture
    true_pi = np.array([0.4, 0.35, 0.25])
    true_mu = np.array([-2.5, 0.5, 3.0])
    true_sigma = np.array([0.5, 0.8, 0.6])
    n = 2000
    K_true = 3
    z = rng.choice(K_true, size=n, p=true_pi)
    x = np.array([rng.normal(true_mu[zi], true_sigma[zi]) for zi in z])
    print(f"  n = {n}, K_true = {K_true}")
    print(f"  Truth pi    = {true_pi},  mu = {true_mu},  sigma = {true_sigma}")

    fit = em_gmm_1d(x, K=3, seed=1)
    # Sort by mu for identifiability
    order = np.argsort(fit["mu"])
    print(f"\n  MLE (K=3) iters = {fit['iters']}, log-lik = {fit['loglik']:.2f}")
    print(f"  pi    = {np.round(fit['pi'][order], 3)}")
    print(f"  mu    = {np.round(fit['mu'][order], 3)}")
    print(f"  sigma = {np.round(fit['sigma'][order], 3)}")

    # Model selection via BIC
    print("\n  Model selection (BIC):")
    for k in [2, 3, 4, 5]:
        best_ll = -np.inf
        for s in range(5):
            f = em_gmm_1d(x, K=k, seed=s)
            if f["loglik"] > best_ll:
                best_ll = f["loglik"]
        p_params = k - 1 + 2 * k         # pi (k-1) + mu (k) + sigma (k)
        bic = p_params * np.log(n) - 2 * best_ll
        print(f"    K = {k}:  log-lik = {best_ll:.2f}   BIC = {bic:.2f}")

    print("\n--- library cross-check (mclust R; sklearn.mixture.GaussianMixture Python) ---")
