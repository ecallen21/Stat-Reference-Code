"""Savage-Dickey density-ratio Bayes factor (Reference Sec 47.59).

Dickey 1971 'The weighted likelihood ratio'; Wagenmakers, Lodewyckx,
Kuriyal & Grasman 2010 'Bayesian hypothesis testing for
psychologists: A tutorial on the Savage-Dickey method'. For NESTED
models H_0: theta = theta_0 vs H_1: theta free, the Bayes factor
simplifies to the density ratio at the null point:

    BF_{01}  =  p(theta_0 | y, H_1) / p(theta_0 | H_1)

Only requires posterior + prior evaluated at the null value -- no
marginal-likelihood integral, no bridge sampling.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm    # closed-form conjugate posterior


def savage_dickey_normal_mean(y, mu0=0.0, prior_mu=0.0, prior_tau2=1.0, sigma2=1.0):
    """BF_01 for H_0: mu = mu0 vs H_1: mu ~ N(prior_mu, prior_tau2).

    Data y ~ N(mu, sigma2), n observations. Conjugate posterior is
    N(mu_post, tau2_post) with closed form.
    """
    n = len(y)
    ybar = float(np.mean(y))
    tau2_post = 1.0 / (1.0 / prior_tau2 + n / sigma2)
    mu_post = tau2_post * (prior_mu / prior_tau2 + n * ybar / sigma2)
    prior_pdf = norm.pdf(mu0, loc=prior_mu, scale=np.sqrt(prior_tau2))
    post_pdf = norm.pdf(mu0, loc=mu_post, scale=np.sqrt(tau2_post))
    bf01 = post_pdf / prior_pdf
    return {"bf01": float(bf01), "bf10": float(1 / bf01),
            "mu_post": float(mu_post), "tau2_post": float(tau2_post),
            "prior_pdf": float(prior_pdf), "post_pdf": float(post_pdf)}


if __name__ == "__main__":
    print("=== Savage-Dickey Bayes factor (Dickey 1971; Wagenmakers 2010) ===\n")
    rng = np.random.default_rng(0)
    prior_mu, prior_tau2, sigma2 = 0.0, 4.0, 1.0

    for effect in [0.0, 0.2, 0.5, 1.0]:
        y = effect + rng.normal(size=50) * np.sqrt(sigma2)
        r = savage_dickey_normal_mean(y, mu0=0.0,
                                        prior_mu=prior_mu, prior_tau2=prior_tau2,
                                        sigma2=sigma2)
        interp = ("H0" if r["bf01"] > 3 else "no" if r["bf01"] > 1 / 3 else "H1")
        print(f"  true mu={effect:+.2f}  ybar={y.mean():+.3f}   "
              f"BF_01 = {r['bf01']:8.3f}   BF_10 = {r['bf10']:8.3f}   "
              f"post mu = {r['mu_post']:+.3f}   verdict: {interp}")

    print("\n  BF_01 > 3 = moderate evidence for H_0, > 10 = strong;")
    print("  BF_10 > 3 = moderate evidence for H_1 (Jeffreys 1961 scale).")

    print("\n--- library cross-check (BayesFactor / bridgesampling R; PyMC Python) ---")
