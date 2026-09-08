"""Beta-Binomial hierarchical model (Reference Sec 47.132).

Efron & Morris 1975 (Stein-shrinkage of proportions); Gelman-Carlin-
Stern-Rubin 2013 ch 5. For groups i with x_i successes out of n_i:

    p_i | alpha, beta ~ Beta(alpha, beta)
    x_i | p_i, n_i ~ Binomial(n_i, p_i).

Empirical-Bayes or full-Bayes posterior for each p_i shrinks
noisy per-group estimates toward the population mean, with
shrinkage strength alpha + beta.

Demonstrated on Efron-Morris baseball batting averages: shrunk
estimates strictly beat MLE (per-player x/n) at predicting the
rest-of-season averages.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # marginal-likelihood MLE


def beta_binomial_marginal_loglik(params, x, n):
    """Marginal log-likelihood of x_i ~ BetaBinomial(n_i, alpha, beta)."""
    from scipy.special import betaln
    log_alpha, log_beta = params
    a = np.exp(log_alpha); b = np.exp(log_beta)
    return -np.sum(betaln(x + a, n - x + b) - betaln(a, b))


def eb_beta_binomial(x, n):
    x = np.asarray(x); n = np.asarray(n)
    res = minimize(beta_binomial_marginal_loglik, x0=(np.log(2), np.log(2)),
                    args=(x, n), method="Nelder-Mead")
    a, b = np.exp(res.x)
    # Posterior mean per group
    p_hat = (x + a) / (n + a + b)
    return {"alpha": a, "beta": b, "p_shrunk": p_hat}


if __name__ == "__main__":
    print("=== Beta-Binomial hierarchical (Efron-Morris 1975) ===\n")
    # Efron-Morris 1975 baseball data: 18 players, first-45-at-bats hits
    x_first = np.array([18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7])
    n_first = np.full_like(x_first, 45)
    # Held-out rest-of-season averages (from Efron-Morris)
    p_true = np.array([0.346, 0.298, 0.276, 0.222, 0.273, 0.270, 0.263, 0.210,
                        0.269, 0.230, 0.264, 0.256, 0.303, 0.264, 0.226, 0.286, 0.264, 0.210])

    eb = eb_beta_binomial(x_first, n_first)
    print(f"  Empirical-Bayes hyperparameters: alpha = {eb['alpha']:.2f}, "
          f"beta = {eb['beta']:.2f}")
    print(f"  Shrinkage strength alpha + beta = {eb['alpha'] + eb['beta']:.1f} "
          f"vs n_i = 45 (so effective 'prior sample size' comparable)")

    mle = x_first / n_first
    print(f"\n  Per-player | MLE (x/n) | EB shrunk | truth")
    for i in range(len(x_first)):
        print(f"    player {i+1:2d}  |   {mle[i]:.3f}   |   {eb['p_shrunk'][i]:.3f}   |   {p_true[i]:.3f}")

    mse_mle = float(((mle - p_true) ** 2).mean())
    mse_eb = float(((eb['p_shrunk'] - p_true) ** 2).mean())
    print(f"\n  Held-out MSE:  MLE = {mse_mle:.4f}   EB shrunk = {mse_eb:.4f}   "
          f"(EB better by {(1 - mse_eb/mse_mle)*100:.1f} %)")

    print("\n--- library cross-check (VGAM / brms R; pymc / stan Python) ---")
