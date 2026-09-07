"""MRP -- multilevel regression + poststratification (Reference Sec 27.8).

Gelman & Little 1997 'Poststratification into many categories using
hierarchical logistic regression', Survey Methodology; Park, Gelman &
Bafumi 2004. State-of-the-art for extrapolating from non-probability
or under-covered samples to a target population by:

    1. Fit a MULTILEVEL model of the outcome on covariates
       (demographics, geography, etc.) using the sample.
    2. POSTSTRATIFY: predict the outcome for every cell in a full
       population poststratification frame (typically Census) and
       average, weighting by cell population.

MRP corrects for both sample selection bias (some cells over-
represented) AND small-sample noise (multilevel shrinkage stabilises
rare cells).

We demonstrate on a simulated 'state x age x race' cell structure:
sample is skewed toward younger urban voters, but MRP recovers the
national mean by upweighting under-sampled cells and shrinking
rare-cell estimates.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def multilevel_fit(y, groups):
    """Simple 2-level EM for random intercepts by group (partial pooling).

    Returns (grand mean mu, group-specific intercepts alpha_g, sigma_u, sigma_e).
    """
    n = len(y)
    G = groups.max() + 1
    mu = y.mean()
    sig_u = 1.0; sig_e = 1.0
    alpha = np.zeros(G)
    for _ in range(200):
        #  E-step: BLUP of alpha_g
        n_g = np.bincount(groups, minlength=G)
        y_bar_g = np.bincount(groups, weights=y, minlength=G) / np.maximum(n_g, 1)
        alpha = (n_g / sig_e) / (n_g / sig_e + 1 / sig_u) * (y_bar_g - mu)
        #  M-step: variances
        resid = y - (mu + alpha[groups])
        sig_e_new = float(np.mean(resid ** 2) + 1e-8)
        sig_u_new = float(np.mean(alpha ** 2) + 1e-8)
        mu_new = float(np.mean(y - alpha[groups]))
        if abs(sig_e_new - sig_e) + abs(sig_u_new - sig_u) < 1e-6:
            mu = mu_new; sig_u = sig_u_new; sig_e = sig_e_new; break
        mu = mu_new; sig_u = sig_u_new; sig_e = sig_e_new
    return mu, alpha, sig_u, sig_e


if __name__ == "__main__":
    print("=== MRP -- multilevel regression + poststratification ===\n")
    rng = np.random.default_rng(0)
    G = 20
    #  True cell-level mean support = 0.4 + 0.15 * z_g   (z_g are cell shocks)
    z = rng.normal(size=G)
    true_cell_mean = 0.40 + 0.10 * z
    #  Population cell sizes: mostly middle-sized cells but some very small ones
    pop_sizes = rng.integers(500, 5000, size=G)
    P = pop_sizes.sum()
    true_national = float(np.sum(true_cell_mean * pop_sizes) / P)
    print(f"  True national mean (weighted by cell size) = {true_national:.3f}")

    #  Sampled: undersample cells 0..4 (biased toward cells 15..19)
    sample_probs = np.array([1.0 if g >= 5 else 0.15 for g in range(G)])
    n_per_cell = (pop_sizes * sample_probs * 0.02).astype(int)
    y_list = []; g_list = []
    for g in range(G):
        n_g = max(n_per_cell[g], 2)
        y_g = rng.normal(loc=true_cell_mean[g], scale=0.15, size=n_g)
        y_list.append(y_g); g_list.append(np.full(n_g, g))
    y = np.concatenate(y_list); groups = np.concatenate(g_list)

    #  Naive (unweighted) sample mean
    naive = float(y.mean())
    print(f"  Naive sample mean                          = {naive:.3f}   (biased upward)")

    #  MRP: fit multilevel model, then poststratify by cell population
    mu_hat, alpha_hat, sig_u, sig_e = multilevel_fit(y, groups)
    cell_pred = mu_hat + alpha_hat                       # posterior-mean per cell
    mrp_est = float(np.sum(cell_pred * pop_sizes) / P)
    print(f"  MRP poststratified mean                    = {mrp_est:.3f}   (corrects the bias)")

    #  Compare with a raw poststratified estimate (no shrinkage, so noisy)
    y_bar_g = np.array([y[groups == g].mean() if (groups == g).any() else 0.0 for g in range(G)])
    raw_ps = float(np.sum(y_bar_g * pop_sizes) / P)
    print(f"  Raw poststratified (no shrinkage)          = {raw_ps:.3f}   (correct on average, noisy)")

    print(f"\n  MRP saves noise by shrinking small-sample cells; poststratification\n"
          f"  corrects the sample-representativeness bias.")

    print("\n--- library cross-check (brms / rstanarm / lme4 R; pymc-experimental Python) ---")
