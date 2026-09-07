"""Poisson-gamma empirical Bayes (Reference Sec 24.19).

Clayton & Kaldor 1987 'Empirical Bayes estimates of age-standardized
relative risks for use in disease mapping', Biometrics. Classic
'shrinkage' estimator for small-area disease rates.

Model:
    Y_i | theta_i ~ Poisson(E_i * theta_i)         (observed events)
    theta_i       ~ Gamma(alpha, alpha / mu)       (unknown site risks)

Marginally Y_i ~ Negative Binomial. Posterior for theta_i:
    theta_i | Y_i ~ Gamma(alpha + Y_i, alpha / mu + E_i)
    E[theta_i | Y_i] = (alpha + Y_i) / (alpha / mu + E_i)

Empirical Bayes: estimate (mu, alpha) from the data (method of moments
or MLE) then plug in to get shrunken SMRs = theta_hat_i.

Effect: raw SMR = Y_i / E_i has huge variance for small E_i; the
Poisson-gamma shrunken estimate pulls it toward the overall mean mu
proportionally to how weak the evidence is.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar    # 1-D MLE
from scipy.special import gammaln


def fit_neg_binom_mom(Y, E):
    """Method-of-moments for the mean mu and shape alpha (NB2 parameterisation)."""
    r = Y / E
    mu = np.mean(r * E) / np.mean(E)
    v = np.mean((r - mu) ** 2)
    #  Var(r) approx mu / E + mu^2 / alpha  ->  alpha = mu^2 / (Var(r) - mean(mu/E))
    residual = v - np.mean(mu / E)
    if residual <= 0:
        alpha = 1e6
    else:
        alpha = mu ** 2 / residual
    return mu, alpha


def fit_neg_binom_mle(Y, E, alpha_bounds=(1e-2, 1e5)):
    """MLE of (alpha) with mu = sum(Y) / sum(E) (weighted MLE)."""
    mu = np.sum(Y) / np.sum(E)

    def nll(alpha):
        r = alpha * mu / E
        p = alpha / (alpha + r * E)      # p = alpha / (alpha + mu * E)
        p = alpha / (alpha + mu * E)     # cleaner
        ll = (gammaln(Y + alpha) - gammaln(alpha) - gammaln(Y + 1)
              + alpha * np.log(p) + Y * np.log(1 - p + 1e-300))
        return -ll.sum()

    r = minimize_scalar(nll, bounds=alpha_bounds, method="bounded")
    return mu, r.x


def shrunken_smr(Y, E, mu, alpha):
    """Posterior mean E[theta | Y] under Poisson-gamma."""
    return (alpha + Y) / (alpha / mu + E)


if __name__ == "__main__":
    print("=== Poisson-gamma empirical Bayes -- shrunken SMRs ===\n")
    rng = np.random.default_rng(0)
    n_areas = 200

    #  Expected cases E_i vary widely across small / large areas
    E = rng.gamma(shape=2.0, scale=5.0, size=n_areas)   # mean ~ 10, tail small
    #  True area-specific risks theta_i ~ Gamma(mean=1, alpha=5)
    alpha_true = 5.0
    mu_true = 1.0
    theta = rng.gamma(shape=alpha_true, scale=mu_true / alpha_true, size=n_areas)
    #  Observed events
    Y = rng.poisson(lam=E * theta)

    #  Raw SMR
    raw = Y / E

    #  Fit and shrink
    mu_hat, alpha_hat = fit_neg_binom_mle(Y, E)
    smr_shrunk = shrunken_smr(Y, E, mu_hat, alpha_hat)

    print(f"  True (mu, alpha)          = ({mu_true:.2f}, {alpha_true:.2f})")
    print(f"  Estimated (mu, alpha)     = ({mu_hat:.3f}, {alpha_hat:.2f})")

    #  Mean absolute error against true theta
    mae_raw = float(np.mean(np.abs(raw - theta)))
    mae_shrunk = float(np.mean(np.abs(smr_shrunk - theta)))
    print(f"\n  MAE (raw SMR vs truth)    = {mae_raw:.3f}")
    print(f"  MAE (shrunken vs truth)   = {mae_shrunk:.3f}   "
          f"({(1 - mae_shrunk / mae_raw) * 100:.0f}% reduction)")

    #  Where does shrinkage hit hardest? Small-E areas.
    small = E < np.quantile(E, 0.25)
    print(f"\n  Small-area subset (bottom 25 % of E):")
    print(f"    raw MAE      = {np.mean(np.abs(raw[small] - theta[small])):.3f}")
    print(f"    shrunken MAE = {np.mean(np.abs(smr_shrunk[small] - theta[small])):.3f}")

    print("\n--- library cross-check (SpatialEpi R, INLA R, DCluster R; pymc Python) ---")
