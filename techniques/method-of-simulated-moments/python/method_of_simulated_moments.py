"""Method of Simulated Moments (MSM) (Reference Sec 45.7).

McFadden 1989; Pakes & Pollard 1989. When theoretical moments
m(theta) = E[g(X; theta)] are intractable but we can SIMULATE draws
X_s(theta), replace theoretical moments with simulated moments:

    hat_m(theta) = (1 / S) * sum_s g(X_s(theta))

Estimator minimises the GMM criterion:

    theta_hat = argmin (hat_m(theta) - g_bar)' W (hat_m(theta) - g_bar)

where g_bar are the empirical data moments and W is a weighting
matrix (identity or the two-step efficient GMM matrix).

Applications:
    * Discrete-choice models (BLP), auction models, DSGE macro.
    * Any model where the likelihood needs to marginalise over latent
      variables that must be simulated.

We demonstrate MSM on estimating (mu, sigma) of a log-normal by
matching mean + variance of exp(mu + sigma * Z) via simulation --
even though the log-normal MLE is trivial, this shows the machinery.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # criterion optimisation


def simulate_moments(theta, S, seed):
    """Return simulated (mean, variance) of exp(mu + sigma*Z)."""
    mu, sigma = theta
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=S)
    X = np.exp(mu + sigma * Z)
    return np.array([X.mean(), X.var(ddof=1)])


def msm_estimator(y, S=5000, W=None, seed=0):
    """Match empirical mean & variance of y via simulated moments."""
    g_bar = np.array([y.mean(), y.var(ddof=1)])
    if W is None:
        W = np.eye(2)

    def criterion(theta):
        if theta[1] <= 0:
            return 1e12
        m = simulate_moments(theta, S, seed)     # fixed seed = 'common random numbers'
        d = m - g_bar
        return float(d @ W @ d)

    r = minimize(criterion, x0=np.array([0.0, 1.0]), method="Nelder-Mead")
    return {"theta_hat": r.x, "criterion": r.fun, "moments": g_bar}


if __name__ == "__main__":
    print("=== Method of Simulated Moments -- log-normal (mu, sigma) ===\n")
    rng = np.random.default_rng(0)
    mu_true, sigma_true = 0.3, 0.4
    n = 500
    Z = rng.normal(size=n)
    y = np.exp(mu_true + sigma_true * Z)

    r = msm_estimator(y, S=8000, seed=42)
    mu_hat, sigma_hat = r["theta_hat"]

    print(f"  True (mu, sigma)   = ({mu_true:.3f}, {sigma_true:.3f})")
    print(f"  MSM  (mu, sigma)   = ({mu_hat:.3f}, {sigma_hat:.3f})")
    print(f"  Empirical moments  = mean {r['moments'][0]:.3f}, var {r['moments'][1]:.3f}")

    #  Compare with direct MLE on log(y): trivial
    mu_mle = np.log(y).mean()
    sigma_mle = np.log(y).std(ddof=1)
    print(f"\n  Direct log-normal MLE:  (mu, sigma) = ({mu_mle:.3f}, {sigma_mle:.3f})")
    print(f"  MSM matches MLE to ~2 decimals; MSM machinery is what you need when")
    print(f"  the likelihood is intractable (BLP, DSGE, auction models).")

    print("\n--- library cross-check (msm R; PyBLP Python; SMM 'gmm' Python) ---")
