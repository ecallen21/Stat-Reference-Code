"""First-hitting-time (FHT) survival model (Reference Sec 11.28).

Whitmore 1986, 1995; Aalen, Borgan & Gjessing 2008 'Survival and Event
History Analysis'. Models the failure time as the FIRST TIME a latent
health/damage process X(t) hits an absorbing boundary. For Brownian
motion X(t) with drift -mu (mu > 0 wear rate) starting at
X(0) = y0 > 0 that fails when X = 0, the inverse-Gaussian
distribution results:

    T ~ IG(y0 / mu, y0^2 / sigma^2)
    E[T]     = y0 / mu
    Var(T)   = y0 * sigma^2 / mu^3

Density:
    f(t) = y0 / sqrt(2 pi sigma^2 t^3) * exp(-(y0 - mu t)^2 / (2 sigma^2 t))

Attractive interpretation for medical / reliability data: latent
'wear' variable, drift = decline rate, boundary = failure event.
Allows CURE fraction: P(T = infinity) = 1 - exp(-2 y0 mu / sigma^2)
if drift can be negative (upward-drifting particles may never hit 0).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import invgauss
from scipy.optimize import minimize


def fit_ig_mle(t):
    """MLE for inverse-Gaussian (mu, lambda). Closed form."""
    mu_hat = t.mean()
    lam_hat = 1.0 / (np.mean(1 / t) - 1 / mu_hat)
    return {"mu": float(mu_hat), "lambda": float(lam_hat)}


def fit_fht_censored(t, e, seed=0):
    """MLE for right-censored IG using scipy invgauss."""
    def nll(params):
        mu, lam = np.exp(params)
        try:
            pdf = invgauss.pdf(t, mu / lam, scale=lam)     # scipy IG param
            cdf = invgauss.cdf(t, mu / lam, scale=lam)
        except Exception:
            return 1e12
        ll = np.sum(e * np.log(pdf + 1e-300) + (1 - e) * np.log(1 - cdf + 1e-300))
        return -ll
    r = minimize(nll, [np.log(t.mean()), np.log(t.var() * 5)], method="Nelder-Mead")
    mu, lam = np.exp(r.x)
    return {"mu": float(mu), "lambda": float(lam), "loglik": -r.fun}


if __name__ == "__main__":
    print("=== First-hitting-time model (inverse-Gaussian survival) ===\n")
    rng = np.random.default_rng(0)

    #  Simulate uncensored inverse-Gaussian
    mu_true = 3.0; lam_true = 10.0
    t = rng.wald(mean=mu_true, scale=lam_true, size=1000)
    r_unc = fit_ig_mle(t)
    print(f"  Uncensored MLE:  mu = {r_unc['mu']:.3f}  lambda = {r_unc['lambda']:.3f}   "
          f"(truth {mu_true}, {lam_true})")

    #  Censored
    c = rng.exponential(5.0, size=1000)
    e = (t <= c).astype(int)
    t_obs = np.minimum(t, c)
    r_cen = fit_fht_censored(t_obs, e)
    print(f"  Censored  ({(1 - e.mean()) * 100:.0f}% censoring): "
          f"mu = {r_cen['mu']:.3f}  lambda = {r_cen['lambda']:.3f}")

    print("\n  IG hazard shape: unimodal (rises then falls).")
    print("  Contrast: Weibull hazard is monotone; log-normal is unimodal but heavier-tailed.")

    print("\n--- library cross-check (SMPracticals R, threg R; scipy.stats.invgauss Python) ---")
