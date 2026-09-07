"""Accelerated Failure Time (AFT) models (Reference Sec 11.25).

Kalbfleisch & Prentice 2002. Parametric survival regression on the
LOG scale of failure time rather than on the hazard:

    log T = X @ beta + sigma * W

with W a standard error distribution. Common choices:

    W ~ Extreme Value min   ->  T ~ Weibull   (AFT + PH)
    W ~ Logistic            ->  T ~ Log-logistic
    W ~ Normal              ->  T ~ Log-normal
    W ~ EV_min (sigma=1)    ->  T ~ Exponential (special case)

Interpretation: exp(beta_j) is the ACCELERATION FACTOR -- a unit
change in X_j multiplies survival time by exp(beta_j). Contrast with
Cox (multiplies hazard by exp(beta_j)).

Right-censored log-likelihood (Weibull AFT):

    log L = sum_i { e_i * [-log(sigma * T_i) + z_i - exp(z_i)]
                    + (1 - e_i) * [-exp(z_i)] }

with z_i = (log T_i - X_i @ beta) / sigma.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # MLE


def weibull_aft_loglik(params, X, log_t, e):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    z = (log_t - X @ beta) / sigma
    log_S = -np.exp(z)                      # log survival
    log_f = -log_sigma - log_t + z + log_S  # log density
    log_h = log_f - log_S                   # log hazard
    return -np.sum(e * log_f + (1 - e) * log_S)


def lognormal_aft_loglik(params, X, log_t, e):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    z = (log_t - X @ beta) / sigma
    from scipy.stats import norm
    log_pdf = norm.logpdf(z) - log_sigma - log_t
    log_S = norm.logsf(z)
    return -np.sum(e * log_pdf + (1 - e) * log_S)


def fit_aft(t, e, X, dist="weibull"):
    """Fit AFT model by MLE. X should already include intercept."""
    log_t = np.log(t)
    p = X.shape[1]
    theta0 = np.r_[np.zeros(p), 0.0]

    if dist == "weibull":
        f = weibull_aft_loglik
    elif dist == "lognormal":
        f = lognormal_aft_loglik
    else:
        raise ValueError(dist)

    r = minimize(f, theta0, args=(X, log_t, e), method="L-BFGS-B")
    beta = r.x[:-1]
    sigma = np.exp(r.x[-1])
    return {"beta": beta, "sigma": sigma,
            "accel_factors": np.exp(beta),
            "loglik": -r.fun, "dist": dist}


if __name__ == "__main__":
    print("=== Accelerated failure time (AFT) models ===\n")
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, size=n)

    #  Simulate Weibull AFT: log T = 2.0 + 0.5*x1 - 0.8*x2 + 0.4*W
    #  (W ~ Gumbel_min, sigma=0.4). exp(0.5)=1.65 accel factor per SD of x1.
    sigma_true = 0.4
    #  Gumbel min: W = log(-log(U))
    U = rng.uniform(size=n)
    W = np.log(-np.log(U))
    log_t = 2.0 + 0.5 * x1 - 0.8 * x2 + sigma_true * W
    t_full = np.exp(log_t)
    #  Random right-censoring
    c = rng.exponential(20.0, size=n)
    e = (t_full <= c).astype(int)
    t = np.minimum(t_full, c)

    X = np.c_[np.ones(n), x1, x2]
    r_w = fit_aft(t, e, X, "weibull")
    r_ln = fit_aft(t, e, X, "lognormal")

    print(f"  Censoring rate = {(1 - e.mean()) * 100:.1f}%\n")
    print("  Weibull AFT fit:")
    print(f"    beta  = intercept {r_w['beta'][0]:+.3f}  "
          f"x1 {r_w['beta'][1]:+.3f}  x2 {r_w['beta'][2]:+.3f}")
    print(f"    sigma = {r_w['sigma']:.3f}   (truth 0.400)")
    print(f"    accel factors  x1: {r_w['accel_factors'][1]:.3f}  "
          f"x2: {r_w['accel_factors'][2]:.3f}")
    print(f"    Interpretation: 1 SD higher x1 -> survival multiplied by "
          f"{r_w['accel_factors'][1]:.2f};")
    print(f"                     x2=1 -> survival multiplied by "
          f"{r_w['accel_factors'][2]:.2f}")
    print(f"    log-lik = {r_w['loglik']:.2f}")
    print()
    print("  Log-normal AFT (for comparison):")
    print(f"    beta  = {r_ln['beta'].round(3).tolist()}")
    print(f"    sigma = {r_ln['sigma']:.3f}   log-lik = {r_ln['loglik']:.2f}")
    print(f"    Choose model by log-lik / AIC.")

    print("\n--- library cross-check (survreg R; lifelines Python) ---")
