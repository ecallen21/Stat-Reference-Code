"""Count time series (Reference §13.32).

Discrete non-negative-integer time series require count-specific models;
Gaussian ARMA violates the non-negativity and integer support.

INAR(1) - Integer AR via binomial thinning (McKenzie 1985; Al-Osh & Alzaid 1987)
    y_t = alpha o y_{t-1} + eps_t
    where 'alpha o y' means binomial thinning: given y_{t-1}, draw
    Binomial(y_{t-1}, alpha) independently.  eps_t is an integer innovation
    (e.g. Poisson(lambda)).  Marginally y_t has a mixed Poisson-Binomial
    distribution; a Poisson INAR(1) with Poisson(lambda) innovations has
    Poisson(lambda / (1 - alpha)) stationary marginal.

Poisson-AR (aka INGARCH; Ferland-Latour-Oraichi 2006)
    y_t | past ~ Poisson(mu_t)
    mu_t = omega + alpha y_{t-1} + beta mu_{t-1}
    Deterministic conditional-mean recursion; parallels GARCH for counts.

Methods
    - Method-of-moments estimation of alpha via sample lag-1 ACF.
    - Conditional Poisson MLE for INGARCH.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def simulate_inar1(n: int, alpha: float, lam: float, seed: int = 0) -> np.ndarray:
    """Simulate a Poisson-innovation INAR(1) series."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int); y[0] = int(rng.poisson(lam / max(1 - alpha, 1e-6)))
    for t in range(1, n):
        y[t] = int(rng.binomial(y[t - 1], alpha) + rng.poisson(lam))
    return y


def inar1_mom(y) -> dict:
    """Method-of-moments estimator for Poisson-INAR(1).

    alpha_hat = sample lag-1 autocorrelation (as in real-valued AR(1)).
    lambda_hat = ybar (1 - alpha_hat)
    """
    y = np.asarray(y, dtype=float); n = len(y); ybar = y.mean()
    r1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    alpha_hat = max(min(r1, 0.999), 0.0)
    lam_hat = ybar * (1 - alpha_hat)
    return {"alpha": float(alpha_hat), "lambda": float(lam_hat),
            "mean_stationary": float(lam_hat / max(1 - alpha_hat, 1e-6)),
            "method": "Poisson-INAR(1) method of moments"}


def simulate_ingarch(n: int, omega: float, alpha: float, beta: float,
                     seed: int = 0) -> np.ndarray:
    """Simulate INGARCH(1,1): y_t | past ~ Poisson(mu_t), mu_t = omega + alpha y_{t-1} + beta mu_{t-1}."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int); mu = np.full(n, omega / max(1 - alpha - beta, 1e-6))
    y[0] = int(rng.poisson(mu[0]))
    for t in range(1, n):
        mu[t] = omega + alpha * y[t - 1] + beta * mu[t - 1]
        y[t] = int(rng.poisson(mu[t]))
    return y


def fit_ingarch(y) -> dict:
    """Conditional Poisson MLE for INGARCH(1, 1)."""
    from scipy.optimize import minimize
    y = np.asarray(y, dtype=float); n = len(y)
    def neg_ll(theta):
        log_omega, logit_a, logit_b = theta
        omega = math.exp(log_omega)
        a = 1 / (1 + math.exp(-logit_a)) * 0.95
        b = 1 / (1 + math.exp(-logit_b)) * (0.95 - a)
        if b < 0 or omega <= 0: return 1e10
        mu = np.zeros(n); mu[0] = y.mean()
        for t in range(1, n):
            mu[t] = omega + a * y[t - 1] + b * mu[t - 1]
            if mu[t] <= 0: return 1e10
        return -np.sum(y * np.log(mu) - mu)  # (drop constant log(y!))
    res = minimize(neg_ll, [math.log(0.5 * y.mean() + 0.1), 0.0, 0.0], method="Nelder-Mead")
    log_omega, logit_a, logit_b = res.x
    omega = math.exp(log_omega)
    a = 1 / (1 + math.exp(-logit_a)) * 0.95
    b = 1 / (1 + math.exp(-logit_b)) * (0.95 - a)
    return {"omega": float(omega), "alpha": float(a), "beta": float(b),
            "log_likelihood": float(-res.fun),
            "method": "Poisson INGARCH(1,1) conditional MLE"}


if __name__ == "__main__":
    print("=== INAR(1) simulation & MoM estimation ===")
    y = simulate_inar1(1000, alpha=0.6, lam=2.5, seed=0)
    r = inar1_mom(y)
    print(f"  alpha_hat  = {r['alpha']:.3f}  (true 0.6)")
    print(f"  lambda_hat = {r['lambda']:.3f}  (true 2.5)")
    print(f"  stationary mean estimate = {r['mean_stationary']:.3f}  (true = 6.25)")

    print("\n=== INGARCH(1,1) simulation & conditional Poisson MLE ===")
    y = simulate_ingarch(1000, omega=1.0, alpha=0.4, beta=0.3, seed=0)
    r = fit_ingarch(y)
    print(f"  omega = {r['omega']:.3f}  (true 1.0)")
    print(f"  alpha = {r['alpha']:.3f}  (true 0.4)")
    print(f"  beta  = {r['beta']:.3f}  (true 0.3)")
    print(f"  log-lik = {r['log_likelihood']:.3f}")

    print("\n--- library cross-check (tscount, R only) ---")
    print("  R's `tscount::tsglm(family='poisson', model=list(past_obs=1, past_mean=1))`")
    print("  is the reference INGARCH implementation.")
