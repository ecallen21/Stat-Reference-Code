"""Stochastic Volatility (Reference §13.40).

Latent-state alternative to GARCH: log-variance follows its own AR(1)
STOCHASTIC process rather than a deterministic function of past y and past
variance.

    y_t     = exp(h_t / 2) * eps_t          eps_t ~ N(0, 1)
    h_t     = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t   eta_t ~ N(0, 1)

Parameters: mu (mean log-var), phi in (-1, 1) (persistence), sigma_eta.

Contrast with GARCH(1,1): GARCH has deterministic variance recursion; SV
has stochastic variance -- allows random shocks to volatility independent
of return shocks.  Fits leverage effect and extreme-move clustering better
in equity data (Kim, Shephard & Chib 1998).

Estimation
    - Kalman-filter-based quasi-MLE on the LOG-SQUARED returns:
        log y_t^2 = h_t + log eps_t^2
      log eps_t^2 has known distribution (approx Normal with fixed mean/var).
    - Full MCMC (mixture-of-normals approximation, Kim-Shephard-Chib 1998).
    - Particle filter for the exact likelihood.

The demo below simulates SV and uses a simple bootstrap particle filter
to filter the latent volatility given (mu, phi, sigma_eta).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def simulate_sv(T: int, mu: float = -8.0, phi: float = 0.97, sigma_eta: float = 0.15,
                seed: int = 0):
    """Simulate a stochastic-volatility process."""
    rng = np.random.default_rng(seed)
    h = np.zeros(T); h[0] = mu
    for t in range(1, T):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * rng.normal()
    y = np.exp(h / 2) * rng.normal(size=T)
    return y, h


def particle_filter_sv(y, mu: float, phi: float, sigma_eta: float,
                       n_particles: int = 2000, seed: int = 0) -> dict:
    """Bootstrap particle filter for the latent log-volatility h_t.

    Returns filtered mean h_t | y_{1:t} and log marginal likelihood estimate.
    """
    y = np.asarray(y, dtype=float); T = len(y)
    rng = np.random.default_rng(seed)
    h = rng.normal(mu, sigma_eta / math.sqrt(max(1 - phi ** 2, 1e-6)), n_particles)
    filt_mean = np.empty(T); log_lik = 0.0
    for t in range(T):
        if t > 0:
            h = mu + phi * (h - mu) + sigma_eta * rng.normal(size=n_particles)
        # Weight by observation likelihood
        log_w = -0.5 * (h + y[t] ** 2 / np.exp(h) + math.log(2 * math.pi))
        log_w_max = log_w.max()
        w = np.exp(log_w - log_w_max)
        log_lik += log_w_max + math.log(w.mean())
        w /= w.sum()
        filt_mean[t] = float(np.sum(w * h))
        # Resample (systematic resampling would be nicer; multinomial is simplest)
        idx = rng.choice(n_particles, size=n_particles, p=w)
        h = h[idx]
    return {"filtered_h": filt_mean,
            "filtered_vol": np.exp(filt_mean / 2),
            "log_lik": float(log_lik),
            "n_particles": int(n_particles), "T": int(T),
            "method": "Bootstrap particle filter for SV latent state"}


def qmle_sv(y, seed: int = 0) -> dict:
    """Quasi-MLE via Kalman filter on log-squared returns (approx Normal noise)."""
    from scipy.optimize import minimize
    y = np.asarray(y, dtype=float); T = len(y)
    y2 = np.log(y ** 2 + 1e-10)
    # log(chi-square_1) has mean -1.27 and var pi^2/2 ~ 4.93
    c_const = -1.2704; obs_var = math.pi ** 2 / 2
    def neg_ll(theta):
        mu, phi, log_sigma_eta = theta
        if not (-0.999 < phi < 0.999): return 1e10
        sigma_eta = math.exp(log_sigma_eta)
        # Kalman filter for h_t
        h_pred = mu; P_pred = sigma_eta ** 2 / max(1 - phi ** 2, 1e-6)
        ll = 0.0
        for t in range(T):
            # Observation: y2_t = h_t + c + eps  with Var(eps) = obs_var
            innov = y2[t] - (h_pred + c_const)
            S = P_pred + obs_var
            ll += -0.5 * (math.log(2 * math.pi * S) + innov ** 2 / S)
            K = P_pred / S
            h_filt = h_pred + K * innov
            P_filt = (1 - K) * P_pred
            # Predict
            h_pred = mu + phi * (h_filt - mu)
            P_pred = phi ** 2 * P_filt + sigma_eta ** 2
        return -ll
    res = minimize(neg_ll, [np.log(np.var(y) + 1e-10), 0.9, math.log(0.2)], method="Nelder-Mead")
    mu, phi, log_sigma_eta = res.x
    return {"mu": float(mu), "phi": float(phi),
            "sigma_eta": float(math.exp(log_sigma_eta)),
            "log_likelihood": float(-res.fun),
            "method": "Quasi-MLE via Kalman filter on log(y^2)"}


if __name__ == "__main__":
    y, h_true = simulate_sv(500, mu=-8.0, phi=0.97, sigma_eta=0.15, seed=0)
    print(f"=== Simulated SV: T = {len(y)}, mu=-8.0, phi=0.97, sigma_eta=0.15 ===")

    print("\n=== Particle filter with TRUE parameters ===")
    r = particle_filter_sv(y, mu=-8.0, phi=0.97, sigma_eta=0.15, n_particles=2000, seed=1)
    print(f"  log-lik = {r['log_lik']:.2f}")
    print(f"  mean |filtered_h - true h| = {np.mean(np.abs(r['filtered_h'] - h_true)):.3f}")

    print("\n=== Quasi-MLE parameter estimation ===")
    q = qmle_sv(y)
    print(f"  mu_hat        = {q['mu']:.3f}   (true -8.0)")
    print(f"  phi_hat       = {q['phi']:.3f}  (true 0.97)")
    print(f"  sigma_eta_hat = {q['sigma_eta']:.3f}  (true 0.15)")
    print(f"  log-lik       = {q['log_likelihood']:.2f}")

    print("\n--- library cross-check (statsmodels UnobservedComponents or arch, not exact SV) ---")
    print("  Full-Bayesian SV: see rstan / cmdstanr with the built-in stochastic-volatility example,")
    print("  or PyMC's pymc.StudentT + AR(1) latent formulation.")
