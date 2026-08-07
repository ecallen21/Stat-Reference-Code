"""Gibbs sampler (Reference §14.7).

Special case of MH where the proposal is EXACTLY the full conditional of
each parameter, giving acceptance probability 1.  When each full conditional
is tractable (conjugate structure), Gibbs sweeps are cheap and mix well.

Prototype problem: Normal-Inverse-Gamma
    y_i | mu, sigma^2 ~ Normal(mu, sigma^2)
    mu | sigma^2      ~ Normal(mu_0, sigma^2 / kappa_0)
    sigma^2           ~ InvGamma(alpha_0, beta_0)

Full conditionals:
    mu | sigma^2, y  ~ Normal(mu_n, sigma^2 / kappa_n)
        mu_n = (kappa_0 mu_0 + n ybar) / (kappa_0 + n)
        kappa_n = kappa_0 + n
    sigma^2 | mu, y  ~ InvGamma(alpha_0 + n/2,  beta_0 + 0.5 sum (y_i - mu)^2)

Extended demo: two-level normal hierarchical model (Gelman "8 schools" flavor)
    y_j    ~ Normal(theta_j, sigma_j^2)         (sigma_j known)
    theta_j ~ Normal(mu, tau^2)
    mu     ~ Normal(0, 100^2)
    tau^2  ~ InvGamma(0.5, 0.5)

Full conditionals: theta_j given (mu, tau) is Normal; mu given (theta, tau) is
Normal; tau^2 given (theta, mu) is InvGamma.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def gibbs_normal_ig(y, mu_0: float = 0.0, kappa_0: float = 1.0,
                    alpha_0: float = 0.5, beta_0: float = 0.5,
                    n_iter: int = 5000, seed: int = 0) -> dict:
    """Gibbs sampler for Normal likelihood with Normal-InvGamma prior."""
    y = np.asarray(y, dtype=float); n = len(y); ybar = float(y.mean())
    rng = np.random.default_rng(seed)
    mu = ybar; sig2 = float(y.var(ddof=1))
    mus = np.empty(n_iter); sig2s = np.empty(n_iter)
    for t in range(n_iter):
        # mu | sig2, y
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * ybar) / kappa_n
        mu = rng.normal(mu_n, math.sqrt(sig2 / kappa_n))
        # sig2 | mu, y
        alpha_n = alpha_0 + n / 2
        beta_n = beta_0 + 0.5 * np.sum((y - mu) ** 2)
        sig2 = 1 / rng.gamma(alpha_n, 1 / beta_n)
        mus[t] = mu; sig2s[t] = sig2
    burn = n_iter // 5
    return {"mu_samples": mus[burn:], "sig2_samples": sig2s[burn:],
            "mu_mean": float(mus[burn:].mean()),
            "sig2_mean": float(sig2s[burn:].mean()),
            "mu_95": tuple(np.quantile(mus[burn:], [0.025, 0.975])),
            "sig2_95": tuple(np.quantile(sig2s[burn:], [0.025, 0.975])),
            "n": int(n), "n_iter": int(n_iter),
            "method": "Gibbs on Normal-InverseGamma"}


def gibbs_hierarchical_normal(y, sigma_known, mu_prior_var: float = 1e4,
                              tau2_prior_alpha: float = 0.5, tau2_prior_beta: float = 0.5,
                              n_iter: int = 5000, seed: int = 0) -> dict:
    """Two-level hierarchical Normal model (Gelman '8-schools' style).

    y[j]         : observed group mean for group j
    sigma_known  : known SE of y[j]
    """
    y = np.asarray(y, dtype=float); sigma = np.asarray(sigma_known, dtype=float)
    J = len(y)
    rng = np.random.default_rng(seed)
    mu = float(y.mean()); tau2 = float(y.var(ddof=1))
    theta = y.copy()
    thetas = np.empty((n_iter, J)); mus = np.empty(n_iter); tau2s = np.empty(n_iter)
    for t in range(n_iter):
        # theta_j | mu, tau, y : Normal
        v_j = 1 / (1 / sigma ** 2 + 1 / tau2)
        m_j = v_j * (y / sigma ** 2 + mu / tau2)
        theta = rng.normal(m_j, np.sqrt(v_j))
        # mu | theta, tau : Normal
        prec = J / tau2 + 1 / mu_prior_var
        m = (theta.sum() / tau2) / prec
        mu = rng.normal(m, math.sqrt(1 / prec))
        # tau^2 | theta, mu : InvGamma
        a = tau2_prior_alpha + J / 2
        b = tau2_prior_beta + 0.5 * np.sum((theta - mu) ** 2)
        tau2 = 1 / rng.gamma(a, 1 / b)
        thetas[t] = theta; mus[t] = mu; tau2s[t] = tau2
    burn = n_iter // 5
    theta_mean = thetas[burn:].mean(0)
    return {"theta_posterior_mean": theta_mean,
            "theta_95": np.quantile(thetas[burn:], [0.025, 0.975], axis=0).T,
            "mu_mean": float(mus[burn:].mean()),
            "tau_mean": float(np.sqrt(tau2s[burn:]).mean()),
            "shrinkage_factor": float(1 - np.var(theta_mean) / np.var(y)),
            "n_groups": int(J), "n_iter": int(n_iter),
            "method": "Gibbs on hierarchical Normal (partial pooling)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Gibbs on Normal-InverseGamma, n = 30, true mu=4, sigma^2=2 ===")
    y = rng.normal(4, math.sqrt(2), 30)
    r = gibbs_normal_ig(y, mu_0=0.0, kappa_0=0.01, alpha_0=0.5, beta_0=0.5, seed=0)
    print(f"  mu   posterior mean: {r['mu_mean']:.3f}  95% CI: ({r['mu_95'][0]:.3f}, {r['mu_95'][1]:.3f})")
    print(f"  sig2 posterior mean: {r['sig2_mean']:.3f}  95% CI: ({r['sig2_95'][0]:.3f}, {r['sig2_95'][1]:.3f})")

    print("\n=== Gibbs on 8-school hierarchical Normal (Rubin data) ===")
    y_schools = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sig_schools = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    r = gibbs_hierarchical_normal(y_schools, sig_schools, n_iter=8000, seed=0)
    print(f"  raw y:              {y_schools}")
    print(f"  posterior theta:    {np.round(r['theta_posterior_mean'], 2)}")
    print(f"  overall mu: {r['mu_mean']:.3f}  between-school tau: {r['tau_mean']:.3f}")
    print(f"  shrinkage factor: {r['shrinkage_factor']:.3f}")

    print("\n--- cross-check: conjugate posterior mean when kappa_0 -> 0 ---")
    print(f"  y.mean()  = {y.mean():.3f} (Gibbs mu_mean = {r['mu_mean']:.3f} on schools)")
