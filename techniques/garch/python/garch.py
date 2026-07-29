"""GARCH(1, 1) via MLE (Reference §13.11; §13.33 multivariate note).

Financial return series show VOLATILITY CLUSTERING: large moves cluster in
time. GARCH models the conditional variance sigma_t^2 as a recursion:

    r_t = mu + eps_t         eps_t = sigma_t z_t,  z_t ~ N(0, 1)
    sigma_t^2  =  omega  +  alpha eps_{t-1}^2  +  beta sigma_{t-1}^2

    Constraints for a valid model: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        (last one ensures a stationary conditional-variance recursion).

Fitted by maximum likelihood on the normal-density log-likelihood of r_t:
    log L = -0.5 sum [log(2 pi) + log(sigma_t^2) + eps_t^2 / sigma_t^2]

MULTIVARIATE GARCH (§13.33):
    DCC (Dynamic Conditional Correlation, Engle 2002), BEKK (Engle-Kroner 1995),
    CCC (Constant Conditional Correlation, Bollerslev 1990). All handle
    time-varying volatility across multiple assets. The `arch` Python package
    (also mgarch) supports these; not implemented from scratch here.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize    # constrained MLE


def _garch11_neg_ll(params, r):
    """Negative log-likelihood for GARCH(1,1) with normal innovations.

    params: (log_omega, logit_alpha, logit_beta_sum_share, mu)
    We reparameterize alpha + beta to be < 1 via a share of (1 - epsilon).
    """
    log_omega, l_alpha, l_beta, mu = params
    omega = math.exp(log_omega)
    alpha = 1.0 / (1.0 + math.exp(-l_alpha)) * 0.9    # (0, 0.9)
    beta = 1.0 / (1.0 + math.exp(-l_beta)) * (0.999 - alpha)
    n = len(r)
    eps = r - mu
    sigma2 = np.empty(n)
    sigma2[0] = float(np.var(eps))                    # unconditional variance
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.clip(sigma2, 1e-12, None)
    ll = -0.5 * np.sum(math.log(2 * math.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return -float(ll)


def fit_garch11(r) -> dict:
    """Fit GARCH(1,1) with normal innovations by MLE."""
    r = np.asarray(r, dtype=float)
    theta0 = [math.log(np.var(r) * 0.1), 0.0, 1.0, float(r.mean())]
    res = optimize.minimize(_garch11_neg_ll, theta0, args=(r,),
                             method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
    log_omega, l_alpha, l_beta, mu = res.x
    omega = math.exp(log_omega)
    alpha = 1.0 / (1.0 + math.exp(-l_alpha)) * 0.9
    beta = 1.0 / (1.0 + math.exp(-l_beta)) * (0.999 - alpha)
    # unconditional variance = omega / (1 - alpha - beta)  (if stationary)
    persist = alpha + beta
    uncond_var = omega / (1 - persist) if persist < 1 else float("inf")

    # Compute conditional sigma^2 series
    n = len(r); eps = r - mu
    sigma2 = np.empty(n); sigma2[0] = float(np.var(eps))
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

    return {"mu": float(mu),
            "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
            "persistence_alpha_plus_beta": float(persist),
            "unconditional_variance": float(uncond_var),
            "log_lik": float(-res.fun),
            "conditional_sigma2_head": sigma2[:5].tolist(),
            "conditional_sigma2_tail": sigma2[-5:].tolist(),
            "n": int(n),
            "method": "GARCH(1,1) with normal innovations via MLE"}


def library_versions(r):
    try:
        from arch import arch_model
        am = arch_model(r, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
        res = am.fit(disp="off")
        return {"arch package omega": float(res.params["omega"]),
                "arch package alpha[1]": float(res.params["alpha[1]"]),
                "arch package beta[1]": float(res.params["beta[1]"])}
    except Exception as ex:
        return {"arch (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(37)
    n = 1000
    # Simulate GARCH(1,1) with omega=0.05, alpha=0.1, beta=0.85
    omega_t, alpha_t, beta_t = 0.05, 0.1, 0.85
    r = np.zeros(n); sigma2 = np.zeros(n)
    sigma2[0] = omega_t / (1 - alpha_t - beta_t)
    r[0] = math.sqrt(sigma2[0]) * rng.normal()
    for t in range(1, n):
        sigma2[t] = omega_t + alpha_t * r[t - 1] ** 2 + beta_t * sigma2[t - 1]
        r[t] = math.sqrt(sigma2[t]) * rng.normal()

    print("=== GARCH(1,1) fit (true omega=0.05, alpha=0.1, beta=0.85) ===")
    fit = fit_garch11(r)
    print(f"  mu     = {fit['mu']:+.4f}")
    print(f"  omega  = {fit['omega']:.4f}")
    print(f"  alpha  = {fit['alpha']:.4f}")
    print(f"  beta   = {fit['beta']:.4f}")
    print(f"  persistence (alpha + beta) = {fit['persistence_alpha_plus_beta']:.4f}")
    print(f"  unconditional variance = {fit['unconditional_variance']:.4f}   (true = {omega_t/(1-alpha_t-beta_t):.4f})")
    print(f"  log-lik = {fit['log_lik']:.2f}")

    print("\n--- library (arch) ---")
    for k, v in library_versions(r).items():
        print(f"  {k}: {v}")
