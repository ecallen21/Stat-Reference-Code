"""Metropolis-Hastings MCMC (Reference §14.6).

Generic-purpose MCMC sampler for arbitrary log-density log p(theta).  A
Markov chain whose stationary distribution is the target posterior.

Random-walk Metropolis
    Propose theta' = theta + eps    (eps ~ N(0, Sigma_prop))
    Accept with probability min(1, p(theta') / p(theta)).

Adaptive proposal (Haario et al. 2001)
    Every K iterations, update Sigma_prop to be a scaled sample covariance
    of the chain so far.  Efficient scaling factor is (2.38)^2 / d for a
    d-dimensional target (Roberts, Gelman, Gilks 1997).

Diagnostics
    - Trace plots (visual).
    - Acceptance rate (target ~0.234 for high-d, ~0.44 for 1-d).
    - Autocorrelation and effective sample size (ESS = N / (1 + 2 sum rho_k)).
    - Gelman-Rubin R-hat across multiple chains: near 1.0 = converged.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def metropolis_hastings(log_target, theta0, n_iter: int = 5000, prop_sd=1.0,
                        adapt: bool = True, seed: int = 0) -> dict:
    """Random-walk Metropolis sampler with optional Haario adaptation.

    log_target : callable, theta -> log p(theta) (unnormalized).
    theta0     : starting value (scalar or array).
    prop_sd    : initial proposal std (scalar or diagonal vector).
    """
    rng = np.random.default_rng(seed)
    theta = np.atleast_1d(np.asarray(theta0, dtype=float))
    d = theta.shape[0]
    Sigma = np.eye(d) * (np.asarray(prop_sd, dtype=float) ** 2 if np.ndim(prop_sd) else prop_sd ** 2)
    samples = np.empty((n_iter, d))
    n_accept = 0
    log_p = log_target(theta)
    for t in range(n_iter):
        prop = theta + rng.multivariate_normal(np.zeros(d), Sigma)
        log_p_prop = log_target(prop)
        if math.log(rng.uniform()) < log_p_prop - log_p:
            theta, log_p = prop, log_p_prop
            n_accept += 1
        samples[t] = theta
        if adapt and t >= 200 and t % 50 == 0:
            emp_cov = np.cov(samples[max(0, t - 500):t + 1].T)
            if emp_cov.ndim == 0: emp_cov = emp_cov.reshape(1, 1)
            Sigma = (2.38 ** 2 / d) * (emp_cov + 1e-6 * np.eye(d))
    accept_rate = n_accept / n_iter
    return {"samples": samples,
            "acceptance_rate": float(accept_rate),
            "final_proposal_cov": Sigma,
            "n_iter": int(n_iter), "dim": int(d),
            "method": "Random-walk Metropolis" + (" with Haario adaptation" if adapt else "")}


def effective_sample_size(x) -> float:
    """ESS via truncated autocorrelation sum (Geyer's initial monotone)."""
    x = np.asarray(x, dtype=float)
    n = len(x); x = x - x.mean()
    if x.std() == 0: return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    acf = acf[:min(n, 500)]
    # Sum until first negative pair (Geyer)
    s = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair < 0: break
        s += pair
    return float(n / (1 + 2 * s))


def gelman_rubin(chains) -> float:
    """R-hat for a list of 1-D chains of equal length."""
    chains = [np.asarray(c) for c in chains]
    m = len(chains); n = len(chains[0])
    means = np.array([c.mean() for c in chains])
    vars_ = np.array([c.var(ddof=1) for c in chains])
    W = vars_.mean()
    B = n * means.var(ddof=1)
    var_hat = (1 - 1 / n) * W + B / n
    return float(math.sqrt(var_hat / W)) if W > 0 else float("nan")


if __name__ == "__main__":
    # Target: N(3, 1.5^2)
    def log_target(theta):
        return -0.5 * ((theta[0] - 3) / 1.5) ** 2

    print("=== MH on N(3, 1.5^2) target, 5000 iter, adaptive ===")
    r = metropolis_hastings(log_target, theta0=[0.0], n_iter=5000, prop_sd=0.5, seed=0)
    burn = 500
    x = r["samples"][burn:, 0]
    print(f"  acceptance rate: {r['acceptance_rate']:.3f}")
    print(f"  sample mean: {x.mean():.3f}  (true = 3)")
    print(f"  sample sd:   {x.std():.3f}   (true = 1.5)")
    print(f"  ESS: {effective_sample_size(x):.0f} / {len(x)} samples")

    # Multi-chain R-hat
    chains = [metropolis_hastings(log_target, theta0=[start], n_iter=3000, seed=i)["samples"][500:, 0]
              for i, start in enumerate([-5, 0, 5])]
    print(f"  R-hat (3 chains): {gelman_rubin(chains):.4f} (should be near 1)")

    print("\n=== MH on 2-D correlated Gaussian ===")
    Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    def log_target2(theta):
        return -0.5 * theta @ Sigma_inv @ theta

    r = metropolis_hastings(log_target2, theta0=[0.0, 0.0], n_iter=8000, prop_sd=0.5, seed=1)
    x = r["samples"][1000:]
    print(f"  acceptance rate: {r['acceptance_rate']:.3f}")
    print(f"  empirical cov:\n{np.cov(x.T).round(3)}")
    print(f"  true cov:\n{Sigma}")
