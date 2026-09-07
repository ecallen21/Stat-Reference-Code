"""Network meta-analysis (Reference Sec 22.15).

Lumley 2002.  Simultaneous meta-analysis of multiple treatments
using both DIRECT (head-to-head) and INDIRECT (via common
comparators) evidence.

Compact demo: 3 treatments (A, B, C); studies compare pairs.
Contrast-based (Bucher for closed triangles) vs arm-based
(GLM with study random effect + treatment fixed effects) approach.
Here we implement a random-effects contrast-based approach solving:
    for each study i comparing T_i1 vs T_i2:
      y_i = mu[T_i1] - mu[T_i2] + u_i + eps_i
      u_i ~ N(0, tau^2)   eps_i ~ N(0, sigma_i^2)

Reference treatment fixed to 0.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize


def nma(y, se, arms1, arms2, K):
    """Random-effects contrast-based NMA.

    y, se     : study-level effect + SE
    arms1/2   : treatment indices (0 = reference)
    K         : number of treatments (arm 0 fixed to mu=0)
    """
    K_free = K - 1                                # exclude reference
    def nll(params):
        mu = np.concatenate([[0.0], params[:K_free]])
        tau2 = np.exp(2 * params[K_free])
        var = se ** 2 + tau2
        pred = mu[arms1] - mu[arms2]
        ll = -0.5 * (np.log(2 * np.pi * var) + (y - pred) ** 2 / var).sum()
        return -ll
    x0 = np.zeros(K_free + 1)
    r = minimize(nll, x0=x0, method="Nelder-Mead")
    mu = np.concatenate([[0.0], r.x[:K_free]])
    tau = float(np.exp(r.x[K_free]))
    return {"mu": mu, "tau": tau, "loglik": float(-r.fun)}


if __name__ == "__main__":
    print("=== Network meta-analysis (contrast-based, random-effects) ===\n")
    rng = np.random.default_rng(0)
    K = 3                # A=0, B=1, C=2
    true_mu = np.array([0.0, 0.5, -0.3])
    tau = 0.10
    n_studies_per_pair = 6
    y, se, arms1, arms2 = [], [], [], []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        for _ in range(n_studies_per_pair):
            se_i = rng.uniform(0.08, 0.20)
            eff = true_mu[a] - true_mu[b] + rng.normal(0, tau) + rng.normal(0, se_i)
            y.append(eff); se.append(se_i); arms1.append(a); arms2.append(b)
    y = np.array(y); se = np.array(se); arms1 = np.array(arms1); arms2 = np.array(arms2)

    r = nma(y, se, arms1, arms2, K)
    print(f"  True mu = {true_mu}")
    print(f"  NMA estimated mu (ref A=0) = {r['mu'].round(3)}")
    print(f"  Estimated tau = {r['tau']:.3f}")

    # Rank treatments by mu (higher = better)
    order = np.argsort(-r["mu"])
    print(f"\n  Treatment ranking (by mu, higher better): {order.tolist()}\n")
    print("--- library cross-check (R netmeta::netmeta, gemtc; Python custom + BUGS/JAGS) ---")
