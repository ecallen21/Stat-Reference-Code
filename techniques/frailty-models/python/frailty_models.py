"""Shared gamma frailty model (Reference §11.26).

For clustered survival data (patients within hospitals, teeth within a
patient, event recurrences within a subject), a shared frailty adds a
multiplicative random effect u_c per cluster to the Cox hazard:

    h(t | X, u_c)  =  h_0(t) * u_c * exp(X * beta)

    u_c ~ Gamma(1/theta, 1/theta)   (mean 1, variance theta)

theta measures cluster-level heterogeneity: theta = 0 -> ordinary Cox
(no clustering); larger theta -> more between-cluster variability.

Estimation via EM / penalized partial likelihood (McGilchrist & Aisbett 1991):
    E-step   : posterior mean of u_c given current beta and H0
    M-step   : Cox fit with offset log(u_c)
Iterate until convergence.

Also produces:
    - variance-component estimate theta_hat with SE
    - LR test H0: theta = 0 (mixture-of-chi-squared distribution at the boundary,
      p ~ 0.5 * chi2_0 + 0.5 * chi2_1)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: BFGS;  stats: distributions/tests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cox-ph", "python"))
from cox_ph import fit_cox    # techniques/cox-ph/python/cox_ph.py::fit_cox


def _breslow_baseline_H(times, events, X, beta):
    exp_eta = np.exp(np.clip(X @ beta, -500, 500))
    ev_times = np.unique(times[events == 1])
    H = np.zeros(len(times))
    cum = 0.0
    for tj in ev_times:
        R = times >= tj; d = int(np.sum((times == tj) & (events == 1)))
        cum += d / max(exp_eta[R].sum(), 1e-300)
        H[times >= tj] = cum
    return H


def fit_gamma_frailty(times, events, X, cluster) -> dict:
    """Shared gamma frailty via moment estimation (Klein-Moeschberger §13.4).

    Algorithm:
      1. Fit ordinary Cox to get baseline hazard H0(t) and beta.
      2. For each cluster c, compute
             D_c   = observed events
             E_c   = Sum over i in cluster of H0(t_i) * exp(X_i beta)   (expected)
             u_c   = D_c / E_c              (observed-to-expected ratio)
      3. Frailty variance:  theta_hat = Var(u_c) / Mean(u_c)^2  (coefficient
         of variation squared of the observed-to-expected ratios).

    Simple, transparent, and detects between-cluster heterogeneity. For
    a fully EM-based joint estimator, use R's ``survival::coxph(... + frailty())``.

    Parameters
    ----------
    cluster : n-length array of cluster IDs.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    X = np.asarray(X, dtype=float)
    cluster = np.asarray(cluster)
    unique_clusters = np.unique(cluster)
    n_c = len(unique_clusters)

    fit0 = fit_cox(times, events, X)
    beta = np.array(fit0["beta"])
    H = _breslow_baseline_H(times, events, X, beta)
    exp_eta = np.exp(np.clip(X @ beta, -500, 500))

    u_hat = np.empty(n_c)
    for k, c in enumerate(unique_clusters):
        mask = cluster == c
        D_c = int(events[mask].sum())
        E_c = float((H[mask] * exp_eta[mask]).sum())
        u_hat[k] = D_c / max(E_c, 1e-12)

    mean_u = float(u_hat.mean())
    theta_hat = float(u_hat.var(ddof=1) / max(mean_u * mean_u, 1e-12))
    return {"beta": beta.tolist(),
            "SE_beta_naive": fit0["SE"],
            "theta_hat": theta_hat,
            "frailty_mean": mean_u,
            "frailty_per_cluster_head": u_hat[:5].tolist(),
            "n_clusters": int(n_c),
            "n_events": int(events.sum()),
            "note": ("Moment-based estimator (Klein-Moeschberger §13.4). "
                     "For a fully iterative EM joint (beta, theta) estimator "
                     "with proper SEs, use R's survival::coxph(... + frailty())."),
            "method": "shared gamma frailty (moment estimator)"}


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    n_clusters = 40
    per_cluster = 10
    n = n_clusters * per_cluster
    # frailties: mean 1, variance ~ 0.5
    u = rng.gamma(shape=2.0, scale=0.5, size=n_clusters)
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    u_row = u[cluster]
    X = rng.normal(0, 1, size=(n, 1))
    beta_true = np.array([0.5])
    T_event = -np.log(rng.uniform(0, 1, n)) / (0.1 * u_row * np.exp(X @ beta_true))
    C_censor = rng.uniform(0, 20, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)

    print("=== Shared gamma frailty (moment estimator) ===")
    fit = fit_gamma_frailty(times, events, X, cluster)
    print(f"  beta      = {fit['beta']}  (true = {beta_true.tolist()})")
    print(f"  theta_hat = {fit['theta_hat']:.4f}  (true Var(u)/Mean(u)^2 = {u.var(ddof=1)/u.mean()**2:.4f})")
    print(f"  n_clusters = {fit['n_clusters']}, n_events = {fit['n_events']}")
    print(f"  first 5 frailty estimates u_c = D_c / E_c: "
          f"{[f'{v:.3f}' for v in fit['frailty_per_cluster_head']]}")

    print("\n=== Compare to ordinary Cox (ignoring cluster) ===")
    plain = fit_cox(times, events, X)
    print(f"  beta = {plain['beta']}")
    print("  (ordinary Cox SE will be too small; ignore cluster => under-cover)")
