"""Cox regression with time-varying covariates (Reference Sec 11.10).

The Cox model with time-varying x(t):

    lambda(t | x) = lambda_0(t) * exp(beta * x(t))

Partial likelihood with time-varying covariates uses the
counting-process form: sum over event times t_j of

    beta * x_i(t_j) - log sum_{i in R(t_j)} exp(beta * x_i(t_j))

We implement a compact version with (start, stop, event, x) rows
and a single continuous time-varying covariate.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar


def cox_ptvc_ll(beta, start, stop, event, x):
    """Partial log-likelihood for a single time-varying covariate."""
    order = np.argsort(stop)
    stop_s = stop[order]; ev_s = event[order]; x_s = x[order]; start_s = start[order]
    ll = 0.0
    for j in range(len(stop_s)):
        if ev_s[j] == 0: continue
        t = stop_s[j]
        # At-risk set at t: rows where start < t <= stop
        risk = (start_s < t) & (stop_s >= t)
        eta_i = beta * x_s[j]
        log_sum = np.log(np.exp(beta * x_s[risk]).sum() + 1e-300)
        ll += eta_i - log_sum
    return ll


def fit_cox_tvc(start, stop, event, x):
    res = minimize_scalar(lambda b: -cox_ptvc_ll(b, start, stop, event, x),
                           bounds=(-5, 5), method="bounded")
    return {"beta_hat": float(res.x), "log_partial_lik": float(-res.fun)}


if __name__ == "__main__":
    print("=== Cox regression with a time-varying covariate ===\n")
    rng = np.random.default_rng(0)
    n = 500
    # Time-varying x: exposure switches at random time
    start_all = []; stop_all = []; ev_all = []; x_all = []
    beta_true = 0.7
    for i in range(n):
        # Baseline hazard 0.10; TV covariate switches at 3 (from 0 to 1)
        # For simplicity, generate event time from piecewise exponential
        u = rng.random()
        switch = rng.uniform(1, 5)
        # Before switch: x=0, hazard = 0.10; after: x=1, hazard = 0.10 * exp(beta)
        lam0 = 0.10; lam1 = lam0 * np.exp(beta_true)
        T_pre = -np.log(u) / lam0
        if T_pre <= switch:
            T = T_pre
        else:
            # Survived to switch; residual under lam1
            u2 = rng.random()
            T = switch - np.log(u2) / lam1
        C = 10.0
        obs = min(T, C); event = int(T <= C)
        # Two rows if switch inside observation
        if switch < obs:
            start_all += [0.0, switch]; stop_all += [switch, obs]
            ev_all += [0, event]; x_all += [0.0, 1.0]
        else:
            start_all.append(0.0); stop_all.append(obs)
            ev_all.append(event); x_all.append(0.0)
    start = np.array(start_all); stop = np.array(stop_all)
    event = np.array(ev_all); x = np.array(x_all)

    r = fit_cox_tvc(start, stop, event, x)
    print(f"  True beta = {beta_true}")
    print(f"  Estimated beta = {r['beta_hat']:+.3f}   (HR = {np.exp(r['beta_hat']):.3f})\n")

    print("--- library cross-check (R survival::coxph + Surv(start, stop, event); Python lifelines CoxTimeVaryingFitter) ---")
