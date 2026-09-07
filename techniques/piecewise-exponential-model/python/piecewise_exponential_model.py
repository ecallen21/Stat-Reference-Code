"""Piecewise exponential model (Reference Sec 11.26).

Friedman 1982 'Piecewise exponential models for survival data with
covariates', Ann Stat; Holford 1980. Approximates any continuous
hazard by a step function: partition follow-up time into K intervals
(a_0 = 0, a_1, ..., a_K = inf) with constant hazard within each:

    h(t | x) = lambda_k * exp(x' * beta)   for t in [a_{k-1}, a_k)

Fit via Poisson GLM on split data: expand each subject into one row
per interval they enter, with offset log(exposure time) and outcome
= event indicator in that interval.

Advantages:
    * Any continuous hazard is approximated arbitrarily well as K
      grows (nonparametric baseline).
    * Time-varying covariates handled by including them as extra
      time-dependent design columns.
    * Easily extended to competing risks / relative-survival /
      Bayesian settings.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # MLE


def split_person_time(t, e, x, cuts):
    """Split each subject at cut times; return (event, log_time, x_expand, interval_id)."""
    rows_e = []; rows_tt = []; rows_x = []; rows_k = []
    K = len(cuts) - 1
    for i in range(len(t)):
        for k in range(K):
            a, b = cuts[k], cuts[k + 1]
            if t[i] <= a:
                continue
            tt = min(t[i], b) - a
            rows_tt.append(tt)
            rows_e.append(int(e[i] == 1 and t[i] <= b and t[i] > a))
            rows_x.append(x[i])
            rows_k.append(k)
    return (np.array(rows_e), np.array(rows_tt),
            np.array(rows_x), np.array(rows_k))


def piecewise_exp_fit(events, log_time, x, k_idx, K):
    """Poisson MLE for lambda_k, beta."""
    n = len(events)
    p = x.shape[1]
    p_full = K + p

    def nll(theta):
        lam = np.exp(theta[:K])
        beta = theta[K:]
        eta = np.log(lam[k_idx]) + x @ beta + np.log(log_time + 1e-12)
        mu = np.exp(eta)
        return -np.sum(events * eta - mu)

    theta0 = np.r_[np.full(K, -1.0), np.zeros(p)]
    r = minimize(nll, theta0, method="L-BFGS-B")
    lam_hat = np.exp(r.x[:K]); beta_hat = r.x[K:]
    return {"lambda": lam_hat, "beta": beta_hat, "loglik": -r.fun}


if __name__ == "__main__":
    print("=== Piecewise exponential model ===\n")
    rng = np.random.default_rng(0)
    n = 800
    x = rng.binomial(1, 0.4, size=n).reshape(-1, 1).astype(float)
    #  True hazard: piecewise with 3 pieces, HR = 2 for exposed
    def true_h(t, x_i):
        base = np.where(t < 1, 0.2, np.where(t < 3, 0.5, 1.0))
        return base * np.exp(np.log(2) * x_i)

    #  Simulate via inverse CDF (piecewise)
    U = rng.uniform(size=n)
    t_true = np.zeros(n)
    for i in range(n):
        target = -np.log(U[i])
        cum = 0; t = 0
        for start, end, hz in [(0, 1, 0.2), (1, 3, 0.5), (3, 20, 1.0)]:
            h_x = hz * np.exp(np.log(2) * x[i, 0])
            delta = end - start
            if cum + h_x * delta >= target:
                t = start + (target - cum) / h_x
                break
            cum += h_x * delta
        t_true[i] = t
    c = rng.exponential(4.0, size=n)
    e = (t_true <= c).astype(int)
    t_obs = np.minimum(t_true, c)

    cuts = np.array([0.0, 1.0, 3.0, 20.0])
    ev, tt, xexp, kid = split_person_time(t_obs, e, x, cuts)
    r = piecewise_exp_fit(ev, tt, xexp, kid, K=len(cuts) - 1)

    print(f"  Cuts = {cuts}")
    print(f"  True lambda = [0.20, 0.50, 1.00]")
    print(f"  Est  lambda = {r['lambda'].round(3).tolist()}")
    print(f"  True HR (x)  = {2.0:.2f}")
    print(f"  Est  HR (x)  = {np.exp(r['beta'][0]):.3f}")

    print("\n--- library cross-check (survival::pyears + glm(family=poisson) R;\n"
          "                          lifelines PiecewiseExponentialRegressionFitter Python) ---")
