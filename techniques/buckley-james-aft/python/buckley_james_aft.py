"""Buckley-James semi-parametric AFT (Reference Sec 11.32).

Buckley & James 1979 'Linear regression with censored data',
Biometrika. Semi-parametric AFT: fits log T = X * beta + eps without
parameterising the residual distribution.

Iterative KM-imputation:

    1. Initialise beta (e.g., OLS on complete cases).
    2. Compute residuals e_i = log T_i - X_i * beta.
    3. Build KM of the residuals treating censoring status;
       impute censored residuals with the conditional MEAN
           E[e | e > e_censored]  via the KM.
    4. Refit OLS on imputed y* = X * beta + e_imputed -> new beta.
    5. Iterate until convergence.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def km_conditional_mean(residuals, event, threshold):
    """Estimate E[e | e > threshold] via a Kaplan-Meier of the residuals."""
    order = np.argsort(residuals)
    r = residuals[order]; e = event[order]
    n = len(r); S = 1.0
    #  Build survival function on residuals
    times = []; surv = []
    at_risk = n
    for i in range(n):
        if e[i]:
            S *= (1 - 1 / at_risk)
        times.append(r[i]); surv.append(S)
        at_risk -= 1
    times = np.array(times); surv = np.array(surv)
    #  Restricted mean beyond threshold: integral from threshold to max of S(t) dt
    mask = times > threshold
    if mask.sum() == 0:
        return threshold
    #  Approximate integral via trapezoidal rule + max area beyond
    t_seg = times[mask]; s_seg = surv[mask]
    prev_S = surv[np.searchsorted(times, threshold) - 1] if threshold > times[0] else 1.0
    if prev_S <= 0:
        return threshold
    #  Conditional survival S(t | e > threshold) = S(t) / S(threshold)
    S_cond = s_seg / prev_S
    #  E[e | e > threshold] = threshold + integral_0^inf S_cond(u + threshold) du
    from numpy import trapezoid                # NumPy >= 2.0 name
    ints = trapezoid(np.r_[1.0, S_cond], np.r_[threshold, t_seg])
    return threshold + ints


def buckley_james(t, e, X, max_iter=30, tol=1e-4):
    y = np.log(t)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]        # initial OLS
    for it in range(max_iter):
        resid = y - X @ beta
        y_impute = y.copy()
        for i in range(len(t)):
            if e[i] == 0:
                y_impute[i] = X[i] @ beta + km_conditional_mean(resid, e, resid[i])
        beta_new = np.linalg.lstsq(X, y_impute, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return {"beta": beta, "iterations": it + 1}


if __name__ == "__main__":
    print("=== Buckley-James semi-parametric AFT (1979) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    #  True: log T = 2 + 0.5 x + eps, eps ~ Gumbel_min (Weibull AFT)
    U = rng.uniform(size=n)
    W = np.log(-np.log(U))
    log_t = 2.0 + 0.5 * x + 0.4 * W
    t = np.exp(log_t)
    c = rng.exponential(15.0, size=n)
    e = (t <= c).astype(int)
    t = np.minimum(t, c)
    X = np.c_[np.ones(n), x]

    r = buckley_james(t, e, X)
    print(f"  Censoring rate = {(1 - e.mean()) * 100:.1f}%")
    print(f"  True beta      = (2.0, 0.5)")
    print(f"  BJ estimate    = ({r['beta'][0]:.3f}, {r['beta'][1]:.3f})")
    print(f"  Converged in {r['iterations']} iterations")

    print("\n--- library cross-check (bujar / RegularizedSCA R; from-scratch Python) ---")
