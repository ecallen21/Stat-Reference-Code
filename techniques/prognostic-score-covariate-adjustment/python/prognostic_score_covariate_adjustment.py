"""Prognostic-score covariate adjustment (Reference Sec 44.15).

Hansen 2008 'The prognostic analogue of the propensity score',
Biometrika; Schuler et al. 2022 'Increasing the efficiency of
randomised trial estimates via linear adjustment for a prognostic
score' (PROCOVA / EMA CHMP 2024). In a randomised trial, adjusting
the outcome for a MODEL-BASED PROGNOSTIC SCORE built from CONTROL-
GROUP outcomes (typically ML on historical / observational data)
reduces variance of the treatment effect estimate.

Estimator:
    1. Fit m_hat(x) = E[Y_control | X] on external / historical data
       or the concurrent control arm.
    2. ANCOVA: Y = beta_0 + beta_1 * T + beta_2 * m_hat(X) + eps.
       The coefficient beta_1 is the adjusted treatment effect.

Efficiency gain vs unadjusted difference-in-means:

    Var(delta_adj)  /  Var(delta_unadj)  ~  1 - R^2(m_hat)

so a prognostic score with R^2 = 0.5 halves the variance.

EMA CHMP (2024) has issued qualification opinion allowing PROCOVA in
regulatory clinical trial submissions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def procova_adjustment(y, t, m_hat):
    """ANCOVA-style adjusted ATE with prognostic score m_hat."""
    X = np.c_[np.ones(len(y)), t, m_hat]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    #  Sandwich SE for the treatment coefficient
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros_like(XtX_inv)
    for i in range(n):
        u = resid[i]; xi = X[i]
        S += u ** 2 * np.outer(xi, xi)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return {"beta": beta, "ATE": beta[1], "SE_ATE": se}


def unadjusted_ate(y, t):
    d = np.mean(y[t == 1]) - np.mean(y[t == 0])
    n1 = (t == 1).sum(); n0 = (t == 0).sum()
    var = np.var(y[t == 1], ddof=1) / n1 + np.var(y[t == 0], ddof=1) / n0
    return {"ATE": d, "SE_ATE": np.sqrt(var)}


if __name__ == "__main__":
    print("=== PROCOVA / prognostic-score covariate adjustment ===\n")
    rng = np.random.default_rng(0)

    #  Historical (external) control data: n_hist patients, X observed
    n_hist = 5000; p = 5
    X_hist = rng.normal(size=(n_hist, p))
    beta_true = np.array([1.2, -0.6, 0.4, 0.0, 0.3])
    #  Historical outcomes under control
    y_hist_control = X_hist @ beta_true + rng.normal(scale=1.0, size=n_hist)

    #  Fit prognostic score via OLS on historical data
    coefs_prog, *_ = np.linalg.lstsq(np.c_[np.ones(n_hist), X_hist], y_hist_control, rcond=None)

    #  Concurrent trial: n patients, randomised (1:1)
    n = 400
    X = rng.normal(size=(n, p))
    t = rng.integers(0, 2, size=n)
    tau_true = 0.6                       # true ATE
    y = X @ beta_true + tau_true * t + rng.normal(scale=1.0, size=n)
    m_hat = np.c_[np.ones(n), X] @ coefs_prog

    unadj = unadjusted_ate(y, t)
    adj = procova_adjustment(y, t, m_hat)

    print(f"  True ATE = {tau_true:.3f}\n")
    print(f"  Unadjusted            ATE = {unadj['ATE']:+.3f}   SE = {unadj['SE_ATE']:.3f}")
    print(f"  PROCOVA (prognostic)  ATE = {adj['ATE']:+.3f}   SE = {adj['SE_ATE']:.3f}")
    ratio = (adj['SE_ATE'] / unadj['SE_ATE']) ** 2
    print(f"\n  Variance ratio adj/unadj = {ratio:.3f}   -> effective-sample-size gain "
          f"{1 / ratio:.2f}x")

    print("\n--- library cross-check (RATES / procova R; unlearn-ai/procova Python) ---")
