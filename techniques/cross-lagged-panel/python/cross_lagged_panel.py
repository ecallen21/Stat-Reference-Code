"""Cross-Lagged Panel Model + RI-CLPM (Reference §12.10; also covers §12.18).

Two variables X and Y measured at 2+ time points on the same subjects.
Question: does X cause Y, does Y cause X, or both / neither?

Classic (2-wave) CLPM:
    X_t = a_x + phi_x X_{t-1} + beta_{yx} Y_{t-1} + e_x
    Y_t = a_y + phi_y Y_{t-1} + beta_{xy} X_{t-1} + e_y

    phi_x, phi_y  = AUTOREGRESSIVE (stability of each variable)
    beta_yx, beta_xy = CROSS-LAGGED (Y_{t-1} predicting X_t and vice versa)

Interpretation: beta_xy significant + beta_yx not => X leads Y (Granger-causal
in the panel sense).

**RI-CLPM** (Random-Intercept CLPM; Hamaker, Kuiper, Grasman 2015):
Adds a subject-level RANDOM INTERCEPT for each variable, so the CLPM effects
are WITHIN-subject rather than between+within. Fixes a well-known problem
where cross-lagged effects in the classic CLPM confound stable trait
differences with time-varying change. If you have >= 3 waves and want
"true" within-person Granger-style effects, use RI-CLPM.

This file:
    - Estimates classic 2-wave CLPM by two OLS regressions.
    - Estimates a simplified RI-CLPM via per-subject centering.
    - Notes that a full RI-CLPM SEM fit needs at least 3 waves and specialized
      software (`lavaan` in R, `semopy` in Python).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _ols_with_se(X, y):
    """Plain OLS returning beta + SE + t + p (2-sided)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, p = X.shape
    sigma2 = (resid ** 2).sum() / (n - p)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = beta / np.where(se > 0, se, 1e-12)
    p_val = 2 * stats.t.sf(np.abs(t), n - p)
    return beta, se, t, p_val


def clpm_two_wave(X_wave1, Y_wave1, X_wave2, Y_wave2) -> dict:
    """Classic two-wave cross-lagged panel model via two OLS regressions."""
    X1 = np.asarray(X_wave1, dtype=float); Y1 = np.asarray(Y_wave1, dtype=float)
    X2 = np.asarray(X_wave2, dtype=float); Y2 = np.asarray(Y_wave2, dtype=float)
    n = len(X1)
    # Regression 1: X_2 = a + phi_x X_1 + beta_yx Y_1 + e
    D = np.column_stack([np.ones(n), X1, Y1])
    b_x, se_x, t_x, p_x = _ols_with_se(D, X2)
    # Regression 2: Y_2 = a + phi_y Y_1 + beta_xy X_1 + e
    D = np.column_stack([np.ones(n), Y1, X1])
    b_y, se_y, t_y, p_y = _ols_with_se(D, Y2)
    return {"X_regression": {
                "intercept": float(b_x[0]), "SE_intercept": float(se_x[0]),
                "autoreg_phi_x": float(b_x[1]), "SE_phi_x": float(se_x[1]),
                "cross_lag_from_Y_beta_yx": float(b_x[2]),
                "SE_beta_yx": float(se_x[2]),
                "p_beta_yx": float(p_x[2])},
            "Y_regression": {
                "intercept": float(b_y[0]), "SE_intercept": float(se_y[0]),
                "autoreg_phi_y": float(b_y[1]), "SE_phi_y": float(se_y[1]),
                "cross_lag_from_X_beta_xy": float(b_y[2]),
                "SE_beta_xy": float(se_y[2]),
                "p_beta_xy": float(p_y[2])},
            "n_subjects": n,
            "method": "classic 2-wave cross-lagged panel model (two OLS)"}


def ri_clpm_within_person(X_waves, Y_waves) -> dict:
    """Simplified RI-CLPM via person-centering.

    Parameters
    ----------
    X_waves : n x T array of X observations (rows = subjects, cols = time).
    Y_waves : n x T array of Y observations.
    """
    X = np.asarray(X_waves, dtype=float); Y = np.asarray(Y_waves, dtype=float)
    if X.shape != Y.shape or X.shape[1] < 2:
        raise ValueError("X_waves and Y_waves must be same n x T (T >= 2)")
    n, T = X.shape
    # Person-center: subtract each subject's mean across time
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    # Stack lagged pairs: (t-1, t) for t = 1..T-1
    X_prev = Xc[:, :-1].flatten(); Y_prev = Yc[:, :-1].flatten()
    X_curr = Xc[:, 1:].flatten();  Y_curr = Yc[:, 1:].flatten()
    n_pairs = len(X_prev)
    # Regressions on within-person residuals (dropping the intercept since centered)
    D = np.column_stack([X_prev, Y_prev])
    b_x, se_x, t_x, p_x = _ols_with_se(D, X_curr)
    D = np.column_stack([Y_prev, X_prev])
    b_y, se_y, t_y, p_y = _ols_with_se(D, Y_curr)
    return {"X_regression_within": {
                "autoreg_phi_x": float(b_x[0]), "SE_phi_x": float(se_x[0]),
                "cross_lag_from_Y_beta_yx_within": float(b_x[1]),
                "SE_beta_yx_within": float(se_x[1]),
                "p_beta_yx_within": float(p_x[1])},
            "Y_regression_within": {
                "autoreg_phi_y": float(b_y[0]), "SE_phi_y": float(se_y[0]),
                "cross_lag_from_X_beta_xy_within": float(b_y[1]),
                "SE_beta_xy_within": float(se_y[1]),
                "p_beta_xy_within": float(p_y[1])},
            "n_subjects": n, "n_waves": T, "n_lagged_pairs": n_pairs,
            "note": ("simplified RI-CLPM via within-person centering; a full "
                     "SEM fit (lavaan/semopy) treats the random intercept as "
                     "a latent variable and gives proper standard errors."),
            "method": "RI-CLPM approximation (person-centered OLS)"}


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    n = 300
    # True model: X leads Y (beta_xy > 0), Y does NOT lead X
    X1 = rng.normal(0, 1, n)
    Y1 = 0.3 * X1 + rng.normal(0, 1, n)
    X2 = 0.6 * X1 + 0.0 * Y1 + rng.normal(0, 1, n)          # no Y -> X effect
    Y2 = 0.5 * Y1 + 0.4 * X1 + rng.normal(0, 1, n)          # strong X -> Y effect

    print("=== Classic 2-wave CLPM (true: X -> Y, no Y -> X) ===")
    fit = clpm_two_wave(X1, Y1, X2, Y2)
    xr = fit["X_regression"]; yr = fit["Y_regression"]
    print(f"  X_2 = {xr['intercept']:+.4f} + {xr['autoreg_phi_x']:+.4f} X_1 "
          f"+ {xr['cross_lag_from_Y_beta_yx']:+.4f} Y_1     (p Y->X: {xr['p_beta_yx']:.3g})")
    print(f"  Y_2 = {yr['intercept']:+.4f} + {yr['autoreg_phi_y']:+.4f} Y_1 "
          f"+ {yr['cross_lag_from_X_beta_xy']:+.4f} X_1     (p X->Y: {yr['p_beta_xy']:.3g})")

    # RI-CLPM approx (needs 3+ waves for interesting results; here we fake 3 waves)
    print("\n=== RI-CLPM approximation with 3 fabricated waves ===")
    X3 = 0.6 * X2 + rng.normal(0, 1, n)
    Y3 = 0.5 * Y2 + 0.4 * X2 + rng.normal(0, 1, n)
    Xw = np.column_stack([X1, X2, X3])
    Yw = np.column_stack([Y1, Y2, Y3])
    ri = ri_clpm_within_person(Xw, Yw)
    print(f"  within-person X_t = {ri['X_regression_within']['autoreg_phi_x']:+.4f} X_{{t-1}} "
          f"+ {ri['X_regression_within']['cross_lag_from_Y_beta_yx_within']:+.4f} Y_{{t-1}}"
          f"   (p Y->X within: {ri['X_regression_within']['p_beta_yx_within']:.3g})")
    print(f"  within-person Y_t = {ri['Y_regression_within']['autoreg_phi_y']:+.4f} Y_{{t-1}} "
          f"+ {ri['Y_regression_within']['cross_lag_from_X_beta_xy_within']:+.4f} X_{{t-1}}"
          f"   (p X->Y within: {ri['Y_regression_within']['p_beta_xy_within']:.3g})")
