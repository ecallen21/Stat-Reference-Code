"""Multivariate longitudinal analysis (Reference §12.14).

Two (or more) longitudinal outcomes measured on the same subjects over
time.  Analyzing them separately loses cross-outcome dependence and gives
inefficient inference.  Joint modelling captures:
    - correlated random effects across outcomes,
    - correlated residuals within a visit,
    - potentially different fixed-effect structures per outcome.

Bivariate LMM (linear-mixed-model) formulation
    y1_ij = X1_ij beta1 + Z1_ij b1_i + e1_ij
    y2_ij = X2_ij beta2 + Z2_ij b2_i + e2_ij
    (b1_i, b2_i) ~ N(0, D)          D is a JOINT random-effect covariance
    (e1_ij, e2_ij) ~ N(0, Sigma)    within-visit residual cov

Estimation via a stacked long form (subject-visit-outcome) plus a suitable
random-effect design matrix, then a standard LMM.  We take a two-stage
shortcut here: fit each outcome separately with random-intercepts, then
estimate the between-outcome random-effect correlation from the estimated
BLUPs.  Full joint MLE requires nlme/lme4 in R (or a Stan/PyMC model).

Contrast with joint-longitudinal-survival models (§12.10, deferred) which
tie a longitudinal biomarker to survival.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def two_stage_bivariate_lmm(subject, time, y1, y2) -> dict:
    """Two-stage bivariate random-intercept LMM estimator.

    Stage 1: fit y_k = alpha_k + beta_k * time + b_{ki} + e_{kij} for each outcome.
    Stage 2: correlate the BLUPs of b_1 and b_2 across subjects.
    """
    subject = np.asarray(subject); time = np.asarray(time, dtype=float)
    y1 = np.asarray(y1, dtype=float); y2 = np.asarray(y2, dtype=float)
    subs = np.unique(subject); N = len(subs)

    def fit_one(y):
        """Random-intercept model via profile MLE using the standard shrinkage formula."""
        # Fit fixed effects by OLS ignoring correlation (starting point)
        X = np.column_stack([np.ones_like(time), time])
        beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta_ols
        # Estimate sigma^2 (within) and tau^2 (between-subject)
        subj_means = np.array([resid[subject == s].mean() for s in subs])
        ns = np.array([np.sum(subject == s) for s in subs])
        tau2 = max(float(subj_means.var(ddof=1) - resid.var(ddof=1) / ns.mean()), 1e-6)
        sig2 = float(resid.var(ddof=1))
        # BLUP of random intercept per subject
        blup = np.array([subj_means[i] * (ns[i] * tau2) / (ns[i] * tau2 + sig2) for i in range(N)])
        return {"beta": beta_ols, "tau2": tau2, "sigma2": sig2, "blup": blup}

    f1 = fit_one(y1); f2 = fit_one(y2)
    corr_random = float(np.corrcoef(f1["blup"], f2["blup"])[0, 1])
    # Within-visit residual correlation
    r1 = y1 - (f1["beta"][0] + f1["beta"][1] * time)
    r2 = y2 - (f2["beta"][0] + f2["beta"][1] * time)
    corr_within = float(np.corrcoef(r1, r2)[0, 1])
    return {"outcome1": f1, "outcome2": f2,
            "corr_random_intercept": corr_random,
            "corr_within_residual": corr_within,
            "n_subjects": int(N),
            "method": "Two-stage bivariate random-intercept LMM (approximate)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 100  # subjects
    T = 5    # visits per subject
    subject = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N).astype(float)

    # True random intercepts: correlated across outcomes
    rho_b = 0.6; sig_b = 1.0
    D = sig_b ** 2 * np.array([[1, rho_b], [rho_b, 1]])
    b = rng.multivariate_normal([0, 0], D, N)

    # True within-visit residual correlation
    rho_e = 0.3; sig_e = 0.5
    Sigma = sig_e ** 2 * np.array([[1, rho_e], [rho_e, 1]])

    y1 = np.zeros(N * T); y2 = np.zeros(N * T)
    for i in range(N):
        for j in range(T):
            eps = rng.multivariate_normal([0, 0], Sigma)
            k = i * T + j
            y1[k] = 2.0 + 0.5 * time[k] + b[i, 0] + eps[0]
            y2[k] = 1.0 - 0.3 * time[k] + b[i, 1] + eps[1]

    r = two_stage_bivariate_lmm(subject, time, y1, y2)
    print("=== Bivariate random-intercept LMM ===")
    print(f"  outcome 1: beta = {r['outcome1']['beta'].round(3)}, tau^2 = {r['outcome1']['tau2']:.3f}, sigma^2 = {r['outcome1']['sigma2']:.3f}")
    print(f"  outcome 2: beta = {r['outcome2']['beta'].round(3)}, tau^2 = {r['outcome2']['tau2']:.3f}, sigma^2 = {r['outcome2']['sigma2']:.3f}")
    print(f"\n  corr(random intercepts) est: {r['corr_random_intercept']:.3f}  (true 0.60)")
    print(f"  corr(within-visit residuals) est: {r['corr_within_residual']:.3f}  (true ~ 0.30 combined with random-effect corr)")
