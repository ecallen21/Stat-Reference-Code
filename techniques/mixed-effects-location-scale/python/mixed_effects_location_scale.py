"""Mixed-Effects Location-Scale (MELS) model (Reference §12.21).

Standard LMM assumes constant within-subject variance. MELS lets EACH SUBJECT
have their own residual variance -- captures individuals who are highly
variable ("volatile") vs. highly consistent.

Model:
    y_{ij}  =  X_{ij}' beta  +  u_{0i}  +  eps_{ij}
    u_{0i}  ~  N(0, sigma_u^2)
    eps_{ij} ~ N(0, sigma_{eps,i}^2)      subject-specific residual variance

    log sigma_{eps,i}^2  =  gamma_0  +  gamma_1' Z_{i}  +  v_i         v_i ~ N(0, sigma_v^2)

So there are TWO random effects per subject: one for the mean level (u_0i), one
for the log-residual-SD (v_i). Fitting is done via joint MLE / adaptive quadrature.

This file implements a simplified MELS:
    1. Fit ordinary LMM to get beta and per-subject BLUP intercepts.
    2. Compute per-subject residual variance from within-subject residuals.
    3. Report the empirical variance of log(residual variance) across subjects as
       an estimate of sigma_v^2 (the location-scale random-effect variance).

For a full joint MELS fit see R's `mixregls` (Hedeker & Nordgren 2013).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "linear-mixed-models", "python"))
from linear_mixed_models import fit_lmm    # techniques/linear-mixed-models/python/linear_mixed_models.py::fit_lmm


def fit_mels_two_stage(y, X, cluster_ids) -> dict:
    """Two-stage MELS: LMM fit for mean + empirical per-subject residual SD.

    Returns
    -------
    dict with fixed effects (beta), subject-level location random effect (sigma_u^2),
    per-subject residual variances, and their between-subject variance
    (sigma_v^2 estimate).
    """
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    Z = np.ones((len(y), 1))
    fit = fit_lmm(y, X, Z, cluster_ids)
    beta = np.array(fit["beta"])
    blups = fit["blups_head"]                          # only first 5, but we need all
    # Recompute BLUPs for all subjects
    sigma_u2 = fit["sigma_u2"]; sigma2 = fit["sigma2"]
    unique = np.unique(cluster_ids)
    all_blups = np.empty(len(unique))
    per_subj_var = np.empty(len(unique))
    per_subj_n = np.empty(len(unique), dtype=int)
    for k, c in enumerate(unique):
        m = cluster_ids == c
        y_i = y[m]; X_i = X[m]; n_i = m.sum()
        # Best linear predictor of u_i given data
        V_i = sigma_u2 * np.ones((n_i, n_i)) + sigma2 * np.eye(n_i)
        V_inv = np.linalg.inv(V_i)
        u = sigma_u2 * np.ones(n_i) @ V_inv @ (y_i - X_i @ beta)
        all_blups[k] = float(u)
        # Per-subject residual after subtracting fixed + BLUP
        r_i = y_i - X_i @ beta - u
        per_subj_var[k] = float(np.var(r_i, ddof=1)) if n_i > 1 else float("nan")
        per_subj_n[k] = int(n_i)

    log_sd_per_subject = np.log(np.sqrt(np.clip(per_subj_var, 1e-12, None)))
    # Between-subject SD of log-residual-SD
    sigma_v = float(log_sd_per_subject.std(ddof=1))
    gamma_0 = float(log_sd_per_subject.mean())
    return {"beta_fixed": beta.tolist(),
            "sigma_u2": float(sigma_u2),
            "sigma_v_est_sd_of_log_resid_SD": sigma_v,
            "gamma_0_mean_of_log_resid_SD": gamma_0,
            "per_subject_residual_var_head": per_subj_var[:5].tolist(),
            "per_subject_residual_var_stats": {
                "min": float(per_subj_var.min()),
                "median": float(np.median(per_subj_var)),
                "max": float(per_subj_var.max())},
            "n_clusters": int(len(unique)),
            "note": ("Two-stage MELS approximation; full joint MLE (Hedeker "
                     "Nordgren 2013) is in R's mixregls package."),
            "method": "two-stage MELS approximation"}


if __name__ == "__main__":
    rng = np.random.default_rng(59)
    n_clusters = 60; n_per = 10; n = n_clusters * n_per
    cluster_ids = np.repeat(np.arange(n_clusters), n_per)
    u0 = rng.normal(0, 0.7, n_clusters)                   # intercept random effect
    # subject-specific log-SD: gamma_0 = log(0.5) = -0.69; v_i ~ N(0, 0.4)
    v_i = rng.normal(0, 0.4, n_clusters)
    log_sd_i = -0.69 + v_i
    sd_i = np.exp(log_sd_i)                                # per-subject residual SD
    x = rng.normal(0, 1, n)
    y = 1.0 + 0.4 * x + u0[cluster_ids] + rng.normal(0, 1, n) * sd_i[cluster_ids]
    X = np.column_stack([np.ones(n), x])

    print("=== MELS two-stage fit ===")
    fit = fit_mels_two_stage(y, X, cluster_ids)
    print(f"  fixed effects beta = {fit['beta_fixed']}  (true = [1.0, 0.4])")
    print(f"  sigma_u^2 (intercept RE) = {fit['sigma_u2']:.4f}  (true = 0.49)")
    print(f"  gamma_0 (mean log residual SD) = {fit['gamma_0_mean_of_log_resid_SD']:+.4f}  (true = -0.69)")
    print(f"  sigma_v (SD of log residual SD across subjects) = {fit['sigma_v_est_sd_of_log_resid_SD']:.4f}  (true = 0.40)")
    print(f"  per-subject residual variance range: {fit['per_subject_residual_var_stats']}")
