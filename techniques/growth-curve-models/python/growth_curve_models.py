"""Growth Curve Models via LMM (Reference §12.4).

Growth models represent individual trajectories over time with random
intercepts and slopes:

    y_{ij}  =  (beta_0 + u_{0i})  +  (beta_1 + u_{1i}) * t_{ij}  +  eps_{ij}

fixed effects (beta_0, beta_1) = grand-average intercept and slope
random effects (u_0, u_1) ~ N(0, G) = subject-specific deviations

Quadratic extension:
    y_{ij}  =  (beta_0 + u_{0i})  +  (beta_1 + u_{1i}) t_{ij}
                                  +  (beta_2 + u_{2i}) t_{ij}^2  +  eps_{ij}

Equivalent to a latent growth model (LGM) in SEM parameterization.

Per-subject BLUPs give each subject's own trajectory, useful for individual-
level prediction and visualization.

We reuse the LMM fitter from techniques/linear-mixed-models.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "linear-mixed-models", "python"))
from linear_mixed_models import fit_lmm    # techniques/linear-mixed-models/python/linear_mixed_models.py::fit_lmm


def fit_growth_curve_linear(y, time, subject_ids, extra_X=None) -> dict:
    """Random-intercept + random-slope linear growth model.

    Parameters
    ----------
    y : outcome, length n.
    time : time variable, length n.
    subject_ids : subject identifier per row.
    extra_X : optional additional fixed-effect covariates (n x q).
    """
    y = np.asarray(y, dtype=float); time = np.asarray(time, dtype=float)
    if extra_X is None:
        X = np.column_stack([np.ones(len(y)), time])
    else:
        X = np.column_stack([np.ones(len(y)), time, np.asarray(extra_X, dtype=float)])
    Z = np.column_stack([np.ones(len(y)), time])          # random int + slope
    fit = fit_lmm(y, X, Z, subject_ids)
    fit["method"] = "linear growth-curve model (random int + slope) via LMM REML"
    return fit


def fit_growth_curve_quadratic(y, time, subject_ids) -> dict:
    """Random-intercept + random-slope-on-time + random-slope-on-time^2."""
    y = np.asarray(y, dtype=float); time = np.asarray(time, dtype=float)
    X = np.column_stack([np.ones(len(y)), time, time ** 2])
    # For simplicity: random-int + random-slope-on-time only (not on t^2, to
    # keep the fit stable). Extension: use Z with time^2 for a 3x3 G matrix.
    Z = np.column_stack([np.ones(len(y)), time])
    fit = fit_lmm(y, X, Z, subject_ids)
    fit["method"] = "quadratic growth-curve model (random int + slope; fixed quadratic)"
    return fit


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    n_subj = 40; n_time = 6
    subject_ids = np.repeat(np.arange(n_subj), n_time)
    time = np.tile(np.arange(n_time, dtype=float), n_subj)
    # Random intercept + random slope
    u0 = rng.normal(0, 0.8, n_subj)
    u1 = rng.normal(0, 0.2, n_subj)
    beta0_true, beta1_true = 50.0, 2.0
    y = (beta0_true + u0[subject_ids]) + (beta1_true + u1[subject_ids]) * time + rng.normal(0, 0.5, n_subj * n_time)

    print("=== Linear growth curve ===")
    fit = fit_growth_curve_linear(y, time, subject_ids)
    print(f"  fixed effects (intercept, slope) = {fit['beta']}  (true = [50.0, 2.0])")
    print(f"  SE = {fit['SE_beta']}")
    print(f"  sigma^2 (residual) = {fit['sigma2']:.4f}  (true = 0.25)")
    print(f"  G matrix (random-effect cov):")
    for row in fit["G_matrix"]:
        print(f"    {row}")
    print(f"  BLUPs (first 3 subjects, [intercept-dev, slope-dev]):")
    for k, v in list(fit["blups_head"].items())[:3]:
        print(f"    subject {k}: {v}")

    # Quadratic
    print("\n=== Quadratic growth curve ===")
    y2 = y + 0.1 * time ** 2                             # add mild curvature
    fit_q = fit_growth_curve_quadratic(y2, time, subject_ids)
    print(f"  fixed effects (int, linear, quad) = {[f'{b:.4f}' for b in fit_q['beta']]}")
    print(f"  (true linear = 2.0, true quadratic ~ 0.1)")
