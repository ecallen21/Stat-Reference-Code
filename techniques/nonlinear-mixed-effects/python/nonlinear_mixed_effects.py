"""Nonlinear Mixed-Effects Models via two-stage estimation (Reference §12.12).

Standard NLME model:
    y_{ij}  =  f(theta_i, t_{ij})  +  eps_{ij}
    theta_i =  theta_pop  +  b_i             b_i ~ N(0, D)

Common f: pharmacokinetic 1-compartment (exp decay), logistic growth,
Michaelis-Menten, ...

Two-stage estimation (this file):
    Stage 1 : fit per-subject nonlinear regression -> theta_i_hat
    Stage 2 : meta-analyze the theta_i_hats -> theta_pop and D

Transparent and standalone, but LESS efficient than a joint (one-stage)
NLME fit that borrows strength across subjects (that requires numerical
integration over the random effects -- see R's nlme::nlme() or Python's
statsmodels doesn't ship this natively).

Cross-check note: for a full joint fit, use `nlme::nlme` in R.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint for functions

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize    # least-squares nonlinear fit


def fit_nlme_two_stage(y, time, subject_ids, model: Callable, theta0,
                        param_names=None) -> dict:
    """Two-stage NLME.

    Parameters
    ----------
    y : outcome per row.
    time : time per row.
    subject_ids : subject id per row.
    model : callable(theta_vector, t_vector) -> predicted y.
    theta0 : starting values for the parameter vector.
    """
    y = np.asarray(y, dtype=float); time = np.asarray(time, dtype=float)
    subject_ids = np.asarray(subject_ids)
    unique = np.unique(subject_ids)
    p = len(theta0)
    if param_names is None:
        param_names = [f"theta{i}" for i in range(p)]

    per_subject = {}
    theta_matrix = []
    for s in unique:
        m = subject_ids == s
        y_i = y[m]; t_i = time[m]
        try:
            res = optimize.least_squares(
                lambda th: model(th, t_i) - y_i, x0=theta0,
                max_nfev=1000
            )
            theta_i = res.x
            per_subject[s.item() if hasattr(s, "item") else s] = theta_i.tolist()
            theta_matrix.append(theta_i)
        except Exception as ex:
            per_subject[s.item() if hasattr(s, "item") else s] = None
    theta_matrix = np.array(theta_matrix)
    # Stage 2: population mean and covariance
    theta_pop = theta_matrix.mean(axis=0)
    D = np.cov(theta_matrix.T, ddof=1) if theta_matrix.shape[0] > 1 else np.zeros((p, p))
    se_pop = np.sqrt(np.clip(np.diag(D), 0, None) / len(theta_matrix))
    return {"theta_pop": dict(zip(param_names, theta_pop.tolist())),
            "SE_pop": dict(zip(param_names, se_pop.tolist())),
            "D_matrix": D.tolist(),
            "n_subjects_fit": int(theta_matrix.shape[0]),
            "per_subject_head": {k: v for k, v in list(per_subject.items())[:5]},
            "method": "two-stage NLME (per-subject NLS + meta of parameters)"}


# --- Common nonlinear models --------------------------------------------

def logistic_growth(theta, t):
    """theta = (asymptote, growth_rate, midpoint)"""
    A, r, m = theta
    return A / (1 + np.exp(-r * (t - m)))


def exp_decay(theta, t):
    """theta = (initial, decay_rate)"""
    y0, k = theta
    return y0 * np.exp(-k * t)


if __name__ == "__main__":
    rng = np.random.default_rng(43)
    n_subj = 40; n_time = 8
    subject_ids = np.repeat(np.arange(n_subj), n_time)
    time = np.tile(np.linspace(0, 10, n_time), n_subj)
    # True population parameters for a logistic growth (asymptote, rate, midpoint)
    true_pop = np.array([100.0, 0.6, 5.0])
    D_true = np.diag([100.0, 0.02, 0.5])         # subject-level variance components
    b = rng.multivariate_normal(np.zeros(3), D_true, n_subj)
    y = np.empty(n_subj * n_time)
    for i in range(n_subj):
        m = subject_ids == i
        theta_i = true_pop + b[i]
        y[m] = logistic_growth(theta_i, time[m]) + rng.normal(0, 2, m.sum())

    print("=== Two-stage NLME on logistic growth (true pop = [100, 0.6, 5.0]) ===")
    fit = fit_nlme_two_stage(y, time, subject_ids, logistic_growth,
                              theta0=[80.0, 0.5, 4.0],
                              param_names=["asymptote", "rate", "midpoint"])
    print(f"  theta_pop = {fit['theta_pop']}")
    print(f"  SE_pop    = {fit['SE_pop']}")
    print(f"  fit subjects = {fit['n_subjects_fit']}")
    print(f"  first 3 subject fits: {list(fit['per_subject_head'].items())[:3]}")
