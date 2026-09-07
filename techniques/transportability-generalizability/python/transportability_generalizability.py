"""Transportability / generalizability (Reference Sec 15.39).

Cole & Stuart 2010, Westreich et al. 2017. Standardises an
experimental causal effect from a STUDY sample to a TARGET population
that differs in the distribution of effect modifiers X.

Setup:
    S = 1 for study sample, 0 for target sample.
    Y = outcome, T = randomised treatment (study only).
    X = effect modifiers -- covariates whose distribution differs
        between S=1 and S=0 AND that interact with T.

Inverse-odds weighting (IOW) estimator:

    w_i = P(S = 0 | X = x_i) / P(S = 1 | X = x_i)     for study units

    E[Y(1) | S = 0] = sum_i w_i * 1[T=1] * Y_i / sum_i w_i * 1[T=1]
    E[Y(0) | S = 0] = sum_i w_i * 1[T=0] * Y_i / sum_i w_i * 1[T=0]

Transported ATE = the difference of the two.

Assumptions (Cole-Stuart):
  * Randomisation of T | S=1.
  * Positivity across S at every X value in the target.
  * Effect-measure transportability: all effect modifiers included in X.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # logistic fit


def fit_logistic(x, y):
    x1 = np.c_[np.ones(len(y)), x]

    def nll(b):
        p = 1 / (1 + np.exp(-(x1 @ b)))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize(nll, np.zeros(x1.shape[1]), method="L-BFGS-B")
    return 1 / (1 + np.exp(-(x1 @ r.x)))


def transported_ate(y, t, x, s):
    """Inverse-odds weighting for transportability.

    y, t: outcome and treatment (defined for S=1 only; for S=0 units,
    y and t values are ignored but arrays are aligned.)
    """
    p_s1 = fit_logistic(x, s)                   # P(S=1 | X)
    #  weights ONLY for S=1 units; multiply by P(S=0)/P(S=1)
    p_s0 = 1 - p_s1
    w = np.zeros_like(p_s1)
    mask = s == 1
    w[mask] = p_s0[mask] / p_s1[mask]

    #  Hajek estimator on treated / control study units
    num_1 = np.sum(w * (mask) * (t == 1) * y)
    den_1 = np.sum(w * (mask) * (t == 1))
    num_0 = np.sum(w * (mask) * (t == 0) * y)
    den_0 = np.sum(w * (mask) * (t == 0))
    E1 = num_1 / den_1
    E0 = num_0 / den_0
    return {"E_Y1_target": E1, "E_Y0_target": E0, "ATE_target": E1 - E0}


if __name__ == "__main__":
    print("=== Transportability -- IOW (Cole-Stuart) ===\n")
    rng = np.random.default_rng(0)
    n_study = 2000
    n_target = 4000

    #  Study sample: X ~ N(0, 1). Target: X ~ N(1.5, 1) -- shifted right.
    x_study = rng.normal(0, 1, size=n_study)
    x_target = rng.normal(1.5, 1, size=n_target)
    x = np.r_[x_study, x_target]
    s = np.r_[np.ones(n_study), np.zeros(n_target)]

    #  Randomised treatment in study
    t_study = rng.integers(0, 2, size=n_study)
    t = np.r_[t_study, np.zeros(n_target)]        # ignored for target

    #  Effect modification: tau(x) = 1 + x. Study ATE at x=0 => 1.
    #  Target ATE at x=1.5 => 1 + 1.5 = 2.5.
    tau = 1.0 + x_study
    y0 = 0.5 * x_study + rng.normal(scale=0.5, size=n_study)
    y_study = y0 + tau * t_study
    y = np.r_[y_study, np.zeros(n_target)]

    #  Naive study ATE (ignores target distribution)
    naive = np.mean(y_study[t_study == 1]) - np.mean(y_study[t_study == 0])
    print(f"  Study ATE (naive, applies at X ~ N(0,1))     = {naive:+.3f}   (truth 1.00)")

    r = transported_ate(y, t, x.reshape(-1, 1), s)
    print(f"  Transported ATE to target (X ~ N(1.5, 1))    = {r['ATE_target']:+.3f}   (truth 2.50)")

    print("\n--- library cross-check (generalize R; causallib Python) ---")
