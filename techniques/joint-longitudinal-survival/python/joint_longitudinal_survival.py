"""Joint modelling of longitudinal and survival outcomes (Reference §12.10).

Longitudinal biomarker y_ij measured over time on subject i, plus a survival
outcome (event time T_i with censoring).  The biomarker is
    (i)  measured with error,
    (ii) often observed less frequently near the event,
    (iii) related to the hazard of the event.

Naive analyses:
    - Cox with last-observation-carried-forward: biased when spacing is
      informative.
    - Cox with time-varying observed y_ij: no measurement-error correction,
      biased toward null.

Joint model
    y_ij = X_ij beta + m_i(t_ij) + eps_ij,   m_i(t) = z_i(t) b_i  (random effect)
    h_i(t) = h_0(t) exp(alpha * m_i(t) + gamma^T W_i)
    where m_i(t) is the TRUE (unobserved) biomarker trajectory and alpha
    quantifies the association between the current biomarker level and
    the instantaneous hazard.

Estimation
    Full joint MLE / MCMC (JM / JMbayes in R, PyMC).  We implement a
    TWO-STAGE approximation (Tsiatis-Davidian 2004): fit the LME first,
    then use the BLUPs of m_i(t) as a time-varying covariate in Cox.
    Biased under strong association (informative censoring feedback);
    full joint MLE removes the bias.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def _fit_random_intercept_slope(subject, time, y):
    """Fit y_ij = alpha + beta * t_ij + b0_i + b1_i * t_ij via method-of-moments approx.

    Returns per-subject BLUPs and a callable m_hat(subject, t).
    """
    subject = np.asarray(subject); time = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    subs = np.unique(subject); N = len(subs)
    # Fit fixed effects by OLS ignoring correlation
    X = np.column_stack([np.ones_like(time), time])
    beta_fix, *_ = np.linalg.lstsq(X, y, rcond=None)
    # Per-subject residual intercept + slope estimates via subject-wise regression
    b = np.zeros((N, 2))
    for i, s in enumerate(subs):
        mask = subject == s
        if mask.sum() >= 2:
            Xi = X[mask]; yi = y[mask]
            beta_i, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            b[i] = beta_i - beta_fix
    # Shrinkage toward zero (crude EB): shrink by rho = var_between / (var_between + var_within/n_i)
    # Skipped for simplicity; use raw b_i's here.
    def m_hat(sub_id, t):
        i = int(np.where(subs == sub_id)[0][0])
        return beta_fix[0] + beta_fix[1] * t + b[i, 0] + b[i, 1] * t
    return {"beta_fixed": beta_fix, "b_subject": b, "subjects": subs, "m_hat": m_hat}


def _cox_partial_like(time_event, event, X_tv_fn) -> dict:
    """Fit Cox PH with a single TIME-VARYING covariate X_tv_fn(i, t) via Breslow ties."""
    from scipy.optimize import minimize
    time_event = np.asarray(time_event, dtype=float); event = np.asarray(event, dtype=int)
    n = len(time_event); ord_ = np.argsort(time_event)
    def neg_ll(alpha):
        alpha = float(alpha[0])
        ll = 0.0
        for idx in np.where(event == 1)[0]:
            t = time_event[idx]
            at_risk = np.where(time_event >= t)[0]
            eta = np.array([alpha * X_tv_fn(i, t) for i in at_risk])
            ll += alpha * X_tv_fn(idx, t) - math.log(np.exp(eta).sum())
        return -ll
    res = minimize(neg_ll, [0.0], method="Nelder-Mead")
    alpha_hat = float(res.x[0])
    # SE via numerical Hessian
    eps = 1e-4
    h = (neg_ll([alpha_hat + eps]) - 2 * neg_ll([alpha_hat]) + neg_ll([alpha_hat - eps])) / (eps ** 2)
    se = math.sqrt(1 / max(h, 1e-10))
    return {"alpha": alpha_hat, "se": float(se),
            "z": alpha_hat / se, "p": float(2 * stats.norm.sf(abs(alpha_hat / se))),
            "n_events": int(event.sum())}


def two_stage_joint(subject, time_long, y_long, subject_survival,
                    time_event, event) -> dict:
    """Two-stage joint model: LME -> plug into Cox with time-varying m_hat(t)."""
    lme = _fit_random_intercept_slope(subject, time_long, y_long)
    subs = lme["subjects"]
    def X_tv_fn(i, t):
        return lme["m_hat"](subject_survival[i], t)
    cox = _cox_partial_like(time_event, event, X_tv_fn)
    return {"lme_beta": lme["beta_fixed"],
            "cox_alpha": cox["alpha"], "cox_se": cox["se"],
            "cox_z": cox["z"], "cox_p": cox["p"],
            "n_events": cox["n_events"],
            "method": "Two-stage joint (LME BLUPs -> Cox with time-varying m_hat)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 150
    # Longitudinal: 3 visits per subject at t = 0, 1, 2
    subject = np.repeat(np.arange(N), 3)
    time_long = np.tile([0, 1, 2], N).astype(float)
    b0 = rng.normal(0, 1, N); b1 = rng.normal(0, 0.5, N)
    y_long = 2 + 0.5 * time_long + b0[subject] + b1[subject] * time_long + rng.normal(0, 0.5, len(subject))
    # Survival: hazard depends on true trajectory at the event time
    # h(t) = 0.05 * exp(alpha * m_i(t)); alpha = 0.8 true
    alpha_true = 0.8
    time_event = np.zeros(N); event = np.ones(N, dtype=int)
    for i in range(N):
        u = rng.uniform()
        # Integrated hazard = 0.05 * exp(alpha (2 + b0_i)) * (exp(alpha (0.5 + b1_i) t) - 1) / (alpha (0.5 + b1_i))
        # Approximate by numerical inversion via grid search
        grid = np.arange(0, 20, 0.05)
        m_vals = 2 + 0.5 * grid + b0[i] + b1[i] * grid
        cum = np.cumsum(0.05 * np.exp(alpha_true * m_vals) * 0.05)
        t_i = grid[np.searchsorted(cum, -math.log(u))] if -math.log(u) < cum[-1] else grid[-1]
        c_i = rng.uniform(0, 10)  # censoring
        if c_i < t_i: time_event[i] = c_i; event[i] = 0
        else: time_event[i] = t_i; event[i] = 1

    print(f"=== N = {N}, events = {int(event.sum())} ===")

    print("\n=== Naive Cox on subject baseline y(0) ===")
    y0 = np.array([y_long[subject == i][0] for i in range(N)])
    r_naive = _cox_partial_like(time_event, event,
                                lambda i, t: y0[i])
    print(f"  alpha_naive = {r_naive['alpha']:.3f}, SE = {r_naive['se']:.3f}, p = {r_naive['p']:.4f}")

    print("\n=== Two-stage joint (LME + Cox with time-varying m_hat) ===")
    subject_surv = np.arange(N)
    r = two_stage_joint(subject, time_long, y_long, subject_surv, time_event, event)
    print(f"  LME fixed intercept, slope = {r['lme_beta'].round(3)}   (true 2.0, 0.5)")
    print(f"  Cox alpha_hat = {r['cox_alpha']:.3f}, SE = {r['cox_se']:.3f}, p = {r['cox_p']:.4f}  (true {alpha_true})")

    print("\n--- library cross-check (JM in R; JMbayes; pyjm) ---")
    print("  Python has no first-class joint-model package; use JM::jointModel or")
    print("  JMbayes2::jm in R for the full joint MLE.")
