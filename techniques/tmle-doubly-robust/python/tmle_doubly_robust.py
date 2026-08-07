"""Targeted Maximum Likelihood Estimation (TMLE) + AIPW (Reference §15.11).

Doubly-robust semiparametric estimator of the average treatment effect (ATE)
under strong ignorability.

Setup
    Y (outcome), A (binary treatment), W (confounders)
    True targets:
        Q_bar(A, W) = E[Y | A, W]              outcome regression
        g(W)        = Pr(A = 1 | W)             propensity score
    ATE = E[Q_bar(1, W) - Q_bar(0, W)]

AIPW / DR estimator (Robins-Rotnitzky-Zhao 1994)
    hat_ATE_AIPW = mean(
        Q_bar(1, W) - Q_bar(0, W)
      + A (Y - Q_bar(1, W)) / g(W)
      - (1 - A) (Y - Q_bar(0, W)) / (1 - g(W))
    )
    Doubly-robust: consistent if EITHER Q_bar OR g is correctly specified.

TMLE (van der Laan-Rubin 2006)
    Uses a TARGETING STEP: after obtaining Q_bar and g, update Q_bar to
    solve the efficient influence function equation.  Fluctuation submodel:
        logit Q_bar^*(A, W) = logit Q_bar(A, W) + eps * H(A, W)
        H(A, W) = A / g(W) - (1 - A) / (1 - g(W))
    Fit eps by weighted logistic (or linear for continuous Y).
    Update Q_bar; recompute ATE using plug-in formula:
        hat_ATE_TMLE = mean(Q_bar^*(1, W) - Q_bar^*(0, W))

Both share the double-robustness property; TMLE is a plug-in estimator
that respects the boundaries of the parameter space.

The demo below implements AIPW and a simple linear-Y TMLE.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _fit_ps(W, A):
    """Logistic propensity."""
    def neg_ll(beta):
        z = W @ beta
        return -np.sum(A * z - np.logaddexp(0, z))
    res = minimize(neg_ll, np.zeros(W.shape[1]), method="BFGS")
    return 1 / (1 + np.exp(-(W @ res.x)))


def _fit_outcome(W, A, Y):
    """OLS Q_bar(A, W) with a treatment-covariate interaction (main + T-interaction)."""
    X = np.column_stack([W, A[:, None], (W * A[:, None])])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    def predict(A_new, W_new=W):
        X_new = np.column_stack([W_new, A_new[:, None], (W_new * A_new[:, None])])
        return X_new @ beta
    return predict


def aipw_ate(W, A, Y, g_clip: float = 0.01) -> dict:
    """AIPW / doubly-robust ATE."""
    W = np.asarray(W, dtype=float); A = np.asarray(A, dtype=int); Y = np.asarray(Y, dtype=float)
    g = np.clip(_fit_ps(W, A), g_clip, 1 - g_clip)
    Q_hat = _fit_outcome(W, A.astype(float), Y)
    Q1 = Q_hat(np.ones(len(A))); Q0 = Q_hat(np.zeros(len(A)))
    dr = Q1 - Q0 + A * (Y - Q1) / g - (1 - A) * (Y - Q0) / (1 - g)
    ate = float(dr.mean())
    se = float(np.std(dr, ddof=1) / math.sqrt(len(dr)))
    return {"ATE_AIPW": ate, "SE": se,
            "ci_95": (ate - 1.96 * se, ate + 1.96 * se),
            "mean_Q1": float(Q1.mean()), "mean_Q0": float(Q0.mean()),
            "method": "AIPW / doubly-robust ATE"}


def tmle_ate(W, A, Y, g_clip: float = 0.01) -> dict:
    """TMLE for continuous Y (linear fluctuation submodel)."""
    W = np.asarray(W, dtype=float); A = np.asarray(A, dtype=int); Y = np.asarray(Y, dtype=float)
    g = np.clip(_fit_ps(W, A), g_clip, 1 - g_clip)
    Q_hat = _fit_outcome(W, A.astype(float), Y)
    Q_A = Q_hat(A.astype(float))
    H = A / g - (1 - A) / (1 - g)   # clever covariate
    # Fluctuation: Y ~ Q_A + eps * H   -> OLS on residual
    eps = float((H @ (Y - Q_A)) / (H @ H))
    Q1_star = Q_hat(np.ones(len(A))) + eps * (1 / g)
    Q0_star = Q_hat(np.zeros(len(A))) - eps * (1 / (1 - g))
    ate = float(np.mean(Q1_star - Q0_star))
    # Influence-function SE
    IC = (Q1_star - Q0_star - ate
          + A * (Y - Q1_star) / g - (1 - A) * (Y - Q0_star) / (1 - g))
    se = float(np.std(IC, ddof=1) / math.sqrt(len(IC)))
    return {"ATE_TMLE": ate, "SE": se, "epsilon": eps,
            "ci_95": (ate - 1.96 * se, ate + 1.96 * se),
            "method": "TMLE (continuous Y, linear fluctuation)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000
    w1 = rng.normal(size=n); w2 = rng.normal(size=n)
    logit_e = -0.5 + 1.0 * w1 - 0.5 * w2
    A = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_e))).astype(int)
    Y = 1 + 2 * A + 0.7 * w1 - 0.4 * w2 + 0.3 * A * w1 + rng.normal(0, 1, n)
    W = np.column_stack([np.ones(n), w1, w2])

    print(f"=== ATE estimation (n = {n}, true ATE ~ 2 + 0.3 * E[w1] = 2.0) ===")
    print("\n=== Naive difference in means (biased) ===")
    print(f"  Y_treat - Y_control = {Y[A == 1].mean() - Y[A == 0].mean():.3f}")

    print("\n=== AIPW / doubly-robust ATE ===")
    r = aipw_ate(W, A, Y)
    print(f"  ATE_AIPW = {r['ATE_AIPW']:.3f}   SE = {r['SE']:.3f}")
    print(f"  95% CI   = ({r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f})")

    print("\n=== TMLE ATE ===")
    r = tmle_ate(W, A, Y)
    print(f"  ATE_TMLE = {r['ATE_TMLE']:.3f}   SE = {r['SE']:.3f}   epsilon = {r['epsilon']:.4f}")
    print(f"  95% CI   = ({r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f})")

    print("\n--- library cross-check (R tmle::tmle; Python zEpid, econml) ---")
    print("  R: tmle::tmle(Y = Y, A = A, W = data.frame(w1, w2))")
