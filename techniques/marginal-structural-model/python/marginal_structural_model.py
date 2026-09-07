"""Marginal Structural Model (Reference Sec 15.19).

Robins-Hernan-Brumback 2000.  For TIME-VARYING treatment with TIME-
VARYING CONFOUNDING that is also affected by past treatment,
standard adjustment (regression, matching) is BIASED.

MSM procedure:
  1. Fit a TREATMENT model at each time  t:  P(A_t | history).
  2. IPTW = product of inverse treatment probabilities over t.
     Stabilised IPTW divides numerator by marginal probability.
  3. Fit the outcome model on the WEIGHTED pseudo-population; the
     estimated coefficients are marginal (population-averaged)
     causal effects.

Compact demo: 2 time points, 1 outcome, treatment-confounder
feedback.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression, LinearRegression


def stabilised_iptw_2tp(A0, A1, L0, L1):
    """2-time-point stabilised IPTW weights."""
    # Time 0
    m0 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(L0[:, None], A0)
    p0 = m0.predict_proba(L0[:, None])[:, 1]
    p0_num = np.full_like(p0, A0.mean())
    # Time 1: model conditional on L0, A0, L1
    X1 = np.column_stack([L0, A0, L1])
    m1 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(X1, A1)
    p1 = m1.predict_proba(X1)[:, 1]
    # Numerator conditional on A0 only (Robins stabilisation)
    p1_num_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(A0[:, None], A1)
    p1_num = p1_num_model.predict_proba(A0[:, None])[:, 1]

    def _prob(y, p): return np.where(y == 1, p, 1 - p)
    w = _prob(A0, p0_num) * _prob(A1, p1_num) / (_prob(A0, p0) * _prob(A1, p1))
    return np.clip(w, 0.05, 20)


if __name__ == "__main__":
    print("=== Marginal Structural Model with stabilised IPTW ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    # Time 0
    L0 = rng.normal(0, 1, n)
    p_A0 = 1 / (1 + np.exp(-(0.4 * L0)))
    A0 = (rng.random(n) < p_A0).astype(int)
    # Time 1 confounder depends on A0 (treatment-confounder feedback)
    L1 = 0.3 * L0 - 0.5 * A0 + rng.normal(0, 1, n)
    p_A1 = 1 / (1 + np.exp(-(0.4 * L1 + 0.5 * A0)))
    A1 = (rng.random(n) < p_A1).astype(int)
    # Outcome: sum of treatments has true marginal effect 0.4 per period
    y = 1.0 + 0.4 * (A0 + A1) + 0.3 * L0 + 0.2 * L1 + rng.normal(0, 1, n)

    # Naive: regress y on (A0, A1, L0, L1) -> conditional effect (biased if collider on L1)
    Xn = np.column_stack([A0, A1, L0, L1])
    from numpy.linalg import lstsq
    beta_naive, *_ = lstsq(np.column_stack([np.ones(n), Xn]), y, rcond=None)
    print(f"  Naive OLS coef (A0, A1): {beta_naive[1]:+.3f}, {beta_naive[2]:+.3f}"
          f"   (biased if L1 is a mediator/collider)")

    # MSM: IPTW pseudo-population + outcome model on (A0, A1) only
    w = stabilised_iptw_2tp(A0, A1, L0, L1)
    X_msm = np.column_stack([np.ones(n), A0, A1])
    W = np.diag(w)
    beta_msm = np.linalg.solve(X_msm.T @ W @ X_msm, X_msm.T @ W @ y)
    print(f"  MSM   IPTW coef (A0, A1): {beta_msm[1]:+.3f}, {beta_msm[2]:+.3f}   (true 0.4 each)\n")

    print("--- library cross-check (R ipw + geeglm; Python zepid.causal.gformula.MSMIPTW) ---")
