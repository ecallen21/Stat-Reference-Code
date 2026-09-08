"""MR-Egger regression (Reference Sec 47.64).

Bowden, Davey Smith & Burgess 2015 'Mendelian randomization with
invalid instruments: effect estimation and bias detection through
Egger regression', Int J Epidemiol 44(2). Extends inverse-variance-
weighted (IVW) Mendelian randomization by including an INTERCEPT:

    beta_YG_j = alpha_0 + beta_MR * beta_XG_j + eps_j,   weighted by 1/se_YG_j^2

  * alpha_0 = 0  under the InSIDE assumption (no directional
    pleiotropy).
  * If alpha_0 != 0 the IVW estimate is biased by average
    horizontal pleiotropy; MR-Egger's slope beta_MR is still
    consistent.

Weaker assumption but wider CIs; run alongside IVW + MR-PRESSO.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ivw_mr(bx, by, se_by):
    """Inverse-variance-weighted MR (no intercept)."""
    w = 1.0 / se_by ** 2
    beta = float(np.sum(w * bx * by) / np.sum(w * bx * bx))
    se = float(np.sqrt(1.0 / np.sum(w * bx * bx)))
    return {"beta": beta, "se": se}


def mr_egger(bx, by, se_by):
    """Weighted-linear MR-Egger with intercept alpha_0."""
    w = 1.0 / se_by ** 2
    X = np.column_stack([np.ones_like(bx), bx])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ by
    coef = np.linalg.solve(XtWX, XtWy)
    resid = by - X @ coef
    sigma2 = float((w * resid ** 2).sum() / (len(bx) - 2))
    cov = sigma2 * np.linalg.inv(XtWX)
    return {"intercept": float(coef[0]), "beta": float(coef[1]),
            "se_intercept": float(np.sqrt(cov[0, 0])),
            "se_beta": float(np.sqrt(cov[1, 1]))}


if __name__ == "__main__":
    print("=== MR-Egger regression (Bowden-Davey Smith-Burgess 2015) ===\n")
    rng = np.random.default_rng(0)
    K = 40                                  # number of SNP instruments
    bx = rng.normal(0.10, 0.03, size=K)     # SNP-exposure effects
    se_bx = 0.005 * np.ones(K)              # tiny SE on exposure side
    se_by = 0.015 * np.ones(K)              # SE on outcome side

    beta_true = 0.30
    # Scenario 1: no pleiotropy -- IVW and MR-Egger agree
    by_null = beta_true * bx + rng.normal(0, se_by, K)
    ivw = ivw_mr(bx, by_null, se_by)
    egg = mr_egger(bx, by_null, se_by)
    print(f"  Scenario 1 (no pleiotropy):")
    print(f"    IVW      beta = {ivw['beta']:.3f} +/- {ivw['se']:.3f}")
    print(f"    MR-Egger beta = {egg['beta']:.3f} +/- {egg['se_beta']:.3f}"
          f"   alpha = {egg['intercept']:+.4f} +/- {egg['se_intercept']:.4f}")

    # Scenario 2: directional pleiotropy alpha0 = 0.02 across all instruments
    alpha0 = 0.02
    by_dir = alpha0 + beta_true * bx + rng.normal(0, se_by, K)
    ivw = ivw_mr(bx, by_dir, se_by)
    egg = mr_egger(bx, by_dir, se_by)
    print(f"\n  Scenario 2 (directional pleiotropy alpha_0 = {alpha0}):")
    print(f"    IVW      beta = {ivw['beta']:.3f}     (BIASED upward)")
    print(f"    MR-Egger beta = {egg['beta']:.3f}     (consistent under InSIDE)")
    print(f"    MR-Egger alpha= {egg['intercept']:+.4f} +/- {egg['se_intercept']:.4f}   "
          f"(detects pleiotropy)")

    print("\n--- library cross-check (MendelianRandomization / TwoSampleMR R) ---")
