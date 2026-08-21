"""Mendelian randomization (Reference §15.x extra).

MR uses genetic variants (SNPs) as instrumental variables for an exposure X
to estimate the causal effect on outcome Y under the classical IV assumptions:
    (i)  relevance: SNP -> X
    (ii) exchangeability: SNP _|_ confounders
    (iii) exclusion: SNP affects Y only through X.

Given per-SNP summary statistics
    beta_X_k = SNP_k -> X effect,   SE_X_k
    beta_Y_k = SNP_k -> Y effect,   SE_Y_k
the per-SNP Wald ratio is
    r_k = beta_Y_k / beta_X_k
and the causal effect estimators are:

  * IVW           = weighted mean of r_k with weights 1 / (SE_Y_k / beta_X_k)^2.
  * MR-Egger      = intercept + slope regression Y ~ X across SNPs;
                    a non-zero intercept flags directional pleiotropy.
  * Weighted median = median of r_k weighted by inverse variance; robust to
                       up to 50% invalid instruments.
  * Weighted mode  = mode-based estimator; robust to plurality-valid IVs.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def ivw(beta_X, beta_Y, se_Y):
    beta_X = np.asarray(beta_X, dtype=float); beta_Y = np.asarray(beta_Y, dtype=float)
    se_Y = np.asarray(se_Y, dtype=float)
    w = (beta_X ** 2) / (se_Y ** 2)                        # 1 / (SE_Y / bX)^2
    r = beta_Y / beta_X
    est = float((w * r).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    return {"estimate": est, "se": se}


def mr_egger(beta_X, beta_Y, se_Y):
    """Weighted regression of beta_Y on beta_X with intercept."""
    w = 1.0 / (np.asarray(se_Y, dtype=float) ** 2)
    X = np.column_stack([np.ones_like(beta_X), beta_X])
    WX = X * w[:, None]
    beta_hat = np.linalg.solve(X.T @ WX, X.T @ (w * beta_Y))
    resid = beta_Y - X @ beta_hat
    sigma2 = float((w * resid ** 2).sum() / (len(beta_X) - 2))
    cov = sigma2 * np.linalg.inv(X.T @ WX)
    return {"intercept": float(beta_hat[0]),
            "intercept_se": float(np.sqrt(cov[0, 0])),
            "slope": float(beta_hat[1]),
            "slope_se": float(np.sqrt(cov[1, 1]))}


def weighted_median(beta_X, beta_Y, se_Y):
    r = np.asarray(beta_Y, dtype=float) / np.asarray(beta_X, dtype=float)
    w = (np.asarray(beta_X, dtype=float) ** 2) / (np.asarray(se_Y, dtype=float) ** 2)
    order = np.argsort(r)
    rs = r[order]; ws = w[order]
    cw = np.cumsum(ws) - 0.5 * ws
    cw = cw / ws.sum()
    # linear interpolation to find r at cw = 0.5
    below = np.where(cw <= 0.5)[0][-1]
    above = below + 1 if below + 1 < len(cw) else below
    if below == above:
        return {"estimate": float(rs[below])}
    x0, x1 = cw[below], cw[above]
    return {"estimate": float(rs[below] + (rs[above] - rs[below]) *
                              (0.5 - x0) / (x1 - x0))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 40                                                 # number of SNPs
    causal_effect = 0.5
    beta_X = np.abs(rng.normal(0.15, 0.05, K))             # instrument strengths
    se_X = np.full(K, 0.02)
    # perfect IV: beta_Y = causal * beta_X + noise
    beta_Y_valid = causal_effect * beta_X + rng.normal(scale=0.02, size=K)
    se_Y = np.full(K, 0.02)

    # scenario 2: 20% of SNPs have direct (pleiotropic) effects on Y
    pleiotropy = np.zeros(K)
    invalid = rng.choice(K, size=K // 5, replace=False)
    pleiotropy[invalid] = rng.normal(loc=0.10, scale=0.03, size=len(invalid))    # positive shift
    beta_Y_pleio = causal_effect * beta_X + pleiotropy + rng.normal(scale=0.02, size=K)

    for name, bY in [("valid IVs (no pleiotropy)", beta_Y_valid),
                     ("20% invalid IVs (directional pleiotropy)", beta_Y_pleio)]:
        print(f"\n=== {name} (true causal = {causal_effect}) ===")
        w = ivw(beta_X, bY, se_Y)
        e = mr_egger(beta_X, bY, se_Y)
        med = weighted_median(beta_X, bY, se_Y)
        print(f"  IVW              = {w['estimate']:+.3f} (SE {w['se']:.3f})")
        print(f"  MR-Egger slope   = {e['slope']:+.3f} (SE {e['slope_se']:.3f})   "
              f"intercept = {e['intercept']:+.3f} (SE {e['intercept_se']:.3f})")
        print(f"  weighted median  = {med['estimate']:+.3f}")

    print("\n--- library cross-check (R MendelianRandomization::mr_ivw / mr_egger / mr_median) ---")
