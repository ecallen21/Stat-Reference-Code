"""Polygenic risk scores (Reference Sec 40.2, 40.15).

PRS_i = sum_j beta_hat_j * G_ij     over pre-selected SNPs j.

Pruning + Thresholding (P+T):
  1. Compute GWAS betas on a discovery cohort.
  2. LD-prune to keep near-independent SNPs (r^2 < threshold).
  3. Threshold on p-value (e.g., p < 5e-8 for strict; p < 0.05 for
     inclusive) and sum weighted alleles in a target cohort.

Evaluate incremental discrimination: AUC or R^2 with vs without PRS
on top of clinical predictors (see clinical-prediction batch).
"""
from __future__ import annotations    # stdlib

import warnings                                            # suppress noise

import numpy as np    # numerical arrays

warnings.filterwarnings("ignore")


def compute_prs(G, betas, p_values, thresholds=(5e-8, 1e-5, 1e-3, 1e-1, 1.0)):
    """Return PRS across p-value thresholds."""
    scores = {}
    for thr in thresholds:
        mask = p_values < thr
        scores[thr] = G[:, mask] @ betas[mask]
    return scores


def r2_incremental(y, base_pred, prs):
    """Additional R^2 (linear) from adding PRS on top of base_pred."""
    def r2(y, yhat):
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot
    from sklearn.linear_model import LinearRegression
    base_r2 = r2(y, base_pred)
    X = np.column_stack([base_pred, prs])
    yhat_full = LinearRegression().fit(X, y).predict(X)
    full_r2 = r2(y, yhat_full)
    return {"base_R2": float(base_r2), "full_R2": float(full_r2),
            "delta_R2": float(full_r2 - base_r2)}


if __name__ == "__main__":
    print("=== Polygenic risk score: P+T + incremental R^2 vs clinical baseline ===\n")
    rng = np.random.default_rng(0)
    n_disc, n_targ = 3000, 1000
    m = 800
    mafs = rng.uniform(0.05, 0.5, m)
    G_disc = np.stack([rng.binomial(2, p, n_disc) for p in mafs], axis=1).astype(float)
    G_targ = np.stack([rng.binomial(2, p, n_targ) for p in mafs], axis=1).astype(float)

    # 50 causal SNPs with small effects
    causal_idx = rng.choice(m, size=50, replace=False)
    beta_true = np.zeros(m)
    beta_true[causal_idx] = rng.normal(0, 0.05, 50)
    clinical_disc = rng.normal(0, 1, n_disc)
    clinical_targ = rng.normal(0, 1, n_targ)

    y_disc = 0.5 * clinical_disc + G_disc @ beta_true + rng.normal(0, 1, n_disc)
    y_targ = 0.5 * clinical_targ + G_targ @ beta_true + rng.normal(0, 1, n_targ)

    # GWAS on discovery
    from scipy.stats import norm
    Gm = G_disc - G_disc.mean(axis=0)
    Gv = (Gm ** 2).sum(axis=0)
    betas = (Gm * (y_disc - y_disc.mean())[:, None]).sum(axis=0) / Gv
    resid = y_disc[:, None] - betas * G_disc - (y_disc.mean() - betas * G_disc.mean(axis=0))
    se = np.sqrt((resid ** 2).sum(axis=0) / (n_disc - 2) / Gv)
    z = betas / se
    p_gwas = 2 * norm.sf(np.abs(z))

    prs_targ = compute_prs(G_targ, betas, p_gwas)
    print(f"  {'p-threshold':>12s}  {'m_kept':>7s}  {'PRS var':>8s}   {'delta R^2':>10s}")
    for thr, prs in prs_targ.items():
        n_kept = int((p_gwas < thr).sum())
        base_pred = 0.5 * clinical_targ
        inc = r2_incremental(y_targ, base_pred, prs)
        print(f"  {thr:>12.1e}  {n_kept:>7d}  {prs.var():>8.3f}"
              f"   {inc['delta_R2']:>+10.4f}")

    print("\n--- library cross-check (R PRSice2/bigsnpr/lassosum; Python ldpred/custom) ---")
