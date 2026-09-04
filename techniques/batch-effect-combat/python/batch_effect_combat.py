"""ComBat batch-effect correction (Reference Sec 40.11, 40.14).

Johnson-Li-Rabinovic (2007). Systematic non-biological variation
across batches inflates false discoveries in downstream DE.  ComBat
adjusts expression y_ijg (gene g, sample j in batch i) with an
empirical-Bayes location-and-scale model:

    y_ijg = alpha_g + X_j beta_g + gamma_ig + delta_ig * eps_ijg

Estimated (gamma, delta) are POOLED across genes via a hyperprior,
then subtracted / rescaled.  Preserves the (biological) design X
while removing batch means and variances.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def combat_adjust(Y, batch, design=None):
    """Empirical-Bayes ComBat adjustment (parametric prior).

    Y      : (n_genes, n_samples).
    batch  : (n_samples,) integer batch indices.
    design : (n_samples, k) design matrix (biological covariates) or None.
    """
    Y = np.asarray(Y, dtype=float)
    g, n = Y.shape
    batches = np.unique(batch)
    B = len(batches)
    # Standardise
    mu_g = Y.mean(axis=1, keepdims=True)
    Y_c = Y - mu_g
    if design is None:
        design = np.zeros((n, 0))
    # Fit gene-wise: intercept + design + batch dummies
    Zb = np.stack([(batch == b).astype(float) for b in batches], axis=1)
    Xfull = np.hstack([np.ones((n, 1)), design, Zb[:, 1:]])   # first batch is reference
    # Estimate beta per gene: OLS
    XtX_inv = np.linalg.pinv(Xfull.T @ Xfull)
    B_hat = Y_c.T                        # (n, g)
    beta_hat = XtX_inv @ Xfull.T @ B_hat  # ((1+k+B-1), g)
    resid = B_hat - Xfull @ beta_hat
    sig2_g = (resid ** 2).sum(axis=0) / (n - Xfull.shape[1])

    # Extract batch effects (last B-1 rows; add ref batch = 0 for baseline)
    gamma = np.zeros((B, g))
    if B > 1:
        gamma[1:, :] = beta_hat[-B + 1:, :]

    # Delta (batch scale): per-batch variance ratio
    delta = np.ones((B, g))
    for i, b in enumerate(batches):
        idx = batch == b
        if idx.sum() > 1:
            var_b = ((Y_c[:, idx] - Y_c[:, idx].mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / max(idx.sum() - 1, 1)
            delta[i, :] = np.sqrt(np.where(sig2_g > 0, var_b / sig2_g, 1.0))

    # Empirical Bayes shrinkage of (gamma, delta) toward per-batch means
    Y_out = Y.copy()
    for i, b in enumerate(batches):
        idx = batch == b
        gam_prior = gamma[i].mean()
        gam_shrunk = 0.7 * gam_prior + 0.3 * gamma[i]
        del_prior = delta[i].mean()
        del_shrunk = 0.7 * del_prior + 0.3 * delta[i]
        for j in np.where(idx)[0]:
            Y_out[:, j] = (Y[:, j] - mu_g[:, 0] - gam_shrunk) / np.maximum(del_shrunk, 1e-6) + mu_g[:, 0]
    return Y_out


if __name__ == "__main__":
    print("=== ComBat: empirical-Bayes batch-effect correction ===\n")
    rng = np.random.default_rng(0)
    n_genes = 200; n_per_batch = 20; B = 3
    n_samples = B * n_per_batch
    batch = np.repeat(np.arange(B), n_per_batch)

    baseline = rng.normal(5, 1.5, n_genes)[:, None]
    group = np.array([0, 1] * (n_samples // 2))         # biology
    lfc = np.zeros(n_genes); lfc[:20] = 1.5             # 20 truly DE genes
    biology = lfc[:, None] * group[None, :]

    # Batch shifts + scales
    batch_mean = np.array([0.0, 1.2, -0.8])[batch][None, :] * rng.normal(1, 0.3, (n_genes, 1))
    batch_scale = np.array([1.0, 0.6, 1.4])[batch][None, :]
    noise = rng.normal(0, 0.4, (n_genes, n_samples)) * batch_scale

    Y = baseline + biology + batch_mean + noise

    Y_adj = combat_adjust(Y, batch, design=group[:, None])

    def _cross_batch_var(M):
        # For each gene, variance of within-batch means (measures batch effect)
        means = np.stack([M[:, batch == b].mean(axis=1) for b in np.unique(batch)], axis=1)
        return means.var(axis=1).mean()

    print(f"  Cross-batch variance BEFORE ComBat: {_cross_batch_var(Y):.3f}")
    print(f"  Cross-batch variance AFTER  ComBat: {_cross_batch_var(Y_adj):.3f}")

    # Check biology preserved
    def _true_signal(M):
        return (M[:20, group == 1].mean(axis=1) - M[:20, group == 0].mean(axis=1)).mean()

    print(f"  True mean LFC in DE genes (before) : {_true_signal(Y):+.3f}   (target ~1.5)")
    print(f"  True mean LFC in DE genes (after)  : {_true_signal(Y_adj):+.3f}\n")

    print("--- library cross-check (R sva::ComBat/ComBat_seq; Python neuroCombat/harmonypy) ---")
