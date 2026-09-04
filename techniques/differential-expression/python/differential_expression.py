"""Differential expression analysis (Reference Sec 40.3).

Two paradigms for RNA-seq / microarray gene-level tests:

  MICROARRAY / normalised log-expression -> LINEAR MODEL + moderated
  t-statistic (limma-style empirical Bayes variance shrinkage).

  RNA-seq COUNTS -> NEGATIVE BINOMIAL GLM (DESeq2, edgeR) with
  dispersion shrinkage across genes.

Both borrow strength across genes to stabilise per-gene variance
estimates when replicates are few (typical: 3-8 per condition).

Here we implement the limma-style moderated t on log2-transformed
data and compare against the plain per-gene t-test.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # t, chi2


def moderated_t(X, group, prior_df=4.0):
    """Limma-style moderated t.

    X       : (n_genes, n_samples) expression matrix.
    group   : (n_samples,) binary group indicator.
    prior_df: empirical-Bayes degrees of freedom for prior on variance.
    """
    a = X[:, group == 0]; b = X[:, group == 1]
    na, nb = a.shape[1], b.shape[1]
    mean_diff = b.mean(axis=1) - a.mean(axis=1)
    va = a.var(axis=1, ddof=1); vb = b.var(axis=1, ddof=1)
    s2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    df = na + nb - 2
    # Empirical Bayes prior variance = mean(s2) with prior_df d.f.
    s2_prior = np.mean(s2)
    s2_post = (prior_df * s2_prior + df * s2) / (prior_df + df)
    df_post = prior_df + df
    se_mod = np.sqrt(s2_post * (1 / na + 1 / nb))
    t_mod = mean_diff / se_mod
    p_mod = 2 * stats.t.sf(np.abs(t_mod), df=df_post)

    # BH-FDR
    order = np.argsort(p_mod)
    p_sorted = p_mod[order]
    m = len(p_mod)
    q_sorted = np.minimum.accumulate((p_sorted * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty_like(q_sorted); q[order] = q_sorted
    return {"log2FC": mean_diff, "t_mod": t_mod, "p": p_mod, "qBH": q,
            "s2_prior": float(s2_prior)}


if __name__ == "__main__":
    print("=== Differential expression: limma-style moderated t + BH-FDR ===\n")
    rng = np.random.default_rng(0)
    n_genes = 500; n_per_grp = 5
    # 30 truly DE genes with log2FC = +/- 1.5
    de_idx = rng.choice(n_genes, size=30, replace=False)
    lfc_true = np.zeros(n_genes)
    lfc_true[de_idx] = rng.choice([-1.5, 1.5], size=30)

    baseline = rng.normal(6.0, 1.5, n_genes)     # log2 expression per gene
    X_a = baseline[:, None] + rng.normal(0, 0.4, (n_genes, n_per_grp))
    X_b = baseline[:, None] + lfc_true[:, None] + rng.normal(0, 0.4, (n_genes, n_per_grp))
    X = np.hstack([X_a, X_b])
    group = np.array([0] * n_per_grp + [1] * n_per_grp)

    # Plain (unmoderated) t
    a = X[:, group == 0]; b = X[:, group == 1]
    t_plain, p_plain = stats.ttest_ind(a, b, axis=1, equal_var=True)

    res = moderated_t(X, group)
    print(f"  n_genes = {n_genes}, n_per_group = {n_per_grp}, true DE = 30")
    print(f"  Empirical-Bayes prior variance = {res['s2_prior']:.3f}\n")

    def _summary(qs, name):
        sig = qs < 0.05
        tp = int(sig[de_idx].sum())
        fp = int(sig.sum() - tp)
        print(f"    {name:>18s}: {int(sig.sum()):>3d} calls   TP = {tp:>2d}   FP = {fp:>3d}")

    # BH for plain t
    p_plain_arr = np.asarray(p_plain)
    order = np.argsort(p_plain_arr); p_s = p_plain_arr[order]; m = len(p_s)
    q_s = np.minimum.accumulate((p_s * m / np.arange(1, m + 1))[::-1])[::-1]
    q_plain = np.empty_like(q_s); q_plain[order] = q_s
    _summary(q_plain, "plain t + BH")
    _summary(res["qBH"], "moderated t + BH")
    print("\n  Moderated t typically gains power at small replicate counts.\n")

    print("--- library cross-check (R limma::eBayes / DESeq2 / edgeR; Python pydeseq2) ---")
