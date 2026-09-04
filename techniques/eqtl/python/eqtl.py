"""eQTL analysis (Reference Sec 40.16).

Expression quantitative trait locus: test whether a SNP explains a
gene's expression.  Cis-eQTL (within ~1Mb of the gene) vs trans-eQTL
(distal or on different chromosome).

MatrixEQTL (Shabalin 2012) vectorises SNP x gene linear regression.
Here we implement a compact per-SNP per-gene linear model, report
top cis pairs by p, and apply BH-FDR across the pairs tested.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # t + BH


def matrix_eqtl(G, Y, cis_map=None):
    """Vectorised SNP x gene regression.

    G       : (n_samples, m_snps)
    Y       : (n_samples, g_genes)
    cis_map : optional list of (snp_idx, gene_idx) pairs; restricts tests.
    """
    n, m = G.shape; _, g = Y.shape
    Gm = G - G.mean(axis=0); Ym = Y - Y.mean(axis=0)
    if cis_map is None:
        cis_map = [(s, g_) for s in range(m) for g_ in range(g)]
    beta, se, p = [], [], []
    for s, g_ in cis_map:
        gsnp = Gm[:, s]; y = Ym[:, g_]
        Sxx = (gsnp ** 2).sum()
        if Sxx == 0:
            beta.append(0.0); se.append(np.inf); p.append(1.0); continue
        b = (gsnp * y).sum() / Sxx
        yhat = b * gsnp
        sig2 = ((y - yhat) ** 2).sum() / (n - 2)
        s_e = np.sqrt(sig2 / Sxx)
        t = b / s_e
        p.append(2 * stats.t.sf(np.abs(t), df=n - 2))
        beta.append(b); se.append(s_e)
    return np.array(beta), np.array(se), np.array(p), cis_map


def bh_fdr(p):
    order = np.argsort(p); p_s = p[order]
    m = len(p)
    q_s = np.minimum.accumulate((p_s * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty_like(q_s); q[order] = q_s
    return q


if __name__ == "__main__":
    print("=== eQTL: per SNP-gene linear regression (MatrixEQTL-style) ===\n")
    rng = np.random.default_rng(0)
    n = 300
    m_snps = 40; g_genes = 20
    mafs = rng.uniform(0.1, 0.5, m_snps)
    G = np.stack([rng.binomial(2, p, n) for p in mafs], axis=1).astype(float)

    # Baseline expression + 3 true cis-eQTLs (SNP -> gene)
    Y = rng.normal(0, 1, (n, g_genes))
    truth = [(3, 5, 0.6), (10, 12, -0.5), (25, 18, 0.7)]
    for s, g_, effect in truth:
        Y[:, g_] += effect * G[:, s]

    # Restrict tests to a "cis window" of +/- 3 SNPs per gene
    cis_map = [(s, g_) for g_ in range(g_genes)
               for s in range(max(0, g_ - 3), min(m_snps, g_ + 4))]

    beta, se, p, pairs = matrix_eqtl(G, Y, cis_map=cis_map)
    q = bh_fdr(p)
    hits = [(pairs[i], float(beta[i]), float(p[i]), float(q[i]))
            for i in np.argsort(p)[:6]]
    print(f"  n_samples = {n}, m_snps = {m_snps}, g_genes = {g_genes}")
    print(f"  Cis-window tests = {len(cis_map)}")
    print(f"  Top hits by p (SNP -> gene):")
    for pair, b, p_, q_ in hits:
        print(f"    SNP {pair[0]:>3d} -> gene {pair[1]:>3d}   beta = {b:+.3f}   p = {p_:.2e}   qBH = {q_:.3f}")

    print("\n--- library cross-check (R MatrixEQTL, QTLtools; Python tensorqtl, hail) ---")
