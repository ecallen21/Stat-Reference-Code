"""Genome-wide association studies (Reference Sec 40.1, 40.23).

Per-SNP regression of a phenotype on genotype (additive model,
minor-allele count 0/1/2) for M variants.  Two universal QC / analysis
outputs:

  * PER-SNP p-value + effect size.
  * Genome-wide significance threshold p < 5e-8 (Bonferroni ~ 1e6
    independent tests).
  * Genomic-control lambda = median(chi^2) / 0.4549 -- inflation of
    the null; lambda > 1.05 suggests population structure or cryptic
    relatedness.

Manhattan plot (chr position vs -log10 p) and Q-Q plot (observed vs
expected -log10 p) are the visual summaries.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # chi2 for genomic control


def gwas_scan(G, y):
    """Vectorised linear-regression scan over M SNPs (columns of G).

    Returns per-SNP beta, SE, p, chi^2, plus genomic inflation lambda.
    """
    n, m = G.shape
    # Standardise y for beta interpretability
    Gm = G - G.mean(axis=0)
    Gv = (Gm ** 2).sum(axis=0)
    Gv = np.where(Gv > 0, Gv, np.nan)
    beta = (Gm * (y - y.mean())[:, None]).sum(axis=0) / Gv
    resid = y[:, None] - (beta * G) - (y.mean() - beta * G.mean(axis=0))
    sigma2 = (resid ** 2).sum(axis=0) / (n - 2)
    se = np.sqrt(sigma2 / Gv)
    z = beta / se
    chi2 = z ** 2
    p = 2 * stats.norm.sf(np.abs(z))
    lam = float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))
    return {"beta": beta, "se": se, "chi2": chi2, "p": p,
            "genomic_lambda": lam, "n_snps": m, "n_samples": n}


if __name__ == "__main__":
    print("=== GWAS: per-SNP regression + genomic inflation lambda ===\n")
    rng = np.random.default_rng(0)
    n = 800; m = 500
    mafs = rng.uniform(0.05, 0.5, m)
    G = np.stack([rng.binomial(2, p, n) for p in mafs], axis=1).astype(float)

    # 3 true causal SNPs, small effects; rest null
    causal = [50, 200, 400]
    beta = np.zeros(m); beta[causal] = [0.35, -0.28, 0.30]
    y = G @ beta + rng.normal(0, 1, n)

    res = gwas_scan(G, y)
    threshold = 5e-8       # canonical genome-wide
    thr_used = 0.05 / m    # Bonferroni for this small demo
    sig = np.where(res["p"] < thr_used)[0]

    print(f"  n samples = {n}, m SNPs = {m}")
    print(f"  Genomic inflation lambda = {res['genomic_lambda']:.3f}   (target ~1.0)")
    print(f"  Bonferroni threshold (0.05 / m) = {thr_used:.2e}")
    print(f"  True causal SNPs        : {causal}")
    print(f"  Sig hits above threshold: {sig.tolist()}")
    for i in causal:
        print(f"    SNP {i}: beta = {res['beta'][i]:+.3f}   p = {res['p'][i]:.2e}")

    print("\n--- library cross-check (R qqman/GENESIS; Python hail/pandas-plink) ---")
