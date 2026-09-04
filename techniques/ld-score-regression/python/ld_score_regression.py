"""LD score regression (Reference Sec 40.17).

Bulik-Sullivan et al. 2015.  From GWAS summary statistics (chi^2 per
SNP) and an LD reference panel:

    E[ chi^2_j ] = N * h2_g / M * l_j + N * a + 1

  where l_j = LD score = sum_k r^2_jk over SNPs k in LD with j,
  M = # SNPs, N = sample size, h2_g = SNP heritability, a = confounding
  bias inflation.

Regressing chi^2 on l_j:
  * SLOPE  -> proportional to h2_g.
  * INTERCEPT -> 1 + N * a.  > 1 indicates confounding (pop.
    stratification, cryptic relatedness), not polygenicity.

This DISTINGUISHES confounding from true polygenic signal in GWAS
inflation lambda -- landmark contribution.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ldsc(chi2, ld_scores, N, M):
    """Weighted OLS regression: chi^2 ~ N * h2/M * l + intercept."""
    y = chi2; x = ld_scores
    # Weights ~ 1 / (1 + N * h2/M * l)^2 iterated once; here use unit weights.
    A = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    intercept = float(beta[0])
    slope = float(beta[1])
    h2 = slope * M / N
    return {"intercept": intercept, "slope": slope, "h2": h2}


if __name__ == "__main__":
    print("=== LD score regression: h^2 + confounding intercept ===\n")
    rng = np.random.default_rng(0)
    M = 5000; N = 20000
    l = rng.gamma(2.5, 20, M)             # positive LD scores
    h2_true = 0.30                        # SNP heritability
    a_true = 0.002                        # small confounding

    mean_chi2 = 1 + N * a_true + N * h2_true * l / M
    chi2 = rng.chisquare(df=1) * mean_chi2 / 1.0
    # simulate correlated non-centrality
    chi2 = rng.noncentral_chisquare(df=1, nonc=N * h2_true * l / M) + N * a_true + 1

    r = ldsc(chi2, l, N=N, M=M)
    print(f"  True h2 = {h2_true:.3f}   estimated h2 = {r['h2']:.3f}")
    print(f"  True intercept 1 + N*a = {1 + N * a_true:.3f}   estimated = {r['intercept']:.3f}")
    print(f"  Slope = {r['slope']:.5f}\n")

    # Confounding-vs-polygenicity case: same lambda_GC but different intercept
    chi2_pop_strat = rng.noncentral_chisquare(df=1, nonc=1.0) + 0.02 * N
    print("  Genomic inflation (lambda_GC) alone cannot distinguish polygenicity")
    print("  from confounding.  LDSC intercept can:")
    print("    intercept ~= 1  -> genuine polygenic signal.")
    print("    intercept >> 1  -> confounding (population structure).\n")
    print("--- library cross-check (R GenomicSEM/bigsnpr; Python ldsc/hail) ---")
