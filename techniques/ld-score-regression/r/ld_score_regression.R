# LD score regression (Reference Sec 40.17)
# Native Python via ldsc (Broad); R via GenomicSEM / bigsnpr.
# Run with:  Rscript ld_score_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  GenomicSEM                     -- multivariate LDSC + genomic-SEM models\n")
  cat("  bigsnpr::snp_ldsc              -- LDSC on GWAS summary stats\n")
  cat("  GCTA (external)                -- GREML / bivariate heritability\n")
  cat("Python:\n")
  cat("  ldsc (Broad, munge_sumstats.py + ldsc.py)\n")
  cat("  hail::ld_score_regression      -- distributed LDSC\n")
  cat("  custom                          -- reference weighted regression\n")
  cat("Refs: Bulik-Sullivan et al. (2015) 'LD score regression distinguishes\n")
  cat("      confounding from polygenicity in GWAS', Nat Genet; Yang et al. (2010)\n")
  cat("      'Common SNPs explain a large proportion of the heritability for human\n")
  cat("      height', Nat Genet.\n")
}
