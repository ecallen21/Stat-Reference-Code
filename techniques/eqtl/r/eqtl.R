# eQTL analysis (Reference Sec 40.16)
# Native R via MatrixEQTL; Python tensorqtl + custom.
# Run with:  Rscript eqtl.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MatrixEQTL::Matrix_eQTL_main    -- ultra-fast SNP x gene regression\n")
  cat("  QTLtools (external CLI)         -- production pipeline w/ permutations\n")
  cat("  qvalue::qvalue                  -- Storey q-values across pairs\n")
  cat("Python:\n")
  cat("  tensorqtl                       -- GPU-accelerated cis / trans eQTL\n")
  cat("  hail::linear_regression_rows    -- distributed eQTL / GWAS\n")
  cat("  pandas-plink + statsmodels      -- ad-hoc analyses\n")
  cat("Refs: Shabalin, A.A. (2012) 'Matrix eQTL: ultra fast eQTL analysis via large\n")
  cat("      matrix operations', Bioinformatics; GTEx Consortium (2020) 'The GTEx\n")
  cat("      Consortium atlas of genetic regulatory effects across human tissues',\n")
  cat("      Science.\n")
}
