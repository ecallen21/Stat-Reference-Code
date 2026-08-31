# Probabilistic PCA (Reference Sec 25.10)
# Native R via pcaMethods; Python via sklearn.
# Run with:  Rscript probabilistic_pca.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pcaMethods (Bioconductor)    -- PPCA + BPCA + NIPALS with missing-value support\n")
  cat("  MASS::mvrnorm                 -- adjacent: multivariate normal sampling\n")
  cat("Python:\n")
  cat("  sklearn.decomposition.PCA     -- basic PCA (special case sigma^2 -> 0)\n")
  cat("  probabilistic-pca (pip pkg)   -- EM PPCA + Bayesian variants\n")
  cat("  scikit-learn IterativeImputer -- EM-based imputation cousin\n")
  cat("Refs: Tipping, M.E. & Bishop, C.M. (1999) 'Probabilistic principal component\n")
  cat("      analysis', JRSS-B; Bishop, C.M. (1999) 'Bayesian PCA', NeurIPS.\n")
}
