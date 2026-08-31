# High-dim covariance estimation (Reference Sec 32.11)
# Native R via corpcor / CovTools; Python via sklearn.
# Run with:  Rscript covariance_estimation_highdim.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  corpcor                      -- Schafer-Strimmer shrinkage covariance\n")
  cat("  CovTools                     -- banded / tapered / thresholded covariance\n")
  cat("  glasso, huge                 -- sparse precision matrix\n")
  cat("  spcov                         -- Xue-Ma-Zou sparse covariance L1 penalty\n")
  cat("Python:\n")
  cat("  sklearn.covariance.LedoitWolf / OAS / ShrunkCovariance\n")
  cat("  sklearn.covariance.GraphicalLasso  (sparse precision)\n")
  cat("Refs: Ledoit, O. & Wolf, M. (2004) 'A well-conditioned estimator for\n")
  cat("      large-dimensional covariance matrices', JMVA;\n")
  cat("      Bickel, P.J. & Levina, E. (2008) 'Regularized estimation of large\n")
  cat("      covariance matrices', Annals of Statistics.\n")
}
