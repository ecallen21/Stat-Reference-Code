# Debiased LASSO (Reference Sec 32.4)
# Native R via hdi; Python via reticulate.
# Run with:  Rscript debiased_lasso.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  hdi::lasso.proj              -- Zhang-Zhang / van-de-Geer debiased LASSO CIs\n")
  cat("  hdi::boot.lasso.proj         -- residual bootstrap version\n")
  cat("  selectiveInference           -- post-selection inference after LASSO\n")
  cat("Python:\n")
  cat("  celer / hdlasso               -- fast implementations\n")
  cat("  statsmodels QuantReg + BH     -- adjacent selection inference\n")
  cat("Refs: Zhang, C.-H. & Zhang, S. (2014) 'Confidence intervals for low\n")
  cat("      dimensional parameters in high dimensional linear models', JRSS-B;\n")
  cat("      van de Geer, S. et al. (2014) 'On asymptotically optimal confidence\n")
  cat("      regions and tests for high-dimensional models', Annals of Statistics.\n")
}
