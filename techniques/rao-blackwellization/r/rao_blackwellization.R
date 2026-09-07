# Rao-Blackwellization (Reference Sec 45.11)
# From-scratch technique; no dedicated package.
# Run with:  Rscript rao_blackwellization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- technique applied inside samplers\n")
  cat("  MCMCpack / rjags / rstan: Rao-Blackwellised estimators via conditional means\n")
  cat("  coda::mcmc + custom summary function\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see rao_blackwellization.py)\n")
  cat("  pymc, numpyro: use conditional means in posterior summaries\n")
  cat("  scikit-learn: expected-value replacements in ensembles (soft outputs)\n")
  cat("Refs: Rao, C.R. (1945) 'Information and the accuracy attainable in the\n")
  cat("      estimation of statistical parameters', Bull Calcutta Math Soc 37;\n")
  cat("      Blackwell, D. (1947) 'Conditional expectation and unbiased sequential\n")
  cat("      estimation', Ann Math Stat 18; Casella & Robert (1996) 'Rao-\n")
  cat("      Blackwellisation of sampling schemes', Biometrika 83(1).\n")
}
