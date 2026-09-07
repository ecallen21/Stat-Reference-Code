# Entropy balancing (Reference Sec 15.55)
# Native R via WeightIt / ebal; Python custom + econml.
# Run with:  Rscript entropy_balancing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  WeightIt::weightit(method='ebal') -- Hainmueller entropy balancing\n")
  cat("  ebal                                -- original Hainmueller implementation\n")
  cat("  survey::calibrate                   -- calibration weights (related)\n")
  cat("Python:\n")
  cat("  econml (residual estimator with balancing weights)\n")
  cat("  custom (Newton exponential tilting)\n")
  cat("Refs: Hainmueller, J. (2012) 'Entropy balancing for causal effects: a\n")
  cat("      multivariate reweighting method to produce balanced samples in\n")
  cat("      observational studies', Political Analysis.\n")
}
