# Sample Entropy (Reference Sec 47.329)
# Native R via nonlinearTseries / pracma; Python via antropy / nolds / from-scratch.
# Run with:  Rscript sample_entropy_signal.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  nonlinearTseries::sampleEntropy   -- HRV / EEG entropy\n")
  cat("  pracma::sample_entropy            -- utility function\n")
  cat("  TSEntropies                        -- entropy measures for time series\n")
  cat("Python:\n")
  cat("  antropy.sample_entropy\n")
  cat("  nolds.sampen\n")
  cat("  pyeeg.samp_entropy\n")
  cat("  From-scratch (see sample_entropy_signal.py)\n")
  cat("Refs: Richman, J.S. & Moorman, J.R. (2000) 'Physiological time-series\n")
  cat("      analysis using approximate entropy and sample entropy',\n")
  cat("      Am J Physiol 278(6).\n")
}
