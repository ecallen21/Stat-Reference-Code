# Data-drift detection: PSI + KS + Wasserstein (Reference Ch 32 MLOps)
# Native R for the metrics; Python for the full monitoring stack.
# Run with:  Rscript data_drift_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  driftR                      -- PSI + KS drift monitoring\n")
  cat("  causaldrf, wass1r           -- Wasserstein-1 helpers\n")
  cat("  drifter                     -- concept + data drift detectors\n")
  cat("Python:\n")
  cat("  evidently                   -- PSI + KS + Wasserstein per-feature + dashboards\n")
  cat("  alibi-detect                -- MMD, LSDD, KS, ChiSq, and multivariate drift\n")
  cat("  whylogs                     -- profile-based drift monitoring\n")
  cat("  nannyML                     -- performance-estimation + drift\n")
  cat("Refs: Wu, D. & Olson, D. (2010) 'A Comparison of Stability Measures for\n")
  cat("      Financial Time Series', Journal of Risk Model Validation.\n")
  cat("      Rabanser, S., Gunnemann, S. & Lipton, Z. (2019)\n")
  cat("      'Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift', NeurIPS.\n")
}
