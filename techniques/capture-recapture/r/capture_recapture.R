# Capture-recapture (Reference Sec 38.11)
# Native R via Rcapture; Python custom.
# Run with:  Rscript capture_recapture.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Rcapture (closedp, openp, robustd) -- log-linear models for CR\n")
  cat("  CARE1                          -- multiple-list CR with covariates\n")
  cat("  multimark                      -- multiple mark types (SECR)\n")
  cat("  secr                           -- spatially explicit CR\n")
  cat("Python:\n")
  cat("  custom                         -- Lincoln-Petersen, Chapman, Schnabel\n")
  cat("Refs: Chao, A. (2001) Statistics in Medicine; International Working Group for\n")
  cat("      Disease Monitoring and Forecasting (1995) AJE; Otis et al. (1978)\n")
  cat("      'Statistical inference from capture data on closed animal populations'.\n")
}
