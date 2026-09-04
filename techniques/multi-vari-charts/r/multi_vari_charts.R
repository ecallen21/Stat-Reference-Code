# Multi-vari charts (Reference Sec 37.12)
# Native R via SixSigma; Python custom.
# Run with:  Rscript multi_vari_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SixSigma::ss.ci                -- multi-vari plot\n")
  cat("  qualityTools                   -- process/quality tools\n")
  cat("Python:\n")
  cat("  custom                         -- variance decomposition\n")
  cat("Refs: Seder, L.A. (1950) 'Diagnosis with diagrams', Industrial Quality Control;\n")
  cat("      Montgomery, D.C. (2013) Introduction to Statistical Quality Control, 7th ed.\n")
}
