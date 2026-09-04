# Multivariate control charts (Reference Sec 37.4)
# Native R via qcc::mqcc; Python via multivariate-quality-control.
# Run with:  Rscript multivariate_control_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc::mqcc                    -- Hotelling T^2 for multivariate SPC\n")
  cat("  MSQC                          -- MEWMA, MCUSUM, T-squared, dc\n")
  cat("  IQCC                          -- Individual observations + subgroups\n")
  cat("Python:\n")
  cat("  multivariate-quality-control  -- Hotelling T^2, MEWMA\n")
  cat("  pyspc                          -- basic multivariate SPC\n")
  cat("Refs: Hotelling, H. (1947) 'Multivariate quality control', in Techniques of\n")
  cat("      Statistical Analysis; Mason, R.L. & Young, J.C. (2002) 'Multivariate\n")
  cat("      Statistical Process Control with Industrial Applications', SIAM.\n")
}
