# EWMA control chart (Reference Sec 37.3)
# Native R via qcc::ewma; Python via pyspc.
# Run with:  Rscript ewma_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc::ewma                     -- Roberts EWMA reference\n")
  cat("  spc                            -- ARL tables + design points\n")
  cat("Python:\n")
  cat("  pyspc                          -- EWMA helper\n")
  cat("  statsmodels.tsa.holtwinters   -- adjacent exponential smoothing\n")
  cat("Refs: Roberts, S.W. (1959) 'Control chart tests based on geometric moving\n")
  cat("      averages', Technometrics; Montgomery, D.C. (2020) 'Introduction to\n")
  cat("      Statistical Quality Control', 8th ed., Wiley, Ch. 9.\n")
}
