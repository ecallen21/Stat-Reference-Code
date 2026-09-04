# CUSUM control chart (Reference Sec 37.2)
# Native R via qcc::cusum; Python via pyspc.
# Run with:  Rscript cusum_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc::cusum                    -- tabular CUSUM with mask\n")
  cat("  spc                            -- ARL and design-point tables\n")
  cat("  rQCC                           -- self-starting CUSUM\n")
  cat("Python:\n")
  cat("  pyspc                          -- tabular CUSUM helper\n")
  cat("  scikit-multiflow.drift_detection.PageHinkley -- adjacent drift detector\n")
  cat("Refs: Page, E.S. (1954) 'Continuous inspection schemes', Biometrika;\n")
  cat("      Hawkins, D.M. & Olwell, D.H. (1998) 'Cumulative Sum Charts and\n")
  cat("      Charting for Quality Improvement', Springer.\n")
}
