# Shewhart control charts (Reference Sec 37.1)
# Native R via qcc / qcc::qcc; Python via pyspc.
# Run with:  Rscript shewhart_control_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc                          -- Scrucca's SPC reference: x-bar, R, S, p, np, c, u\n")
  cat("  qcr                           -- Regression + control charts\n")
  cat("  IQCC                          -- Nonparametric + risk-adjusted charts\n")
  cat("Python:\n")
  cat("  pyspc                          -- x-bar, R, S, p, np, c, u charts\n")
  cat("  spc                            -- broader SPC toolkit\n")
  cat("  matplotlib + custom          -- manual\n")
  cat("Refs: Shewhart, W. (1931) 'Economic Control of Quality of Manufactured Product',\n")
  cat("      Van Nostrand; Montgomery, D.C. (2020) 'Introduction to Statistical Quality\n")
  cat("      Control', 8th ed., Wiley.\n")
}
