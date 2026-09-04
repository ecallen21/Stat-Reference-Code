# Pareto charts (Reference Sec 37.14)
# Native R via qcc; Python custom.
# Run with:  Rscript pareto_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc::pareto.chart              -- Pareto chart + cumulative line\n")
  cat("  qualityTools::paretoChart      -- alternative\n")
  cat("Python:\n")
  cat("  custom                         -- sorted counts + cumulative %\n")
  cat("Refs: Juran, J.M. (1954) 'Universals in management planning and controlling',\n")
  cat("      Management Review; Montgomery, D.C. (2013) Introduction to Statistical\n")
  cat("      Quality Control, 7th ed.\n")
}
