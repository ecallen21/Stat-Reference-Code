# Forecast evaluation: TS CV + accuracy metrics + reconciliation
# (Reference §13.23, §13.31, §13.35, §13.36, §13.45, §13.51)
# R via forecast::tsCV + forecast::accuracy + fable + hts::htsboot / MinT.
# Run with:  Rscript forecast_evaluation_cv.R

if (sys.nframe() == 0) {
  cat("Recommended R packages for time-series forecasting evaluation:\n\n")
  cat("  forecast::tsCV(y, forecastfunction, h = 1)\n")
  cat("  forecast::accuracy(fitted, actual)\n")
  cat("  fabletools::accuracy(fable, test_data)\n\n")
  cat("Hierarchical reconciliation:\n")
  cat("  fable::reconcile() with min_trace() / bottom_up() / top_down()\n")
  cat("  hts::MinT() for the classical MinT (Wickramasuriya 2019).\n")
}
