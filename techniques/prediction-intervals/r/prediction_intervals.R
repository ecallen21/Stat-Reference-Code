# Prediction intervals vs confidence intervals (Reference Sec 39.14)
# Native R via stats::predict.lm; Python statsmodels + custom.
# Run with:  Rscript prediction_intervals.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::predict.lm(interval='prediction' | 'confidence')\n")
  cat("  rms::Predict                    -- prediction + CI/PI for rms fits\n")
  cat("  ciTools::add_pi                 -- prediction intervals for many model classes\n")
  cat("Python:\n")
  cat("  statsmodels.regression.linear_model.wls_prediction_std\n")
  cat("  sklearn                        -- no direct PI (use MAPIE for CP-based PI)\n")
  cat("  custom                         -- linear model PI + CI from scratch\n")
  cat("Refs: Steyerberg, E.W. (2019) Clinical Prediction Models, 2nd ed., Springer, Ch 15;\n")
  cat("      Harrell, F.E. (2015) Regression Modeling Strategies, 2nd ed., Springer, Ch 5.\n")
}
