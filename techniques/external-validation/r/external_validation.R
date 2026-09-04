# External validation (Reference Sec 39.5)
# Native R via rms::val.prob; Python sklearn + custom.
# Run with:  Rscript external_validation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::val.prob                  -- AUC + calibration slope, CITL, Brier\n")
  cat("  riskRegression::Score          -- multi-model AUC + Brier + ICI\n")
  cat("  predtools::calibration_plot    -- calibration by decile / spline\n")
  cat("Python:\n")
  cat("  sklearn.metrics (roc_auc_score, brier_score_loss)\n")
  cat("  sklearn.calibration.calibration_curve\n")
  cat("  custom                         -- CITL + slope + intercept\n")
  cat("Refs: Steyerberg, E.W. & Harrell, F.E. (2016) 'Prediction models need\n")
  cat("      appropriate internal, internal-external, and external validation', JCE;\n")
  cat("      Collins et al. (2012) 'External validation of multivariable prediction\n")
  cat("      models: a systematic review', BMJ.\n")
}
