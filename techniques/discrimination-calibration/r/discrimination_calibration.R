# Discrimination vs calibration (Reference Sec 39.17)
# Native R via rms + pROC; Python sklearn + custom.
# Run with:  Rscript discrimination_calibration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::validate                  -- bootstrap-corrected AUC + calibration slope\n")
  cat("  rms::val.prob                  -- calibration slope, intercept, ICI, E-max\n")
  cat("  pROC                           -- AUC + CI + DeLong test\n")
  cat("  CalibrationCurves              -- val.prob.ci.2 with belt CIs\n")
  cat("Python:\n")
  cat("  sklearn.metrics.roc_auc_score / brier_score_loss\n")
  cat("  sklearn.calibration.calibration_curve\n")
  cat("  custom                         -- ICI (Austin-Steyerberg 2019)\n")
  cat("Refs: Steyerberg et al. (2010) 'Assessing the performance of prediction models',\n")
  cat("      Epidemiology; Van Calster et al. (2019) 'Calibration: the Achilles heel of\n")
  cat("      predictive analytics', BMC Medicine.\n")
}
