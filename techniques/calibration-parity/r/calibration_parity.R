# Calibration / predictive parity (Reference Ch 31 Fairness)
# Native R via fairness packages; Python via fairlearn / aif360.
# Run with:  Rscript calibration_parity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness                    -- pred_rate_parity() = PPV parity\n")
  cat("  fairml                      -- pre-/post-processing calibrators\n")
  cat("  rms::val.prob               -- reliability diagrams per subgroup\n")
  cat("Python:\n")
  cat("  fairlearn.metrics.selection_rate + calibration_curve per group\n")
  cat("  aif360.metrics.ClassificationMetric  (differential_fairness, positive_predictive_value)\n")
  cat("Refs: Chouldechova, A. (2017) 'Fair Prediction with Disparate Impact:\n")
  cat("      A Study of Bias in Recidivism Prediction Instruments', Big Data;\n")
  cat("      Kleinberg, J., Mullainathan, S. & Raghavan, M. (2017)\n")
  cat("      'Inherent Trade-offs in the Fair Determination of Risk Scores', ITCS.\n")
}
