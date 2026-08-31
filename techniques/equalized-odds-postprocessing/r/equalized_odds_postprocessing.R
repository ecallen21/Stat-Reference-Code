# Equalized-odds postprocessing (Reference Ch 31 Fairness)
# Native R via fairness packages; Python via aif360 / fairlearn.
# Run with:  Rscript equalized_odds_postprocessing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness                    -- group-specific threshold helpers\n")
  cat("  fairml                      -- SVM/LR with post-hoc constraints\n")
  cat("Python:\n")
  cat("  aif360.algorithms.postprocessing.EqOddsPostprocessing  (Hardt 2016 reference)\n")
  cat("  aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing (Pleiss 2017)\n")
  cat("  fairlearn.postprocessing.ThresholdOptimizer  (constraint='equalized_odds')\n")
  cat("Refs: Hardt, M., Price, E. & Srebro, N. (2016)\n")
  cat("      'Equality of Opportunity in Supervised Learning', NeurIPS.\n")
}
