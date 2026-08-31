# Equalized odds (Reference Ch 31 Fairness)
# Native R via fairness packages; Python via fairlearn.
# Run with:  Rscript equalized_odds.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness                    -- equal_odds, fnr_parity, fpr_parity\n")
  cat("  fairml                      -- SVM/logistic with equalised-odds constraint\n")
  cat("  mlr3fairness                -- MeasureFairness with 'equalized_odds' key\n")
  cat("Python:\n")
  cat("  fairlearn.metrics.equalized_odds_difference / equalized_odds_ratio\n")
  cat("  aif360.metrics.ClassificationMetric.equal_opportunity_difference\n")
  cat("Refs: Hardt, M., Price, E. & Srebro, N. (2016) 'Equality of Opportunity in\n")
  cat("      Supervised Learning', NeurIPS.\n")
}
