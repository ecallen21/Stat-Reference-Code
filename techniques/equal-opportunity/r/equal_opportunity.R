# Equal opportunity (Reference Ch 31 Fairness)
# Native R via fairness packages; Python via fairlearn / aif360.
# Run with:  Rscript equal_opportunity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness                    -- equal_odds() reports FNR / FPR parity components\n")
  cat("  fairml                      -- SVM/LR with equal-opportunity constraint\n")
  cat("  mlr3fairness                -- MeasureFairness 'equal_opportunity'\n")
  cat("Python:\n")
  cat("  fairlearn.metrics.true_positive_rate_difference / _ratio\n")
  cat("  aif360.metrics.ClassificationMetric.equal_opportunity_difference\n")
  cat("Refs: Hardt, M., Price, E. & Srebro, N. (2016)\n")
  cat("      'Equality of Opportunity in Supervised Learning', NeurIPS.\n")
}
