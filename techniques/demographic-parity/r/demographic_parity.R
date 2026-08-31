# Demographic parity / four-fifths rule (Reference Ch 31 Fairness)
# Base R + fairness packages.
# Run with:  Rscript demographic_parity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness                    -- demographic_parity, statistical_parity, dp_ratio\n")
  cat("  fairml                      -- pre-, in-, post-processing fair models\n")
  cat("  mlr3fairness                -- fairness metrics + measures for mlr3 learners\n")
  cat("Python:\n")
  cat("  fairlearn.metrics.demographic_parity_difference / demographic_parity_ratio\n")
  cat("  aif360.metrics.BinaryLabelDatasetMetric.disparate_impact\n")
  cat("Refs: Uniform Guidelines on Employee Selection Procedures (1978)\n")
  cat("      -- the 'four-fifths rule' (Sec. 4D).\n")
  cat("      Feldman, M. et al. (2015) 'Certifying and Removing Disparate Impact', KDD.\n")
}
