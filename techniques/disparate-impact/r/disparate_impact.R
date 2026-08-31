# Disparate impact ratio / four-fifths rule (Reference Ch 31 Fairness)
# Native R via fairness packages; SciPy for the two-proportion CI in Python.
# Run with:  Rscript disparate_impact.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness                    -- prop.test-based CIs for DI\n")
  cat("  fairml                      -- Feldman DI-repair preprocessing\n")
  cat("  fairmodels                  -- disparate_impact() metric\n")
  cat("Python:\n")
  cat("  aif360.metrics.BinaryLabelDatasetMetric.disparate_impact\n")
  cat("  fairlearn.metrics.demographic_parity_ratio\n")
  cat("  aif360.algorithms.preprocessing.DisparateImpactRemover  (Feldman 2015)\n")
  cat("Refs: Uniform Guidelines on Employee Selection Procedures (1978) 29 CFR 1607;\n")
  cat("      Feldman, M. et al. (2015) 'Certifying and Removing Disparate Impact', KDD.\n")
}
