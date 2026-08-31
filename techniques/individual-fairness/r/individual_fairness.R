# Individual fairness / Lipschitz criterion (Reference Ch 31 Fairness)
# Native R for the diagnostic; Python for in-training enforcement.
# Run with:  Rscript individual_fairness.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairness / fairmodels        -- individual-fairness metrics on top of any predictor\n")
  cat("  proxy, philentropy           -- pairwise task-metric distances\n")
  cat("Python:\n")
  cat("  aif360.algorithms.inprocessing.PrejudiceRemover   (Kamishima 2012, adjacent)\n")
  cat("  sen-fair-consistency (Yurochkin 2020 SenSR / SenSeI)\n")
  cat("  fairtorch                    -- Lipschitz penalty for pytorch predictors\n")
  cat("Refs: Dwork, C., Hardt, M., Pitassi, T., Reingold, O. & Zemel, R. (2012)\n")
  cat("      'Fairness Through Awareness', ITCS.\n")
  cat("      Yurochkin, M., Bower, A. & Sun, Y. (2020)\n")
  cat("      'Training Individually Fair ML Models with Sensitive Subspace Robustness', ICLR.\n")
}
