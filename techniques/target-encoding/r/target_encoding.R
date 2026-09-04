# Target encoding (Reference Sec 41.11)
# Native R via vtreat / recipes; Python category_encoders + custom.
# Run with:  Rscript target_encoding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  vtreat::mkCrossFrameCExperiment / designTreatmentsC\n")
  cat("  recipes::step_lencode_mixed / step_lencode_bayes / step_woe (from embed)\n")
  cat("  embed                             -- embeddings + supervised encodings\n")
  cat("Python:\n")
  cat("  category_encoders (TargetEncoder, LeaveOneOutEncoder, WOEEncoder, BinaryEncoder)\n")
  cat("  sklearn.preprocessing (limited)\n")
  cat("Refs: Micci-Barreca (2001) 'A preprocessing scheme for high-cardinality\n")
  cat("      categorical attributes in classification and prediction problems',\n")
  cat("      ACM SIGKDD Explorations.\n")
}
