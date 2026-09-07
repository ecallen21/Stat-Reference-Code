# F-beta optimal decision threshold (Reference Sec 26.9)
# Native R via yardstick / probably; Python via scikit-learn.
# Run with:  Rscript f1_optimal_threshold.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  yardstick::f_meas                 -- F-beta / F1 metric\n")
  cat("  probably::threshold_perf           -- sweep threshold and pick optimum\n")
  cat("  pROC / caret::confusionMatrix     -- classic threshold sweeps\n")
  cat("Python:\n")
  cat("  sklearn.metrics.fbeta_score / precision_recall_fscore_support\n")
  cat("  sklearn.metrics.precision_recall_curve + argmax\n")
  cat("  imbalanced-learn (metrics for imbalanced datasets)\n")
  cat("Refs: van Rijsbergen, C.J. (1979) Information Retrieval; Chinchor, N.\n")
  cat("      (1992) 'MUC-4 evaluation metrics'; Powers (2011) Evaluation review.\n")
}
