# Cross-entropy / log-loss (Reference Sec 34.6)
# Native R: base R + MLmetrics; Python: sklearn / torch.
# Run with:  Rscript cross_entropy_log_loss.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MLmetrics::LogLoss / MultiLogLoss  -- binary and categorical\n")
  cat("  yardstick::mn_log_loss             -- tidymodels wrapper\n")
  cat("  torch (R port)                     -- nnf_cross_entropy\n")
  cat("Python:\n")
  cat("  sklearn.metrics.log_loss           -- binary + multi-class\n")
  cat("  torch.nn.CrossEntropyLoss          -- softmax + NLL fused\n")
  cat("  tf.keras.losses.CategoricalCrossentropy\n")
  cat("Refs: Shannon 1948; Good, I.J. (1952) 'Rational decisions', JRSS-B (proper\n")
  cat("      scoring rules).\n")
}
