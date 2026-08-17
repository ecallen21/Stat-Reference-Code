# Isolation Forest + one-class SVM + robust covariance (Reference §26.18)
# R via isotree, solitude, or e1071::svm(type = "one-classification").
# Run with:  Rscript isolation_forest_anomaly.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- rbind(matrix(rnorm(200 * 3), 200, 3), matrix(rnorm(10 * 3, mean = 5, sd = 0.5), 10, 3))
  if (requireNamespace("isotree", quietly = TRUE)) {
    cat("=== isotree::isolation.forest ===\n")
    fit <- isotree::isolation.forest(X, ntrees = 100)
    scores <- predict(fit, X)
    detected <- which(scores > quantile(scores, 0.95))
    cat(sprintf("  recall on true anomalies (rows 201-210): %.3f\n",
                mean((201:210) %in% detected)))
  }
}
