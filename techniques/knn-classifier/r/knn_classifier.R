# k-Nearest Neighbors (Reference §26.11)
# R via class::knn or FNN::knn.
# Run with:  Rscript knn_classifier.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- rbind(matrix(rnorm(100 * 2), 100, 2),
             matrix(rnorm(100 * 2, mean = 4), 100, 2),
             cbind(rnorm(100, 2), rnorm(100, 4)))
  y <- factor(rep(0:2, each = 100))
  if (requireNamespace("class", quietly = TRUE)) {
    cat("=== class::knn (5-NN) ===\n")
    pred <- class::knn(X, X, y, k = 5)
    cat(sprintf("  accuracy = %.3f\n", mean(pred == y)))
  }
}
