# Naive Bayes (Reference §26.10)
# R via e1071::naiveBayes or naivebayes::naive_bayes.
# Run with:  Rscript naive_bayes.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- rbind(matrix(rnorm(100 * 2), 100, 2),
             matrix(rnorm(100 * 2, mean = 4), 100, 2),
             cbind(rnorm(100, 2), rnorm(100, 4)))
  y <- factor(rep(0:2, each = 100))
  if (requireNamespace("e1071", quietly = TRUE)) {
    cat("=== e1071::naiveBayes (Gaussian) ===\n")
    fit <- e1071::naiveBayes(X, y)
    cat(sprintf("  accuracy = %.3f\n", mean(predict(fit, X) == y)))
  }
}
