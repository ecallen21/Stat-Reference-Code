# Support Vector Machine (Reference §26.9)
# R via e1071::svm (LIBSVM wrapper) or kernlab::ksvm.
# Run with:  Rscript svm_classifier.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- rbind(matrix(rnorm(100 * 2), 100, 2),
             matrix(rnorm(100 * 2, mean = 3), 100, 2))
  y <- factor(c(rep(-1, 100), rep(1, 100)))
  if (requireNamespace("e1071", quietly = TRUE)) {
    cat("=== e1071::svm (linear) ===\n")
    fit <- e1071::svm(X, y, kernel = "linear")
    cat(sprintf("  accuracy = %.3f\n", mean(predict(fit, X) == y)))
    cat("\n=== e1071::svm (RBF) ===\n")
    fit <- e1071::svm(X, y, kernel = "radial")
    cat(sprintf("  accuracy = %.3f\n", mean(predict(fit, X) == y)))
  }
}
