# Model stacking / super learner (Reference §26.14)
# R via SuperLearner (van der Laan) or stacks (tidymodels).
# Run with:  Rscript model_stacking.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 400
  X <- matrix(runif(n * 2, -3, 3), n, 2)
  y <- sin(X[, 1]) + 0.5 * X[, 2] + rnorm(n, 0, 0.3)
  df <- data.frame(y = y, x1 = X[, 1], x2 = X[, 2])
  if (requireNamespace("SuperLearner", quietly = TRUE)) {
    cat("=== SuperLearner ===\n")
    fit <- SuperLearner::SuperLearner(y, data.frame(x1 = X[, 1], x2 = X[, 2]),
                                       SL.library = c("SL.lm", "SL.rpart", "SL.knn"))
    cat(sprintf("  CV MSE = %.4f\n", min(fit$cvRisk)))
  }
}
