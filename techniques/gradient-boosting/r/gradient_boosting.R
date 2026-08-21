# Gradient Boosting (Reference §26.8)
# R via gbm or xgboost (production) / lightgbm.
# Run with:  Rscript gradient_boosting.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 400
  X <- matrix(runif(n * 2, -3, 3), n, 2)
  y <- sin(X[, 1]) + 0.5 * X[, 2] + rnorm(n, 0, 0.3)
  df <- data.frame(y = y, x1 = X[, 1], x2 = X[, 2])
  if (requireNamespace("gbm", quietly = TRUE)) {
    cat("=== gbm::gbm ===\n")
    fit <- gbm::gbm(y ~ x1 + x2, data = df, distribution = "gaussian",
                    n.trees = 100, interaction.depth = 3, shrinkage = 0.1)
    cat(sprintf("  in-sample RMSE = %.3f\n",
                sqrt(mean((y - predict(fit, df, n.trees = 100)) ^ 2))))
  }
}
