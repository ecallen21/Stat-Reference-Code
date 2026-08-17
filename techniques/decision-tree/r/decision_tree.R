# CART Decision Tree (Reference §26.6)
# R via rpart (Breiman-Friedman-Olshen-Stone).
# Run with:  Rscript decision_tree.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- matrix(runif(400 * 2, -3, 3), 400, 2)
  y <- sin(X[, 1]) + 0.5 * X[, 2] + rnorm(400, 0, 0.3)
  df <- data.frame(y = y, x1 = X[, 1], x2 = X[, 2])
  if (requireNamespace("rpart", quietly = TRUE)) {
    cat("=== rpart regression tree ===\n")
    fit <- rpart::rpart(y ~ x1 + x2, data = df, control = rpart::rpart.control(maxdepth = 6))
    cat(sprintf("  in-sample RMSE = %.3f\n", sqrt(mean((y - predict(fit)) ^ 2))))
  }
}
