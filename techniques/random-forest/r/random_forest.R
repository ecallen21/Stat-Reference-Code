# Random Forest (Reference §26.7; Breiman 2001)
# R via randomForest or ranger.
# Run with:  Rscript random_forest.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- rbind(matrix(rnorm(100 * 2), 100, 2),
             matrix(rnorm(100 * 2, mean = 4), 100, 2),
             cbind(rnorm(100, 2), rnorm(100, 4)))
  y <- factor(rep(0:2, each = 100))
  df <- data.frame(y = y, x1 = X[, 1], x2 = X[, 2])
  if (requireNamespace("randomForest", quietly = TRUE)) {
    cat("=== randomForest::randomForest ===\n")
    fit <- randomForest::randomForest(y ~ x1 + x2, data = df, ntree = 100)
    print(fit)
  }
}
