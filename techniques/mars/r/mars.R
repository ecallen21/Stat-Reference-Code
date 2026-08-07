# Multivariate Adaptive Regression Splines (Reference §5.28)
# R via earth::earth (Milborrow's port of Friedman's MARS).
# Run with:  Rscript mars.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300
  x1 <- runif(n, -3, 3); x2 <- runif(n, -3, 3)
  y <- pmax(x1 - 0.5, 0) - 2 * pmax(-x1 - 1, 0) + 0.5 * abs(x2) + rnorm(n, 0, 0.3)
  df <- data.frame(y = y, x1 = x1, x2 = x2)
  if (requireNamespace("earth", quietly = TRUE)) {
    cat("=== earth::earth ===\n")
    print(summary(earth::earth(y ~ x1 + x2, data = df, degree = 2)))
  }
}
