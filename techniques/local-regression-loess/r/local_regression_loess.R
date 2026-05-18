# Local Regression / LOESS (Reference §6.22)
# From-scratch base R plus stats::loess / stats::lowess.
# Run with:  Rscript local_regression_loess.R
#
# Inputs:  x, y, x_grid, span, degree

tricube <- function(u) ifelse(abs(u) < 1, (1 - abs(u)^3)^3, 0)

loess_fit_scratch <- function(x, y, x_grid, span = 0.5, degree = 1) {
  n <- length(x); q <- max(2, ceiling(span * n))
  out <- numeric(length(x_grid))
  for (i in seq_along(x_grid)) {
    x0 <- x_grid[i]; d <- abs(x - x0)
    h <- max(sort(d)[q], 1e-15); w <- tricube(d / h)
    X_loc <- cbind(1, sapply(seq_len(degree), function(p) (x - x0)^p))
    WX <- X_loc * w
    beta <- as.vector(solve(t(WX) %*% X_loc, t(WX) %*% y))
    out[i] <- beta[1]
  }
  list(x_grid = x_grid, y_smoothed = out, span = span, degree = degree)
}

# Library: stats::loess (default) and stats::lowess (older Cleveland)
library_demo <- function(x, y, grid) {
  fit <- loess(y ~ x, span = 0.5, degree = 2)
  cat("stats::loess at first 5 grid pts:\n")
  print(predict(fit, newdata = data.frame(x = grid[1:5])))
}

if (sys.nframe() == 0) {
  set.seed(9)
  x <- seq(0, 10, length.out = 200); y <- sin(x) + 0.3 * x + rnorm(200, 0, 0.4)
  grid <- seq(0, 10, length.out = 30)
  cat("=== span = 0.3, degree = 1 ===\n")
  print(loess_fit_scratch(x, y, grid, span = 0.3, degree = 1)$y_smoothed[1:10])
  cat("\n=== span = 0.7, degree = 2 ===\n")
  print(loess_fit_scratch(x, y, grid, span = 0.7, degree = 2)$y_smoothed[1:10])
  cat("\n--- library ---\n"); library_demo(x, y, grid)
}
