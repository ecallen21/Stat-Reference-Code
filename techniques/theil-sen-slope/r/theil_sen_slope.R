# Theil-Sen Slope Estimator (Reference §6.32)
# From-scratch base R plus mblm::mblm or RobustLinearReg::theil_sen_regression.
# Run with:  Rscript theil_sen_slope.R

theil_sen_slope_scratch <- function(x, y) {
  n <- length(x); pairs <- combn(n, 2)
  slopes <- numeric(0)
  for (k in seq_len(ncol(pairs))) {
    i <- pairs[1, k]; j <- pairs[2, k]
    if (x[j] != x[i]) slopes <- c(slopes, (y[j] - y[i]) / (x[j] - x[i]))
  }
  slope <- median(slopes); intercept <- median(y - slope * x)
  list(intercept = intercept, slope = slope, n_pairs = length(slopes),
       method = "Theil-Sen")
}

theil_sen_ci_scratch <- function(x, y, conf = 0.95) {
  n <- length(x)
  pairs <- combn(n, 2)
  slopes <- numeric(0)
  for (k in seq_len(ncol(pairs))) {
    i <- pairs[1, k]; j <- pairs[2, k]
    if (x[j] != x[i]) slopes <- c(slopes, (y[j] - y[i]) / (x[j] - x[i]))
  }
  slopes <- sort(slopes); N <- length(slopes)
  var_S <- n * (n - 1) * (2 * n + 5) / 18
  z <- qnorm(0.5 + conf / 2); half <- z * sqrt(var_S)
  M1 <- max(1, floor((N - half) / 2) + 1)
  M2 <- min(N, floor((N + half) / 2))
  list(slope_estimate = median(slopes),
       ci_lower = slopes[M1], ci_upper = slopes[M2],
       conf = conf, method = "Theil-Sen (Sen 1968 CI)")
}

library_demo <- function(x, y) {
  if (requireNamespace("mblm", quietly = TRUE)) {
    cat("mblm::mblm:\n"); print(summary(mblm::mblm(y ~ x, repeated = FALSE)))
  } else cat("(install 'mblm' for theil_sen)\n")
}

if (sys.nframe() == 0) {
  set.seed(11); n <- 60
  x <- runif(n, 0, 10); y <- 1 + 2 * x + rnorm(n)
  y[1] <- 50; y[6] <- -30
  print(theil_sen_slope_scratch(x, y))
  print(theil_sen_ci_scratch(x, y))
  cat("OLS slope:", coef(lm(y ~ x))[2], "\n\n--- library ---\n"); library_demo(x, y)
}
