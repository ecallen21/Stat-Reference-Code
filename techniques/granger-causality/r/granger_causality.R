# Granger causality (Reference §13.50)
# Base R via lmtest::grangertest.
# Run with:  Rscript granger_causality.R

if (sys.nframe() == 0) {
  set.seed(31); n <- 500
  x <- numeric(n); y <- numeric(n); x[1] <- rnorm(1); y[1] <- rnorm(1)
  for (t in 2:n) {
    x[t] <- 0.5 * x[t - 1] + rnorm(1)
    y[t] <- 0.3 * y[t - 1] + 0.6 * x[t - 1] + rnorm(1)
  }
  if (requireNamespace("lmtest", quietly = TRUE)) {
    cat("=== X -> Y ===\n"); print(lmtest::grangertest(y ~ x, order = 5))
    cat("\n=== Y -> X ===\n"); print(lmtest::grangertest(x ~ y, order = 5))
  }
}
