# Truncated regression (Reference §5.18)
# R via truncreg::truncreg.
# Run with:  Rscript truncated_regression.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- numeric(); y <- numeric()
  while (length(y) < 500) {
    xn <- rnorm(200); yn <- 1 + 2 * xn + rnorm(200)
    keep <- yn < 3
    X <- c(X, xn[keep]); y <- c(y, yn[keep])
  }
  X <- X[1:500]; y <- y[1:500]
  cat("=== Naive OLS on truncated sample ===\n")
  print(coef(lm(y ~ X)))
  if (requireNamespace("truncreg", quietly = TRUE)) {
    cat("\n=== truncreg::truncreg (upper truncation at 3) ===\n")
    print(summary(truncreg::truncreg(y ~ X, point = 3, direction = "right")))
  }
}
