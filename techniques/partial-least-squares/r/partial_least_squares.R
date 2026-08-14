# PLS Regression (Reference §5.31)
# R via pls::plsr.
# Run with:  Rscript partial_least_squares.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 100; p <- 20
  F <- matrix(rnorm(n * 3), n, 3); load <- matrix(rnorm(3 * p), 3, p)
  y <- as.numeric(F %*% c(2, -1, 0.5) + rnorm(n, 0, 0.5))
  X <- F %*% load + matrix(rnorm(n * p, 0, 0.5), n, p)
  df <- data.frame(y = y, X = X)
  if (requireNamespace("pls", quietly = TRUE)) {
    cat("=== pls::plsr with CV to select components ===\n")
    fit <- pls::plsr(y ~ X, data = df, ncomp = 8, validation = "CV")
    print(summary(fit))
  }
}
