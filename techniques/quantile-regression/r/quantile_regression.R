# Quantile regression (Reference §5.15)
# R via quantreg::rq (Koenker's canonical implementation).
# Run with:  Rscript quantile_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  x <- rnorm(n)
  y <- 1 + 2 * x + (1 + 0.8 * x) * rnorm(n)
  if (requireNamespace("quantreg", quietly = TRUE)) {
    cat("=== quantreg::rq at tau = 0.1, 0.25, 0.5, 0.75, 0.9 ===\n")
    fit <- quantreg::rq(y ~ x, tau = c(0.1, 0.25, 0.5, 0.75, 0.9))
    print(summary(fit))
  }
}
