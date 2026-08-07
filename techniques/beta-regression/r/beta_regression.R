# Beta regression (Reference §5.20)
# R via betareg::betareg.
# Run with:  Rscript beta_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 400
  x <- rnorm(n); mu <- plogis(0 + 1.2 * x); phi <- 20
  y <- rbeta(n, mu * phi, (1 - mu) * phi)
  if (requireNamespace("betareg", quietly = TRUE)) {
    cat("=== betareg::betareg ===\n")
    print(summary(betareg::betareg(y ~ x)))
  }
}
