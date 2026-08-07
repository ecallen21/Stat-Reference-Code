# Instrumental variables / 2SLS (Reference §5.22)
# R via AER::ivreg or ivreg::ivreg.
# Run with:  Rscript iv_2sls.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  z <- rnorm(n); u <- rnorm(n)
  x <- 0.5 + 1.5 * z + 0.8 * u + rnorm(n, 0, 0.5)
  y <- 1 + 2 * x + 0.6 * u + rnorm(n, 0, 0.5)
  cat("=== Naive OLS (biased) ===\n")
  print(coef(lm(y ~ x)))
  if (requireNamespace("AER", quietly = TRUE)) {
    cat("\n=== AER::ivreg (2SLS with z as instrument) ===\n")
    fit <- AER::ivreg(y ~ x | z)
    print(summary(fit, diagnostics = TRUE))
  }
}
