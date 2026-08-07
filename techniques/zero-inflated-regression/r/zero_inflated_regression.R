# Zero-inflated and hurdle count regression (Reference §5.24)
# R via pscl::zeroinfl and pscl::hurdle.
# Run with:  Rscript zero_inflated_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  x <- rnorm(n); z <- rnorm(n)
  pi_true <- plogis(-1 + 0.5 * z)
  mu_true <- exp(0.5 + 0.6 * x)
  y <- ifelse(runif(n) < pi_true, 0, rpois(n, mu_true))
  df <- data.frame(y = y, x = x, z = z)
  if (requireNamespace("pscl", quietly = TRUE)) {
    cat("=== pscl::zeroinfl (ZIP) ===\n")
    print(summary(pscl::zeroinfl(y ~ x | z, data = df, dist = "poisson")))
    cat("\n=== pscl::hurdle (Poisson hurdle) ===\n")
    print(summary(pscl::hurdle(y ~ x | z, data = df, dist = "poisson", zero.dist = "binomial")))
  }
}
