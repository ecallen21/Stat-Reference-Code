# Fractional-logit / fractional response (Reference §5.26)
# R via glm(family = quasibinomial) + sandwich::vcovHC.
# Run with:  Rscript fractional_logit.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu <- plogis(0.8 * x1 - 0.4 * x2)
  y <- rbeta(n, mu * 20, (1 - mu) * 20)
  y[sample(n, 30)] <- 0; y[sample(n, 20)] <- 1
  fit <- glm(y ~ x1 + x2, family = quasibinomial())
  print(summary(fit))
  if (requireNamespace("sandwich", quietly = TRUE) &&
      requireNamespace("lmtest", quietly = TRUE)) {
    cat("\n=== HC0 sandwich SEs ===\n")
    print(lmtest::coeftest(fit, vcov = sandwich::vcovHC(fit, type = "HC0")))
  }
}
