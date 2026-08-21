# Multivariate Multiple Regression (Reference §9.20)
# Base R lm(Y ~ X) with multivariate response + car::Anova.
# Run with:  Rscript multivariate_multiple_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  E <- MASS::mvrnorm(n, c(0, 0, 0), matrix(c(1, 0.5, 0.3, 0.5, 1, 0.2, 0.3, 0.2, 1), 3, 3))
  Y <- cbind(1 + 0.5 * x1 + 1.5 * x2, 2 + 0.3 * x1 - 0.6 * x2, -0.5 + 0.7 * x1 + 0.4 * x2) + E
  colnames(Y) <- c("y1", "y2", "y3")
  fit <- lm(Y ~ x1 + x2)
  cat("=== summary(lm(Y ~ X)) ===\n")
  print(summary(fit))
  if (requireNamespace("car", quietly = TRUE)) {
    cat("\n=== car::Anova with multivariate tests (Wilks/Pillai) ===\n")
    print(car::Anova(fit, test.statistic = "Wilks"))
  }
}
