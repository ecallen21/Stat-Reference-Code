# Inverse Probability of Treatment Weighting (Reference §15.7)
# R via WeightIt / survey::svyglm or PSweight.
# Run with:  Rscript inverse_probability_weighting.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 800
  x1 <- rnorm(n); x2 <- rnorm(n)
  T <- as.integer(runif(n) < plogis(-0.5 + x1 - 0.5 * x2))
  Y <- 1 + 2 * T + 0.5 * x1 + 0.3 * x2 + rnorm(n)
  df <- data.frame(T = T, x1 = x1, x2 = x2, Y = Y)
  cat("=== Naive ===\n")
  cat(sprintf("  %.3f\n", mean(Y[T == 1]) - mean(Y[T == 0])))
  if (requireNamespace("WeightIt", quietly = TRUE) && requireNamespace("survey", quietly = TRUE)) {
    w <- WeightIt::weightit(T ~ x1 + x2, data = df, method = "ps", estimand = "ATE",
                             stabilize = TRUE)
    design <- survey::svydesign(ids = ~1, weights = w$weights, data = df)
    fit <- survey::svyglm(Y ~ T, design = design)
    cat("\n=== IPTW ATE via WeightIt + survey::svyglm ===\n")
    print(summary(fit))
  }
}
