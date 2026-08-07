# Non-inferiority tests (Reference §17.7)
# R via base t.test for means, DescTools::BinomDiffCI for proportions.
# Run with:  Rscript non_inferiority_test.R

if (sys.nframe() == 0) {
  set.seed(0); margin <- 1.0
  y_new <- rnorm(60, 9.8, 2); y_std <- rnorm(60, 10.0, 2)
  cat("=== NI on means (margin = 1.0, higher is better) ===\n")
  fit <- t.test(y_new, y_std, alternative = "greater", mu = -margin)
  print(fit)
  cat(sprintf("  reject H0 (delta <= -margin)? p = %.4f\n", fit$p.value))

  cat("\n=== NI on proportions (82/100 vs 80/100, margin = 0.10) ===\n")
  if (requireNamespace("DescTools", quietly = TRUE)) {
    print(DescTools::BinomDiffCI(82, 100, 80, 100, sides = "left"))
  } else {
    # Manual Farrington-Manning would go here; simplified Wald below
    p1 <- 0.82; p2 <- 0.80
    se <- sqrt(p1 * (1 - p1) / 100 + p2 * (1 - p2) / 100)
    z <- (p1 - p2 + 0.10) / se
    cat(sprintf("  Wald z = %.3f, one-sided p = %.4f\n", z, 1 - pnorm(z)))
  }
}
