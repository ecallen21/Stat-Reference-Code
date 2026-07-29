# Linear Mixed Models via REML (Reference §12.2 + many folded-in guides)
# Base R via lme4 (authoritative implementation).
# Run with:  Rscript linear_mixed_models.R

if (sys.nframe() == 0) {
  set.seed(11); n_cl <- 40; n_per <- 8; n <- n_cl * n_per
  cluster <- rep(1:n_cl, each = n_per)
  u <- rnorm(n_cl, 0, 0.8); x <- rnorm(n)
  y <- 1.0 + 0.5 * x + u[cluster] + rnorm(n, 0, 0.4)
  df <- data.frame(y = y, x = x, cluster = factor(cluster))
  if (requireNamespace("lme4", quietly = TRUE)) {
    cat("=== lme4::lmer (REML, random intercept) ===\n")
    fit <- lme4::lmer(y ~ x + (1 | cluster), data = df, REML = TRUE)
    print(summary(fit))
    cat("\n=== ICC ===\n")
    vc <- as.data.frame(lme4::VarCorr(fit))
    icc <- vc$vcov[1] / sum(vc$vcov)
    cat("ICC =", round(icc, 4), "\n")
    cat("\n=== BLUPs (first 5) ===\n")
    print(head(lme4::ranef(fit)$cluster, 5))
  } else {
    cat("=== nlme::lme fallback ===\n")
    if (requireNamespace("nlme", quietly = TRUE)) {
      print(summary(nlme::lme(y ~ x, random = ~ 1 | cluster, data = df, method = "REML")))
    }
  }
}
