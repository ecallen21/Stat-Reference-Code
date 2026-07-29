# Kenward-Roger + Satterthwaite denominator df for LMM (Reference §12.17)
# Base R via pbkrtest::KRmodcomp (true KR) or lmerTest::contest (Satterthwaite).
# Run with:  Rscript kenward_roger.R

if (sys.nframe() == 0) {
  set.seed(47); n_c <- 20; n_per <- 5; n <- n_c * n_per
  cluster <- rep(1:n_c, each = n_per); u <- rnorm(n_c, 0, 0.8); x <- rnorm(n)
  y <- 1.0 + 0.4 * x + u[cluster] + rnorm(n, 0, 0.5)
  df <- data.frame(y = y, x = x, cluster = factor(cluster))
  if (requireNamespace("lme4", quietly = TRUE)) {
    fit <- lme4::lmer(y ~ x + (1 | cluster), data = df, REML = TRUE)
    if (requireNamespace("lmerTest", quietly = TRUE)) {
      cat("=== lmerTest::contest (Satterthwaite) ===\n")
      print(summary(lmerTest::lmer(y ~ x + (1 | cluster), data = df)))
    }
    if (requireNamespace("pbkrtest", quietly = TRUE)) {
      cat("\n=== pbkrtest::KRmodcomp (true KR) ===\n")
      fit0 <- lme4::lmer(y ~ 1 + (1 | cluster), data = df, REML = TRUE)
      print(pbkrtest::KRmodcomp(fit, fit0))
    }
  }
}
